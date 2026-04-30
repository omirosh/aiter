# The kernels in this file are adapted from vLLM:
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_unified_attention.py
import triton
import torch
from aiter.ops.triton.utils.device_info import get_num_sms
import math
from aiter.ops.triton._triton_kernels.attention.unified_attention import (
    kernel_unified_attention_2d,
    kernel_unified_attention_3d,
    reduce_segments,
)

from aiter.ops.triton._triton_kernels.flash_attn_triton_amd.utils import get_arch


def select_2d_config(
    block_size,
    head_size,
    sliding_window,
    all_decode,
    max_seqlen_q,
    max_seqlen_k,
    num_queries_per_kv,
    num_2d_prgms,
):
    arch = get_arch()

    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )

    TILE_SIZE = 32 if arch.name == "gfx1201" else 16 if arch.is_rdna else 64
    waves_per_eu = 8 if arch.name == "gfx1151" else 6 if arch.is_rdna else 2

    max_num_stages_2d = 2 if head_size > 128 else 4

    # base prefill, for short cases
    if not all_decode:
        num_stages_2d, num_warps = 1, 2
    # pure decode config
    else:
        # to not have masking when loading KV
        TILE_SIZE = min(64, triton.next_power_of_2(block_size))
        if arch.is_rdna:
            num_stages_2d, num_warps = 1, 4
        else:
            num_stages_2d, num_warps = 3, 2

    # large prefill config
    if max_seqlen_q >= 256:
        BLOCK_M = 64 if arch.is_rdna else 128
        num_stages_2d, num_warps = 1, 4

    BLOCK_Q = BLOCK_M // num_queries_per_kv
    num_stages_2d = min(max_num_stages_2d, num_stages_2d)

    return {
        "BLOCK_M": BLOCK_M,
        "BLOCK_Q": BLOCK_Q,
        "TILE_SIZE": TILE_SIZE,
        "num_warps": num_warps,
        "num_stages": num_stages_2d,
        "waves_per_eu": waves_per_eu,
    }


def select_3d_config(
    head_size, block_size, element_size, max_seqlen_k, target_num_prgms, num_2d_prgms
):
    reduce_num_warps = 2
    attn_warps = 2
    TILE_SIZE = min(64, triton.next_power_of_2(block_size))
    # MAX_SEGMENTS = min(128, math.ceil(max_seqlen_k / TILE_SIZE))
    num_segments = math.ceil(target_num_prgms / num_2d_prgms)
    num_segments = triton.next_power_of_2(num_segments)
    num_segments = min(num_segments, 128)
    MIN_SEGMENTS = 16 if TILE_SIZE <= 16 else 8
    num_segments = max(num_segments, MIN_SEGMENTS)
    if num_segments == MIN_SEGMENTS:
        reduce_num_warps = 1
    attn_config = {
        "TILE_SIZE": TILE_SIZE,
        "NUM_SEGMENTS_PER_SEQ": num_segments,
        "num_warps": attn_warps,
        "num_stages": 1,
        "waves_per_eu": 2,
    }
    reduce_config = {
        "TILE_SIZE": TILE_SIZE,
        "NUM_SEGMENTS_PER_SEQ": num_segments,
        "num_warps": reduce_num_warps,
        "num_stages": 1,
        "waves_per_eu": 2,
    }
    return attn_config, reduce_config


def use_2d_kernel(
    head_size,
    sliding_window,
    all_decode,
    max_seqlen_q,
    max_seqlen_k,
    target_num_prgms,
    num_2d_prgms,
):
    return (
        (sliding_window > 0)
        or (max_seqlen_k <= 512)
        or (num_2d_prgms > target_num_prgms)
    )


def unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    # Optional tensor for sinks
    sinks=None,
    # KV-cache memory layout. "NHD" (default) treats `k` and `v` as the
    # standard `[num_blocks, page_size, num_kv_heads, head_size]` paged
    # cache with strides taken from the tensors. "SHUFFLE" reinterprets
    # them as the AMD MFMA-friendly layout
    #   K: [num_blocks, num_kv_heads, head_size // x, page_size, x]
    #   V: [num_blocks, num_kv_heads, page_size // x, head_size, x]
    # where `x = 16 // kv_cache.element_size()`. The kernel computes
    # SHUFFLE addresses internally; the strides of `k`/`v` are ignored
    # in that path. This mirrors `cp_mha_gather_cache_kernel`'s SHUFFLE
    # arm and is intended for vLLM's `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT`
    # path.
    kv_cache_layout="NHD",
):
    assert causal, "Only causal attention is supported"
    assert kv_cache_layout in ("NHD", "SHUFFLE"), (
        f"unified_attention: kv_cache_layout must be 'NHD' or 'SHUFFLE', "
        f"got {kv_cache_layout!r}"
    )
    # `x` is computed from kv element size and only used by the SHUFFLE
    # arm of the kernels. For NHD it's a no-op constant.
    if kv_cache_layout == "SHUFFLE":
        x_kv = 16 // k.element_size()
        # Sanity: SHUFFLE V layout requires page_size to be a multiple
        # of x (so that the (slot // x, slot % x) split is exact). Same
        # constraint as the SHUFFLE write/gather kernels.
        block_size_kv = v.shape[1]
        assert block_size_kv % x_kv == 0, (
            f"SHUFFLE layout requires page_size ({block_size_kv}) "
            f"divisible by x={x_kv}"
        )
        assert k.shape[3] % x_kv == 0, (
            f"SHUFFLE layout requires head_size ({k.shape[3]}) divisible "
            f"by x={x_kv}"
        )
    else:
        x_kv = 1

    if sinks is not None:
        assert sinks.shape[0] == q.shape[1], "Sinks must be num_query_heads size"

    use_alibi_slopes = alibi_slopes is not None
    use_qq_bias = qq_bias is not None
    SLIDING_WINDOW = 1 + window_size[0]

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    assert BLOCK_Q >= 1
    # Ideally we would launch with kernel with:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)] blocks.
    # However, it is slow to realize the query_lens on cpu.
    # Instead we use upper-bound:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)]
    #   <= \sum_i[floor(query_len[i] / BLOCK_Q) + 1]
    #    = \sum_i[floor(query_len[i] / BLOCK_Q)] + num_seqs
    #   <= floor(\sum_i(query_len[i]) / BLOCK_Q) + num_seqs
    #    = floor(q.shape[0] / BLOCK_Q) + num_seqs
    cu_count = get_num_sms()
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
    target_num_prgms = cu_count * 4
    num_2d_prgms = total_num_q_blocks * num_kv_heads
    ALL_DECODE = int(max_seqlen_q) == 1
    # if batch contains a prefill
    # SHUFFLE-layout diagnostic: the 2d kernel's SHUFFLE arm trips a
    # GPU memory-access fault on lm_eval gsm8k (max_seqlen_k <= 512
    # routes through use_2d_kernel(...) = True; bench-serve at
    # ISL=1000 always landed on the 3d kernel, which is clean). The
    # in-kernel address arithmetic and the partial-tile masking
    # behavior have both been audited without finding a smoking gun.
    # As a *temporary* probe, force SHUFFLE+verify off the 2d kernel
    # entirely so we can localize the fault to the 2d arm vs. some
    # other kernel-selection-independent issue. If lm_eval clears
    # under this branch, the bug is 2d-arm-specific. If it still
    # faults, it's elsewhere (drafter dispatch, async-spec, etc.).
    force_3d = kv_cache_layout == "SHUFFLE"
    if not force_3d and use_2d_kernel(
        head_size,
        SLIDING_WINDOW,
        ALL_DECODE,
        max_seqlen_q,
        max_seqlen_k,
        target_num_prgms,
        num_2d_prgms,
    ):
        config = select_2d_config(
            block_size,
            head_size,
            SLIDING_WINDOW,
            ALL_DECODE,
            max_seqlen_q,
            max_seqlen_k,
            num_queries_per_kv,
            num_2d_prgms,
        )
        assert config["BLOCK_Q"] >= 1
        total_num_q_blocks = q.shape[0] // config["BLOCK_Q"] + num_seqs

        kernel_unified_attention_2d[
            (
                num_kv_heads,
                total_num_q_blocks,
            )
        ](
            output_ptr=out,
            query_ptr=q,
            key_cache_ptr=k,
            value_cache_ptr=v,
            sink_ptr=sinks,
            block_tables_ptr=block_table,
            seq_lens_ptr=seqused_k,
            alibi_slopes_ptr=alibi_slopes,
            qq_bias_ptr=qq_bias,
            scale=softmax_scale,
            q_descale_ptr=q_descale,
            k_descale_ptr=k_descale,
            v_descale_ptr=v_descale,
            out_scale_ptr=output_scale,
            softcap=softcap,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            block_table_stride=block_table.stride(0),
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
            BLOCK_SIZE=block_size,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            USE_ALIBI_SLOPES=use_alibi_slopes,
            USE_QQ_BIAS=use_qq_bias,
            USE_SOFTCAP=(softcap > 0),
            USE_SINKS=(sinks is not None),
            SLIDING_WINDOW=SLIDING_WINDOW,
            stride_k_cache_0=k.stride(0),
            stride_k_cache_1=k.stride(1),
            stride_k_cache_2=k.stride(2),
            stride_k_cache_3=k.stride(3),
            stride_v_cache_0=v.stride(0),
            stride_v_cache_1=v.stride(1),
            stride_v_cache_2=v.stride(2),
            stride_v_cache_3=v.stride(3),
            query_start_len_ptr=cu_seqlens_q,
            num_seqs=num_seqs,
            ALL_DECODE=ALL_DECODE,
            KV_CACHE_LAYOUT=kv_cache_layout,
            X=x_kv,
            **config,
        )

    else:
        attn_config, reduce_config = select_3d_config(
            head_size,
            block_size,
            q.element_size(),
            max_seqlen_k,
            target_num_prgms,
            num_2d_prgms,
        )
        NUM_SEGMENTS = attn_config["NUM_SEGMENTS_PER_SEQ"]
        segm_output = torch.empty(
            q.shape[0],
            num_query_heads,
            NUM_SEGMENTS,
            triton.next_power_of_2(head_size),
            dtype=torch.float32,
            device=q.device,
        )
        segm_max = torch.empty(
            q.shape[0],
            num_query_heads,
            NUM_SEGMENTS,
            dtype=torch.float32,
            device=q.device,
        )
        segm_expsum = torch.empty(
            q.shape[0],
            num_query_heads,
            NUM_SEGMENTS,
            dtype=torch.float32,
            device=q.device,
        )

        kernel_unified_attention_3d[(total_num_q_blocks, num_kv_heads, NUM_SEGMENTS)](
            segm_output_ptr=segm_output,
            segm_max_ptr=segm_max,
            segm_expsum_ptr=segm_expsum,
            query_ptr=q,
            key_cache_ptr=k,
            value_cache_ptr=v,
            sink_ptr=sinks,
            block_tables_ptr=block_table,
            seq_lens_ptr=seqused_k,
            alibi_slopes_ptr=alibi_slopes,
            qq_bias_ptr=qq_bias,
            scale=softmax_scale,
            q_descale_ptr=q_descale,
            k_descale_ptr=k_descale,
            v_descale_ptr=v_descale,
            softcap=softcap,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            block_table_stride=block_table.stride(0),
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
            BLOCK_SIZE=block_size,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            USE_ALIBI_SLOPES=use_alibi_slopes,
            USE_QQ_BIAS=use_qq_bias,
            USE_SOFTCAP=(softcap > 0),
            USE_SINKS=(sinks is not None),
            SLIDING_WINDOW=SLIDING_WINDOW,
            stride_k_cache_0=k.stride(0),
            stride_k_cache_1=k.stride(1),
            stride_k_cache_2=k.stride(2),
            stride_k_cache_3=k.stride(3),
            stride_v_cache_0=v.stride(0),
            stride_v_cache_1=v.stride(1),
            stride_v_cache_2=v.stride(2),
            stride_v_cache_3=v.stride(3),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            num_seqs=num_seqs,
            BLOCK_M=BLOCK_M,
            ALL_DECODE=ALL_DECODE,
            KV_CACHE_LAYOUT=kv_cache_layout,
            X=x_kv,
            **attn_config,
        )
        reduce_segments[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=segm_output,
            segm_max_ptr=segm_max,
            segm_expsum_ptr=segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            out_scale_ptr=output_scale,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            block_table_stride=block_table.stride(0),
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            **reduce_config,
        )
