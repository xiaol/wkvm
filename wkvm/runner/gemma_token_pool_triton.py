"""Triton decode kernels for native Gemma token-pool attention."""

from __future__ import annotations

from functools import lru_cache
import os


def _triton_modules():
    import triton
    import triton.language as tl

    return triton, tl


triton, tl = _triton_modules()


@triton.jit
def _token_pool_store_kv_kernel(
    key_states,
    value_states,
    key_buffer,
    value_buffer,
    slot_mapping,
    k_stride_t: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_d: tl.constexpr,
    v_stride_t: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_d: tl.constexpr,
    kb_stride_t: tl.constexpr,
    kb_stride_h: tl.constexpr,
    kb_stride_d: tl.constexpr,
    vb_stride_t: tl.constexpr,
    vb_stride_h: tl.constexpr,
    vb_stride_d: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_elems: tl.constexpr,
):
    token_idx = tl.program_id(0)
    tile_idx = tl.program_id(1)
    offsets = tile_idx * block_elems + tl.arange(0, block_elems)
    valid = offsets < (num_heads * head_dim)
    head = offsets // head_dim
    dim = offsets % head_dim
    slot = tl.load(slot_mapping + token_idx).to(tl.int64)
    write_mask = valid & (slot >= 0)

    key_tile = tl.load(
        key_states
        + token_idx * k_stride_t
        + head * k_stride_h
        + dim * k_stride_d,
        mask=valid,
        other=0.0,
    )
    value_tile = tl.load(
        value_states
        + token_idx * v_stride_t
        + head * v_stride_h
        + dim * v_stride_d,
        mask=valid,
        other=0.0,
    )
    tl.store(
        key_buffer
        + slot * kb_stride_t
        + head * kb_stride_h
        + dim * kb_stride_d,
        key_tile,
        mask=write_mask,
    )
    tl.store(
        value_buffer
        + slot * vb_stride_t
        + head * vb_stride_h
        + dim * vb_stride_d,
        value_tile,
        mask=write_mask,
    )


@triton.jit
def _token_pool_gqa_decode_kernel(
    query,
    keys,
    values,
    kv_indptr,
    kv_indices,
    output,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_t: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_t: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_d: tl.constexpr,
    v_stride_t: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_d: tl.constexpr,
    o_stride_b: tl.constexpr,
    o_stride_h: tl.constexpr,
    o_stride_d: tl.constexpr,
    scaling: tl.constexpr,
    groups: tl.constexpr,
    head_dim: tl.constexpr,
    block_d: tl.constexpr,
    block_n: tl.constexpr,
):
    row = tl.program_id(0)
    q_head = tl.program_id(1)
    kv_head = q_head // groups
    d_offsets = tl.arange(0, block_d)
    n_offsets = tl.arange(0, block_n)

    q = tl.load(
        query
        + row * q_stride_b
        + q_head * q_stride_h
        + d_offsets * q_stride_d,
        mask=d_offsets < head_dim,
        other=0.0,
    ).to(tl.float32)

    start = tl.load(kv_indptr + row)
    end = tl.load(kv_indptr + row + 1)
    seq_len = end - start

    m_i = tl.full((), -float("inf"), tl.float32)
    l_i = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((block_d,), tl.float32)

    offset = 0
    while offset < seq_len:
        valid_n = n_offsets + offset < seq_len
        token_ids = tl.load(
            kv_indices + start + offset + n_offsets,
            mask=valid_n,
            other=0,
        )
        k = tl.load(
            keys
            + token_ids[:, None] * k_stride_t
            + kv_head * k_stride_h
            + d_offsets[None, :] * k_stride_d,
            mask=valid_n[:, None] & (d_offsets[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(k * q[None, :], axis=1) * scaling
        scores = tl.where(valid_n, scores, -float("inf"))

        block_m = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, block_m)
        p = tl.exp(scores - m_new)
        alpha = tl.exp(m_i - m_new)

        v = tl.load(
            values
            + token_ids[:, None] * v_stride_t
            + kv_head * v_stride_h
            + d_offsets[None, :] * v_stride_d,
            mask=valid_n[:, None] & (d_offsets[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new
        offset += block_n

    out = acc / l_i
    tl.store(
        output + row * o_stride_b + q_head * o_stride_h + d_offsets * o_stride_d,
        out,
        mask=d_offsets < head_dim,
    )


@triton.jit
def _token_pool_gqa_decode_grouped_kernel(
    query,
    keys,
    values,
    kv_indptr,
    kv_indices,
    output,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_t: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_t: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_d: tl.constexpr,
    v_stride_t: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_d: tl.constexpr,
    o_stride_b: tl.constexpr,
    o_stride_h: tl.constexpr,
    o_stride_d: tl.constexpr,
    scaling: tl.constexpr,
    groups: tl.constexpr,
    head_dim: tl.constexpr,
    block_d: tl.constexpr,
    block_n: tl.constexpr,
    block_g: tl.constexpr,
    input_precision: tl.constexpr,
    native_dot: tl.constexpr,
):
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    group_offsets = tl.arange(0, block_g)
    d_offsets = tl.arange(0, block_d)
    n_offsets = tl.arange(0, block_n)
    valid_g = group_offsets < groups
    q_heads = kv_head * groups + group_offsets

    q = tl.load(
        query
        + row * q_stride_b
        + q_heads[:, None] * q_stride_h
        + d_offsets[None, :] * q_stride_d,
        mask=valid_g[:, None] & (d_offsets[None, :] < head_dim),
        other=0.0,
    )
    if not native_dot:
        q = q.to(tl.float32)

    start = tl.load(kv_indptr + row)
    end = tl.load(kv_indptr + row + 1)
    seq_len = end - start

    m_i = tl.full((block_g,), -float("inf"), tl.float32)
    l_i = tl.zeros((block_g,), tl.float32)
    acc = tl.zeros((block_g, block_d), tl.float32)

    offset = 0
    while offset < seq_len:
        valid_n = n_offsets + offset < seq_len
        token_ids = tl.load(
            kv_indices + start + offset + n_offsets,
            mask=valid_n,
            other=0,
        )
        k = tl.load(
            keys
            + token_ids[:, None] * k_stride_t
            + kv_head * k_stride_h
            + d_offsets[None, :] * k_stride_d,
            mask=valid_n[:, None] & (d_offsets[None, :] < head_dim),
            other=0.0,
        )
        if not native_dot:
            k = k.to(tl.float32)
        scores = tl.dot(k, tl.trans(q), input_precision=input_precision).to(tl.float32) * scaling
        scores = tl.where(valid_n[:, None] & valid_g[None, :], scores, -float("inf"))

        block_m = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, block_m)
        p = tl.exp(scores - m_new[None, :])
        alpha = tl.exp(m_i - m_new)

        v = tl.load(
            values
            + token_ids[:, None] * v_stride_t
            + kv_head * v_stride_h
            + d_offsets[None, :] * v_stride_d,
            mask=valid_n[:, None] & (d_offsets[None, :] < head_dim),
            other=0.0,
        )
        if not native_dot:
            v = v.to(tl.float32)
            p_v = p
        else:
            p_v = p.to(v.dtype)
        acc = acc * alpha[:, None] + tl.dot(
            tl.trans(p_v),
            v,
            input_precision=input_precision,
        ).to(tl.float32)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new
        offset += block_n

    out = acc / l_i[:, None]
    tl.store(
        output
        + row * o_stride_b
        + q_heads[:, None] * o_stride_h
        + d_offsets[None, :] * o_stride_d,
        out,
        mask=valid_g[:, None] & (d_offsets[None, :] < head_dim),
    )


@triton.jit
def _token_pool_gqa_decode_grouped_split_stage1_kernel(
    query,
    keys,
    values,
    kv_indptr,
    kv_indices,
    partial_m,
    partial_l,
    partial_acc,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_t: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_t: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_d: tl.constexpr,
    v_stride_t: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_d: tl.constexpr,
    pm_stride_b: tl.constexpr,
    pm_stride_h: tl.constexpr,
    pm_stride_s: tl.constexpr,
    pm_stride_g: tl.constexpr,
    pl_stride_b: tl.constexpr,
    pl_stride_h: tl.constexpr,
    pl_stride_s: tl.constexpr,
    pl_stride_g: tl.constexpr,
    pa_stride_b: tl.constexpr,
    pa_stride_h: tl.constexpr,
    pa_stride_s: tl.constexpr,
    pa_stride_g: tl.constexpr,
    pa_stride_d: tl.constexpr,
    scaling: tl.constexpr,
    groups: tl.constexpr,
    head_dim: tl.constexpr,
    block_d: tl.constexpr,
    block_n: tl.constexpr,
    block_g: tl.constexpr,
    split_size: tl.constexpr,
    input_precision: tl.constexpr,
    native_dot: tl.constexpr,
):
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    group_offsets = tl.arange(0, block_g)
    d_offsets = tl.arange(0, block_d)
    n_offsets = tl.arange(0, block_n)
    valid_g = group_offsets < groups
    q_heads = kv_head * groups + group_offsets

    q = tl.load(
        query
        + row * q_stride_b
        + q_heads[:, None] * q_stride_h
        + d_offsets[None, :] * q_stride_d,
        mask=valid_g[:, None] & (d_offsets[None, :] < head_dim),
        other=0.0,
    )
    if not native_dot:
        q = q.to(tl.float32)

    start = tl.load(kv_indptr + row)
    end = tl.load(kv_indptr + row + 1)
    seq_len = end - start
    split_start = split_id * split_size
    split_end = tl.minimum(seq_len, split_start + split_size)
    split_len = tl.maximum(split_end - split_start, 0)

    m_i = tl.full((block_g,), -float("inf"), tl.float32)
    l_i = tl.zeros((block_g,), tl.float32)
    acc = tl.zeros((block_g, block_d), tl.float32)

    offset = 0
    while offset < split_len:
        valid_n = n_offsets + offset < split_len
        token_ids = tl.load(
            kv_indices + start + split_start + offset + n_offsets,
            mask=valid_n,
            other=0,
        )
        k = tl.load(
            keys
            + token_ids[:, None] * k_stride_t
            + kv_head * k_stride_h
            + d_offsets[None, :] * k_stride_d,
            mask=valid_n[:, None] & (d_offsets[None, :] < head_dim),
            other=0.0,
        )
        if not native_dot:
            k = k.to(tl.float32)
        scores = tl.dot(k, tl.trans(q), input_precision=input_precision).to(tl.float32) * scaling
        scores = tl.where(valid_n[:, None] & valid_g[None, :], scores, -float("inf"))

        block_m = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, block_m)
        p = tl.exp(scores - m_new[None, :])
        alpha = tl.exp(m_i - m_new)

        v = tl.load(
            values
            + token_ids[:, None] * v_stride_t
            + kv_head * v_stride_h
            + d_offsets[None, :] * v_stride_d,
            mask=valid_n[:, None] & (d_offsets[None, :] < head_dim),
            other=0.0,
        )
        if not native_dot:
            v = v.to(tl.float32)
            p_v = p
        else:
            p_v = p.to(v.dtype)
        acc = acc * alpha[:, None] + tl.dot(
            tl.trans(p_v),
            v,
            input_precision=input_precision,
        ).to(tl.float32)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new
        offset += block_n

    tl.store(
        partial_m
        + row * pm_stride_b
        + kv_head * pm_stride_h
        + split_id * pm_stride_s
        + group_offsets * pm_stride_g,
        m_i,
        mask=valid_g,
    )
    tl.store(
        partial_l
        + row * pl_stride_b
        + kv_head * pl_stride_h
        + split_id * pl_stride_s
        + group_offsets * pl_stride_g,
        l_i,
        mask=valid_g,
    )
    tl.store(
        partial_acc
        + row * pa_stride_b
        + kv_head * pa_stride_h
        + split_id * pa_stride_s
        + group_offsets[:, None] * pa_stride_g
        + d_offsets[None, :] * pa_stride_d,
        acc,
        mask=valid_g[:, None] & (d_offsets[None, :] < head_dim),
    )


@triton.jit
def _token_pool_gqa_decode_grouped_split_stage2_kernel(
    partial_m,
    partial_l,
    partial_acc,
    output,
    seq_lens,
    pm_stride_b: tl.constexpr,
    pm_stride_h: tl.constexpr,
    pm_stride_s: tl.constexpr,
    pm_stride_g: tl.constexpr,
    pl_stride_b: tl.constexpr,
    pl_stride_h: tl.constexpr,
    pl_stride_s: tl.constexpr,
    pl_stride_g: tl.constexpr,
    pa_stride_b: tl.constexpr,
    pa_stride_h: tl.constexpr,
    pa_stride_s: tl.constexpr,
    pa_stride_g: tl.constexpr,
    pa_stride_d: tl.constexpr,
    o_stride_b: tl.constexpr,
    o_stride_h: tl.constexpr,
    o_stride_d: tl.constexpr,
    groups: tl.constexpr,
    head_dim: tl.constexpr,
    block_d: tl.constexpr,
    block_g: tl.constexpr,
    max_splits: tl.constexpr,
    split_size: tl.constexpr,
    has_seq_lens: tl.constexpr,
):
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    group_offsets = tl.arange(0, block_g)
    d_offsets = tl.arange(0, block_d)
    valid_g = group_offsets < groups
    q_heads = kv_head * groups + group_offsets

    m_i = tl.full((block_g,), -float("inf"), tl.float32)
    l_i = tl.zeros((block_g,), tl.float32)
    acc = tl.zeros((block_g, block_d), tl.float32)

    row_splits = max_splits
    if has_seq_lens:
        row_seq_len = tl.load(seq_lens + row)
        row_splits = tl.minimum(tl.cdiv(row_seq_len, split_size), max_splits)

    split_id = 0
    while split_id < row_splits:
        m_s = tl.load(
            partial_m
            + row * pm_stride_b
            + kv_head * pm_stride_h
            + split_id * pm_stride_s
            + group_offsets * pm_stride_g,
            mask=valid_g,
            other=-float("inf"),
        ).to(tl.float32)
        l_s = tl.load(
            partial_l
            + row * pl_stride_b
            + kv_head * pl_stride_h
            + split_id * pl_stride_s
            + group_offsets * pl_stride_g,
            mask=valid_g,
            other=0.0,
        ).to(tl.float32)
        acc_s = tl.load(
            partial_acc
            + row * pa_stride_b
            + kv_head * pa_stride_h
            + split_id * pa_stride_s
            + group_offsets[:, None] * pa_stride_g
            + d_offsets[None, :] * pa_stride_d,
            mask=valid_g[:, None] & (d_offsets[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)

        active = valid_g & (l_s > 0.0)
        m_new = tl.maximum(m_i, m_s)
        alpha = tl.where(active, tl.exp(m_i - m_new), 1.0)
        beta = tl.where(active, tl.exp(m_s - m_new), 0.0)
        acc = acc * alpha[:, None] + acc_s * beta[:, None]
        l_i = l_i * alpha + l_s * beta
        m_i = tl.where(active, m_new, m_i)
        split_id += 1

    out = acc / l_i[:, None]
    tl.store(
        output
        + row * o_stride_b
        + q_heads[:, None] * o_stride_h
        + d_offsets[None, :] * o_stride_d,
        out,
        mask=valid_g[:, None] & (d_offsets[None, :] < head_dim),
    )


@triton.jit
def _token_pool_paged_gqa_decode_grouped_kernel(
    query,
    keys,
    values,
    block_tables,
    block_table_lens,
    selected_start_positions,
    seq_lens,
    output,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_t: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_t: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_d: tl.constexpr,
    v_stride_t: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_d: tl.constexpr,
    bt_stride_b: tl.constexpr,
    bt_stride_m: tl.constexpr,
    o_stride_b: tl.constexpr,
    o_stride_h: tl.constexpr,
    o_stride_d: tl.constexpr,
    scaling: tl.constexpr,
    groups: tl.constexpr,
    head_dim: tl.constexpr,
    block_d: tl.constexpr,
    block_n: tl.constexpr,
    block_g: tl.constexpr,
    page_size: tl.constexpr,
    input_precision: tl.constexpr,
    native_dot: tl.constexpr,
):
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    group_offsets = tl.arange(0, block_g)
    d_offsets = tl.arange(0, block_d)
    n_offsets = tl.arange(0, block_n)
    valid_g = group_offsets < groups
    q_heads = kv_head * groups + group_offsets

    q = tl.load(
        query
        + row * q_stride_b
        + q_heads[:, None] * q_stride_h
        + d_offsets[None, :] * q_stride_d,
        mask=valid_g[:, None] & (d_offsets[None, :] < head_dim),
        other=0.0,
    )
    if not native_dot:
        q = q.to(tl.float32)

    seq_len = tl.load(seq_lens + row)
    selected_start = tl.load(selected_start_positions + row)
    selected_start_block = selected_start // page_size
    block_table_len = tl.load(block_table_lens + row)

    m_i = tl.full((block_g,), -float("inf"), tl.float32)
    l_i = tl.zeros((block_g,), tl.float32)
    acc = tl.zeros((block_g, block_d), tl.float32)

    offset = 0
    while offset < seq_len:
        row_offsets = offset + n_offsets
        valid_n = row_offsets < seq_len
        logical_pos = selected_start + row_offsets
        logical_block = logical_pos // page_size
        block_table_offsets = logical_block - selected_start_block
        valid_block = block_table_offsets < block_table_len
        physical_blocks = tl.load(
            block_tables + row * bt_stride_b + block_table_offsets * bt_stride_m,
            mask=valid_n & valid_block,
            other=0,
        )
        token_ids = physical_blocks * page_size + (logical_pos % page_size)

        k = tl.load(
            keys
            + token_ids[:, None] * k_stride_t
            + kv_head * k_stride_h
            + d_offsets[None, :] * k_stride_d,
            mask=valid_n[:, None] & valid_block[:, None] & (d_offsets[None, :] < head_dim),
            other=0.0,
        )
        if not native_dot:
            k = k.to(tl.float32)
        scores = tl.dot(k, tl.trans(q), input_precision=input_precision).to(tl.float32) * scaling
        scores = tl.where(
            valid_n[:, None] & valid_block[:, None] & valid_g[None, :],
            scores,
            -float("inf"),
        )

        block_m = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, block_m)
        p = tl.exp(scores - m_new[None, :])
        alpha = tl.exp(m_i - m_new)

        v = tl.load(
            values
            + token_ids[:, None] * v_stride_t
            + kv_head * v_stride_h
            + d_offsets[None, :] * v_stride_d,
            mask=valid_n[:, None] & valid_block[:, None] & (d_offsets[None, :] < head_dim),
            other=0.0,
        )
        if not native_dot:
            v = v.to(tl.float32)
            p_v = p
        else:
            p_v = p.to(v.dtype)
        acc = acc * alpha[:, None] + tl.dot(
            tl.trans(p_v),
            v,
            input_precision=input_precision,
        ).to(tl.float32)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new
        offset += block_n

    out = acc / l_i[:, None]
    tl.store(
        output
        + row * o_stride_b
        + q_heads[:, None] * o_stride_h
        + d_offsets[None, :] * o_stride_d,
        out,
        mask=valid_g[:, None] & (d_offsets[None, :] < head_dim),
    )


@triton.jit
def _token_pool_paged_request_table_gqa_decode_grouped_kernel(
    query,
    keys,
    values,
    req_pool_indices,
    request_block_tables,
    block_table_lens,
    selected_start_positions,
    seq_lens,
    output,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_t: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_t: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_d: tl.constexpr,
    v_stride_t: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_d: tl.constexpr,
    rbt_stride_r: tl.constexpr,
    rbt_stride_m: tl.constexpr,
    o_stride_b: tl.constexpr,
    o_stride_h: tl.constexpr,
    o_stride_d: tl.constexpr,
    scaling: tl.constexpr,
    groups: tl.constexpr,
    head_dim: tl.constexpr,
    block_d: tl.constexpr,
    block_n: tl.constexpr,
    block_g: tl.constexpr,
    page_size: tl.constexpr,
    input_precision: tl.constexpr,
    native_dot: tl.constexpr,
):
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    group_offsets = tl.arange(0, block_g)
    d_offsets = tl.arange(0, block_d)
    n_offsets = tl.arange(0, block_n)
    valid_g = group_offsets < groups
    q_heads = kv_head * groups + group_offsets

    q = tl.load(
        query
        + row * q_stride_b
        + q_heads[:, None] * q_stride_h
        + d_offsets[None, :] * q_stride_d,
        mask=valid_g[:, None] & (d_offsets[None, :] < head_dim),
        other=0.0,
    )
    if not native_dot:
        q = q.to(tl.float32)

    req_slot = tl.load(req_pool_indices + row)
    seq_len = tl.load(seq_lens + row)
    selected_start = tl.load(selected_start_positions + row)
    selected_start_block = selected_start // page_size
    block_table_len = tl.load(block_table_lens + row)

    m_i = tl.full((block_g,), -float("inf"), tl.float32)
    l_i = tl.zeros((block_g,), tl.float32)
    acc = tl.zeros((block_g, block_d), tl.float32)

    offset = 0
    while offset < seq_len:
        row_offsets = offset + n_offsets
        valid_n = row_offsets < seq_len
        logical_pos = selected_start + row_offsets
        logical_block = logical_pos // page_size
        block_table_offsets = logical_block - selected_start_block
        valid_block = block_table_offsets < block_table_len
        physical_blocks = tl.load(
            request_block_tables
            + req_slot * rbt_stride_r
            + logical_block * rbt_stride_m,
            mask=valid_n & valid_block,
            other=0,
        )
        token_ids = physical_blocks * page_size + (logical_pos % page_size)

        k = tl.load(
            keys
            + token_ids[:, None] * k_stride_t
            + kv_head * k_stride_h
            + d_offsets[None, :] * k_stride_d,
            mask=valid_n[:, None] & valid_block[:, None] & (d_offsets[None, :] < head_dim),
            other=0.0,
        )
        if not native_dot:
            k = k.to(tl.float32)
        scores = tl.dot(k, tl.trans(q), input_precision=input_precision).to(tl.float32) * scaling
        scores = tl.where(
            valid_n[:, None] & valid_block[:, None] & valid_g[None, :],
            scores,
            -float("inf"),
        )

        block_m = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, block_m)
        p = tl.exp(scores - m_new[None, :])
        alpha = tl.exp(m_i - m_new)

        v = tl.load(
            values
            + token_ids[:, None] * v_stride_t
            + kv_head * v_stride_h
            + d_offsets[None, :] * v_stride_d,
            mask=valid_n[:, None] & valid_block[:, None] & (d_offsets[None, :] < head_dim),
            other=0.0,
        )
        if not native_dot:
            v = v.to(tl.float32)
            p_v = p
        else:
            p_v = p.to(v.dtype)
        acc = acc * alpha[:, None] + tl.dot(
            tl.trans(p_v),
            v,
            input_precision=input_precision,
        ).to(tl.float32)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new
        offset += block_n

    out = acc / l_i[:, None]
    tl.store(
        output
        + row * o_stride_b
        + q_heads[:, None] * o_stride_h
        + d_offsets[None, :] * o_stride_d,
        out,
        mask=valid_g[:, None] & (d_offsets[None, :] < head_dim),
    )


@triton.jit
def _token_pool_paged_gqa_decode_grouped_split_stage1_kernel(
    query,
    keys,
    values,
    block_tables,
    block_table_lens,
    selected_start_positions,
    seq_lens,
    partial_m,
    partial_l,
    partial_acc,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_t: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_t: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_d: tl.constexpr,
    v_stride_t: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_d: tl.constexpr,
    bt_stride_b: tl.constexpr,
    bt_stride_m: tl.constexpr,
    pm_stride_b: tl.constexpr,
    pm_stride_h: tl.constexpr,
    pm_stride_s: tl.constexpr,
    pm_stride_g: tl.constexpr,
    pl_stride_b: tl.constexpr,
    pl_stride_h: tl.constexpr,
    pl_stride_s: tl.constexpr,
    pl_stride_g: tl.constexpr,
    pa_stride_b: tl.constexpr,
    pa_stride_h: tl.constexpr,
    pa_stride_s: tl.constexpr,
    pa_stride_g: tl.constexpr,
    pa_stride_d: tl.constexpr,
    scaling: tl.constexpr,
    groups: tl.constexpr,
    head_dim: tl.constexpr,
    block_d: tl.constexpr,
    block_n: tl.constexpr,
    block_g: tl.constexpr,
    page_size: tl.constexpr,
    split_size: tl.constexpr,
    input_precision: tl.constexpr,
    native_dot: tl.constexpr,
):
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    group_offsets = tl.arange(0, block_g)
    d_offsets = tl.arange(0, block_d)
    n_offsets = tl.arange(0, block_n)
    valid_g = group_offsets < groups
    q_heads = kv_head * groups + group_offsets

    q = tl.load(
        query
        + row * q_stride_b
        + q_heads[:, None] * q_stride_h
        + d_offsets[None, :] * q_stride_d,
        mask=valid_g[:, None] & (d_offsets[None, :] < head_dim),
        other=0.0,
    )
    if not native_dot:
        q = q.to(tl.float32)

    seq_len = tl.load(seq_lens + row)
    selected_start = tl.load(selected_start_positions + row)
    selected_start_block = selected_start // page_size
    block_table_len = tl.load(block_table_lens + row)
    split_start = split_id * split_size
    split_end = tl.minimum(seq_len, split_start + split_size)
    split_len = tl.maximum(split_end - split_start, 0)

    m_i = tl.full((block_g,), -float("inf"), tl.float32)
    l_i = tl.zeros((block_g,), tl.float32)
    acc = tl.zeros((block_g, block_d), tl.float32)

    offset = 0
    while offset < split_len:
        row_offsets = split_start + offset + n_offsets
        valid_n = offset + n_offsets < split_len
        logical_pos = selected_start + row_offsets
        logical_block = logical_pos // page_size
        block_table_offsets = logical_block - selected_start_block
        valid_block = (block_table_offsets >= 0) & (block_table_offsets < block_table_len)
        physical_blocks = tl.load(
            block_tables + row * bt_stride_b + block_table_offsets * bt_stride_m,
            mask=valid_n & valid_block,
            other=0,
        )
        token_ids = physical_blocks * page_size + (logical_pos % page_size)

        k = tl.load(
            keys
            + token_ids[:, None] * k_stride_t
            + kv_head * k_stride_h
            + d_offsets[None, :] * k_stride_d,
            mask=valid_n[:, None] & valid_block[:, None] & (d_offsets[None, :] < head_dim),
            other=0.0,
        )
        if not native_dot:
            k = k.to(tl.float32)
        scores = tl.dot(k, tl.trans(q), input_precision=input_precision).to(tl.float32) * scaling
        scores = tl.where(
            valid_n[:, None] & valid_block[:, None] & valid_g[None, :],
            scores,
            -float("inf"),
        )

        block_m = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, block_m)
        p = tl.exp(scores - m_new[None, :])
        alpha = tl.exp(m_i - m_new)

        v = tl.load(
            values
            + token_ids[:, None] * v_stride_t
            + kv_head * v_stride_h
            + d_offsets[None, :] * v_stride_d,
            mask=valid_n[:, None] & valid_block[:, None] & (d_offsets[None, :] < head_dim),
            other=0.0,
        )
        if not native_dot:
            v = v.to(tl.float32)
            p_v = p
        else:
            p_v = p.to(v.dtype)
        acc = acc * alpha[:, None] + tl.dot(
            tl.trans(p_v),
            v,
            input_precision=input_precision,
        ).to(tl.float32)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new
        offset += block_n

    tl.store(
        partial_m
        + row * pm_stride_b
        + kv_head * pm_stride_h
        + split_id * pm_stride_s
        + group_offsets * pm_stride_g,
        m_i,
        mask=valid_g,
    )
    tl.store(
        partial_l
        + row * pl_stride_b
        + kv_head * pl_stride_h
        + split_id * pl_stride_s
        + group_offsets * pl_stride_g,
        l_i,
        mask=valid_g,
    )
    tl.store(
        partial_acc
        + row * pa_stride_b
        + kv_head * pa_stride_h
        + split_id * pa_stride_s
        + group_offsets[:, None] * pa_stride_g
        + d_offsets[None, :] * pa_stride_d,
        acc,
        mask=valid_g[:, None] & (d_offsets[None, :] < head_dim),
    )


@triton.jit
def _dense_padded_gqa_decode_grouped_kernel(
    query,
    keys,
    values,
    attention_mask,
    output,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_t: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_t: tl.constexpr,
    k_stride_d: tl.constexpr,
    v_stride_b: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_t: tl.constexpr,
    v_stride_d: tl.constexpr,
    m_stride_b: tl.constexpr,
    m_stride_t: tl.constexpr,
    o_stride_b: tl.constexpr,
    o_stride_h: tl.constexpr,
    o_stride_d: tl.constexpr,
    scaling: tl.constexpr,
    groups: tl.constexpr,
    key_length: tl.constexpr,
    head_dim: tl.constexpr,
    block_d: tl.constexpr,
    block_n: tl.constexpr,
    block_g: tl.constexpr,
    has_mask: tl.constexpr,
    input_precision: tl.constexpr,
    native_dot: tl.constexpr,
):
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    group_offsets = tl.arange(0, block_g)
    d_offsets = tl.arange(0, block_d)
    n_offsets = tl.arange(0, block_n)
    valid_g = group_offsets < groups
    q_heads = kv_head * groups + group_offsets

    q = tl.load(
        query
        + row * q_stride_b
        + q_heads[:, None] * q_stride_h
        + d_offsets[None, :] * q_stride_d,
        mask=valid_g[:, None] & (d_offsets[None, :] < head_dim),
        other=0.0,
    )
    if not native_dot:
        q = q.to(tl.float32)

    m_i = tl.full((block_g,), -float("inf"), tl.float32)
    l_i = tl.zeros((block_g,), tl.float32)
    acc = tl.zeros((block_g, block_d), tl.float32)

    offset = 0
    while offset < key_length:
        valid_n = n_offsets + offset < key_length
        k = tl.load(
            keys
            + row * k_stride_b
            + kv_head * k_stride_h
            + (offset + n_offsets)[:, None] * k_stride_t
            + d_offsets[None, :] * k_stride_d,
            mask=valid_n[:, None] & (d_offsets[None, :] < head_dim),
            other=0.0,
        )
        if not native_dot:
            k = k.to(tl.float32)
        scores = tl.dot(k, tl.trans(q), input_precision=input_precision).to(tl.float32) * scaling
        if has_mask:
            mask_values = tl.load(
                attention_mask
                + row * m_stride_b
                + (offset + n_offsets) * m_stride_t,
                mask=valid_n,
                other=-float("inf"),
            ).to(tl.float32)
            scores += mask_values[:, None]
        scores = tl.where(valid_n[:, None] & valid_g[None, :], scores, -float("inf"))

        block_m = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, block_m)
        p = tl.exp(scores - m_new[None, :])
        alpha = tl.exp(m_i - m_new)

        v = tl.load(
            values
            + row * v_stride_b
            + kv_head * v_stride_h
            + (offset + n_offsets)[:, None] * v_stride_t
            + d_offsets[None, :] * v_stride_d,
            mask=valid_n[:, None] & (d_offsets[None, :] < head_dim),
            other=0.0,
        )
        if not native_dot:
            v = v.to(tl.float32)
            p_v = p
        else:
            p_v = p.to(v.dtype)
        acc = acc * alpha[:, None] + tl.dot(
            tl.trans(p_v),
            v,
            input_precision=input_precision,
        ).to(tl.float32)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new
        offset += block_n

    out = acc / l_i[:, None]
    tl.store(
        output
        + row * o_stride_b
        + q_heads[:, None] * o_stride_h
        + d_offsets[None, :] * o_stride_d,
        out,
        mask=valid_g[:, None] & (d_offsets[None, :] < head_dim),
    )


@lru_cache(maxsize=32)
def _block_d(head_dim: int) -> int:
    return int(triton.next_power_of_2(int(head_dim)))


@lru_cache(maxsize=16)
def _block_g(groups: int, native_dot: bool = False) -> int:
    block_g = int(triton.next_power_of_2(int(groups)))
    if native_dot:
        block_g = max(block_g, 16)
    return block_g


def _default_block_n(head_dim: int) -> int:
    head_dim = int(head_dim)
    if head_dim >= 1024:
        return 16
    return 32


def _resolve_block_n(
    head_dim: int,
    block_n: int | None,
    *,
    env_names: tuple[str, ...],
) -> int:
    if block_n is None:
        bn = None
        for env_name in env_names:
            env_block_n = os.environ.get(env_name)
            if env_block_n:
                bn = int(env_block_n)
                break
        if bn is None:
            bn = _default_block_n(head_dim)
    else:
        bn = int(block_n)
    if bn < 1:
        raise ValueError("token-pool Triton block_n must be >= 1")
    return bn


def _resolve_num_warps(
    block_d: int,
    *,
    env_names: tuple[str, ...],
) -> int:
    configured = None
    for env_name in env_names:
        env_num_warps = os.environ.get(env_name)
        if env_num_warps:
            configured = int(env_num_warps)
            break
    if configured is None:
        configured = 8 if int(block_d) > 512 else 4
    if configured not in {1, 2, 4, 8}:
        raise ValueError("token-pool Triton num_warps must be one of 1, 2, 4, or 8")
    return configured


def _resolve_store_block_elems(total_elems: int, block_elems: int | None = None) -> int:
    if block_elems is None:
        env_block = os.environ.get("WKVM_TOKEN_POOL_KV_STORE_BLOCK_ELEMS")
        if env_block:
            resolved = int(env_block)
        else:
            resolved = min(2048, int(triton.next_power_of_2(int(total_elems))))
            try:
                import torch

                if torch.cuda.is_available():
                    major, _minor = torch.cuda.get_device_capability()
                    if major < 9:
                        resolved = min(512, resolved)
            except Exception:
                resolved = min(512, resolved)
    else:
        resolved = int(block_elems)
    if resolved < 1:
        raise ValueError("token-pool KV store block_elems must be >= 1")
    return resolved


def token_pool_store_kv(
    key_states,
    value_states,
    key_buffer,
    value_buffer,
    slot_mapping,
    *,
    block_elems: int | None = None,
) -> None:
    """Store current-token K/V into flat token-pool buffers by slot mapping.

    This is the flat-slot analogue of vLLM/SGLang's reshape-and-cache helper:
    one CUDA launch stores both K and V for ``[tokens, kv_heads, head_dim]`` into
    ``[capacity, kv_heads, head_dim]`` buffers according to physical token slots.
    """

    import torch

    if key_states.ndim != 3 or value_states.ndim != 3:
        raise ValueError("token-pool KV store requires [N, H, D] K/V tensors")
    if tuple(key_states.shape) != tuple(value_states.shape):
        raise ValueError("token-pool KV store key/value shapes must match")
    if tuple(key_buffer.shape) != tuple(value_buffer.shape):
        raise ValueError("token-pool KV store key/value buffer shapes must match")
    if tuple(key_states.shape[1:]) != tuple(key_buffer.shape[1:]):
        raise ValueError("token-pool KV store source and buffer head shapes differ")
    if key_states.dtype != key_buffer.dtype or value_states.dtype != value_buffer.dtype:
        raise ValueError("token-pool KV store source and buffer dtypes must match")
    if key_states.device != key_buffer.device or value_states.device != value_buffer.device:
        raise ValueError("token-pool KV store source and buffer devices must match")
    if key_buffer.device != value_buffer.device:
        raise ValueError("token-pool KV store buffers must share a device")
    if not key_states.is_cuda:
        raise RuntimeError("token-pool Triton KV store requires CUDA tensors")
    slots = torch.as_tensor(
        slot_mapping,
        dtype=torch.long,
        device=key_states.device,
    ).reshape(-1)
    if int(slots.numel()) != int(key_states.shape[0]):
        raise ValueError("token-pool KV store slot count must match K/V token count")
    if not slots.is_contiguous():
        slots = slots.contiguous()
    total_elems = int(key_states.shape[1]) * int(key_states.shape[2])
    block = _resolve_store_block_elems(total_elems, block_elems)
    grid = (int(key_states.shape[0]), triton.cdiv(total_elems, block))
    num_warps = 8 if block > 512 else 4
    _token_pool_store_kv_kernel[grid](
        key_states,
        value_states,
        key_buffer,
        value_buffer,
        slots,
        key_states.stride(0),
        key_states.stride(1),
        key_states.stride(2),
        value_states.stride(0),
        value_states.stride(1),
        value_states.stride(2),
        key_buffer.stride(0),
        key_buffer.stride(1),
        key_buffer.stride(2),
        value_buffer.stride(0),
        value_buffer.stride(1),
        value_buffer.stride(2),
        int(key_states.shape[1]),
        int(key_states.shape[2]),
        block,
        num_warps=num_warps,
    )


def _resolve_split_size(split_size: int | None) -> int:
    if split_size is None:
        env_split_size = os.environ.get("WKVM_TOKEN_POOL_TRITON_SPLIT_SIZE")
        split = int(env_split_size) if env_split_size else 512
    else:
        split = int(split_size)
    if split < 1:
        raise ValueError("token-pool split-KV Triton split_size must be >= 1")
    return split


def _resolve_split_min_splits(min_splits: int | None) -> int:
    if min_splits is None:
        env_min_splits = os.environ.get("WKVM_TOKEN_POOL_TRITON_MIN_SPLITS")
        minimum = int(env_min_splits) if env_min_splits else 4
    else:
        minimum = int(min_splits)
    if minimum < 2:
        raise ValueError("token-pool split-KV Triton min_splits must be >= 2")
    return minimum


def _resolve_input_precision(dtype, input_precision: str | None = None) -> str:
    configured = input_precision
    if configured is None:
        configured = os.environ.get("WKVM_TOKEN_POOL_TRITON_INPUT_PRECISION")
    if configured is not None:
        precision = configured.strip().lower()
        if precision in {"auto", "default", ""}:
            configured = None
        elif precision in {"ieee", "tf32", "tf32x3"}:
            return precision
        else:
            raise ValueError(
                "WKVM_TOKEN_POOL_TRITON_INPUT_PRECISION must be one of "
                "auto, default, ieee, tf32, or tf32x3"
            )

    dtype_name = str(dtype).lower()
    if "float32" in dtype_name or "float64" in dtype_name:
        return "ieee"
    return "tf32"


def _resolve_native_dot(dtype, dot_dtype: str | None = None) -> bool:
    configured = dot_dtype
    if configured is None:
        configured = os.environ.get("WKVM_TOKEN_POOL_TRITON_DOT_DTYPE")
    if configured is not None:
        mode = configured.strip().lower()
        if mode in {"auto", "default", ""}:
            configured = None
        elif mode == "native":
            return True
        elif mode == "fp32":
            return False
        else:
            raise ValueError(
                "WKVM_TOKEN_POOL_TRITON_DOT_DTYPE must be one of "
                "auto, default, native, or fp32"
            )

    dtype_name = str(dtype).lower()
    return "float32" not in dtype_name and "float64" not in dtype_name


def token_pool_gqa_decode(
    query_states,
    key_buffer,
    value_buffer,
    kv_indptr,
    kv_indices,
    *,
    num_key_value_groups: int,
    scaling: float,
    block_n: int | None = None,
    input_precision: str | None = None,
    dot_dtype: str | None = None,
    output=None,
):
    """Decode-only GQA attention over token-pool K/V buffers.

    Args:
        query_states: ``[batch, query_heads, 1, head_dim]``.
        key_buffer/value_buffer: ``[token_capacity, kv_heads, head_dim]``.
        kv_indptr/kv_indices: flattened ragged token metadata.
    """

    import torch

    if query_states.ndim != 4 or int(query_states.shape[2]) != 1:
        raise ValueError("token-pool Triton attention requires [B, Hq, 1, D] queries")
    if not query_states.is_cuda:
        raise RuntimeError("token-pool Triton attention requires CUDA tensors")
    if key_buffer.device != query_states.device or value_buffer.device != query_states.device:
        raise ValueError("token-pool K/V buffers must be on the query device")
    if kv_indptr.device != query_states.device or kv_indices.device != query_states.device:
        raise ValueError("token-pool metadata must be on the query device")

    batch = int(query_states.shape[0])
    query_heads = int(query_states.shape[1])
    head_dim = int(query_states.shape[3])
    groups = int(num_key_value_groups)
    if groups < 1 or query_heads % groups:
        raise ValueError("invalid grouped-query head layout")
    if int(key_buffer.shape[1]) * groups != query_heads:
        raise ValueError("token-pool K/V head count does not match query heads")
    if int(key_buffer.shape[2]) != head_dim or int(value_buffer.shape[2]) != head_dim:
        raise ValueError("token-pool K/V head_dim does not match query head_dim")

    output_shape = (batch, 1, query_heads, head_dim)
    if output is None:
        output = torch.empty(
            output_shape,
            dtype=query_states.dtype,
            device=query_states.device,
        )
    elif tuple(output.shape) != output_shape:
        raise ValueError("token-pool Triton output buffer has the wrong shape")
    elif output.dtype != query_states.dtype or output.device != query_states.device:
        raise ValueError("token-pool Triton output buffer must match query dtype/device")
    native_dot = _resolve_native_dot(query_states.dtype, dot_dtype)
    bd = _block_d(head_dim)
    bg = _block_g(groups, native_dot)
    bn = _resolve_block_n(
        head_dim,
        block_n,
        env_names=("WKVM_TOKEN_POOL_TRITON_BLOCK_N",),
    )
    precision = _resolve_input_precision(query_states.dtype, input_precision)
    num_warps = _resolve_num_warps(
        bd,
        env_names=("WKVM_TOKEN_POOL_TRITON_NUM_WARPS",),
    )
    _token_pool_gqa_decode_grouped_kernel[(batch, int(key_buffer.shape[1]))](
        query_states,
        key_buffer,
        value_buffer,
        kv_indptr,
        kv_indices,
        output,
        query_states.stride(0),
        query_states.stride(1),
        query_states.stride(2),
        query_states.stride(3),
        key_buffer.stride(0),
        key_buffer.stride(1),
        key_buffer.stride(2),
        value_buffer.stride(0),
        value_buffer.stride(1),
        value_buffer.stride(2),
        output.stride(0),
        output.stride(2),
        output.stride(3),
        float(scaling),
        groups,
        head_dim,
        bd,
        bn,
        bg,
        precision,
        native_dot,
        num_warps=num_warps,
    )
    return output


def token_pool_gqa_decode_split_kv(
    query_states,
    key_buffer,
    value_buffer,
    kv_indptr,
    kv_indices,
    *,
    num_key_value_groups: int,
    scaling: float,
    max_seq_len: int | None = None,
    split_size: int | None = None,
    min_splits: int | None = None,
    block_n: int | None = None,
    input_precision: str | None = None,
    dot_dtype: str | None = None,
    seq_lens=None,
    workspace=None,
    output=None,
):
    """Decode-only GQA attention with split-KV parallelism over flat metadata."""

    import torch

    if query_states.ndim != 4 or int(query_states.shape[2]) != 1:
        raise ValueError("token-pool split-KV Triton attention requires [B, Hq, 1, D] queries")
    if not query_states.is_cuda:
        raise RuntimeError("token-pool split-KV Triton attention requires CUDA tensors")
    if key_buffer.device != query_states.device or value_buffer.device != query_states.device:
        raise ValueError("token-pool K/V buffers must be on the query device")
    if kv_indptr.device != query_states.device or kv_indices.device != query_states.device:
        raise ValueError("token-pool metadata must be on the query device")

    batch = int(query_states.shape[0])
    query_heads = int(query_states.shape[1])
    head_dim = int(query_states.shape[3])
    groups = int(num_key_value_groups)
    if groups < 1 or query_heads % groups:
        raise ValueError("invalid grouped-query head layout")
    kv_heads = int(key_buffer.shape[1])
    if kv_heads * groups != query_heads:
        raise ValueError("token-pool K/V head count does not match query heads")
    if int(key_buffer.shape[2]) != head_dim or int(value_buffer.shape[2]) != head_dim:
        raise ValueError("token-pool K/V head_dim does not match query head_dim")

    output_shape = (batch, 1, query_heads, head_dim)
    if output is None:
        output = torch.empty(
            output_shape,
            dtype=query_states.dtype,
            device=query_states.device,
        )
    elif tuple(output.shape) != output_shape:
        raise ValueError("token-pool split-KV Triton output buffer has the wrong shape")
    elif output.dtype != query_states.dtype or output.device != query_states.device:
        raise ValueError("token-pool split-KV Triton output buffer must match query dtype/device")

    if max_seq_len is None:
        if int(kv_indptr.numel()) != batch + 1:
            raise ValueError("kv_indptr length must match query batch")
        row_lens = kv_indptr[1:] - kv_indptr[:-1]
        max_seq_len = int(row_lens.max().item())
    else:
        max_seq_len = int(max_seq_len)
    if max_seq_len < 1:
        raise ValueError("token-pool split-KV rows must contain at least one KV token")
    seq_lens_for_stage2 = kv_indptr
    has_seq_lens = seq_lens is not None
    if has_seq_lens:
        if seq_lens.device != query_states.device:
            raise ValueError("token-pool split-KV seq_lens must be on the query device")
        seq_lens_for_stage2 = seq_lens.reshape(-1)
        if int(seq_lens_for_stage2.numel()) != batch:
            raise ValueError("token-pool split-KV seq_lens length must match query batch")
        if not seq_lens_for_stage2.is_contiguous():
            seq_lens_for_stage2 = seq_lens_for_stage2.contiguous()
    split = _resolve_split_size(split_size)
    required_splits = _resolve_split_min_splits(min_splits)
    max_splits = (max_seq_len + split - 1) // split
    if max_splits < required_splits:
        return token_pool_gqa_decode(
            query_states,
            key_buffer,
            value_buffer,
            kv_indptr,
            kv_indices,
            num_key_value_groups=groups,
            scaling=scaling,
            block_n=block_n,
            input_precision=input_precision,
            dot_dtype=dot_dtype,
            output=output,
        )

    native_dot = _resolve_native_dot(query_states.dtype, dot_dtype)
    bd = _block_d(head_dim)
    bg = _block_g(groups, native_dot)
    bn = _resolve_block_n(
        head_dim,
        block_n,
        env_names=("WKVM_TOKEN_POOL_TRITON_BLOCK_N",),
    )
    precision = _resolve_input_precision(query_states.dtype, input_precision)
    stats_shape = (batch, kv_heads, max_splits, bg)
    acc_shape = (batch, kv_heads, max_splits, bg, head_dim)
    if workspace is None:
        partial_m = torch.empty(stats_shape, dtype=torch.float32, device=query_states.device)
        partial_l = torch.empty(stats_shape, dtype=torch.float32, device=query_states.device)
        partial_acc = torch.empty(acc_shape, dtype=torch.float32, device=query_states.device)
    else:
        if len(workspace) != 3:
            raise ValueError("token-pool split-KV workspace must be a 3-tuple")
        partial_m, partial_l, partial_acc = workspace
        if tuple(partial_m.shape) != stats_shape or tuple(partial_l.shape) != stats_shape:
            raise ValueError("token-pool split-KV stats workspace has the wrong shape")
        if tuple(partial_acc.shape) != acc_shape:
            raise ValueError("token-pool split-KV accumulator workspace has the wrong shape")
        if (
            partial_m.dtype != torch.float32
            or partial_l.dtype != torch.float32
            or partial_acc.dtype != torch.float32
        ):
            raise ValueError("token-pool split-KV workspace must use float32 tensors")
        if (
            partial_m.device != query_states.device
            or partial_l.device != query_states.device
            or partial_acc.device != query_states.device
        ):
            raise ValueError("token-pool split-KV workspace must be on the query device")

    num_warps = _resolve_num_warps(
        bd,
        env_names=("WKVM_TOKEN_POOL_TRITON_NUM_WARPS",),
    )
    _token_pool_gqa_decode_grouped_split_stage1_kernel[(batch, kv_heads, max_splits)](
        query_states,
        key_buffer,
        value_buffer,
        kv_indptr,
        kv_indices,
        partial_m,
        partial_l,
        partial_acc,
        query_states.stride(0),
        query_states.stride(1),
        query_states.stride(2),
        query_states.stride(3),
        key_buffer.stride(0),
        key_buffer.stride(1),
        key_buffer.stride(2),
        value_buffer.stride(0),
        value_buffer.stride(1),
        value_buffer.stride(2),
        partial_m.stride(0),
        partial_m.stride(1),
        partial_m.stride(2),
        partial_m.stride(3),
        partial_l.stride(0),
        partial_l.stride(1),
        partial_l.stride(2),
        partial_l.stride(3),
        partial_acc.stride(0),
        partial_acc.stride(1),
        partial_acc.stride(2),
        partial_acc.stride(3),
        partial_acc.stride(4),
        float(scaling),
        groups,
        head_dim,
        bd,
        bn,
        bg,
        split,
        precision,
        native_dot,
        num_warps=num_warps,
    )
    _token_pool_gqa_decode_grouped_split_stage2_kernel[(batch, kv_heads)](
        partial_m,
        partial_l,
        partial_acc,
        output,
        seq_lens_for_stage2,
        partial_m.stride(0),
        partial_m.stride(1),
        partial_m.stride(2),
        partial_m.stride(3),
        partial_l.stride(0),
        partial_l.stride(1),
        partial_l.stride(2),
        partial_l.stride(3),
        partial_acc.stride(0),
        partial_acc.stride(1),
        partial_acc.stride(2),
        partial_acc.stride(3),
        partial_acc.stride(4),
        output.stride(0),
        output.stride(2),
        output.stride(3),
        groups,
        head_dim,
        bd,
        bg,
        max_splits,
        split,
        has_seq_lens,
        num_warps=num_warps,
    )
    return output


def token_pool_paged_gqa_decode(
    query_states,
    key_buffer,
    value_buffer,
    block_tables,
    block_table_lens,
    selected_start_positions,
    seq_lens,
    *,
    block_size: int,
    num_key_value_groups: int,
    scaling: float,
    block_n: int | None = None,
    input_precision: str | None = None,
    dot_dtype: str | None = None,
    output=None,
):
    """Decode-only GQA attention over page/block token-pool metadata.

    ``block_tables`` are interpreted relative to ``selected_start_positions``:
    column zero is the physical block containing the selected row start.
    """

    import torch

    if query_states.ndim != 4 or int(query_states.shape[2]) != 1:
        raise ValueError("token-pool paged Triton attention requires [B, Hq, 1, D] queries")
    if not query_states.is_cuda:
        raise RuntimeError("token-pool paged Triton attention requires CUDA tensors")
    if key_buffer.device != query_states.device or value_buffer.device != query_states.device:
        raise ValueError("token-pool K/V buffers must be on the query device")
    for name, tensor in (
        ("block_tables", block_tables),
        ("block_table_lens", block_table_lens),
        ("selected_start_positions", selected_start_positions),
        ("seq_lens", seq_lens),
    ):
        if tensor.device != query_states.device:
            raise ValueError(f"token-pool paged metadata tensor {name} must be on the query device")

    batch = int(query_states.shape[0])
    query_heads = int(query_states.shape[1])
    head_dim = int(query_states.shape[3])
    groups = int(num_key_value_groups)
    page_size = int(block_size)
    if page_size < 1:
        raise ValueError("token-pool paged block_size must be >= 1")
    if groups < 1 or query_heads % groups:
        raise ValueError("invalid grouped-query head layout")
    if int(key_buffer.shape[1]) * groups != query_heads:
        raise ValueError("token-pool K/V head count does not match query heads")
    if int(key_buffer.shape[2]) != head_dim or int(value_buffer.shape[2]) != head_dim:
        raise ValueError("token-pool K/V head_dim does not match query head_dim")
    if block_tables.ndim != 2 or int(block_tables.shape[0]) != batch:
        raise ValueError("block_tables must have shape [batch, max_blocks]")
    for name, tensor in (
        ("block_table_lens", block_table_lens),
        ("selected_start_positions", selected_start_positions),
        ("seq_lens", seq_lens),
    ):
        if tensor.reshape(-1).numel() != batch:
            raise ValueError(f"{name} length must match query batch")

    output_shape = (batch, 1, query_heads, head_dim)
    if output is None:
        output = torch.empty(
            output_shape,
            dtype=query_states.dtype,
            device=query_states.device,
        )
    elif tuple(output.shape) != output_shape:
        raise ValueError("token-pool paged Triton output buffer has the wrong shape")
    elif output.dtype != query_states.dtype or output.device != query_states.device:
        raise ValueError("token-pool paged Triton output buffer must match query dtype/device")

    native_dot = _resolve_native_dot(query_states.dtype, dot_dtype)
    bd = _block_d(head_dim)
    bg = _block_g(groups, native_dot)
    bn = _resolve_block_n(
        head_dim,
        block_n,
        env_names=(
            "WKVM_TOKEN_POOL_PAGED_TRITON_BLOCK_N",
            "WKVM_TOKEN_POOL_TRITON_BLOCK_N",
        ),
    )
    precision = _resolve_input_precision(query_states.dtype, input_precision)
    num_warps = _resolve_num_warps(
        bd,
        env_names=(
            "WKVM_TOKEN_POOL_PAGED_TRITON_NUM_WARPS",
            "WKVM_TOKEN_POOL_TRITON_NUM_WARPS",
        ),
    )
    _token_pool_paged_gqa_decode_grouped_kernel[(batch, int(key_buffer.shape[1]))](
        query_states,
        key_buffer,
        value_buffer,
        block_tables,
        block_table_lens,
        selected_start_positions,
        seq_lens,
        output,
        query_states.stride(0),
        query_states.stride(1),
        query_states.stride(2),
        query_states.stride(3),
        key_buffer.stride(0),
        key_buffer.stride(1),
        key_buffer.stride(2),
        value_buffer.stride(0),
        value_buffer.stride(1),
        value_buffer.stride(2),
        block_tables.stride(0),
        block_tables.stride(1),
        output.stride(0),
        output.stride(2),
        output.stride(3),
        float(scaling),
        groups,
        head_dim,
        bd,
        bn,
        bg,
        page_size,
        precision,
        native_dot,
        num_warps=num_warps,
    )
    return output


def token_pool_paged_request_table_gqa_decode(
    query_states,
    key_buffer,
    value_buffer,
    req_pool_indices,
    request_block_tables,
    block_table_lens,
    selected_start_positions,
    seq_lens,
    *,
    block_size: int,
    num_key_value_groups: int,
    scaling: float,
    block_n: int | None = None,
    input_precision: str | None = None,
    dot_dtype: str | None = None,
    output=None,
):
    """Decode-only paged GQA attention over a persistent request block table."""

    import torch

    if query_states.ndim != 4 or int(query_states.shape[2]) != 1:
        raise ValueError(
            "token-pool request-table paged Triton attention requires "
            "[B, Hq, 1, D] queries"
        )
    if not query_states.is_cuda:
        raise RuntimeError(
            "token-pool request-table paged Triton attention requires CUDA tensors"
        )
    if key_buffer.device != query_states.device or value_buffer.device != query_states.device:
        raise ValueError("token-pool K/V buffers must be on the query device")
    for name, tensor in (
        ("req_pool_indices", req_pool_indices),
        ("request_block_tables", request_block_tables),
        ("block_table_lens", block_table_lens),
        ("selected_start_positions", selected_start_positions),
        ("seq_lens", seq_lens),
    ):
        if tensor.device != query_states.device:
            raise ValueError(
                f"token-pool request-table paged metadata tensor {name} "
                "must be on the query device"
            )

    batch = int(query_states.shape[0])
    query_heads = int(query_states.shape[1])
    head_dim = int(query_states.shape[3])
    groups = int(num_key_value_groups)
    page_size = int(block_size)
    if page_size < 1:
        raise ValueError("token-pool paged block_size must be >= 1")
    if groups < 1 or query_heads % groups:
        raise ValueError("invalid grouped-query head layout")
    if int(key_buffer.shape[1]) * groups != query_heads:
        raise ValueError("token-pool K/V head count does not match query heads")
    if int(key_buffer.shape[2]) != head_dim or int(value_buffer.shape[2]) != head_dim:
        raise ValueError("token-pool K/V head_dim does not match query head_dim")
    if request_block_tables.ndim != 2:
        raise ValueError("request_block_tables must have shape [max_requests, max_pages]")
    for name, tensor in (
        ("req_pool_indices", req_pool_indices),
        ("block_table_lens", block_table_lens),
        ("selected_start_positions", selected_start_positions),
        ("seq_lens", seq_lens),
    ):
        if tensor.reshape(-1).numel() != batch:
            raise ValueError(f"{name} length must match query batch")

    req_pool_indices = req_pool_indices.reshape(-1)
    block_table_lens = block_table_lens.reshape(-1)
    selected_start_positions = selected_start_positions.reshape(-1)
    seq_lens = seq_lens.reshape(-1)
    if not req_pool_indices.is_contiguous():
        req_pool_indices = req_pool_indices.contiguous()
    if not block_table_lens.is_contiguous():
        block_table_lens = block_table_lens.contiguous()
    if not selected_start_positions.is_contiguous():
        selected_start_positions = selected_start_positions.contiguous()
    if not seq_lens.is_contiguous():
        seq_lens = seq_lens.contiguous()
    if not request_block_tables.is_contiguous():
        request_block_tables = request_block_tables.contiguous()

    output_shape = (batch, 1, query_heads, head_dim)
    if output is None:
        output = torch.empty(
            output_shape,
            dtype=query_states.dtype,
            device=query_states.device,
        )
    elif tuple(output.shape) != output_shape:
        raise ValueError(
            "token-pool request-table paged Triton output buffer has the wrong shape"
        )
    elif output.dtype != query_states.dtype or output.device != query_states.device:
        raise ValueError(
            "token-pool request-table paged Triton output buffer must match "
            "query dtype/device"
        )

    native_dot = _resolve_native_dot(query_states.dtype, dot_dtype)
    bd = _block_d(head_dim)
    bg = _block_g(groups, native_dot)
    bn = _resolve_block_n(
        head_dim,
        block_n,
        env_names=(
            "WKVM_TOKEN_POOL_PAGED_TRITON_BLOCK_N",
            "WKVM_TOKEN_POOL_TRITON_BLOCK_N",
        ),
    )
    precision = _resolve_input_precision(query_states.dtype, input_precision)
    num_warps = _resolve_num_warps(
        bd,
        env_names=(
            "WKVM_TOKEN_POOL_PAGED_TRITON_NUM_WARPS",
            "WKVM_TOKEN_POOL_TRITON_NUM_WARPS",
        ),
    )
    _token_pool_paged_request_table_gqa_decode_grouped_kernel[
        (batch, int(key_buffer.shape[1]))
    ](
        query_states,
        key_buffer,
        value_buffer,
        req_pool_indices,
        request_block_tables,
        block_table_lens,
        selected_start_positions,
        seq_lens,
        output,
        query_states.stride(0),
        query_states.stride(1),
        query_states.stride(2),
        query_states.stride(3),
        key_buffer.stride(0),
        key_buffer.stride(1),
        key_buffer.stride(2),
        value_buffer.stride(0),
        value_buffer.stride(1),
        value_buffer.stride(2),
        request_block_tables.stride(0),
        request_block_tables.stride(1),
        output.stride(0),
        output.stride(2),
        output.stride(3),
        float(scaling),
        groups,
        head_dim,
        bd,
        bn,
        bg,
        page_size,
        precision,
        native_dot,
        num_warps=num_warps,
    )
    return output


def token_pool_paged_gqa_decode_split_kv(
    query_states,
    key_buffer,
    value_buffer,
    block_tables,
    block_table_lens,
    selected_start_positions,
    seq_lens,
    *,
    block_size: int,
    num_key_value_groups: int,
    scaling: float,
    max_seq_len: int | None = None,
    split_size: int | None = None,
    min_splits: int | None = None,
    block_n: int | None = None,
    input_precision: str | None = None,
    dot_dtype: str | None = None,
    workspace=None,
    output=None,
):
    """Decode-only paged GQA attention with split-KV parallelism."""

    import torch

    if query_states.ndim != 4 or int(query_states.shape[2]) != 1:
        raise ValueError("token-pool paged split-KV Triton attention requires [B, Hq, 1, D] queries")
    if not query_states.is_cuda:
        raise RuntimeError("token-pool paged split-KV Triton attention requires CUDA tensors")
    if key_buffer.device != query_states.device or value_buffer.device != query_states.device:
        raise ValueError("token-pool K/V buffers must be on the query device")
    for name, tensor in (
        ("block_tables", block_tables),
        ("block_table_lens", block_table_lens),
        ("selected_start_positions", selected_start_positions),
        ("seq_lens", seq_lens),
    ):
        if tensor.device != query_states.device:
            raise ValueError(f"token-pool paged metadata tensor {name} must be on the query device")

    batch = int(query_states.shape[0])
    query_heads = int(query_states.shape[1])
    head_dim = int(query_states.shape[3])
    groups = int(num_key_value_groups)
    page_size = int(block_size)
    if page_size < 1:
        raise ValueError("token-pool paged block_size must be >= 1")
    if groups < 1 or query_heads % groups:
        raise ValueError("invalid grouped-query head layout")
    kv_heads = int(key_buffer.shape[1])
    if kv_heads * groups != query_heads:
        raise ValueError("token-pool K/V head count does not match query heads")
    if int(key_buffer.shape[2]) != head_dim or int(value_buffer.shape[2]) != head_dim:
        raise ValueError("token-pool K/V head_dim does not match query head_dim")
    if block_tables.ndim != 2 or int(block_tables.shape[0]) != batch:
        raise ValueError("block_tables must have shape [batch, max_blocks]")
    for name, tensor in (
        ("block_table_lens", block_table_lens),
        ("selected_start_positions", selected_start_positions),
        ("seq_lens", seq_lens),
    ):
        if tensor.reshape(-1).numel() != batch:
            raise ValueError(f"{name} length must match query batch")

    output_shape = (batch, 1, query_heads, head_dim)
    if output is None:
        output = torch.empty(
            output_shape,
            dtype=query_states.dtype,
            device=query_states.device,
        )
    elif tuple(output.shape) != output_shape:
        raise ValueError("token-pool paged split-KV Triton output buffer has the wrong shape")
    elif output.dtype != query_states.dtype or output.device != query_states.device:
        raise ValueError("token-pool paged split-KV Triton output buffer must match query dtype/device")

    if max_seq_len is None:
        max_seq_len = int(seq_lens.reshape(-1).max().item())
    else:
        max_seq_len = int(max_seq_len)
    if max_seq_len < 1:
        raise ValueError("token-pool paged split-KV rows must contain at least one KV token")
    seq_lens_for_kernel = seq_lens.reshape(-1)
    if not seq_lens_for_kernel.is_contiguous():
        seq_lens_for_kernel = seq_lens_for_kernel.contiguous()
    split = _resolve_split_size(split_size)
    required_splits = _resolve_split_min_splits(min_splits)
    max_splits = (max_seq_len + split - 1) // split
    if max_splits < required_splits:
        return token_pool_paged_gqa_decode(
            query_states,
            key_buffer,
            value_buffer,
            block_tables,
            block_table_lens,
            selected_start_positions,
            seq_lens,
            block_size=page_size,
            num_key_value_groups=groups,
            scaling=scaling,
            block_n=block_n,
            input_precision=input_precision,
            dot_dtype=dot_dtype,
            output=output,
        )

    native_dot = _resolve_native_dot(query_states.dtype, dot_dtype)
    bd = _block_d(head_dim)
    bg = _block_g(groups, native_dot)
    bn = _resolve_block_n(
        head_dim,
        block_n,
        env_names=(
            "WKVM_TOKEN_POOL_PAGED_TRITON_BLOCK_N",
            "WKVM_TOKEN_POOL_TRITON_BLOCK_N",
        ),
    )
    precision = _resolve_input_precision(query_states.dtype, input_precision)
    stats_shape = (batch, kv_heads, max_splits, bg)
    acc_shape = (batch, kv_heads, max_splits, bg, head_dim)
    if workspace is None:
        partial_m = torch.empty(stats_shape, dtype=torch.float32, device=query_states.device)
        partial_l = torch.empty(stats_shape, dtype=torch.float32, device=query_states.device)
        partial_acc = torch.empty(acc_shape, dtype=torch.float32, device=query_states.device)
    else:
        if len(workspace) != 3:
            raise ValueError("token-pool paged split-KV workspace must be a 3-tuple")
        partial_m, partial_l, partial_acc = workspace
        if tuple(partial_m.shape) != stats_shape or tuple(partial_l.shape) != stats_shape:
            raise ValueError("token-pool paged split-KV stats workspace has the wrong shape")
        if tuple(partial_acc.shape) != acc_shape:
            raise ValueError("token-pool paged split-KV accumulator workspace has the wrong shape")
        if (
            partial_m.dtype != torch.float32
            or partial_l.dtype != torch.float32
            or partial_acc.dtype != torch.float32
        ):
            raise ValueError("token-pool paged split-KV workspace must use float32 tensors")
        if (
            partial_m.device != query_states.device
            or partial_l.device != query_states.device
            or partial_acc.device != query_states.device
        ):
            raise ValueError("token-pool paged split-KV workspace must be on the query device")

    num_warps = _resolve_num_warps(
        bd,
        env_names=(
            "WKVM_TOKEN_POOL_PAGED_TRITON_NUM_WARPS",
            "WKVM_TOKEN_POOL_TRITON_NUM_WARPS",
        ),
    )
    _token_pool_paged_gqa_decode_grouped_split_stage1_kernel[(batch, kv_heads, max_splits)](
        query_states,
        key_buffer,
        value_buffer,
        block_tables,
        block_table_lens,
        selected_start_positions,
        seq_lens_for_kernel,
        partial_m,
        partial_l,
        partial_acc,
        query_states.stride(0),
        query_states.stride(1),
        query_states.stride(2),
        query_states.stride(3),
        key_buffer.stride(0),
        key_buffer.stride(1),
        key_buffer.stride(2),
        value_buffer.stride(0),
        value_buffer.stride(1),
        value_buffer.stride(2),
        block_tables.stride(0),
        block_tables.stride(1),
        partial_m.stride(0),
        partial_m.stride(1),
        partial_m.stride(2),
        partial_m.stride(3),
        partial_l.stride(0),
        partial_l.stride(1),
        partial_l.stride(2),
        partial_l.stride(3),
        partial_acc.stride(0),
        partial_acc.stride(1),
        partial_acc.stride(2),
        partial_acc.stride(3),
        partial_acc.stride(4),
        float(scaling),
        groups,
        head_dim,
        bd,
        bn,
        bg,
        page_size,
        split,
        precision,
        native_dot,
        num_warps=num_warps,
    )
    _token_pool_gqa_decode_grouped_split_stage2_kernel[(batch, kv_heads)](
        partial_m,
        partial_l,
        partial_acc,
        output,
        seq_lens_for_kernel,
        partial_m.stride(0),
        partial_m.stride(1),
        partial_m.stride(2),
        partial_m.stride(3),
        partial_l.stride(0),
        partial_l.stride(1),
        partial_l.stride(2),
        partial_l.stride(3),
        partial_acc.stride(0),
        partial_acc.stride(1),
        partial_acc.stride(2),
        partial_acc.stride(3),
        partial_acc.stride(4),
        output.stride(0),
        output.stride(2),
        output.stride(3),
        groups,
        head_dim,
        bd,
        bg,
        max_splits,
        split,
        True,
        num_warps=num_warps,
    )
    return output


def dense_padded_gqa_decode(
    query_states,
    key_states,
    value_states,
    attention_mask=None,
    *,
    num_key_value_groups: int,
    scaling: float,
    block_n: int | None = None,
    input_precision: str | None = None,
    dot_dtype: str | None = None,
    output=None,
):
    """Decode-only GQA attention over dense padded K/V tensors.

    Args:
        query_states: ``[batch, query_heads, 1, head_dim]``.
        key_states/value_states: ``[batch, kv_heads, key_length, head_dim]``.
        attention_mask: optional ``[batch, 1, 1, key_length]`` additive mask.
    """

    import torch

    if query_states.ndim != 4 or int(query_states.shape[2]) != 1:
        raise ValueError("dense padded Triton attention requires [B, Hq, 1, D] queries")
    if key_states.ndim != 4 or value_states.ndim != 4:
        raise ValueError("dense padded K/V tensors must have shape [B, Hkv, T, D]")
    if not query_states.is_cuda:
        raise RuntimeError("dense padded Triton attention requires CUDA tensors")
    if key_states.device != query_states.device or value_states.device != query_states.device:
        raise ValueError("dense padded K/V tensors must be on the query device")

    batch = int(query_states.shape[0])
    query_heads = int(query_states.shape[1])
    key_length = int(key_states.shape[2])
    head_dim = int(query_states.shape[3])
    groups = int(num_key_value_groups)
    if groups < 1 or query_heads % groups:
        raise ValueError("invalid grouped-query head layout")
    if int(key_states.shape[0]) != batch or int(value_states.shape[0]) != batch:
        raise ValueError("dense padded K/V batch does not match query batch")
    if int(key_states.shape[1]) * groups != query_heads:
        raise ValueError("dense padded K/V head count does not match query heads")
    if tuple(value_states.shape[:3]) != tuple(key_states.shape[:3]):
        raise ValueError("dense padded key/value shapes differ")
    if int(key_states.shape[3]) != head_dim or int(value_states.shape[3]) != head_dim:
        raise ValueError("dense padded K/V head_dim does not match query head_dim")
    if key_length < 1:
        raise ValueError("dense padded key_length must be >= 1")

    has_mask = attention_mask is not None
    if has_mask:
        if attention_mask.device != query_states.device:
            raise ValueError("dense padded attention mask must be on the query device")
        if attention_mask.ndim != 4:
            raise ValueError("dense padded attention mask must have shape [B, 1, 1, T]")
        if int(attention_mask.shape[0]) != batch or int(attention_mask.shape[-1]) != key_length:
            raise ValueError("dense padded attention mask shape does not match K/V")
        if int(attention_mask.shape[1]) != 1 or int(attention_mask.shape[2]) != 1:
            raise ValueError("dense padded attention mask must have singleton head/query dims")
    else:
        attention_mask = key_states

    output_shape = (batch, 1, query_heads, head_dim)
    if output is None:
        output = torch.empty(
            output_shape,
            dtype=query_states.dtype,
            device=query_states.device,
        )
    elif tuple(output.shape) != output_shape:
        raise ValueError("dense padded Triton output buffer has the wrong shape")
    elif output.dtype != query_states.dtype or output.device != query_states.device:
        raise ValueError("dense padded Triton output buffer must match query dtype/device")

    native_dot = _resolve_native_dot(query_states.dtype, dot_dtype)
    bd = _block_d(head_dim)
    bg = _block_g(groups, native_dot)
    bn = _resolve_block_n(
        head_dim,
        block_n,
        env_names=("WKVM_DENSE_TRITON_BLOCK_N", "WKVM_TOKEN_POOL_TRITON_BLOCK_N"),
    )
    precision = _resolve_input_precision(query_states.dtype, input_precision)
    num_warps = _resolve_num_warps(
        bd,
        env_names=("WKVM_DENSE_TRITON_NUM_WARPS", "WKVM_TOKEN_POOL_TRITON_NUM_WARPS"),
    )
    _dense_padded_gqa_decode_grouped_kernel[(batch, int(key_states.shape[1]))](
        query_states,
        key_states,
        value_states,
        attention_mask,
        output,
        query_states.stride(0),
        query_states.stride(1),
        query_states.stride(2),
        query_states.stride(3),
        key_states.stride(0),
        key_states.stride(1),
        key_states.stride(2),
        key_states.stride(3),
        value_states.stride(0),
        value_states.stride(1),
        value_states.stride(2),
        value_states.stride(3),
        attention_mask.stride(0),
        attention_mask.stride(3),
        output.stride(0),
        output.stride(2),
        output.stride(3),
        float(scaling),
        groups,
        key_length,
        head_dim,
        bd,
        bn,
        bg,
        bool(has_mask),
        precision,
        native_dot,
        num_warps=num_warps,
    )
    return output
