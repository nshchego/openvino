// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather.hpp"
#include "openvino/core/except.hpp"

using namespace dnnl::impl::cpu;

namespace ov {
namespace intel_cpu {
namespace kernel {

template <x64::cpu_isa_t isa>
const unsigned Gather<isa>::shufMask8bitUni[16]  = {0x0C080400, 0x80808080, 0x80808080, 0x80808080, 0x0C080400, 0x80808080, 0x80808080, 0x80808080,
                                                    0x0C080400, 0x80808080, 0x80808080, 0x80808080, 0x0C080400, 0x80808080, 0x80808080, 0x80808080};

template <x64::cpu_isa_t isa>
const unsigned Gather<isa>::shufMask16bitUni[16] = {0x05040100, 0x0D0C0908, 0x80808080, 0x80808080, 0x05040100, 0x0D0C0908, 0x80808080, 0x80808080,
                                                    0x05040100, 0x0D0C0908, 0x80808080, 0x80808080, 0x05040100, 0x0D0C0908, 0x80808080, 0x80808080};

template <x64::cpu_isa_t isa>
const unsigned Gather<isa>::m_inc_vec[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

const unsigned Gather<x64::avx2>::permMask8bitUni[16] = {0, 4, 1, 5, 2, 6, 3, 7};
const unsigned Gather<x64::avx512_core>::permMask8bitUni[16] = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};

const unsigned Gather<x64::avx2>::permMask16bitUni[16] = {0, 1, 4, 5, 2, 3, 6, 7};
const unsigned Gather<x64::avx512_core>::permMask16bitUni[16] = {0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15};

#define GET_OFF(field) offsetof(GatherCallArgs, field)

// template <>
// Gather<x64::sse41>::Gather(const GatherCompileParams& jcp) : JitKernel(jit_name(), jcp, x64::sse41) {
//     OPENVINO_THROW("Gather kernel does not support SSE4 instructions set.");
// }

template <x64::cpu_isa_t isa>
Gather<isa>::Gather(const GatherCompileParams& jcp) : JitKernel(jit_name(), jcp, isa) {
    vlen = x64::cpu_isa_traits<isa>::vlen;
    m_data_el_per_vec = vlen / m_jcp.data_et_size;
    m_idx_el_per_vec = vlen / m_jcp.idx_et_size;
    m_data_et_shift = m_jcp.data_et_size / 2;
}

template <x64::cpu_isa_t isa>
void Gather<isa>::generate() {
    this->preamble();
    registersPool = RegistersPool::create(isa, {rax, rcx, rsp, rdi, k0});

    r64_src = getReg64();
    r64_dst = getReg64();
    r64_idx = getReg64();
    r64_idx_iter    = getReg64();
    r32_idx_iter    = Xbyak::Reg32(r64_idx_iter.getIdx());
    r64_work_amount = getReg64();

    v_axis_dim        = getVmm();
    v_spec_idx_b      = getVmm();
    v_spec_idx_size_b = getVmm();
    v_zeros           = getVmm();

    mov(r64_src, ptr[regParams + GET_OFF(src)]);
    mov(r64_dst, ptr[regParams + GET_OFF(dst)]);
    mov(r64_idx, ptr[regParams + GET_OFF(indices)]);

    mov(r64_work_amount, ptr[regParams + GET_OFF(work_amount)]);

    uni_vpxor(v_zeros, v_zeros, v_zeros);
    mov(r64_aux_1, ptr[regParams + GET_OFF(axisDim)]);
    uni_vpbroadcastd(v_axis_dim, ptr[r64_aux_1]);

    if (!m_jcp.dynamic_shapes) {
        mov(r64_aux_1, ptr[regParams + GET_OFF(specIndicesSize)]);
        uni_vpbroadcastd(v_spec_idx_size_b, ptr[r64_aux_1]);
        uni_vpslld(v_spec_idx_size_b, v_spec_idx_size_b, idxTypeShift); // multiply by indices type size.

        mov(r64_aux_1, ptr[regParams + GET_OFF(specIdxB)]);
        uni_vmovups(v_spec_idx_b, ptr[r64_aux_1]);

        if (m_jcp.before_axis_size != 1lu) {
            v_src_before_axis_sum_b = getVmm();
            mov(r64_aux_1, ptr[regParams + GET_OFF(dataBeforeAxisSumB)]);
            uni_vmovups(v_src_before_axis_sum_b, ptr[r64_aux_1]);
        }

        if (m_jcp.after_axis_size == 1lu) { // Elementwise case.
            r64_batch_to_axix_iter = getReg64();
            r32_batch_to_axix_iter = Xbyak::Reg32(r64_batch_to_axix_iter.getIdx());
            r64_batch_to_axis_size = getReg64();
            r32_batch_to_axis_size = Xbyak::Reg32(r64_batch_to_axis_size.getIdx());
            v_idx_batch_sum_b = getVmm();

            uni_vmovd(r32_spec_idx_size_b, xmmSpecIdxSizeB);
            if (m_jcp.before_axis_size != 1lu) {
                v_axis_and_after_axis_size_b = getVmm();
                mov(r64_aux_1, ptr[regParams + GET_OFF(axisAndAfterAxisSizeB)]);
                uni_vpbroadcastd(v_axis_and_after_axis_size_b, ptr[r64_aux_1]);
            }

            mov(r64_aux_1, ptr[regParams + GET_OFF(idxBatchSumB)]);
            uni_vmovups(v_idx_batch_sum_b, ptr[r64_aux_1]);

            mov(r64_aux_1, ptr[regParams + GET_OFF(betweenBatchAndAxisSize)]);
            mov(r64_batch_to_axis_size, ptr[r64_aux_1]);
            mov(r64_batch_to_axix_iter, ptr[regParams + GET_OFF(betweenBatchAndAxisIter)]);

            if (m_jcp.spec_idx_size < m_idx_el_per_vec) { // Short case.
                if (m_jcp.spec_idx_size != 1 && m_jcp.spec_idx_size != 2 && m_jcp.spec_idx_size != 4 && m_jcp.spec_idx_size != 8 && m_jcp.spec_idx_size != 16) {
                    v_perm_idx_mask = getVmm();
                    mov(r64_aux_1, ptr[regParams + GET_OFF(permIdxMask)]);
                    uni_vmovups(v_perm_idx_mask, ptr[r64_aux_1]);
                }
                if (m_jcp.before_axis_size != 1lu) {
                    v_before_axis_diff_b = getVmm();
                    mov(r64_aux_1, ptr[regParams + GET_OFF(beforeAxisDiff)]);
                    uni_vmovups(v_before_axis_diff_b, ptr[r64_aux_1]);
                    if (m_jcp.data_et_size != 1) {
                        uni_vpslld(v_before_axis_diff_b, v_before_axis_diff_b, m_data_et_shift); // multiply by data type size
                    }
                }
                if (m_jcp.batch_dims > 0lu) {
                    v_src_after_batch_size_b = getVmm();
                    mov(r64_aux_1, ptr[regParams + GET_OFF(srcAfterBatchSizeB)]);
                    uni_vpbroadcastd(v_src_after_batch_size_b, ptr[r64_aux_1]);
                }

                process(true, false);
            } else { // Long case.
                uni_vmovd(r32_idx_iter, x_spec_idx_b);
                fillVlenVector();

                process(false, false);
            }
        } else { // Blocked case.
            if (m_jcp.after_axis_size <= m_idx_el_per_vec) { // Short case
                v_after_axis_idx_b       = getVmm();
                v_after_axis_perm_mask   = getVmm();
                v_after_axis_size        = getVmm();
                v_spec_idx_diff          = getVmm();
                v_src_after_batch_size_b = getVmm();

                mov(r64_aux_1, ptr[regParams + GET_OFF(afterAxIdxB)]);
                uni_vmovups(v_after_axis_idx_b, ptr[r64_aux_1]);
                mov(r64_aux_1, ptr[regParams + GET_OFF(afterAxisPermMask)]);
                uni_vmovups(v_after_axis_perm_mask, ptr[r64_aux_1]);
                mov(r64_aux_1, ptr[regParams + GET_OFF(specIdxDiff)]);
                uni_vmovups(v_spec_idx_diff, ptr[r64_aux_1]);
                mov(r64_aux_1, ptr[regParams + GET_OFF(srcAfterBatchSizeB)]);
                uni_vpbroadcastd(v_src_after_batch_size_b, ptr[r64_aux_1]);
                mov(r64_aux_1, ptr[regParams + GET_OFF(afterAxisSize)]);
                uni_vpbroadcastd(v_after_axis_size, ptr[r64_aux_1]);

                if (m_jcp.before_axis_size != 1lu) {
                    mov(rSpecIdxAndAfterAxIterB, ptr[regParams + GET_OFF(specIdxAndAfterAxIterB)]);
                    mov(rSpecIdxAndAfterAxSizeB, ptr[regParams + GET_OFF(specIdxAndAfterAxSizeB)]);
                    if (m_jcp.spec_idx_size * m_jcp.after_axis_size < m_idx_el_per_vec) {
                        v_before_axis_diff_b = getVmm();
                        mov(r64_aux_1, ptr[regParams + GET_OFF(beforeAxisDiff)]);
                        uni_vmovups(v_before_axis_diff_b, ptr[r64_aux_1]);
                    } else {
                        v_axis_and_after_axis_size_b = getVmm();
                        mov(r64_aux_1, ptr[regParams + GET_OFF(axisAndAfterAxisSizeB)]);
                        uni_vpbroadcastd(v_axis_and_after_axis_size_b, ptr[r64_aux_1]);
                    }
                    const uint64_t specIdxAndAfterAxisSize = m_jcp.spec_idx_size * m_jcp.after_axis_size;
                    if (specIdxAndAfterAxisSize != 1 && specIdxAndAfterAxisSize != 2 && specIdxAndAfterAxisSize != 4 &&
                            specIdxAndAfterAxisSize != 8 && specIdxAndAfterAxisSize != 16) {
                        v_before_axis_perm_mask = getVmm();
                        mov(r64_aux_1, ptr[regParams + GET_OFF(beforeAxisPermMask)]);
                        uni_vmovups(v_before_axis_perm_mask, ptr[r64_aux_1]);
                    }
                }

                process(true, true);
            } else { // Long case.
                OPENVINO_THROW("Gather kernel does not support static shape with after axis size greater than elements in vector.");
            }
        }
    } else { // Dynamic shapes
        r64_spec_idx_size_b          = getReg64();

        v_after_axis_idx_b           = getVmm();
        v_after_axis_perm_mask       = getVmm();
        v_axis_and_after_axis_size_b = getVmm();
        v_before_axis_diff_b         = getVmm();
        v_idx_batch_sum_b            = getVmm();
        v_perm_idx_mask              = getVmm();
        v_src_before_axis_sum_b      = getVmm();
        v_src_after_batch_size_b     = getVmm();

        auto v_aux_0 = getVmm();
        auto v_aux_1 = getVmm();

        mov(r64_aux_1, ptr[regParams + GET_OFF(start)]);
        uni_vpbroadcastd(v_spec_idx_b, ptr[r64_aux_1]);
        mov(r64_aux_1, reinterpret_cast<uintptr_t>(m_inc_vec));
        uni_vpaddd(v_spec_idx_b, v_spec_idx_b, ptr[r64_aux_1]);
        vcvtdq2ps(v_spec_idx_b, v_spec_idx_b);

        // Formula: specIndices = (start % specIndicesSize) * idxTypeSize
        mov(r64_aux_1, ptr[regParams + GET_OFF(specIndicesSize)]);
        uni_vpbroadcastd(v_spec_idx_size_b, ptr[r64_aux_1]);
        uni_vcvtdq2ps(v_aux_1, v_spec_idx_size_b);
        uni_vdivps(v_src_before_axis_sum_b, v_spec_idx_b, v_aux_1);
        uni_vroundps(v_src_before_axis_sum_b, v_src_before_axis_sum_b, 0x1);
        uni_vfnmadd231ps(v_spec_idx_b, v_src_before_axis_sum_b, v_aux_1);
        uni_vcvtps2dq(v_spec_idx_b, v_spec_idx_b);
        uni_vpslld(v_spec_idx_b, v_spec_idx_b, idxTypeShift); // multiply by indices type size.
        uni_vpslld(v_spec_idx_size_b, v_spec_idx_size_b, idxTypeShift); // multiply by indices type size.
        uni_vmovd(r32_spec_idx_size_b, xmmSpecIdxSizeB);

        mov(r64_aux_1, ptr[regParams + GET_OFF(betweenBatchAndAxisSize)]);
        uni_vpbroadcastd(v_aux_1, ptr[r64_aux_1]);
        uni_vmovd(r32_batch_to_axis_size, xmm_aux_1);
        uni_vcvtdq2ps(v_aux_1, v_aux_1);
        uni_vdivps(v_idx_batch_sum_b, v_src_before_axis_sum_b, v_aux_1);
        uni_vroundps(v_idx_batch_sum_b, v_idx_batch_sum_b, 0x1);
        uni_vfnmadd231ps(v_src_before_axis_sum_b, v_idx_batch_sum_b, v_aux_1);
        uni_vcvtps2dq(v_src_before_axis_sum_b, v_src_before_axis_sum_b);
        uni_vmovd(r32_batch_to_axis_iter, xmmSrcBeforeAxisSum);
        uni_vcvtps2dq(v_idx_batch_sum_b, v_idx_batch_sum_b);

        mov(r64_aux_1, ptr[regParams + GET_OFF(axisAndAfterAxisSizeB)]);
        uni_vpbroadcastd(v_axis_and_after_axis_size_b, ptr[r64_aux_1]);
        // Formula: srcBeforeAxisSum = ((start / specIndicesSize) % betweenBatchAndAxis) * axisAndAfterAxisSize + srcAfterBatchSize * idxBatchSum
        if (m_jcp.before_axis_size != 1lu) {
            uni_vpmulld(v_src_before_axis_sum_b, v_src_before_axis_sum_b, v_axis_and_after_axis_size_b);
            mov(r64_aux_1, ptr[regParams + GET_OFF(srcAfterBatchSizeB)]);
            uni_vpbroadcastd(v_aux_0, ptr[r64_aux_1]);
            uni_vpmulld(v_aux_0, v_aux_0, v_idx_batch_sum_b);
            uni_vpaddd(v_src_before_axis_sum_b, v_src_before_axis_sum_b, v_aux_0);
        }

        // Formula: idxBatchSum = specIdxSize * (start / afterBatchSize)
        uni_vpmulld(v_idx_batch_sum_b, v_idx_batch_sum_b, v_spec_idx_size_b);

        Xbyak::Label lBlock, lEnd;
        mov(regAux2, ptr[regParams + GET_OFF(afterAxSize)]);
        cmp(regAux2, 1);
        jg(lBlock, T_NEAR);
        {
            Xbyak::Label lLessThanVector1, lTail1, lTail2, lE1;

            cmp(r64_spec_idx_size_b, vlen);
            jl(lLessThanVector1, T_NEAR);
                uni_vmovd(r32_idx_iter, x_spec_idx_b);
                fillVlenVector();

                process(false, false);
                jmp(lE1, T_NEAR);
            L(lLessThanVector1);
                mov(r64_aux_1, ptr[regParams + GET_OFF(permIdxMask)]);
                uni_vmovups(v_perm_idx_mask, ptr[r64_aux_1]);
                if (m_jcp.before_axis_size != 1lu) {
                    mov(r64_aux_1, ptr[regParams + GET_OFF(beforeAxisDiff)]);
                    uni_vmovups(v_before_axis_diff_b, ptr[r64_aux_1]);
                    if (m_jcp.data_et_size != 1)
                        uni_vpslld(v_before_axis_diff_b, v_before_axis_diff_b, m_data_et_shift); // multiply by data type size
                }
                mov(r64_aux_1, ptr[regParams + GET_OFF(srcAfterBatchSizeB)]);
                uni_vpbroadcastd(v_src_after_batch_size_b, ptr[r64_aux_1]);

                process(true, false);
            L(lE1);
            jmp(lEnd, T_NEAR);
        }
        L(lBlock); {
            auto xmm_aux_0 = getXmm();
            auto xmm_aux_1 = getXmm();

            mov(r64_aux_1, ptr[regParams + GET_OFF(start)]);
            uni_vpbroadcastd(v_after_axis_idx_b, ptr[r64_aux_1]);
            mov(r64_aux_1, reinterpret_cast<uintptr_t>(m_inc_vec));
            uni_vpaddd(v_after_axis_idx_b, v_after_axis_idx_b, ptr[r64_aux_1]);
            uni_vcvtdq2ps(v_after_axis_idx_b, v_after_axis_idx_b);

            // afterAxIdxB = (start % afterAxSize) * idxTypeSize
            movd(xmm_aux_0, r32_aux_1);
            uni_vpbroadcastd(v_aux_1, xmm_aux_0);
            uni_vcvtdq2ps(v_aux_1, v_aux_1);
            uni_vdivps(v_src_before_axis_sum_b, v_after_axis_idx_b, v_aux_1);
            uni_vroundps(v_src_before_axis_sum_b, v_src_before_axis_sum_b, 0x1);
            uni_vfnmadd231ps(v_after_axis_idx_b, v_src_before_axis_sum_b, v_aux_1);
            uni_vcvtps2dq(v_after_axis_idx_b, v_after_axis_idx_b);
            uni_vpslld(v_after_axis_idx_b, v_after_axis_idx_b, idxTypeShift); // multiply by indices type size.

            Xbyak::Label lLessThanVector2, lTail3, lTail4, lE2;

            cmp(regAux2, m_data_el_per_vec);
            jl(lLessThanVector2, T_NEAR);
                uni_vmovd(r32_idx_iter, x_spec_idx_b);
                fillVlenVector();

//                process(false, true);
                jmp(lE2, T_NEAR);
            L(lLessThanVector2);
                auto& vAux2 = vmmAuxContainer[2];
                // Calculate permute mask
                uni_vmovd(xmm_aux_0, reg32Aux2);
                uni_vpbroadcastd(v_aux_1, xmm_aux_0);
                mov(r64_aux_1, reinterpret_cast<uintptr_t>(&m_idx_el_per_vec));
                uni_vpbroadcastd(v_aux_0, ptr[r64_aux_1]);
                uni_vpsubd(v_after_axis_perm_mask, v_aux_0, v_aux_1);
                mov(r64_aux_1, reinterpret_cast<uintptr_t>(m_inc_vec));
                uni_vpaddd(v_after_axis_perm_mask, v_after_axis_perm_mask, ptr[r64_aux_1]);
                for (int i = 0; i < 6; i++) {
                    if (isa == x64::avx512_core) {
                        Xbyak::Opmask kMask2 = Xbyak::Opmask(vAux2.getIdx());
                        vpcmpgtd(kMask2, v_aux_0, v_after_axis_perm_mask);
                        uni_vpsubd(v_after_axis_perm_mask | kMask2, v_after_axis_perm_mask, v_aux_1);
                    } else {
                        vpcmpgtd(vAux2, v_aux_0, v_after_axis_perm_mask);
                        vpandn(vAux2, vAux2, v_aux_1);
                        uni_vpsubd(v_after_axis_perm_mask, v_after_axis_perm_mask, vAux2);
                    }
                }

                process(true, true);
            L(lE2);
        }
        L(lEnd);
    }

    this->postamble();
}

template <>
void Gather<x64::avx2>::uniVpGatherDd(Vmm& vDst, const Xbyak::Address& srcAddr, Vmask& kMask) {
    vpgatherdd(vDst, srcAddr, kMask);
}

template <>
void Gather<x64::avx512_core>::uniVpGatherDd(Vmm& vDst, const Xbyak::Address& srcAddr, Vmask& kMask) {
    vpgatherdd(vDst | kMask, srcAddr);
}

template <>
void Gather<x64::avx2>::normalizeRawIndices(const Vmask& k_dst_mask, const Vmm& v_raw_indices) {
    auto k_aux_mask = getMask();

    // Compensate negative indices
    if (m_jcp.reverse_indexing) {
        vpcmpgtd(k_aux_mask, v_zeros, v_raw_indices);
        vpand(k_aux_mask, k_aux_mask, v_axis_dim);
        uni_vpaddd(v_raw_indices, v_raw_indices, k_aux_mask);
    }
    // Check boundaries
    vpcmpgtd(k_dst_mask, v_axis_dim, v_raw_indices);
    vpcmpgtd(k_aux_mask, v_zeros, v_raw_indices);
    vpandn(k_dst_mask, k_aux_mask, k_dst_mask);
    // Multiply by type size
    if (m_jcp.data_et_size > 1lu) {
        uni_vpslld(v_raw_indices, v_raw_indices, m_data_et_shift);
    }
}

template <>
void Gather<x64::avx512_core>::normalizeRawIndices(const Vmask& k_dst_mask, const Vmm& v_raw_indices) {
    auto k_aux_mask = getMask();

    // Compensate negative indices.
    if (m_jcp.reverse_indexing) {
        vpcmpgtd(k_aux_mask, v_zeros, v_raw_indices);
        uni_vpaddd(v_raw_indices | k_aux_mask, v_raw_indices, v_axis_dim);
    }
    // Check boundaries.
    vpcmpgtd(k_aux_mask, v_axis_dim, v_raw_indices);
    vpcmpd(k_dst_mask | k_aux_mask, v_zeros, v_raw_indices, 2); // 2 - LE
    // Multiply by type size.
    if (m_jcp.data_et_size > 1lu) {
        uni_vpslld(v_raw_indices, v_raw_indices, m_data_et_shift);
    }
}

template <>
void Gather<x64::avx2>::normWithUpperBound(Vmm& vTarget, Vmm& vMax, Vmask& k_aux_mask) {
    vpcmpgtd(k_aux_mask, vMax, vTarget);
    vpandn(k_aux_mask, k_aux_mask, vMax);
    uni_vpsubd(vTarget, vTarget, k_aux_mask);
}

template <>
void Gather<x64::avx512_core>::normWithUpperBound(Vmm& vTarget, Vmm& vMax, Vmask& k_aux_mask) {
    vpcmpd(k_aux_mask, vMax, vTarget, 2); // 2 -> LE
    uni_vpsubd(vTarget | k_aux_mask, vTarget, vMax);
}

// Returns calculated shifts in vAuxPool[0] and mask in vAuxPool[1].
template <>
void Gather<x64::avx2>::calcSrcShiftLong(const Vmm& v_dst_shifts, const Vmask& k_dst_mask, bool shift_first) {
    auto v_aux_0      = getVmm();
    auto v_aux_1      = getVmm();
    auto v_aux_mask_0 = getMask();

    Xbyak::Label lIdxStride, lExit;
    if (shift_first) {
        uni_vpaddd(v_spec_idx_b, v_spec_idx_b, v_vec_len_b);
    }

    add(r64_idx_iter, vlen);
    cmp(r64_idx_iter, r64_spec_idx_size_b);
    jge(lIdxStride, T_NEAR); {
        auto r64_aux = getReg64();
        auto r32_aux = Xbyak::Reg32(r64_aux.getIdx());

        if (m_jcp.batch_dims > 0lu) {
            uni_vpaddd(v_dst_shifts, v_idx_batch_sum_b, v_spec_idx_b);
            uni_vmovd(r32_aux, Xbyak::Xmm(v_dst_shifts.getIdx()));
        } else {
            uni_vmovd(r32_aux, x_spec_idx_b);
        }
        vmovdqu(v_dst_shifts, ptr[r64_idx + r64_aux]);
        normalizeRawIndices(k_dst_mask, v_dst_shifts);
        if (m_jcp.before_axis_size != 1lu) {
            uni_vpaddd(v_dst_shifts, v_dst_shifts, v_src_before_axis_sum_b);
        }
    }
    jmp(lExit, T_NEAR);
    L(lIdxStride);
        sub(r64_idx_iter, r64_spec_idx_size_b);
        vpcmpeqd(k_dst_mask, v_aux_0, v_aux_0);
        if (shift_first) {
            vpcmpgtd(v_aux_0, v_spec_idx_size_b, v_spec_idx_b);
            vpandn(v_aux_1, v_aux_0, v_spec_idx_size_b);
            uni_vpsubd(v_aux_1, v_spec_idx_b, v_aux_1);
            if (m_jcp.batch_dims > 0lu)
                uni_vpaddd(v_aux_1, v_idx_batch_sum_b, v_aux_1);
            uni_vpsubd(v_spec_idx_b, v_spec_idx_b, v_spec_idx_size_b);
        } else {
            if (m_jcp.batch_dims > 0lu) {
                uni_vpaddd(v_aux_0, v_idx_batch_sum_b, v_spec_idx_b);
                gatherdd(v_dst_shifts, r64_idx, v_aux_0, k_dst_mask);
            } else {
                gatherdd(v_dst_shifts, r64_idx, v_spec_idx_b, k_dst_mask);
            }
            normalizeRawIndices(k_dst_mask, v_dst_shifts);

            uni_vpbroadcastd(v_aux_0, x_spec_idx_b);
            vpcmpgtd(v_aux_1, v_aux_0, v_spec_idx_b);
            vpandn(v_aux_0, v_aux_1, v_spec_idx_size_b);
            uni_vpsubd(v_spec_idx_b, v_spec_idx_b, v_aux_0);

            if (m_jcp.before_axis_size != 1lu) {
                uni_vpaddd(v_dst_shifts, v_dst_shifts, v_src_before_axis_sum_b);
                vpandn(v_aux_0, v_aux_1, v_axis_and_after_axis_size_b);
                uni_vpaddd(v_src_before_axis_sum_b, v_src_before_axis_sum_b, v_aux_0);
            }
        }

        if (m_jcp.batch_dims > 0lu) {
            Xbyak::Label l1;
            inc(r64_batch_to_axix_iter);
            cmp(r64_batch_to_axix_iter, r64_batch_to_axis_size);
            jl(l1, T_NEAR);
                mov(r64_batch_to_axix_iter, 0);
                if (shift_first) {
                    uni_vpaddd(v_idx_batch_sum_b, v_idx_batch_sum_b, v_spec_idx_size_b);
                    vpandn(v_dst_shifts, v_aux_0, v_spec_idx_size_b);
                    uni_vpaddd(v_aux_1, v_aux_1, v_dst_shifts);
                } else {
                    vpandn(v_aux_0, v_aux_1, v_spec_idx_size_b);
                    uni_vpaddd(v_idx_batch_sum_b, v_idx_batch_sum_b, v_aux_0);
                }
            L(l1);
        }

        if (shift_first) {
            gatherdd(v_dst_shifts, r64_idx, v_aux_1, k_dst_mask);
            normalizeRawIndices(k_dst_mask, v_dst_shifts);

            if (m_jcp.before_axis_size != 1lu) {
                vpandn(v_aux_0, v_aux_0, v_axis_and_after_axis_size_b);
                uni_vpaddd(v_aux_0, v_aux_0, v_src_before_axis_sum_b);
                uni_vpaddd(v_src_before_axis_sum_b, v_src_before_axis_sum_b, v_axis_and_after_axis_size_b);

                uni_vpaddd(v_dst_shifts, v_dst_shifts, v_aux_0);
            }
        }
    L(lExit);
}

template <>
void Gather<x64::avx512_core>::calcSrcShiftLong(const Vmm& v_dst_shifts, const Vmask& k_dst_mask, bool shift_first) {
    auto v_aux_0      = getVmm();
    auto v_aux_1      = getVmm();
    auto k_aux_mask_0 = getMask();
    auto k_aux_mask_1 = getMask();

    Xbyak::Label lIdxStride, lExit;
    if (shift_first) {
        uni_vpaddd(v_spec_idx_b, v_spec_idx_b, v_vec_len_b);
    }

    add(r64_idx_iter, vlen);
    cmp(r64_idx_iter, r64_spec_idx_size_b);
    jge(lIdxStride, T_NEAR); {
        auto r64_aux = getReg64();
        auto r32_aux = Xbyak::Reg32(r64_aux.getIdx());

        if (m_jcp.batch_dims > 0lu) {
            uni_vpaddd(v_dst_shifts, v_idx_batch_sum_b, v_spec_idx_b);
            uni_vmovd(r32_aux, Xbyak::Xmm(v_dst_shifts.getIdx()));
        } else {
            uni_vmovd(r32_aux, x_spec_idx_b);
        }
        vmovdqu64(v_dst_shifts, ptr[r64_idx + r64_aux]);
        normalizeRawIndices(k_dst_mask, v_dst_shifts);
        if (m_jcp.before_axis_size != 1lu) {
            uni_vpaddd(v_dst_shifts, v_dst_shifts, v_src_before_axis_sum_b);
        }
    }
    jmp(lExit, T_NEAR);
    L(lIdxStride);
        sub(r64_idx_iter, r64_spec_idx_size_b);
        vpcmpeqd(k_dst_mask, v_dst_shifts, v_dst_shifts);
        if (shift_first) {
            vpcmpd(k_aux_mask_1, v_spec_idx_size_b, v_spec_idx_b, 2); // 2 -> LE
            if (m_jcp.batch_dims > 0lu) {
                uni_vpaddd(v_aux_1, v_idx_batch_sum_b, v_spec_idx_b);
                uni_vpsubd(v_aux_1 | k_aux_mask_1, v_aux_1, v_spec_idx_size_b);
            } else {
                uni_vmovups(v_aux_1, v_spec_idx_b);
                uni_vpsubd(v_aux_1 | k_aux_mask_1, v_spec_idx_b, v_spec_idx_size_b);
            }
            uni_vpsubd(v_spec_idx_b, v_spec_idx_b, v_spec_idx_size_b);
        } else {
            if (m_jcp.batch_dims > 0lu) {
                uni_vpaddd(v_aux_0, v_idx_batch_sum_b, v_spec_idx_b);
                gatherdd(v_dst_shifts, r64_idx, v_aux_0, k_dst_mask);
            } else {
                gatherdd(v_dst_shifts, r64_idx, v_spec_idx_b, k_dst_mask);
            }
            normalizeRawIndices(k_dst_mask, v_dst_shifts);

            uni_vpbroadcastd(v_aux_0, x_spec_idx_b);
            vpcmpd(k_aux_mask_1, v_aux_0, v_spec_idx_b, 2); // 2 -> LE
            uni_vpsubd(v_spec_idx_b | k_aux_mask_1, v_spec_idx_b, v_spec_idx_size_b);

            if (m_jcp.before_axis_size != 1lu) {
                uni_vpaddd(v_dst_shifts, v_dst_shifts, v_src_before_axis_sum_b);
                uni_vpaddd(v_src_before_axis_sum_b | k_aux_mask_1, v_src_before_axis_sum_b, v_axis_and_after_axis_size_b);
            }
        }

        if (m_jcp.batch_dims > 0lu) {
            Xbyak::Label l1;
            inc(r64_batch_to_axix_iter);
            cmp(r64_batch_to_axix_iter, r64_batch_to_axis_size);
            jl(l1, T_NEAR);
                mov(r64_batch_to_axix_iter, 0);
                if (shift_first) {
                    uni_vpaddd(v_idx_batch_sum_b, v_idx_batch_sum_b, v_spec_idx_size_b);
                    uni_vpaddd(v_aux_1 | k_aux_mask_1, v_aux_1, v_spec_idx_size_b);
                } else {
                    uni_vpaddd(v_idx_batch_sum_b | k_aux_mask_1, v_idx_batch_sum_b, v_spec_idx_size_b);
                }
            L(l1);
        }

        if (shift_first) {
            gatherdd(v_dst_shifts, r64_idx, v_aux_1, k_dst_mask);
            normalizeRawIndices(k_dst_mask, v_dst_shifts);

            if (m_jcp.before_axis_size != 1lu) {
                uni_vpaddd(v_dst_shifts, v_dst_shifts, v_src_before_axis_sum_b);
                uni_vpaddd(v_dst_shifts | k_aux_mask_1, v_dst_shifts, v_axis_and_after_axis_size_b);
                uni_vpaddd(v_src_before_axis_sum_b, v_src_before_axis_sum_b, v_axis_and_after_axis_size_b);
            }
        }
    L(lExit);
}

template <x64::cpu_isa_t isa>
void Gather<isa>::calcSrcShiftLongBlock(Vmm* vAuxPool, bool shift_first) {
    // Most likely there will no significant performance gain vs memcpy in reference implementation on big blocks after axis,
    // therefore no time was invested to this case yet.
    OPENVINO_THROW("Unsupported case.");
}

// Requires vAuxPool length 3.
// Returns calculated shifts in vAuxPool[0] and mask in vAuxPool[1].
template <x64::cpu_isa_t isa>
void Gather<isa>::calcSrcShiftShort(Vmm* vAuxPool, bool shift_first) {
    auto& v_dst_shifts = vAuxPool[0];
    auto& k_dst_mask = masksContainer[vAuxPool[1].getIdx()];
    auto& v_aux_0 = vAuxPool[2];

    if (shift_first) {
        if (m_jcp.before_axis_size != 1lu)
            uni_vpaddd(v_src_before_axis_sum_b, v_src_before_axis_sum_b, v_before_axis_diff_b);
        // No sense to permute if specIdxSize is one of {1, 2, 4, 8, 16}. 0 is reserved for dynamic case.
        if (m_jcp.spec_idx_size != 1 && m_jcp.spec_idx_size != 2 && m_jcp.spec_idx_size != 4 && m_jcp.spec_idx_size != 8 && m_jcp.spec_idx_size != 16) {
            vpermd(v_spec_idx_b, v_perm_idx_mask, v_spec_idx_b);
            if (m_jcp.before_axis_size != 1lu)
                vpermd(v_before_axis_diff_b, v_perm_idx_mask, v_before_axis_diff_b);
        }
    }

    vpcmpeqd(k_dst_mask, v_aux_0, v_aux_0);
    if (m_jcp.batch_dims > 0lu) {
        // Calculate indices batch sum.
        uni_vcvtdq2ps(v_aux_0, v_src_before_axis_sum_b);
        uni_vcvtdq2ps(v_dst_shifts, v_src_after_batch_size_b);
        uni_vdivps(v_aux_0, v_aux_0, v_dst_shifts);
        uni_vroundps(v_aux_0, v_aux_0, 0x1);
        uni_vcvtps2dq(v_aux_0, v_aux_0);

        uni_vpmulld(v_aux_0, v_aux_0, v_spec_idx_size_b);
        uni_vpaddd(v_aux_0, v_aux_0, v_spec_idx_b);

        uniVpGatherDd(v_dst_shifts, ptr[r64_idx + v_aux_0], k_dst_mask);
    } else {
        uniVpGatherDd(v_dst_shifts, ptr[r64_idx + v_spec_idx_b], k_dst_mask);
    }

    auto& k_aux_mask_0 = masksContainer[v_aux_0.getIdx()];
    normalizeRawIndices(k_dst_mask, v_dst_shifts);
    if (m_jcp.before_axis_size != 1lu)
        uni_vpaddd(v_dst_shifts, v_dst_shifts, v_src_before_axis_sum_b);
}

// Requires vAuxPool length 4.
// Returns calculated shifts in vAuxPool[0] and mask in vAuxPool[1].
template <x64::cpu_isa_t isa>
void Gather<isa>::calcSrcShiftShortBlock(Vmm* vAuxPool, bool shift_first) {
    auto& v_dst_shifts = vAuxPool[0];
    auto& k_dst_mask = masksContainer[vAuxPool[1].getIdx()];
    auto& v_aux_0 = vAuxPool[2];
    auto& v_aux_1 = vAuxPool[3];
    auto& k_aux_mask_0 = masksContainer[v_aux_0.getIdx()];
    const uint64_t specIdxAndAfterAxisSize = m_jcp.spec_idx_size * m_jcp.after_axis_size;

    if (shift_first) {
        if (m_jcp.spec_idx_size != 1) {
            uni_vpaddd(v_spec_idx_b, v_spec_idx_b, v_spec_idx_diff);
            normWithUpperBound(v_spec_idx_b, v_spec_idx_size_b, k_aux_mask_0);
        }
        // No sense to permute if afterAxisSize is one of {1, 2, 4, 8, 16}. 0 is reserved for dynamic case.
        if (m_jcp.after_axis_size != 1 && m_jcp.after_axis_size != 2 && m_jcp.after_axis_size != 4 && m_jcp.after_axis_size != 8 && m_jcp.after_axis_size != 16) {
            vpermd(v_after_axis_idx_b, v_after_axis_perm_mask, v_after_axis_idx_b);
            if (m_jcp.spec_idx_size != 1)
                vpermd(v_spec_idx_diff, v_after_axis_perm_mask, v_spec_idx_diff);
        }

        if (m_jcp.before_axis_size != 1lu) {
            if (!m_jcp.dynamic_shapes) {
                if (specIdxAndAfterAxisSize > 0lu && specIdxAndAfterAxisSize <= m_idx_el_per_vec) {
                    uni_vpaddd(v_src_before_axis_sum_b, v_src_before_axis_sum_b, v_before_axis_diff_b);
                    uni_vmovups(v_aux_1, v_src_before_axis_sum_b);
                    if (specIdxAndAfterAxisSize != 1 && specIdxAndAfterAxisSize != 2 && specIdxAndAfterAxisSize != 4 &&
                            specIdxAndAfterAxisSize != 8 && specIdxAndAfterAxisSize != 16) {
                        vpermd(v_before_axis_diff_b, v_before_axis_perm_mask, v_before_axis_diff_b);
                    }
                } else {
                    Xbyak::Label lBeforeAxStep, lBeforeAxStepEnd;
                    add(rSpecIdxAndAfterAxIterB, m_idx_el_per_vec * m_jcp.data_et_size);
                    cmp(rSpecIdxAndAfterAxIterB, rSpecIdxAndAfterAxSizeB);
                    jl(lBeforeAxStep, T_NEAR);
                        sub(rSpecIdxAndAfterAxIterB, rSpecIdxAndAfterAxSizeB);

                        vpmulld(v_aux_0, v_spec_idx_b, v_after_axis_size);
                        uni_vpaddd(v_aux_0, v_aux_0, v_after_axis_idx_b);
                        auto xmm_aux_0 = getXmm();
                        uni_vpbroadcastd(v_aux_1, xmm_aux_0);
                        if (isa == x64::avx512_core) {
                            Xbyak::Opmask kMask0 = Xbyak::Opmask(k_aux_mask_0.getIdx());
                            vpcmpgtd(kMask0, v_aux_1, v_aux_0);
                            uni_vmovups(v_aux_1, v_src_before_axis_sum_b);
                            uni_vpaddd(v_aux_1 | kMask0, v_src_before_axis_sum_b, v_axis_and_after_axis_size_b);
                        } else {
                            vpcmpgtd(v_aux_1, v_aux_1, v_aux_0);
                            vpand(v_aux_1, v_aux_1, v_axis_and_after_axis_size_b);
                            uni_vpaddd(v_aux_1, v_src_before_axis_sum_b, v_aux_1);
                        }
                        uni_vpaddd(v_src_before_axis_sum_b, v_src_before_axis_sum_b, v_axis_and_after_axis_size_b);
                        jmp(lBeforeAxStepEnd);
                    L(lBeforeAxStep);
                        uni_vmovups(v_aux_1, v_src_before_axis_sum_b);
                    L(lBeforeAxStepEnd);
                }
            } else {
            }
        }
    } else {
        if (m_jcp.before_axis_size != 1lu) {
            uni_vmovups(v_aux_1, v_src_before_axis_sum_b);
            if (specIdxAndAfterAxisSize > m_idx_el_per_vec) {
                // Broadcast the last element.
                if (isa == x64::avx512_core) {
                    vshuff64x2(v_src_before_axis_sum_b, v_src_before_axis_sum_b, v_src_before_axis_sum_b, 0xFF);
                } else {
                    vpermq(v_src_before_axis_sum_b, v_src_before_axis_sum_b, 0xFF);
                }
                vpshufd(v_src_before_axis_sum_b, v_src_before_axis_sum_b, 0xFF);

                Xbyak::Label lBeforeAxStepEnd1;
                add(rSpecIdxAndAfterAxIterB, m_idx_el_per_vec * m_jcp.data_et_size);
                cmp(rSpecIdxAndAfterAxIterB, rSpecIdxAndAfterAxSizeB);
                jl(lBeforeAxStepEnd1, T_NEAR);
                    sub(rSpecIdxAndAfterAxIterB, rSpecIdxAndAfterAxSizeB);
                cmp(rSpecIdxAndAfterAxIterB, 0);
                jne(lBeforeAxStepEnd1, T_NEAR);
                    uni_vpaddd(v_src_before_axis_sum_b, v_src_before_axis_sum_b, v_axis_and_after_axis_size_b);
                L(lBeforeAxStepEnd1);
            }
        }
    }

    vpcmpeqd(k_dst_mask, v_aux_0, v_aux_0);
    if (m_jcp.batch_dims > 0lu) {
        // Calculate indices batch sum.
        uni_vcvtdq2ps(v_aux_0, v_aux_1);
        uni_vcvtdq2ps(v_dst_shifts, v_src_after_batch_size_b);
        uni_vdivps(v_aux_0, v_aux_0, v_dst_shifts);
        uni_vroundps(v_aux_0, v_aux_0, 0x1);
        uni_vcvtps2dq(v_aux_0, v_aux_0);

        uni_vpmulld(v_aux_0, v_aux_0, v_spec_idx_size_b);
        uni_vpaddd(v_aux_0, v_aux_0, v_spec_idx_b);

        uniVpGatherDd(v_dst_shifts, ptr[r64_idx + v_aux_0], k_dst_mask);
    } else {
        uniVpGatherDd(v_dst_shifts, ptr[r64_idx + v_spec_idx_b], k_dst_mask);
    }

    normalizeRawIndices(k_dst_mask, v_dst_shifts);

    if (m_jcp.after_axis_size != 1lu) {
        vpmulld(v_dst_shifts, v_dst_shifts, v_after_axis_size);
        uni_vpaddd(v_dst_shifts, v_dst_shifts, v_after_axis_idx_b);
    }
    if (m_jcp.before_axis_size != 1lu)
        uni_vpaddd(v_dst_shifts, v_dst_shifts, v_aux_1);
}

template <x64::cpu_isa_t isa>
void Gather<isa>::process(bool isShortIdx, bool blocked) {
    Xbyak::Label lTailProc, lEndProc;
    cmp(r64_work_amount, m_data_el_per_vec);
    jl(lTailProc, T_NEAR);
        if (m_jcp.data_et_size == 4)
            process32b(isShortIdx, blocked);
        else if (m_jcp.data_et_size == 2)
            process16b(isShortIdx, blocked);
        else if (m_jcp.data_et_size == 1)
            process8b(isShortIdx, blocked);
    jmp(lEndProc, T_NEAR);
    L(lTailProc);
        tail(isShortIdx, false, blocked);
    L(lEndProc);
}

template <x64::cpu_isa_t isa>
void Gather<isa>::process32b(bool isShortIdx, bool blocked) {
    Xbyak::Label lDstIdxLoop, lTail;

    // First iteration
    shiftIdxAndGather(vmmAuxContainer, isShortIdx, false, blocked);
    uni_vmovups(ptr[r64_dst], vmmAuxContainer[2]);

    // Main loop
    L(lDstIdxLoop);
    {
        add(r64_dst, vlen);
        sub(r64_work_amount, m_data_el_per_vec);
        cmp(r64_work_amount, m_data_el_per_vec);
        jl(lTail, T_NEAR);

        shiftIdxAndGather(vmmAuxContainer, isShortIdx, true, blocked);
        uni_vmovups(ptr[r64_dst], vmmAuxContainer[2]);

        jmp(lDstIdxLoop, T_NEAR);
    }

    L(lTail);
    tail(isShortIdx, true, blocked);
}

template <x64::cpu_isa_t isa>
void Gather<isa>::process16b(bool isShortIdx, bool blocked) {
    Xbyak::Label lDstIdxLoop1, lTail;

    Vmm vShufMask, vPermMask, vBuff0;
    if (isa == x64::avx512_core) {
        vPermMask = vmmAuxContainer[7];
        vShufMask = vmmAuxContainer[8];
        vBuff0    = vmmAuxContainer[9];
    } else {
        vPermMask = vmmAuxContainer[1];
        vShufMask = vmmAuxContainer[4];
        vBuff0    = vmmAuxContainer[5];
    }

    mov(r64_aux_1, reinterpret_cast<uintptr_t>(shufMask16bitUni));
    uni_vmovups(vShufMask, ptr[r64_aux_1]);

    // First iteration
    shiftIdxAndGather(vmmAuxContainer, isShortIdx, false, blocked);
    vpshufb(vBuff0, vmmAuxContainer[2], vShufMask);

    shiftIdxAndGather(vmmAuxContainer, isShortIdx, true, blocked);
    vpshufb(vmmAuxContainer[0], vmmAuxContainer[2], vShufMask);

    vshufps(vmmAuxContainer[0], vBuff0, vmmAuxContainer[0], 0x44);
    // vPermMask(vmm1) is override in shiftIdxAndGather, load the mask here for correctness
    mov(r64_aux_1, reinterpret_cast<uintptr_t>(permMask16bitUni));
    uni_vmovups(vPermMask, ptr[r64_aux_1]);
    vpermd(vmmAuxContainer[0], vPermMask, vmmAuxContainer[0]);

    uni_vmovups(ptr[r64_dst], vmmAuxContainer[0]);

    // Main loop.
    L(lDstIdxLoop1);
    {
        add(r64_dst, vlen);
        sub(r64_work_amount, m_data_el_per_vec);
        cmp(r64_work_amount, m_data_el_per_vec);
        jl(lTail, T_NEAR);

        shiftIdxAndGather(vmmAuxContainer, isShortIdx, true, blocked);
        vpshufb(vBuff0, vmmAuxContainer[2], vShufMask);

        shiftIdxAndGather(vmmAuxContainer, isShortIdx, true, blocked);
        vpshufb(vmmAuxContainer[0], vmmAuxContainer[2], vShufMask);

        vshufps(vmmAuxContainer[0], vBuff0, vmmAuxContainer[0], 0x44);
        if (isa == x64::avx2) {
            // Register vPermMask is invalidated by shiftIdxAndGather and must be initialized again.
            mov(r64_aux_1, reinterpret_cast<uintptr_t>(permMask16bitUni));
            uni_vmovups(vPermMask, ptr[r64_aux_1]);
        }
        vpermd(vmmAuxContainer[0], vPermMask, vmmAuxContainer[0]);

        uni_vmovups(ptr[r64_dst], vmmAuxContainer[0]);

        jmp(lDstIdxLoop1, T_NEAR);
    }

    L(lTail);
    tail(isShortIdx, true, blocked);
}

template <x64::cpu_isa_t isa>
void Gather<isa>::process8b(bool isShortIdx, bool blocked) {
    Xbyak::Label lDstIdxLoop1, lTail;

    Vmm vShufMask, vPermMask, vBuff0, vBuff1;
    if (isa == x64::avx512_core) {
        vPermMask = vmmAuxContainer[7];
        vShufMask = vmmAuxContainer[8];
        vBuff0    = vmmAuxContainer[9];
        vBuff1    = vmmAuxContainer[10];
    } else {
        vPermMask = vmmAuxContainer[1];
        vShufMask = vmmAuxContainer[4];
        vBuff0    = vmmAuxContainer[5];
        vBuff1    = vmmAuxContainer[6];
    }
    mov(r64_aux_1, reinterpret_cast<uintptr_t>(shufMask8bitUni));
    uni_vmovups(vShufMask, ptr[r64_aux_1]);

    // First iteration
    shiftIdxAndGather(vmmAuxContainer, isShortIdx, false, blocked);
    vpshufb(vBuff0, vmmAuxContainer[2], vShufMask);

    shiftIdxAndGather(vmmAuxContainer, isShortIdx, true, blocked);
    vpshufb(vmmAuxContainer[0], vmmAuxContainer[2], vShufMask);

    vshufps(vBuff0, vBuff0, vmmAuxContainer[0], 0x0);

    shiftIdxAndGather(vmmAuxContainer, isShortIdx, true, blocked);
    vpshufb(vBuff1, vmmAuxContainer[2], vShufMask);

    shiftIdxAndGather(vmmAuxContainer, isShortIdx, true, blocked);
    vpshufb(vmmAuxContainer[0], vmmAuxContainer[2], vShufMask);

    vshufps(vBuff1, vBuff1, vmmAuxContainer[0], 0x0);
    vshufps(vmmAuxContainer[0], vBuff0, vBuff1, 0x88);

    mov(r64_aux_1, reinterpret_cast<uintptr_t>(permMask8bitUni));
    uni_vmovups(vPermMask, ptr[r64_aux_1]);

    vpermd(vmmAuxContainer[0], vPermMask, vmmAuxContainer[0]);

    uni_vmovups(ptr[r64_dst], vmmAuxContainer[0]);

    // Main loop.
    L(lDstIdxLoop1);
    {
        add(r64_dst, vlen);
        sub(r64_work_amount, m_data_el_per_vec);
        cmp(r64_work_amount, m_data_el_per_vec);
        jl(lTail, T_NEAR);

        shiftIdxAndGather(vmmAuxContainer, isShortIdx, true, blocked);
        vpshufb(vBuff0, vmmAuxContainer[2], vShufMask);

        shiftIdxAndGather(vmmAuxContainer, isShortIdx, true, blocked);
        vpshufb(vmmAuxContainer[0], vmmAuxContainer[2], vShufMask);

        vshufps(vBuff0, vBuff0, vmmAuxContainer[0], 0x0);

        shiftIdxAndGather(vmmAuxContainer, isShortIdx, true, blocked);
        vpshufb(vBuff1, vmmAuxContainer[2], vShufMask);

        shiftIdxAndGather(vmmAuxContainer, isShortIdx, true, blocked);
        vpshufb(vmmAuxContainer[0], vmmAuxContainer[2], vShufMask);

        vshufps(vmmAuxContainer[0], vBuff1, vmmAuxContainer[0], 0x0);
        vshufps(vmmAuxContainer[0], vBuff0, vmmAuxContainer[0], 0x88);

        if (isa == x64::avx2) {
            // Register vPermMask is invalidated by shiftIdxAndGather and must be initialized again.
            mov(r64_aux_1, reinterpret_cast<uintptr_t>(permMask8bitUni));
            uni_vmovups(vPermMask, ptr[r64_aux_1]);
        }
        vpermd(vmmAuxContainer[0], vPermMask, vmmAuxContainer[0]);

        uni_vmovups(ptr[r64_dst], vmmAuxContainer[0]);

        jmp(lDstIdxLoop1, T_NEAR);
    }

    L(lTail);
    tail(isShortIdx, true, blocked);
}

// Requires vAuxPool length 4.
// Returns gathered data in vAuxPool[2].
template <x64::cpu_isa_t isa>
void Gather<isa>::shiftIdxAndGather(Vmm* vAuxPool, bool isShortIdx, bool shift_first, bool blocked) {
    auto k_gather_mask = getMask();

    if (blocked) {
        if (isShortIdx) {
            calcSrcShiftShortBlock(vAuxPool, shift_first);
        } else {
            calcSrcShiftLongBlock(vAuxPool, shift_first);
        }
    } else {
        if (isShortIdx) {
            calcSrcShiftShort(vAuxPool, shift_first);
        } else {
            calcSrcShiftLong(k_gather_mask, shift_first);
        }
    }

    uni_vmovups(vAuxPool[2], v_zeros);
    uniVpGatherDd(vAuxPool[2], ptr[r64_src + vAuxPool[0]], k_gather_mask);
}

template <x64::cpu_isa_t isa>
void Gather<isa>::tail(bool isShortIdx, bool shift_first, bool blocked) {
    auto& vSrcShift = getVmm();
    auto& kGatherMask = masksContainer[vmmAuxContainer[1].getIdx()];
    auto& v_aux_0 = getVmm();
    auto& v_aux_1 = getVmm();
    auto& k_aux_mask_1 = masksContainer[v_aux_1.getIdx()];
    Xbyak::Label lEnd;

    const int secondStepCycles = 4 / m_jcp.data_et_size;
    for (int p = 0; p < secondStepCycles; p++) {
        cmp(r64_work_amount, 0);
        jle(lEnd, T_NEAR);

        if (isShortIdx) {
            if (blocked) {
                calcSrcShiftShortBlock(vmmAuxContainer, p > 0 || shift_first);
            } else {
                calcSrcShiftShort(vmmAuxContainer, p > 0 || shift_first);
            }
        } else {
            if (blocked) {
                calcSrcShiftLongBlock(vmmAuxContainer, p > 0 || shift_first);
            } else {
                calcSrcShiftLong(vmmAuxContainer, p > 0 || shift_first);
            }
        }

        // fillRestWorkMask(k_aux_mask_1, v_aux_0, r64_work_amount, r64_aux_1, rdx);
        fillRestWorkMask(k_aux_mask_1, r64_work_amount, m_jcp.idx_et_size);

        // Combining masks.
        if (isa == x64::avx512_core) {
            auto kMask1 = Xbyak::Opmask(k_aux_mask_1.getIdx());
            auto kMaskG = Xbyak::Opmask(kGatherMask.getIdx());
            kandd(kMaskG, kMaskG, kMask1);
        } else if (isa == x64::avx2) {
            auto& vGatherMask = vmmAuxContainer[kGatherMask.getIdx()];
            vpand(vGatherMask, vGatherMask, v_aux_1);
        }

        uni_vmovups(v_aux_0, v_zeros);
        uniVpGatherDd(v_aux_0, ptr[r64_src + vSrcShift], kGatherMask);
        if (m_jcp.data_et_size == 4) {
            uni_vmovups_tail(ptr[r64_dst], k_aux_mask_1, v_aux_0);
            sub(r64_work_amount, m_data_el_per_vec);
        } else {
            storeVectorPart(r64_dst, r64_work_amount, v_aux_0);
        }
    }
    L(lEnd);
}

template <x64::cpu_isa_t isa>
void Gather<isa>::storeVectorPart(const Xbyak::Reg64& r64_dst_ptr, const Xbyak::Reg64& r64_el_num, const Vmm& v_src) {
    Xbyak::Label lEnd;
    auto xmm_aux = getXmm();

    for (size_t j = 0lu; j < vlen / vlenXmm; j++) {
        if (isa == x64::avx2) {
            vextracti128(xmm_aux, v_src, j);
        } else if (isa == x64::avx512_core) {
            vextracti64x2(xmm_aux, v_src, j);
        }

        for (int k = 0; k < 4; k++) {
            cmp(r64_el_num, 0);
            jle(lEnd, T_NEAR);

            if (m_jcp.data_et_size == 8) {
                // uni_vpextrd(ptr[r64_dst_ptr], xmm_aux, k);
            } else if (m_jcp.data_et_size == 4) {
                uni_vpextrd(ptr[r64_dst_ptr], xmm_aux, k);
            } else if (m_jcp.data_et_size == 2) {
                uni_vpextrw(ptr[r64_dst_ptr], xmm_aux, k * 2);
            } else if (m_jcp.data_et_size == 1) {
                uni_vpextrb(ptr[r64_dst_ptr], xmm_aux, k * 4);
            }

            add(r64_dst_ptr, m_jcp.data_et_size);
            sub(r64_el_num, 1);
        }
    }
    L(lEnd);
}

template <>
void Gather<x64::avx512_core>::fillVlenVector() {
    if (!v_vec_len_b.isInitialized()) {
        v_vec_len_b = getVmm();
    }
    auto r32_aux = getReg32();
    mov(r32_aux, vlen);
    vpbroadcastd(v_vec_len_b, r32_aux);
}

template <>
void Gather<x64::avx2>::fillVlenVector() {
    if (!v_vec_len_b.isInitialized()) {
        v_vec_len_b = getVmm();
    }
    vpcmpeqd(v_vec_len_b, v_vec_len_b, v_vec_len_b);
    vpsrld(v_vec_len_b, v_vec_len_b, 31);     // Right shift to 1.
    uni_vpslld(v_vec_len_b, v_vec_len_b, 5);  // Left shift to 32.
}

template class Gather<x64::avx2>;
template class Gather<x64::avx512_core>;

}   // namespace kernel
}   // namespace intel_cpu
}   // namespace ov
