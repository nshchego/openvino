// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Gather kernel implements two approaches for indices calculation: "Short" and "Long".
// 1. Short approach is applicable for cases when the number of elements less or equal to vector register length.
// It just uses permutation of current indices vector to retrieve the next.
// 2. Long approach is applicable for cases when the number of elements is greater than vector register length.
// It increases indices in vector on vector length and normalizes upper bound of indices.
//
//                    SUPPORTED CASES
//--------------------------------------------------------------
//  After axis |         AVX512        |         AVX2          |
// (block) size| 32bit | 16bit |  8bit | 32bit | 16bit |  8bit |
//                      STATIC SHAPES
//      1      |   X   |   X   |   X   |   X   |   X   |   X   |
// >1 & <=vlen |   X   |   X   |   X   |   X   |       |       |
//                      DYNAMIC SHAPES
//      1      |   X   |   X   |   X   |   X   |   X   |   X   |
//--------------------------------------------------------------


#pragma once

#include "jit_kernel_base.hpp"
#include "cpu/x64/jit_generator.hpp"
// #include "dnnl_types.h"

namespace ov {
namespace intel_cpu {
namespace kernel {

struct GatherCompileParams {
    uint64_t data_et_size     = 1lu;
    uint64_t idx_et_size      = 4lu; // i32 and i64 are supported
    uint64_t batch_dims       = 0lu;
    uint64_t before_axis_size = 0lu;
    uint64_t spec_idx_size    = 0lu;
    uint64_t after_axis_size  = 0lu;
    bool     dynamic_shapes   = false;
    bool     reverse_indexing = true;
};

struct GatherCallArgs {
    const void* src;
    const void* indices;
    void* dst;
    const int* axisDim;
    const uint64_t* start;
    const uint64_t* specIndicesSize;
    const uint64_t* betweenBatchAndAxisSize;
    const uint64_t* axisAndAfterAxisSizeB;
    const uint64_t* srcAfterBatchSizeB;
    const int* permIdxMask;
    const int* beforeAxisDiff;

    const int* beforeAxisPermMask;
    const int* afterAxIdxB;
    const int* afterAxisPermMask;
    const uint64_t* afterAxisSize;
    const int* specIdxDiff;

    uint64_t work_amount = 0lu;
    uint64_t afterAxSize = 1lu;
    // Blocked short.
    uint64_t specIdxAndAfterAxIterB;
    uint64_t specIdxAndAfterAxSizeB;
    // Only static
    const int* specIdxB;
    const int* idxBatchSumB;
    const int* dataBeforeAxisSumB;
    uint64_t betweenBatchAndAxisIter;
};

// struct jitGatherKernelBase {
//     void (*ker_)(const GatherCallArgs *);
//     void operator()(const GatherCallArgs *args) {
//         assert(ker_);
//         ker_(args);
//     }
//     explicit jitGatherKernelBase(const GatherCompileParams& jcp) : ker_(nullptr), jcp(jcp) {}
//     virtual ~jitGatherKernelBase() {}

//     virtual void create_ker() = 0;
//     uint64_t getVecLen() {
//         return vlen;
//     }
//     uint64_t getDataElPerVec() {
//         return dataElPerVec;
//     }
//     uint64_t getIdxElPerVec() {
//         return idxElPerVec;
//     }
//     virtual bool isSupportedConfiguration(uint64_t afterAxisSize) = 0;

// protected:
//     GatherCompileParams jcp;
//     uint64_t vlen = 0lu;
//     uint64_t dataElPerVec = 0lu;
//     uint64_t idxElPerVec = 0lu;
//     static const unsigned shufMask8bitUni[16];
//     static const unsigned permMask8bitA2[8];
//     static const unsigned permMask8bitA5[16];
//     static const unsigned shufMask16bitUni[16];
//     static const unsigned permMask16bitA2[8];
//     static const unsigned permMask16bitA5[16];
//     static const unsigned incVec[16];

//     int shortPermIdx[16];
//     int shortBeforeAxisDiff[16];
// };

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
class Gather : public JitKernel<GatherCompileParams, GatherCallArgs> {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(Gather)

    explicit Gather(const GatherCompileParams& jcp);

    void generate() override;

protected:
    using Vmm = typename dnnl::impl::utils::conditional<isa == dnnl::impl::cpu::x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    using Vmask = typename dnnl::impl::utils::conditional<isa == dnnl::impl::cpu::x64::avx2, Xbyak::Ymm, Xbyak::Opmask>::type;
    static const uint32_t vlenXmm = dnnl::impl::cpu::x64::cpu_isa_traits<dnnl::impl::cpu::x64::sse41>::vlen;
    static const uint8_t idxTypeShift = 2;
    uint8_t m_data_et_shift = 0;
    uint64_t m_data_el_per_vec = 0lu;
    uint64_t m_idx_el_per_vec = 0lu;

    static const unsigned permMask8bitUni[16];
    static const unsigned permMask16bitUni[16];
    static const unsigned shufMask8bitUni[16];
    static const unsigned shufMask16bitUni[16];
    static const unsigned m_inc_vec[16];// = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    // Suffix b means "In Bytes"
    // 64b registers
    RegistersPool::Reg<Xbyak::Reg64> r64_src;
    RegistersPool::Reg<Xbyak::Reg64> r64_dst;
    // RegistersPool::Reg<Xbyak::Reg64> r64_aux_1;
    // RegistersPool::Reg<Xbyak::Reg64> r64_aux_2;
    RegistersPool::Reg<Xbyak::Reg64> r64_batch_to_axix_iter;
    RegistersPool::Reg<Xbyak::Reg64> r64_batch_to_axis_size;
    RegistersPool::Reg<Xbyak::Reg64> r64_idx;
    RegistersPool::Reg<Xbyak::Reg64> r64_idx_iter;
    RegistersPool::Reg<Xbyak::Reg64> r64_spec_idx_and_after_axis_iter_b;
    RegistersPool::Reg<Xbyak::Reg64> r64_spec_idx_and_after_axis_size_b;
    RegistersPool::Reg<Xbyak::Reg64> r64_spec_idx_size_b;
    RegistersPool::Reg<Xbyak::Reg64> r64_work_amount;

    const Xbyak::Reg64 regParams = Xbyak::Reg64(dnnl::impl::cpu::x64::abi_param_regs[0]);

    // 32b registers
    Xbyak::Reg32 r32_idx_iter;
    Xbyak::Reg32 r32_batch_to_axis_size;
    Xbyak::Reg32 r32_batch_to_axis_iter;
    Xbyak::Reg32 r32_spec_idx_size_b;
    // Xbyak::Reg32 reg32Aux1 = Xbyak::Reg32(regAux1.getIdx());
    // Xbyak::Reg32 reg32Aux2 = Xbyak::Reg32(regAux2.getIdx());

    // Common
    RegistersPool::Reg<Vmm> v_axis_and_after_axis_size_b;
    RegistersPool::Reg<Vmm> v_axis_dim;
    RegistersPool::Reg<Vmm> v_spec_idx_b;
    RegistersPool::Reg<Vmm> v_spec_idx_size_b;
    RegistersPool::Reg<Vmm> v_src_before_axis_sum_b;
    RegistersPool::Reg<Vmm> v_zeros;

    // Only short
    RegistersPool::Reg<Vmm> v_src_after_batch_size_b;
    RegistersPool::Reg<Vmm> v_perm_idx_mask;
    RegistersPool::Reg<Vmm> v_before_axis_diff_b;
    // Blocked short
    RegistersPool::Reg<Vmm> v_spec_idx_diff;
    RegistersPool::Reg<Vmm> v_after_axis_size;
    RegistersPool::Reg<Vmm> v_after_axis_idx_b;
    RegistersPool::Reg<Vmm> v_after_axis_perm_mask;
    RegistersPool::Reg<Vmm> v_before_axis_perm_mask;
    // Only long
    RegistersPool::Reg<Vmm> v_vec_len_b;
    RegistersPool::Reg<Vmm> v_idx_batch_sum_b;

    // XMM
    Xbyak::Xmm xmmSrcBeforeAxisSum = Xbyak::Xmm(vmmSrcBeforeAxisSumB.getIdx());
    Xbyak::Xmm xmmSpecIdxSizeB = Xbyak::Xmm(v_spec_idx_size_b.getIdx());
    Xbyak::Xmm xmmSpecIdxB = Xbyak::Xmm(v_spec_idx_b.getIdx());
    RegistersPool::Reg<Xbyak::Xmm> x_spec_idx_b;


    void calcSrcShiftLong(const Vmm& v_dst_shifts, const Vmask& k_dst_mask, bool shift_first = true);
    void calcSrcShiftLongBlock(Vmm* vAuxPool, bool shift_first = true);
    void calcSrcShiftShort(Vmm* vAuxPool, bool shift_first = true);
    void calcSrcShiftShortBlock(Vmm* vAuxPool, bool shift_first);
    void process(bool isShortIdx, bool blocked);
    void process32b(bool isShortIdx, bool blocked);
    void process16b(bool isShortIdx, bool blocked);
    void process8b(bool isShortIdx, bool blocked);
    void shiftIdxAndGather(Vmm* vAuxPool, bool isShortIdx, bool shift_first, bool blocked);
    void tail(bool isShortIdx, bool shift_first = true, bool blocked = false);
    // Aux functions.
    void normalizeRawIndices(const Vmask& k_dst_mask, const Vmm& v_raw_indices);
    void normWithUpperBound(Vmm& vTarget, Vmm& vMax, Vmask& kAuxMask);
    void storeVectorPart(const Xbyak::Reg64& rDst, const Xbyak::Reg64& r64_el_num, const Vmm& vmmSrc);
    void uniVpGatherDd(Vmm& vDst, const Xbyak::Address& srcAddr, Vmask& vMask);
    void fillVlenVector();

    // const unsigned* permMask8bitUni;
    // const unsigned* permMask16bitUni;
};

}   // namespace kernel
}   // namespace intel_cpu
}   // namespace ov
