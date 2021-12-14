// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu/x64/jit_generator.hpp"
#include <mkldnn_types.h>

namespace MKLDNNPlugin {

struct jGatherConfParams {
    uint64_t dataTypeSize = 1lu;
    bool reverseIndexing = true;
    bool dynamicShapes = false;
    uint64_t dataAfterAxisSize = 1lu;
};

struct gatherJitExecArgs {
    const void* src;
    const void* indices;
    void* dst;
    const int* axisDim;
    const uint64_t* start;
    const uint64_t* specIndicesSize;
    const uint64_t* betweenBatchAndAxisSize;
    const uint64_t* axisAndAfterAxisSizeB;
    const uint64_t* srcAfterBatchSizeB;
    const int* permIdx;
    const int* beforeAxisDiff;
    uint64_t workAmount = 0;
    uint64_t afterAxSize = 0;
    // Only static
    const int* specIdxB;
    const int* idxBatchSumB;
    const int* dataBeforeAxisSumB;
    uint64_t betweenBatchAndAxisIter;
};

struct jitGatherKernelBase {
    void (*ker_)(const gatherJitExecArgs *);
    void operator()(const gatherJitExecArgs *args) {
        assert(ker_);
        ker_(args);
    }
    explicit jitGatherKernelBase(jGatherConfParams jcp) : ker_(nullptr), jcp(jcp) {}
    virtual ~jitGatherKernelBase() {}

    virtual void create_ker() = 0;
    inline uint64_t getVecLen() {
        return vlen;
    }
    inline uint64_t getDataElPerVec() {
        return dataElPerVec;
    }
    inline uint64_t getIdxElPerVec() {
        return idxElPerVec;
    }

protected:
    jGatherConfParams jcp;
    uint64_t vlen;
    uint64_t dataElPerVec;
    uint64_t idxElPerVec;
    static const unsigned shufMask8bitUni[16];
    static const unsigned permMask8bitA2[8];
    static const unsigned permMask8bitA5[16];
    static const unsigned shufMask16bitUni[16];
    static const unsigned permMask16bitA2[8];
    static const unsigned permMask16bitA5[16];
    static const unsigned incVec[16];

    int shortPermIdx[16];
    int shortBeforeAxisDiff[16];
};

template <mkldnn::impl::cpu::x64::cpu_isa_t isa>
struct jitUniGatherKernel : public jitGatherKernelBase, public mkldnn::impl::cpu::x64::jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jitUniGatherKernel)

    explicit jitUniGatherKernel(jGatherConfParams jcp);

    void create_ker() override;
    void generate() override;

protected:
    using Vmm = typename mkldnn::impl::utils::conditional<isa == mkldnn::impl::cpu::x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    using Vmask = typename mkldnn::impl::utils::conditional<isa == mkldnn::impl::cpu::x64::avx2, Xbyak::Ymm, Xbyak::Opmask>::type;
    const uint32_t vlenXmm = mkldnn::impl::cpu::x64::cpu_isa_traits<mkldnn::impl::cpu::x64::sse41>::vlen;
    const uint32_t indicesTypeSize = sizeof(uint32_t);
    const uint32_t idxTypeShift = indicesTypeSize / 2;
    uint32_t dataTypeShift = 0;

    // 64b registers.
    const Xbyak::Reg64& regSrc = r8;
    const Xbyak::Reg64& regDst = r9;
    const Xbyak::Reg64& regIndices = r10;
    const Xbyak::Reg64& regIdxIter = r11;
    const Xbyak::Reg64& regWorkAmount = r12;
    const Xbyak::Reg64& regSpecIdxSizeB = r13;
    const Xbyak::Reg64& regAux1 = r14;
    const Xbyak::Reg64& regAux2 = rsi;
    const Xbyak::Reg64& regBetweenBatchAndAxisIter = r15;
    const Xbyak::Reg64& regBetweenBatchAndAxisSize = rbx;

    const Xbyak::Reg64& regParams = mkldnn::impl::cpu::x64::abi_param1;

    // 32b registers.
    Xbyak::Reg32 reg32IdxIter = Xbyak::Reg32(regIdxIter.getIdx());
    Xbyak::Reg32 reg32SpecIdxSizeB = Xbyak::Reg32(regSpecIdxSizeB.getIdx());
    Xbyak::Reg32 reg32BetweenBatchAndAxisSize = Xbyak::Reg32(regBetweenBatchAndAxisSize.getIdx());
    Xbyak::Reg32 reg32BetweenBatchAndAxisIter = Xbyak::Reg32(regBetweenBatchAndAxisIter.getIdx());
    Xbyak::Reg32 reg32Aux1 = Xbyak::Reg32(regAux1.getIdx());
    Xbyak::Reg32 reg32Aux2 = Xbyak::Reg32(regAux2.getIdx());

    // Opmasks.
    Vmask masksContainer[7] = {Vmask(0), Vmask(1), Vmask(2), Vmask(3), Vmask(4), Vmask(5), Vmask(6)};

    // Auxiliary.
    Vmm vmmAuxContainer[8] = {Vmm(0), Vmm(1), Vmm(2), Vmm(3), Vmm(4), Vmm(5), Vmm(6), /*AVX5*/ Vmm(16)};
    // Common.
    Vmm vmmZeros = Vmm(7);
    Vmm vmmSrcBeforeAxisSumB = Vmm(8);
    Vmm vmmSpecIdxB = Vmm(9);
    Vmm vmmSpecIdxSizeB = Vmm(10);
    Vmm vmmAxisDim = Vmm(11);

    // Only long.
    Vmm vmmAxisAndAfterAxisSizeB = Vmm(12);
    Vmm vmmVecLen = Vmm(13);
    Vmm vmmIdxBatchSumB = Vmm(14);
    // Only short.
    Vmm vmmSrcAfterBatchSizeB = Vmm(12);
    Vmm vmmPermIdxMask = Vmm(13);
    Vmm vmmBeforeAxisDiff = Vmm(14);
    Vmm vmmSpecIdxDiff = Vmm(15);
    Vmm vmmAfterAxisSize = Vmm(5);
    Vmm vmmBlockIdxB = Vmm(6);

    // XMM
    Xbyak::Xmm xmmAuxContainer[6] = {Xbyak::Xmm(0), Xbyak::Xmm(1), Xbyak::Xmm(2), Xbyak::Xmm(3), Xbyak::Xmm(4), Xbyak::Xmm(16)};
    Xbyak::Xmm xmmZeros = Xbyak::Xmm(vmmZeros.getIdx());
    Xbyak::Xmm xmmSrcBeforeAxisSum = Xbyak::Xmm(vmmSrcBeforeAxisSumB.getIdx());
    Xbyak::Xmm xmmSpecIdxSizeB = Xbyak::Xmm(vmmSpecIdxSizeB.getIdx());
    Xbyak::Xmm xmmSpecIdxB = Xbyak::Xmm(vmmSpecIdxB.getIdx());

    // Blocked
//    Vmm vmmBlockIdxB = vmmSpecIndicesInBytes;
//    Vmm vmmSrcBeforeBlockSum = vmmSrcBeforeAxisSum;
//    Vmm vmmBeforeBlockDiff = vmmBeforeAxisDiff;


    void calcSrcShiftLong(Vmm* vAuxPool, Vmask& dstMask, bool shiftFirst = true);
    void calcSrcShiftShort(Vmm* vAuxPool, Vmask& dstMask, bool shiftFirst = true);
    void calcSrcShiftShortBlock(Vmm* vAuxPool, Vmask& dstMask, bool shiftFirst);
    void normalizeRawIndices(Vmm& rawIndices, Vmask& dstMask, Vmask& aux);
    void process32b(bool isShortIdx, bool blocked);
    void process16b(bool isShortIdx, bool blocked);
    void process8b(bool isShortIdx, bool blocked);
//    void processBlock32b(bool isShortIdx, bool shiftFirst, bool blocked);
    void tail(bool isShortIdx, bool shiftFirst = true);
    void fillRestWorkMask(Vmm& vmmMask, Vmm& vmmAux, const Xbyak::Reg64& rWorkRest, const Xbyak::Reg64& rAux0, const Xbyak::Reg64& rAux1);
    void storeScalar(const Xbyak::Reg64& rDst, const Xbyak::Reg64& rToStoreCounter, Vmm& vmmSrc, Vmm& vAux);
    void shiftIdxAndGather(Vmm* vAuxPool, bool isShortIdx, bool shiftFirst, bool blocked);
    void uni_vpgatherdd(Vmm& vDst, const Xbyak::Address& srcAddr, Vmask& vMask);
    void uni_vpcmpeqd(Vmask& vMask, Vmm& vOp0, Vmm& vOp2);

    const unsigned* permMask8bitUni;
    const unsigned* permMask16bitUni;
};

}  // namespace MKLDNNPlugin
