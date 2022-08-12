// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#pragma once

#include "kernel_utils.hpp"
#include "ie_precision.hpp"

namespace ov {
namespace intel_cpu {

enum class InterpolationMode { BILINEAR, BICUBIC, NEAREST };
enum class PaddingMode { ZEROS, BORDER, REFLECTION };

struct jGridSampleConfParams {
    bool dynamicShapes = false;
    bool alignCorners = false;
    InterpolationMode interpolationMode = InterpolationMode::BILINEAR;
    PaddingMode paddingMode = PaddingMode::ZEROS;
    InferenceEngine::Precision inDataPrc;
    InferenceEngine::Precision gridPrc;
//    uint64_t dataTypeSize = 1lu;
//    uint64_t workAmount = 0lu;
    uint64_t batchNum = 0lu;
    uint64_t srcBatchStepB = 0lu;
    uint64_t gridBatchStepB = 0lu;
    uint64_t dstBatchStepB = 0lu;
    float wDenormCoef = 1.f;
    float hDenormCoef = 1.f;
};

struct jGridSamplesExecArgs {
    const void* src;
    const void* grid;
    void* dst;
    uint64_t batchNum = 1lu;
    uint64_t channelsNum = 1lu;
    const float* srcWidthFl;
    const float* srcHeightFl;
    const void* srcBatchStepB;
    uint64_t srcChannelStepB = 0lu;
    uint64_t dstChannelStepB = 0lu;
    const void* dstBatchStepB;
    const void* wDenormCoef;
    const void* hDenormCoef;
    const void* srcWidthB;
    const void* srcHeightMul2Fl;
    const void* srcWidthMul2Fl;
    const void* srcHeightMul2Sub1Fl;
    const void* srcWidthMul2Sub1Fl;
    const void* srcHeightSub1Fl;
    const void* srcWidthSub1Fl;
    const void* halfVal;
    const void* one;
    uint64_t workAmount = 0lu;
};

class jitGridSampleKernelBase: public jitKernelBase {
public:
    void (*ker_)(const jGridSamplesExecArgs *);
    void operator()(const jGridSamplesExecArgs *args) {
        assert(ker_);
        ker_(args);
    }
    explicit jitGridSampleKernelBase(const jGridSampleConfParams& jcp) : ker_(nullptr), jcp(jcp) {}

    virtual void create_ker() = 0;
    uint64_t getVecLen() {
        return vlen;
    }
    uint64_t getDataElPerVec() {
        return dataElPerVec;
    }
    uint64_t getGridElPerVec() {
        return gridElPerVec;
    }

protected:
    jGridSampleConfParams jcp;
    uint64_t dataTypeSize = 1lu;
    uint64_t gridTypeSize = 1lu;
    uint64_t vlen = 0lu;
    uint64_t dataElPerVec = 0lu;
    uint64_t gridElPerVec = 0lu;

    static const unsigned permGridMask32bA2[8];
    static const unsigned permGridMask32bA5[16];
};

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
class jitGridSampleKernel : public jitGridSampleKernelBase {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jitGridSampleKernel)

    explicit jitGridSampleKernel(const jGridSampleConfParams& jcp);

    void create_ker() override;
    void generate() override;

protected:
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41, Xbyak::Xmm,
                                                         isa == dnnl::impl::cpu::x64::avx2,  Xbyak::Ymm,
                                                                                             Xbyak::Zmm>::type;
    using Vmask = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41, Xbyak::Xmm,
                                                           isa == dnnl::impl::cpu::x64::avx2,  Xbyak::Ymm,
                                                                                               Xbyak::Opmask>::type;
//    static const uint32_t vlenXmm = dnnl::impl::cpu::x64::cpu_isa_traits<dnnl::impl::cpu::x64::sse41>::vlen;
//    static const uint32_t gridTypeSize = sizeof(uint32_t);
//    static const uint8_t idxTypeShift = 2;
    uint8_t dataTypeShift = 0;

    // Suffix B means "In Bytes".
    // 64b registers.
    const Xbyak::Reg64& regSrc = r8;
    const Xbyak::Reg64& regDst = r9;
    const Xbyak::Reg64& regGrid = r10;
    const Xbyak::Reg64& regBatch = r11;
    const Xbyak::Reg64& regWorkAmount = r12;
    const Xbyak::Reg64& regDstChannelStepB = r13;
    const Xbyak::Reg64& regChannelsNum = r14;
    const Xbyak::Reg64& regSrcChannelStepB = r15;
    const Xbyak::Reg64& regAux1 = rsi;
    const Xbyak::Reg64& regAux2 = rbx;
    const Xbyak::Reg64& regAux3 = rdx;

    const Xbyak::Reg64 regParams = Xbyak::Reg64(dnnl::impl::cpu::x64::abi_param_regs[0]);

    // 32b registers.
//    Xbyak::Reg32 reg32IdxIter = Xbyak::Reg32(regGridIter.getIdx());
//    Xbyak::Reg32 reg32SpecIdxSizeB = Xbyak::Reg32(regSpecIdxSizeB.getIdx());
//    Xbyak::Reg32 reg32BetweenBatchAndAxisSize = Xbyak::Reg32(regBetweenBatchAndAxisSize.getIdx());
//    Xbyak::Reg32 reg32BetweenBatchAndAxisIter = Xbyak::Reg32(regBetweenBatchAndAxisIter.getIdx());
    Xbyak::Reg32 reg32Aux1 = Xbyak::Reg32(regAux1.getIdx());
//    Xbyak::Reg32 reg32Aux2 = Xbyak::Reg32(regAux2.getIdx());

    // Masks pool. Do not use k0 with gather instruction!
    Vmask masksContainer[8] = {Vmask(0), Vmask(1), Vmask(2), Vmask(3), Vmask(4), Vmask(5), Vmask(6), Vmask(7)};
    // Auxiliary pool.
    const Vmm vAuxContainer[12] =
            {Vmm(0), Vmm(1), Vmm(2), Vmm(3), Vmm(4), Vmm(5), Vmm(6), Vmm(7), /*AVX5*/ Vmm(16), Vmm(17), Vmm(18), Vmm(19)};
    // Common.
    Vmm vZeros       = Vmm(8);
    Vmm vSrcWidthFl  = Vmm(9);
    Vmm vSrcHeightFl = Vmm(10);
    Vmm vWDenormCoef = Vmm(11);
    Vmm vHDenormCoef = Vmm(12);
    Vmm vOnes        = Vmm(14);
    Vmm vPermGridMask = Vmm(15);
    Vmm& vHalf = vWDenormCoef;
    // AVX512
    Vmm vSrcWidthSub1Fl      = Vmm(28);          // for BORDER padding
    Vmm vSrcHeightSub1Fl     = Vmm(29);          // for BORDER padding
    Vmm& vSrcWidthMul2Fl     = vSrcWidthSub1Fl;  // for REFLECTION padding
    Vmm& vSrcHeightMul2Fl    = vSrcHeightSub1Fl; // for REFLECTION padding
    Vmm vSrcWidthMul2Sub1Fl  = Vmm(30);          // for REFLECTION padding
    Vmm vSrcHeightMul2Sub1Fl = Vmm(31);          // for REFLECTION padding
    Vmm& vDataTypeSize       = vSrcWidthSub1Fl;  // for ZEROS padding
    Vmm& vSrcWidthB          = vSrcHeightSub1Fl; // for ZEROS padding

    // XMM
//    Xbyak::Xmm xmmAuxContainer[6] = {Xbyak::Xmm(0), Xbyak::Xmm(1), Xbyak::Xmm(2), Xbyak::Xmm(3), Xbyak::Xmm(4), Xbyak::Xmm(16)};
//    Xbyak::Xmm xmmZeros = Xbyak::Xmm(vZeros.getIdx());
//    Xbyak::Xmm xmmSrcBeforeAxisSum = Xbyak::Xmm(vmmSrcBeforeAxisSumB.getIdx());
//    Xbyak::Xmm xmmSpecIdxSizeB = Xbyak::Xmm(vmmSpecIdxSizeB.getIdx());
//    Xbyak::Xmm xmmSpecIdxB = Xbyak::Xmm(vmmSpecIdxB.getIdx());

    void calcCoordinates(const Vmm* vAuxPool, bool shiftFirst = true);
    void denormalizeRawCoordinates(const Vmm& vWCoord, const Vmm& vHCoord);
    void interpolation(const Vmm* vAuxPool, const Vmm& vWCoord, const Vmm& vHCoord);
    void getPadded(const Vmm* vAuxPool, const Vmm& vWCoord, const Vmm& vHCoord);
    void getZeroMask(const Vmm& vWCoord, const Vmm& vHCoord, const Vmask& kDst, const Vmask& kAux);
    void getBorderPadding(const Vmm& vWCoord, const Vmm& vHCoord, const Vmask& kAux);
    // dim - determines dimension. 0 - width, 1 - height.
    void reflectionNoAlign(const Vmm& vCoord, const Vmm& vAux, const Vmask& kAux, const uint8_t dim);
    void reflectionWithAlign(const Vmm& vCoord, const Vmm& vAux, const Vmask& kAux, const uint8_t dim);

    void calcSrcShiftLongBlock(Vmm* vAuxPool, bool shiftFirst = true);
    void calcSrcShiftShort(Vmm* vAuxPool, bool shiftFirst = true);
    void calcSrcShiftShortBlock(Vmm* vAuxPool, bool shiftFirst);
    void process(bool isShortIdx, bool blocked);
    void process32b(bool isShortIdx, bool blocked);
    void process16b(bool isShortIdx, bool blocked);
    void process8b(bool isShortIdx, bool blocked);
    void shiftIdxAndGather(const Vmm* vAuxPool, bool isShortIdx, bool shiftFirst, bool blocked);
    void tail(bool isShortIdx, bool shiftFirst = true, bool blocked = false);
    // Aux functions.
    void normWithUpperBound(Vmm& vTarget, Vmm& vMax, Vmask& kAuxMask);
    void fillRestWorkMask(Vmask& kMask, Vmm& vAux, const Xbyak::Reg64& rWorkRest, const Xbyak::Reg64& rAux0, const Xbyak::Reg64& rAux1);
    void storeVectorPart(const Xbyak::Reg64& rDst, const Xbyak::Reg64& rToStoreCounter, Vmm& vmmSrc, Vmm& vAux);
    void uniVpGatherDd(Vmm& vDst, const Xbyak::Address& srcAddr, Vmask& vMask);
    void fillVlenVector();

//    const unsigned* permMask8bitUni;
    const unsigned* permGridMaskUni;
};

}   // namespace intel_cpu
}   // namespace ov
