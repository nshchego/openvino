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
    const float* srcWidthF;
    const float* srcHeightF;
    const void* srcBatchStepB;
    uint64_t srcChannelStepB = 0lu;
    uint64_t dstChannelStepB = 0lu;
    uint64_t gridBatchStepB = 0lu;
    uint64_t dstBatchStepB = 0lu;
    const void* wDenormCoef;
    const void* hDenormCoef;
    const void* srcWidthB;
    const void* srcHeightMul2F;
    const void* srcWidthMul2F;
    const void* srcHeightMul2Sub1F;
    const void* srcWidthMul2Sub1F;
    const void* srcHeightSub1F;
    const void* srcWidthSub1F;
//    const void* halfVal;
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

    // Suffix "B" means "In Bytes"
    // 64b registers
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
    const Xbyak::Reg64& regAux4 = regChannelsNum;

    const Xbyak::Reg64 regParams = Xbyak::Reg64(dnnl::impl::cpu::x64::abi_param_regs[0]);

    // 32b registers
    Xbyak::Reg32 reg32Aux1 = Xbyak::Reg32(regAux1.getIdx());
    Xbyak::Reg32 reg32Aux4 = Xbyak::Reg32(regAux4.getIdx());

    // Masks pool. Do not use k0 with gather instruction!
    Vmask masksContainer[8] = {Vmask(0), Vmask(1), Vmask(2), Vmask(3), Vmask(4), Vmask(5), Vmask(6), Vmask(7)};
    // Auxiliary pool
    const Vmm vAuxContainer[14] =
            { /*SSE*/ Vmm(0), Vmm(1), Vmm(2), Vmm(3),
              /*AVX*/ Vmm(8), Vmm(9), Vmm(10), Vmm(11),
              /*AVX5*/ Vmm(16), Vmm(17), Vmm(18), Vmm(19), Vmm(20), Vmm(21)};
    // Common
    Vmm vZeros       = Vmm(4);
    Vmm vOnesF       = Vmm(5);
    Vmm vSrcWidthF   = Vmm(6);
    Vmm vSrcHeightF  = Vmm(7);
    Vmm vWDenormCoef = Vmm(12); // for ALIGN CORNERS
    Vmm vHDenormCoef = Vmm(13); // for ALIGN CORNERS
    Vmm vPermGridMask = Vmm(14);
//    Vmm v = Vmm(15);
    Vmm& vHalf = vWDenormCoef;

    // AVX512
    Vmm vSrcWidthSub1F       = Vmm(23);          // for BORDER padding
    Vmm vSrcHeightSub1F      = Vmm(24);          // for BORDER padding

    Vmm& vSrcWidthMul2F      = vSrcWidthSub1F;   // for REFLECTION padding
    Vmm& vSrcHeightMul2F     = vSrcHeightSub1F;  // for REFLECTION padding
    Vmm vSrcWidthMul2Sub1F   = Vmm(25);          // for REFLECTION padding
    Vmm vSrcHeightMul2Sub1F  = Vmm(26);          // for REFLECTION padding
    Vmm vAbsMask             = Vmm(22);          // for REFLECTION padding

    Vmm& vDataTypeSize       = vSrcWidthSub1F;   // for ZEROS padding
    Vmm& vSrcWidthB          = vSrcHeightSub1F;  // for ZEROS padding

    Vmm vBicubConst          = Vmm(27);          // for BICUBIC
    Vmm vBicub2Const         = Vmm(28);          // for BICUBIC
    Vmm vBicub3Const         = Vmm(29);          // for BICUBIC
    Vmm v2val                = Vmm(30);          // for BICUBIC
    Vmm v2Bicub3Const        = Vmm(31);          // for BICUBIC

    // XMM
//    Xbyak::Xmm xmmAuxContainer[6] = {Xbyak::Xmm(0), Xbyak::Xmm(1), Xbyak::Xmm(2), Xbyak::Xmm(3), Xbyak::Xmm(4), Xbyak::Xmm(16)};
//    Xbyak::Xmm xmmZeros = Xbyak::Xmm(vZeros.getIdx());
//    Xbyak::Xmm xmmSrcBeforeAxisSum = Xbyak::Xmm(vmmSrcBeforeAxisSumB.getIdx());
//    Xbyak::Xmm xmmSpecIdxSizeB = Xbyak::Xmm(vmmSpecIdxSizeB.getIdx());
//    Xbyak::Xmm xmmSpecIdxB = Xbyak::Xmm(vmmSpecIdxB.getIdx());

    void spatialLoop(const Vmm* vAuxPool);
    void getCoordinates(const Vmm& vHCoord, const Vmm& vWCoord, const Vmm& vAux0);
    void denormalizeRawCoordinates(const Vmm& vWCoord, const Vmm& vHCoord);
    void interpolation(const Vmm* vAuxPool, const Vmm& vWCoord, const Vmm& vHCoord);
    void bilinearInterpolation(const Vmm* vAuxPool, const Vmm& vWCoord, const Vmm& vHCoord);
    void bicubicInterpolation(const Vmm* vAuxPool, const Vmm& vWCoord, const Vmm& vHCoord);
    void nearestInterpolation(const Vmm* vAuxPool, const Vmm& vWCoord, const Vmm& vHCoord);
    void getPadded(const Vmm* vAuxPool, const Vmm& vWCoord, const Vmm& vHCoord);
    void zerosPadding(const Vmm& vWCoord, const Vmm& vHCoord, const Vmask& kDst, const Vmask& kAux);
    void zerosPadding0(const Vmm& vCoord, const Vmm& vUpperBound, const Vmask& kDst, const Vmask& kAux);
    void zerosPadding1(const Vmm& vCoord, const Vmm& vUpperBound, const Vmask& kDst, const Vmask& kAux);
    void borderPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const Vmm& vUpperBound, const Vmask& kAux);
    // dim - determines dimension. 0 - width, 1 - height.
    void reflectionPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const Vmm& vAux, const Vmask& kAux, const uint8_t dim);
    void bicubicCoefficients(const Vmm& vCoef, const Vmm& vDX, uint8_t idx);

    void process();
    void tail(bool isShortIdx, bool shiftFirst = true, bool blocked = false);
    // Aux functions.
    void normWithUpperBound(Vmm& vTarget, Vmm& vMax, Vmask& kAuxMask);
    void fillRestWorkMask(Vmask& kMask, Vmm& vAux, const Xbyak::Reg64& rWorkRest, const Xbyak::Reg64& rAux0, const Xbyak::Reg64& rAux1);
    void storeVectorPart(const Xbyak::Reg64& rDst, const Xbyak::Reg64& rToStoreCounter, Vmm& vmmSrc, Vmm& vAux);
    void uniVpGatherDd(Vmm& vDst, const Xbyak::Address& srcAddr, Vmask& vMask);
    void fillVlenVector();

    static const unsigned gridPermMask[isa == dnnl::impl::cpu::x64::sse41 ? 8 : isa == dnnl::impl::cpu::x64::avx512_core ? 16 : 8];
    static const float halfValuesF[isa == dnnl::impl::cpu::x64::sse41 ? 4 : 1];
};

}   // namespace intel_cpu
}   // namespace ov
