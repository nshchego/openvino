// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#pragma once

#include "kernel_utils.hpp"
#include "ie_precision.hpp"
#include <set>

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
    uint64_t srcChannelStepB = 0lu;
    uint64_t dstChannelStepB = 0lu;
    uint64_t srcBatchStepB   = 0lu;
    uint64_t gridBatchStepB  = 0lu;
    uint64_t dstBatchStepB   = 0lu;
    const void* wDenormCoefF;
    const void* hDenormCoefF;
    const void* srcWidthB;
    const void* srcHeightMul2F;
    const void* srcWidthMul2F;
    const void* srcHeightMul2Sub1F;
    const void* srcWidthMul2Sub1F;
    const void* srcHeightSub1F;
    const void* srcWidthSub1F;
    uint64_t workAmount = 0lu;
};

enum coord {
    w, h
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

    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::avx512_core, Xbyak::Zmm,
                                                         isa == dnnl::impl::cpu::x64::sse41,       Xbyak::Xmm,
                                                                                                   Xbyak::Ymm>::type;
    using Vmask = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::avx512_core, Xbyak::Opmask,
                                                           isa == dnnl::impl::cpu::x64::sse41,       Xbyak::Xmm,
                                                                                                     Xbyak::Ymm>::type;
private:
    uint8_t dataTypeShift = 0;

    // Suffix "B" means "In Bytes", "F" - float.
    // 64b registers
    const Xbyak::Reg64& regSrc             = r8;
    const Xbyak::Reg64& regDst             = r9;
    const Xbyak::Reg64& regGrid            = r10;
    const Xbyak::Reg64& regBatch           = r11;
    const Xbyak::Reg64& regChannelsNum     = r12;
    const Xbyak::Reg64& regWorkAmount      = r13;
    const Xbyak::Reg64& regSrcChannelStepB = r14;
    const Xbyak::Reg64& regDstChannelStepB = r15;
    const Xbyak::Reg64& regAux1            = rsi;
    const Xbyak::Reg64& regAux2            = rbx;
    const Xbyak::Reg64& regAux3            = rdx;
    const Xbyak::Reg64& regAux4 = regChannelsNum;
    const Xbyak::Reg64& regAux5            = rbp;

    const Xbyak::Reg64 regParams = Xbyak::Reg64(dnnl::impl::cpu::x64::abi_param_regs[0]);

    // 32b registers
    Xbyak::Reg32 reg32Aux1 = Xbyak::Reg32(regAux1.getIdx());
    Xbyak::Reg32 reg32Aux4 = Xbyak::Reg32(regAux4.getIdx());

    // Masks pool. Do not use k0 with gather instruction!
    Vmask masksContainer[7] = {Vmask(0), Vmask(1), Vmask(2), Vmask(3), Vmask(4), Vmask(5), Vmask(6)};
    const Xbyak::Opmask& kTailMask = k7;

    // Vectors pool
    static const size_t vecNum = isa == dnnl::impl::cpu::x64::avx512_core ? 32 : isa == dnnl::impl::cpu::x64::sse41 ? 8 : 16;
    static const Vmm vPool[vecNum];
    std::set<int> vecSet;

    int srcHeightFIdx = -1;
    int srcWidthFIdx = -1;
    int zerosIdx = -1;
    int onesFIdx = -1;
    int wDenormCoefFIdx = -1;
    int hDenormCoefFIdx = -1;
    int halfFIdx = -1;
    int gridPermMaskIdx = -1;

    // Auxiliary pool
    const Vmm vAuxContainer[14] =
            { /*SSE*/  Vmm(0),  Vmm(1),  Vmm(2),  Vmm(3),
              /*AVX*/  Vmm(8),  Vmm(9),  Vmm(10), Vmm(11),
              /*AVX5*/ Vmm(16), Vmm(17), Vmm(18), Vmm(19), Vmm(20), Vmm(21) };
    // Common.
    Vmm vZeros        = Vmm(4);
    Vmm vOnesF        = Vmm(5);
    Vmm vSrcWidthF    = Vmm(6);
    Vmm vSrcHeightF   = Vmm(7);
    // >= AVX
    Vmm vWDenormCoefF = Vmm(12); // for ALIGN CORNERS
    Vmm vHDenormCoefF = Vmm(13); // for ALIGN CORNERS
    Vmm vPermGridMask = Vmm(14);
    Vmm& vHalfF = vWDenormCoefF;

    // AVX512
    Vmm vSrcWidthSub1F       = Vmm(22);          // for BORDER padding
    Vmm vSrcHeightSub1F      = Vmm(23);          // for BORDER padding

    Vmm& vSrcWidthMul2F      = vSrcWidthSub1F;   // for REFLECTION padding
    Vmm& vSrcHeightMul2F     = vSrcHeightSub1F;  // for REFLECTION padding
    Vmm vSrcWidthMul2Sub1F   = Vmm(24);          // for REFLECTION padding
    Vmm vSrcHeightMul2Sub1F  = Vmm(25);          // for REFLECTION padding
    Vmm vAbsMask             = Vmm(26);          // for REFLECTION padding

    Vmm& vDataTypeSize       = vSrcWidthSub1F;   // for ZEROS padding
    Vmm& vSrcWidthB          = vSrcHeightSub1F;  // for ZEROS padding

    Vmm vBicubConst          = Vmm(27);          // for BICUBIC interpolation
    Vmm vBicub2Const         = Vmm(28);          // for BICUBIC interpolation
    Vmm vBicub3Const         = Vmm(29);          // for BICUBIC interpolation
    Vmm v2val                = Vmm(30);          // for BICUBIC interpolation
    Vmm v2Bicub3Const        = Vmm(31);          // for BICUBIC interpolation

    void process();
    void spatialLoop(const Vmm* vAuxPool);
    void getCoordinates(const Vmm& vHCoord, const Vmm& vWCoord, const Vmm& vAux0);
    void getTailCoordinates(const Vmm& vHCoord, const Vmm& vWCoord, const Vmm* vAuxPool);
    void denormalizeRawCoordinates(const Vmm& vWCoord, const Vmm& vHCoord, const Vmm& vAux);
    void interpolation(const Vmm* vAuxPool, const Vmm& vWCoord, const Vmm& vHCoord, bool tail = false);
    void bilinearInterpolation(const Vmm* vAuxPool, const Vmm& vWCoord, const Vmm& vHCoord, bool tail = false);
    void bicubicInterpolation(const Vmm* vAuxPool, const Vmm& vWCoord, const Vmm& vHCoord, bool tail = false);
    void nearestInterpolation(const Vmm* vAuxPool, const Vmm& vWCoord, const Vmm& vHCoord, bool tail = false);
    void zerosPadding(const Vmm& vWCoord, const Vmm& vHCoord, const Vmask& kDst, const Vmask& kAux);
    void zerosPadding0(const Vmm& vCoord, const Vmm& vUpperBound, const Vmask& kDst, const Vmask& kAux);
    void zerosPadding1(const Vmm& vCoord, const Vmm& vUpperBound, const Vmask& kDst, const Vmask& kAux);
    void borderPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const Vmask& kAux, const coord dim);
    void reflectionPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const Vmm& vAux, const Vmask& kAux, const coord dim);
    void bicubicCoefficients(const Vmm& vCoef, const Vmm& vDX, const Vmm* vAuxPool, const uint8_t idx);
    void tail(const Vmm* vAuxPool);

    // Aux
    void hwShiftPs2dq(const Vmm& vDst, const Vmm& vHCoord,const Vmm& vWCoord, const Vmm& vWidth, const Xbyak::Reg64& rAux);
    int getVecIdx() {
        if (vecSet.empty())
            return -1;
        return *vecSet.erase(vecSet.end());
    };
    int getVecIdx(int& idx) {
        if (vecSet.empty()) {
            idx = -1;
        } else {
            idx = *vecSet.erase(vecSet.begin());
        }
        return idx;
    };
//    void releaseVecIdx(int& idx) {
//        if (idx >= 0 && idx < vecNum) {
//            vecSet.insert(idx);
//            idx = -1;
//        }
//    };
    void releaseVecIdx(int idx) {
        if (idx >= 0 && idx < vecNum)
            vecSet.insert(idx);
    };
};

}   // namespace intel_cpu
}   // namespace ov
