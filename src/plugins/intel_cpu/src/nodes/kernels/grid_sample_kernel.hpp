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
    bool alignCorners  = false;
    InterpolationMode interpolationMode = InterpolationMode::BILINEAR;
    PaddingMode paddingMode = PaddingMode::ZEROS;
    InferenceEngine::Precision inDataPrc;
    InferenceEngine::Precision gridPrc;
    uint64_t batchNum      = 0lu;
    uint64_t srcBatchStepB = 0lu;
};

struct jGridSamplesExecArgs {
    const void* src;
    const void* grid;
    void* dst;
    uint64_t batchNum    = 1lu;
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
    uint64_t vlen         = 16lu;
    uint64_t dataTypeSize = 1lu;
    uint64_t gridTypeSize = 1lu;
    uint64_t dataElPerVec = 1lu;
    uint64_t gridElPerVec = 1lu;
};

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
class jitGridSampleKernel : public jitGridSampleKernelBase {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jitGridSampleKernel)

    explicit jitGridSampleKernel(const jGridSampleConfParams& jcp);

    void create_ker() override;
    void generate() override;

    using Vmm   = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::avx512_core, Xbyak::Zmm,
                                                           isa == dnnl::impl::cpu::x64::sse41,       Xbyak::Xmm,
                                                                                                     Xbyak::Ymm>::type;
    using Vmask = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::avx512_core, Xbyak::Opmask,
                                                           isa == dnnl::impl::cpu::x64::sse41,       Xbyak::Xmm,
                                                                                                     Xbyak::Ymm>::type;
private:
    uint8_t dataTypeShift = 0;

    // Suffix "B" means "In Bytes", "F" - float.
    // 64b register indices.
    int rSrcIdx             = -1;
    int rGridIdx            = -1;
    int rDstIdx             = -1;
    int rBatchIdx           = -1;
    int rChannelNumIdx      = -1;
    int rWorkAmountIdx      = -1;
    int rSrcChannelStepBIdx = -1;
    int rDstChannelStepBIdx = -1;

    const Xbyak::Reg64 regParams = Xbyak::Reg64(dnnl::impl::cpu::x64::abi_param_regs[0]);

    // Masks pool. Do not use k0 with gather instruction!
    Vmask masksContainer[7] = {Vmask(0), Vmask(1), Vmask(2), Vmask(3), Vmask(4), Vmask(5), Vmask(6)};
    const Xbyak::Opmask& kTailMask = k7;
    std::vector<Vmm> vPool;

    // Vector register indices.
    int srcHeightFIdx         = -1;
    int srcWidthFIdx          = -1;
    int zerosIdx              = -1;
    int halfFIdx              = -1;
    int onesFIdx              = -1;
    int wDenormCoefFIdx       = -1;
    int hDenormCoefFIdx       = -1;
    int gridPermMaskIdx       = -1;
    int dataTypeSizeIdx       = -1; // for ZEROS padding
    int srcWidthBIdx          = -1; // for ZEROS padding

    int srcHeightSub1FIdx     = -1; // for BORDER padding
    int srcWidthSub1FIdx      = -1; // for BORDER padding

    int srcHeightMul2FIdx     = -1; // for REFLECTION padding
    int srcWidthMul2FIdx      = -1; // for REFLECTION padding
    int srcHeightMul2Sub1FIdx = -1; // for REFLECTION padding
    int srcWidthMul2Sub1FIdx  = -1; // for REFLECTION padding
    int absMaskIdx            = -1; // for REFLECTION padding

    int const_0_75_idx        = -1; // for BICUBIC interpolation
    int const_1_25_idx        = -1; // for BICUBIC interpolation
    int const_1_50_idx        = -1; // for BICUBIC interpolation
    int const_2_00_idx        = -1; // for BICUBIC interpolation
    int const_2_25_idx        = -1; // for BICUBIC interpolation

    void process();
    void spatialLoop();
    void getCoordinates(const Vmm& vHCoord, const Vmm& vWCoord);
    void getTailCoordinates(const Vmm& vHCoord, const Vmm& vWCoord);
    void denormalizeRawCoordinates(const Vmm& vWCoord, const Vmm& vHCoord);
    void interpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail = false);
    void bilinearInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail = false);
    void bicubicInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail = false);
    void nearestInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail = false);
    void zerosPadding(const Vmask& kDst, const Vmm& vHCoord, const Vmm& vWCoord);
    void zerosPadding0(const Vmask& kDst, const Vmm& vCoord, const Vmm& vUpperBound, const Vmask& kAux);
    void zerosPadding1(const Vmask& kDst, const Vmm& vCoord, const Vmm& vUpperBound, const Vmask& kAux);
    void borderPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const coord dim);
    void reflectionPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const Vmask& kAux, const coord dim);
    void bicubicCoefficients(const Vmm& vCoef, const Vmm& vDX, const uint8_t idx);
    void tail();

    // Aux
    void hwShiftPs2dq(const Vmm& vDst, const Vmm& vHCoord,const Vmm& vWCoord, const Vmm& vWidth);
};

}   // namespace intel_cpu
}   // namespace ov
