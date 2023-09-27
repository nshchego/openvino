// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_kernel_base.hpp"
// #include "ie_precision.hpp"
// #include <set>

namespace ov {
namespace intel_cpu {
namespace kernel {

struct RandomUniformCompileParams {
    element::Type out_data_type = element::f32;
    // bool dynamicShapes  = false;
    // bool dynamicBatch   = false;
    // bool dynamicChannel = false;
    // bool alignCorners  = false;
    // GridSampleInterpolationMode interpolationMode = GridSampleInterpolationMode::BILINEAR;
    // GridSamplePaddingMode paddingMode = GridSamplePaddingMode::ZEROS;
    // InferenceEngine::Precision inDataPrc;
    // InferenceEngine::Precision gridPrc;
    // uint64_t batchNum      = 1lu;
    // uint64_t cannelNum     = 1lu;
    // uint64_t srcBatchStepB = 0lu;
};

struct RandomUniformCallArgs {
    // const void* src;
    void* dst_ptr;
    // uint64_t batchNum    = 1lu;
    // uint64_t channelsNum = 1lu;
    // const float* srcWidthF;
    // const float* srcHeightF;
    // uint64_t srcBatchStepB   = 0lu;
    // uint64_t gridBatchStepB  = 0lu;
    // uint64_t dstBatchStepB   = 0lu;
    // uint64_t srcChannelStepB = 0lu;
    // uint64_t dstChannelStepB = 0lu;
    const void* key_ptr;
    const void* counter_ptr;
    const void* n_ptr;
    const void* min_ptr;
    const void* max_ptr;
    // const void* srcHeightMul2F;
    // const void* srcWidthMul2F;
    // const void* srcHeightMul2Sub1F;
    // const void* srcWidthMul2Sub1F;
    // const void* srcHeightSub1F;
    // const void* srcWidthSub1F;
    // const void* dataTypeSize;
    // const void* buffer;
    uint64_t work_amount = 0lu;
};

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
class RandomUniform : public JitKernel<RandomUniformCompileParams, RandomUniformCallArgs> {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(RandomUniform)

    explicit RandomUniform(const RandomUniformCompileParams& jcp);

    void generate() override;

private:
    using Vmm   = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::avx512_core, Xbyak::Zmm,
                                                           isa == dnnl::impl::cpu::x64::sse41,       Xbyak::Xmm,
                                                                                                     Xbyak::Ymm>::type;
    using Vmask = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::avx512_core, Xbyak::Opmask,
                                                           isa == dnnl::impl::cpu::x64::sse41,       Xbyak::Xmm,
                                                                                                     Xbyak::Ymm>::type;

    uint8_t dataTypeShift = 0;

    // Suffix "B" means "In Bytes", "F" - float.
    // 64b registers.
    RegistersPool::Reg<Xbyak::Reg64> r64_src;
    // RegistersPool::Reg<Xbyak::Reg64> regGrid;
    RegistersPool::Reg<Xbyak::Reg64> r64_dst;
    // RegistersPool::Reg<Xbyak::Reg64> regChannelNum;
    RegistersPool::Reg<Xbyak::Reg64> r64_work_amount;
    // RegistersPool::Reg<Xbyak::Reg64> regSrcChannelStepB;
    // RegistersPool::Reg<Xbyak::Reg64> regDstChannelStepB;

    const Xbyak::Reg64 regParams = Xbyak::Reg64(dnnl::impl::cpu::x64::abi_param_regs[0]);

    // Tail mask.
    RegistersPool::Reg<Vmask> kTailMask;

    // Vector registers.
    RegistersPool::Reg<Vmm> v_max_mul_n_64;
    RegistersPool::Reg<Vmm> v_max_mul_c_64;
    RegistersPool::Reg<Vmm> v_add_low_k;
    RegistersPool::Reg<Vmm> v_add_up_k;
    RegistersPool::Reg<Vmm> v_convert_0;
    RegistersPool::Reg<Vmm> v_convert_1;
    // RegistersPool::Reg<Vmm> v_one;
    RegistersPool::Reg<Vmm> v_n_inc;

    RegistersPool::Reg<Vmm> v_key_64;
    RegistersPool::Reg<Vmm> v_counter_64;
    RegistersPool::Reg<Vmm> v_n_64;
    RegistersPool::Reg<Vmm> v_min;
    RegistersPool::Reg<Vmm> v_max_min;
    // RegistersPool::Reg<Vmm> v_sep_perm;
    // RegistersPool::Reg<Vmm> v_sep_perm_1;
    // RegistersPool::Reg<Vmm> v_sep_perm_2;
    RegistersPool::Reg<Vmm> v_res_perm;
    RegistersPool::Reg<Vmm> v_res_perm_1;

    void initVectors();

    void process();

    void runPhilox(const std::vector<Vmm>& vmm_res, const Vmm& vmm_key, const Vmm& vmm_counter, const Vmm& vmm_n);

    void calculateRound(
        const Vmm& vmm_k_0, const Vmm& vmm_k_1, const Vmm& vmm_c_0, Vmm& vmm_c_1, const Vmm& vmm_n_0, Vmm& vmm_n_1, Vmm& vmm_aux_0, Vmm& vmm_aux_1);

    void raiseKey(const Vmm& vmm_k_0, const Vmm& vmm_k_1);

    void convert(const std::vector<Vmm>& vmm_dst, const std::vector<Vmm>& vmm_src);

    // void spatialLoop();
    // void getCoordinates(const Vmm& vHCoord, const Vmm& vWCoord);
    // void getTailCoordinates(const Vmm& vHCoord, const Vmm& vWCoord);
    // void denormalizeRawCoordinates(const Vmm& vWCoord, const Vmm& vHCoord);
    // void interpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail = false);
    // void bilinearInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail = false);
    // void bicubicInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail = false);
    // void nearestInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail = false);
    // void zerosPadding(const Vmask& kDst, const Vmm& vHCoord, const Vmm& vWCoord);
    // void zerosPaddingW(const Vmask& kDst, const Vmm& vCoord);
    // void zerosPaddingH(const Vmask& kDst, const Vmm& vCoord, const Vmask& kMaskW);
    // void borderPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const coord dim);
    // void reflectionPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const coord dim);
    // void bicubicCoefficients(const Vmm& vCoef, const Vmm& vDX, const uint8_t idx);

    void tail(const std::vector<Vmm>& vmm_dst);

    // Aux
    // void dataTypeShiftPs2Dq(const Vmm& vDst, const Vmm& vSrc);
    // void hwShiftPs2dq(const Vmm& vDst, const Vmm& vHCoord, const Vmm& vWCoord, const Vmm& vWidth);
    static constexpr uint64_t ROUNDS_NUMBER = 10llu;
};


template <template<dnnl::impl::cpu::x64::cpu_isa_t isa> typename KernelT, typename CompileParams, typename CallArgs>
std::shared_ptr<JitKernel<CompileParams, CallArgs>> createInstance(const CompileParams& jcp) {
    std::shared_ptr<JitKernel<CompileParams, CallArgs>> res;

    if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core)) {
        res.reset(new KernelT<dnnl::impl::cpu::x64::avx512_core>(jcp));
    } else if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
        res.reset(new KernelT<dnnl::impl::cpu::x64::avx2>(jcp));
    } else if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::sse41)) {
        res.reset(new KernelT<dnnl::impl::cpu::x64::sse41>(jcp));
    }
    if (!res) {
        OPENVINO_THROW("Could not create JIT kernel.");
    }
    res->create_kernel();

    return res;
}

// template <template<dnnl::impl::cpu::x64::cpu_isa_t isa> typename KernelT>
// std::shared_ptr<KernelT<dnnl::impl::cpu::x64::avx512_core>> createInstance(const typename KernelT<dnnl::impl::cpu::x64::avx512_core>::CompileParams& jcp) {
//     std::shared_ptr<KernelT<isa>> res;

//     if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core)) {
//         res.reset(new KernelT<dnnl::impl::cpu::x64::avx512_core>(jcp));
//     } else if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
//         res.reset(new KernelT<dnnl::impl::cpu::x64::avx2>(jcp));
//     } else if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::sse41)) {
//         res.reset(new KernelT<dnnl::impl::cpu::x64::sse41>(jcp));
//     }
//     if (!res) {
//         OPENVINO_THROW("Could not create JIT kernel.");
//     }
//     res->create_kernel();

//     return res;
// }

}   // namespace kernel
}   // namespace intel_cpu
}   // namespace ov
