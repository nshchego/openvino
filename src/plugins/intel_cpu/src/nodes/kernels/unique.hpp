// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_kernel_base.hpp"
#include "ie_precision.hpp"
#include <set>

namespace ov {
namespace intel_cpu {

enum UniqueOutputIdx {
    UNIQUE_DATA,
    FIRST_UNIQUE_IDX,
    ALL_IDX,
    OCCURRENCES_NUM
};

struct UniqueKernelConfParams {
    bool sorted = false;
    bool flattened = true;
    int axis = 0;
    bool definedOutputs[4] = { true, false, false, false };
    bool dynamicShapes = false;
    InferenceEngine::Precision dataPrc;
};

struct UniqueKernelExecArgs {
    const void* srcPtr;
    void* dstPtr[4];
    uint64_t workAmount = 0lu;
};

class UniqueKernelBase: public JitKernelBase {
public:
    void (*ker_)(const UniqueKernelExecArgs *);
    void operator()(const UniqueKernelExecArgs *args) {
        assert(ker_);
        ker_(args);
    }
    explicit UniqueKernelBase(const UniqueKernelConfParams& jcp) : ker_(nullptr), jcp(jcp) {}

    virtual void create_ker() = 0;
    uint64_t getVecLen() {
        return vlen;
    }
    uint64_t getDataElPerVec() {
        return dataElPerVec;
    }

protected:
    UniqueKernelConfParams jcp;
    uint64_t vlen         = 16lu;
    uint64_t dataTypeSize = 1lu;
    uint64_t dataElPerVec = 1lu;
    uint64_t gridElPerVec = 1lu;
};

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
class UniqueKernel : public UniqueKernelBase {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(UniqueKernel)

    explicit UniqueKernel(const UniqueKernelConfParams& jcp);

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
    const Xbyak::Reg64 regParams = Xbyak::Reg64(dnnl::impl::cpu::x64::abi_param_regs[0]);

    // Suffix "B" means "In Bytes", "F" - float.
    // 64b registers.
    RegistersPool::Reg<Xbyak::Reg64> regSrc;
    RegistersPool::Reg<Xbyak::Reg64> regDst[4];
//    RegistersPool::Reg<Xbyak::Reg64> regDst1;
//    RegistersPool::Reg<Xbyak::Reg64> regDst2;
//    RegistersPool::Reg<Xbyak::Reg64> regDst3;
    RegistersPool::Reg<Xbyak::Reg64> regLeft;
    RegistersPool::Reg<Xbyak::Reg64> regRight;
    RegistersPool::Reg<Xbyak::Reg64> regWorkAmount;

    // Vector registers.
//    RegistersPool::Reg<Vmm> vZeros;
//    RegistersPool::Reg<Vmm> vOnesF;
    RegistersPool::Reg<Vmm> vInc;
    RegistersPool::Reg<Vmm> vSteps;
    RegistersPool::Reg<Vmm> vPermMask;
    RegistersPool::Reg<Vmm> vPermMask2;
    RegistersPool::Reg<Vmm> vPermMask3;
    RegistersPool::Reg<Vmm> vPermMask4;

    // Tail mask.
    RegistersPool::Reg<Vmask> kTailMask;
    RegistersPool::Reg<Vmask> kMaxMask0;
    RegistersPool::Reg<Vmask> kMaxMask1;
    RegistersPool::Reg<Vmask> kMaxMask2;
    RegistersPool::Reg<Vmask> kMaxMask3;
    RegistersPool::Reg<Vmask> kMaxMask4;

    void initVectors();
    void quickSort(const Xbyak::Reg64& rSrc);
    void partition();
    void process();
    void sortVector(const Vmm& vToSort);
    void cmpPerm(const Vmm& vDst, const Vmm& vSrc1, const Vmm& vSrc2, const Vmask& kMaxMask);
//    void spatialLoop();
    void getCoordinates(const Vmm& vHCoord, const Vmm& vWCoord);
    void getTailCoordinates(const Vmm& vHCoord, const Vmm& vWCoord);
    void interpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail = false);
    void tail();

    // Aux
    void dataTypeShiftPs2Dq(const Vmm& vDst, const Vmm& vSrc);
    void hwShiftPs2dq(const Vmm& vDst, const Vmm& vHCoord, const Vmm& vWCoord, const Vmm& vWidth);
};

}   // namespace intel_cpu
}   // namespace ov
