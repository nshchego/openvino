// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "grid_sample_kernel.hpp"
#include "utils/general_utils.h"

using namespace dnnl::impl::cpu;

namespace ov {
namespace intel_cpu {

#define GET_OFF(field) offsetof(jGridSamplesExecArgs, field)
#define vmmRef() vRefWrap<Vmm>(this, vPool[getVecIdx()])

template <x64::cpu_isa_t isa>
jitGridSampleKernel<isa>::jitGridSampleKernel(const jGridSampleConfParams& jcp) :
        jitGridSampleKernelBase(jcp) {
    vlen = x64::cpu_isa_traits<isa>::vlen;
    dataTypeSize = jcp.inDataPrc.size();
    gridTypeSize = jcp.gridPrc.size();
    dataElPerVec = vlen / dataTypeSize;
    gridElPerVec = vlen / gridTypeSize;
    if (dataTypeSize == 2)
        dataTypeShift = 1;
    else if (dataTypeSize == 4)
        dataTypeShift = 2;
    vPool.reserve(vecNum);
    for (int i = 0; i < vecNum; i++) {
        vPool.push_back(Vmm(i));
    }
}

template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::create_ker() {
    auto code = x64::jit_generator::create_kernel();
    if (code != dnnl::impl::status::success)
        IE_THROW() << "Could not create GridSample kernel. Error code: " << std::to_string(code);
    ker_ = (decltype(ker_))jit_ker();
}

template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::generate() {
    this->preamble();

    mov(r64Pool[getRegIdx(rSrcIdx)],  ptr[regParams + GET_OFF(src)]);
    mov(r64Pool[getRegIdx(rGridIdx)], ptr[regParams + GET_OFF(grid)]);
    mov(r64Pool[getRegIdx(rDstIdx)],  ptr[regParams + GET_OFF(dst)]);

    const int auxIdx = getRegIdx();
    const auto& rAux = r64Pool[auxIdx];

    mov(rAux, ptr[regParams + GET_OFF(srcWidthF)]);
    uni_vpbroadcastd(vPool[getVecIdx(srcWidthFIdx)], ptr[(Xbyak::Reg64)rAux]);
    mov(rAux, ptr[regParams + GET_OFF(srcHeightF)]);
    uni_vpbroadcastd(vPool[getVecIdx(srcHeightFIdx)], ptr[(Xbyak::Reg64)rAux]);

    mov(r64Pool[getRegIdx(rSrcChannelStepBIdx)], ptr[regParams + GET_OFF(srcChannelStepB)]);
    mov(r64Pool[getRegIdx(rDstChannelStepBIdx)], ptr[regParams + GET_OFF(dstChannelStepB)]);

    if (one_of(jcp.paddingMode, PaddingMode::ZEROS, PaddingMode::BORDER)) {
        zerosIdx = getVecIdx();
        uni_vpxor(vPool[zerosIdx], vPool[zerosIdx], vPool[zerosIdx]);
    }

    if (one_of(isa, x64::avx512_core, x64::avx2, x64::avx)) {
        if (one_of(jcp.interpolationMode, InterpolationMode::BICUBIC, InterpolationMode::BILINEAR)) {
            static const float onesVal = 1.f;
            mov(rAux, reinterpret_cast<uintptr_t>(&onesVal));
            uni_vpbroadcastd(vPool[getVecIdx(onesFIdx)], ptr[(Xbyak::Reg64)rAux]);
        }

        if (jcp.alignCorners) {
            mov(rAux, ptr[regParams + GET_OFF(wDenormCoefF)]);
            uni_vpbroadcastd(vPool[getVecIdx(wDenormCoefFIdx)], ptr[(Xbyak::Reg64)rAux]);
            mov(rAux, ptr[regParams + GET_OFF(hDenormCoefF)]);
            uni_vpbroadcastd(vPool[getVecIdx(hDenormCoefFIdx)], ptr[(Xbyak::Reg64)rAux]);
        } else {
            static const float halfVal = 0.5f;
            mov(rAux, reinterpret_cast<uintptr_t>(&halfVal));
            uni_vpbroadcastd(vPool[getVecIdx(halfFIdx)], ptr[(Xbyak::Reg64)rAux]);
        }

        if (isa == x64::avx512_core) {
            static const unsigned gridPermMask[16]  = { 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15 };
            mov(rAux, reinterpret_cast<uintptr_t>(gridPermMask));
            uni_vmovups(vPool[getVecIdx(gridPermMaskIdx)], ptr[(Xbyak::Reg64)rAux]);
        } else if (isa == x64::avx2) {
            static const unsigned gridPermMask[8]  = { 0, 2, 4, 6, 1, 3, 5, 7 };
            mov(rAux, reinterpret_cast<uintptr_t>(gridPermMask));
            uni_vmovups(vPool[getVecIdx(gridPermMaskIdx)], ptr[(Xbyak::Reg64)rAux]);
        }

        Xbyak::Reg32 r32Aux(rAux.getIdx());
        if (isa == x64::avx512_core) {
            mov(r64Pool[getRegIdx(rChannelNumIdx)], ptr[regParams + GET_OFF(channelsNum)]);

            if (jcp.paddingMode == PaddingMode::ZEROS) {
                mov(rAux, dataTypeSize);
                vpbroadcastd(vPool[getVecIdx(dataTypeSizeIdx)], r32Aux);
                mov(rAux, ptr[regParams + GET_OFF(srcWidthB)]);
                uni_vpbroadcastd(vPool[getVecIdx(srcWidthBIdx)], ptr[(Xbyak::Reg64)rAux]);
            } else if (jcp.paddingMode == PaddingMode::BORDER) {
                mov(rAux, ptr[regParams + GET_OFF(srcHeightSub1F)]);
                uni_vpbroadcastd(vPool[getVecIdx(srcHeightSub1FIdx)], ptr[(Xbyak::Reg64)rAux]);
                mov(rAux, ptr[regParams + GET_OFF(srcWidthSub1F)]);
                uni_vpbroadcastd(vPool[getVecIdx(srcWidthSub1FIdx)], ptr[(Xbyak::Reg64)rAux]);
            } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
                mov(rAux, ptr[regParams + GET_OFF(srcHeightMul2F)]);
                uni_vpbroadcastd(vPool[getVecIdx(srcHeightMul2FIdx)], ptr[(Xbyak::Reg64)rAux]);
                mov(rAux, ptr[regParams + GET_OFF(srcWidthMul2F)]);
                uni_vpbroadcastd(vPool[getVecIdx(srcWidthMul2FIdx)], ptr[(Xbyak::Reg64)rAux]);
                mov(rAux, ptr[regParams + GET_OFF(srcHeightMul2Sub1F)]);
                uni_vpbroadcastd(vPool[getVecIdx(srcHeightMul2Sub1FIdx)], ptr[(Xbyak::Reg64)rAux]);
                mov(rAux, ptr[regParams + GET_OFF(srcWidthMul2Sub1F)]);
                uni_vpbroadcastd(vPool[getVecIdx(srcWidthMul2Sub1FIdx)], ptr[(Xbyak::Reg64)rAux]);
                if (jcp.alignCorners) {
                    mov(r32Aux, 0x7fffffff);
                    vpbroadcastd(vPool[getVecIdx(absMaskIdx)], r32Aux);
                }
            }

            if (jcp.interpolationMode == InterpolationMode::BICUBIC) {
                mov(r32Aux, 0xbf400000); // -0.75f
                vpbroadcastd(vPool[getVecIdx(const_0_75_idx)], r32Aux);
                mov(r32Aux, 0x3fa00000); // 1.25f
                vpbroadcastd(vPool[getVecIdx(const_1_25_idx)], r32Aux);
                mov(r32Aux, 0x3fc00000); // 1.5f
                vpbroadcastd(vPool[getVecIdx(const_1_50_idx)], r32Aux);
                mov(r32Aux, 0x40000000); // 2.0f
                vpbroadcastd(vPool[getVecIdx(const_2_00_idx)], r32Aux);
                mov(r32Aux, 0x40100000); // 2.25f
                vpbroadcastd(vPool[getVecIdx(const_2_25_idx)], r32Aux);
            }
        }
    } else if (isa == x64::sse41) {
        if (one_of(jcp.interpolationMode, InterpolationMode::BICUBIC, InterpolationMode::BILINEAR)) {
            static const float onesArr[4] = { 1.f, 1.f, 1.f, 1.f };
            mov(rAux, reinterpret_cast<uintptr_t>(onesArr));
            uni_vmovups(vPool[getVecIdx(onesFIdx)], ptr[(Xbyak::Reg64)rAux]);
        }
    }
    releaseRegIdx(auxIdx);

    process();

    this->postamble();
}

template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::process() {
    auto rWorkAmount = r64Ref();
    rWorkAmountIdx = rWorkAmount.getIdx();

    if (jcp.dynamicShapes) {
        Xbyak::Label lBatchLoop, lEnd;
        auto rBatch = r64Ref();
        mov(rBatch, ptr[regParams + GET_OFF(batchNum)]);
        L(lBatchLoop);
        {
            cmp(rBatch, 0);
            jle(lEnd, T_NEAR);

            mov(rWorkAmount, ptr[regParams + GET_OFF(workAmount)]);
            spatialLoop();

            add(r64Pool[rSrcIdx],  ptr[regParams + GET_OFF(srcBatchStepB)]);
            add(r64Pool[rGridIdx], ptr[regParams + GET_OFF(gridBatchStepB)]);
            add(r64Pool[rDstIdx],  ptr[regParams + GET_OFF(dstBatchStepB)]);

            dec(rBatch);
            jmp(lBatchLoop, T_NEAR);
        }
        L(lEnd);
    } else {
        for (uint64_t i = 0lu; i < jcp.batchNum; i++) {
            mov(rWorkAmount, ptr[regParams + GET_OFF(workAmount)]);
            spatialLoop();

            add(r64Pool[rSrcIdx],  jcp.srcBatchStepB);
            add(r64Pool[rGridIdx], ptr[regParams + GET_OFF(gridBatchStepB)]);
            add(r64Pool[rDstIdx],  ptr[regParams + GET_OFF(dstBatchStepB)]);
        }
    }
}

template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::spatialLoop() {
    const Vmm& vHCoord = vPool[getVecIdx()];
    const Vmm& vWCoord = vPool[getVecIdx()];

    Xbyak::Label lSpacialLoop, lTail;
    L(lSpacialLoop);
    {
        cmp(r64Pool[rWorkAmountIdx], dataElPerVec);
        jl(lTail, T_NEAR);

        getCoordinates(vHCoord, vWCoord);
        denormalizeRawCoordinates(vWCoord, vHCoord);
        interpolation(vWCoord, vHCoord);

        sub(r64Pool[rWorkAmountIdx], dataElPerVec);
        add(r64Pool[rDstIdx], vlen);

        jmp(lSpacialLoop, T_NEAR);
    }

    releaseVecIdx(vHCoord.getIdx());
    releaseVecIdx(vWCoord.getIdx());

    L(lTail);
    tail();
}

template <>
void jitGridSampleKernel<x64::avx512_core>::getCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
    auto vAux = vmmRef();

    uni_vpermd(vWCoord, vPool[gridPermMaskIdx], ptr[r64Pool[rGridIdx]]); // Permute to XXXX.XXXX.YYYY.YYYY
    Xbyak::Ymm ymmH(vHCoord.getIdx());
    vextractf64x4(ymmH, vWCoord, 1); // Extract Y component

    add(r64Pool[rGridIdx], vlen);

    uni_vpermd(vAux, vPool[gridPermMaskIdx], ptr[r64Pool[rGridIdx]]); // Permute to XXXX.XXXX.YYYY.YYYY
    Xbyak::Ymm ymmAux(vAux.getIdx());
    vinsertf64x4(vWCoord, vWCoord, ymmAux, 1); // Extract X component
    vextractf64x4(ymmAux, vAux, 1);                // Extract Y component
    vinsertf64x4(vHCoord, vHCoord, ymmAux, 1);

    add(r64Pool[rGridIdx], vlen);
}

template <>
void jitGridSampleKernel<x64::avx2>::getCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
    auto vAux = vmmRef();

    uni_vpermd(vWCoord, vPool[gridPermMaskIdx], ptr[r64Pool[rGridIdx]]); // Permute to XXXX.YYYY
    vperm2i128(vHCoord, vHCoord, vWCoord, 0B00000011); // Extract Y component

    add(r64Pool[rGridIdx], vlen);

    uni_vpermd(vAux, vPool[gridPermMaskIdx], ptr[r64Pool[rGridIdx]]); // Permute to XXXX.YYYY
    vperm2i128(vWCoord, vWCoord, vAux, 0B00100000);    // Extract X component
    vperm2i128(vHCoord, vHCoord, vAux, 0B00110000);    // Extract Y component

    add(r64Pool[rGridIdx], vlen);
}

template <x64::cpu_isa_t isa> // Works for AVX, SSE41
void jitGridSampleKernel<isa>::getCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
    auto vAux = vmmRef();
    Xbyak::Xmm xmmWCoord(vWCoord.getIdx());
    Xbyak::Xmm xmmHCoord(vHCoord.getIdx());
    Xbyak::Xmm xmmAux(vAux.getIdx());
    const uint64_t xmmVlen = x64::cpu_isa_traits<x64::sse41>::vlen;

    uni_vpshufd(xmmWCoord, ptr[r64Pool[rGridIdx]], 0xD8);
    shufpd(xmmHCoord, xmmWCoord, 0x2);

    add(r64Pool[rGridIdx], xmmVlen);

    uni_vpshufd(xmmAux, ptr[r64Pool[rGridIdx]], 0xD8);
    shufpd(xmmWCoord, xmmAux, 0x0);
    shufpd(xmmHCoord, xmmAux, 0x3);

    add(r64Pool[rGridIdx], xmmVlen);

    if (isa == x64::avx) {
        Xbyak::Ymm ymmWCoord(vWCoord.getIdx());
        Xbyak::Ymm ymmHCoord(vHCoord.getIdx());

        vperm2f128(ymmWCoord, ymmWCoord, ymmWCoord, 0x1);
        vperm2f128(ymmHCoord, ymmHCoord, ymmHCoord, 0x1);

        // Here is movups + pshufd instead of vpshufd for two reasons:
        // 1. vpshufd zeroes the rest ov YMM.
        // 2. pshufd does not work with not aligned to YMM Address.
        movups(xmmWCoord, ptr[r64Pool[rGridIdx]]);
        pshufd(xmmWCoord, xmmWCoord, 0xD8);
        shufpd(xmmHCoord, xmmWCoord, 0x2);

        add(r64Pool[rGridIdx], xmmVlen);

        uni_vpshufd(xmmAux, ptr[r64Pool[rGridIdx]], 0xD8);
        shufpd(xmmWCoord, xmmAux, 0x0);
        shufpd(xmmHCoord, xmmAux, 0x3);

        add(r64Pool[rGridIdx], xmmVlen);

        vperm2f128(ymmWCoord, ymmWCoord, ymmWCoord, 0x1);
        vperm2f128(ymmHCoord, ymmHCoord, ymmHCoord, 0x1);
    }
}

template <>
void jitGridSampleKernel<x64::avx512_core>::getTailCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
    Xbyak::Label lEnd, lGridShift, lRest;

    auto vAux = vmmRef();
    auto rAux = r64Ref();
    Xbyak::Ymm ymmH(vHCoord.getIdx());

    mov(rAux, r64Pool[rWorkAmountIdx]);
    sal(rAux, 0x1); // multiply by gridShape[3]
    cmp(r64Pool[rWorkAmountIdx], dataElPerVec / 2);
    jl(lRest, T_NEAR);
    {
        uni_vpermd(vWCoord, vPool[gridPermMaskIdx], ptr[r64Pool[rGridIdx]]);
        vextractf64x4(ymmH, vWCoord, 1); // Extract Y component

        add(r64Pool[rGridIdx], vlen);
        sub(rAux, dataElPerVec);
        cmp(rAux, 0);
        jle(lEnd, T_NEAR);

        fillRestWorkMask(kTailMask, vAux, rAux);
        uni_vmovups((Vmm)vAux | kTailMask, ptr[r64Pool[rGridIdx]]);
        uni_vpermd(vAux, vPool[gridPermMaskIdx], vAux);
        Xbyak::Ymm ymmAux(vAux.getIdx());
        vinsertf64x4(vWCoord, vWCoord, ymmAux, 1); // Extract X component
        vextractf64x4(ymmAux, vAux, 1); // Extract Y component
        vinsertf64x4(vHCoord, vHCoord, ymmAux, 1);

        jmp(lGridShift, T_NEAR);
    }
    L(lRest);
    {
        fillRestWorkMask(kTailMask, vAux, rAux);
        uni_vmovups(vWCoord | kTailMask, ptr[r64Pool[rGridIdx]]);
        uni_vpermd(vWCoord, vPool[gridPermMaskIdx], vWCoord);
        vextractf64x4(ymmH, vWCoord, 1); // Extract Y component
    }

    L(lGridShift);
    if (dataTypeSize > 1)
        sal(rAux, dataTypeShift); // multiply by source data type size.
    add(r64Pool[rGridIdx], rAux);

    L(lEnd);

    fillRestWorkMask(kTailMask, vAux, r64Pool[rWorkAmountIdx]);
}

template <>
void jitGridSampleKernel<x64::avx2>::getTailCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
    Xbyak::Label lRest, lGridShift, lEnd;

    auto vAux  = vmmRef();
    auto rAux  = r64Ref();
    Xbyak::Xmm xmmH(vHCoord.getIdx());

    mov(rAux, r64Pool[rWorkAmountIdx]);
    sal(rAux, 0x1); // multiply by gridShape[3] == 2
    cmp(r64Pool[rWorkAmountIdx], dataElPerVec / 2);
    jl(lRest, T_NEAR);
    {
        uni_vpermd(vWCoord, vPool[gridPermMaskIdx], ptr[r64Pool[rGridIdx]]); // Permute to XXXX.YYYY
        vextractf128(xmmH, vWCoord, 1); // Extract Y component

        add(r64Pool[rGridIdx], vlen);
        sub(rAux, dataElPerVec);
        cmp(rAux, 0);
        jle(lEnd, T_NEAR);

        loadEl2vec32(vAux, r64Pool[rGridIdx], rAux);
        vpermilps(vAux, vAux, 0xD8);
        Xbyak::Xmm xmmAux(vAux.getIdx());
        vinsertf128(vWCoord, vWCoord, xmmAux, 1); // Extract X component
        vextractf128(xmmAux, vAux, 1); // Extract Y component
        vinsertf128(vHCoord, vHCoord, xmmAux, 1);

        jmp(lGridShift, T_NEAR);
    }
    L(lRest);
    {
        loadEl2vec32(vWCoord, r64Pool[rGridIdx], rAux);
        uni_vpermd(vWCoord, vPool[gridPermMaskIdx], vWCoord); // Permute to XXXX.YYYY
        vextractf128(xmmH, vWCoord, 1); // Extract Y component
    }

    L(lGridShift);
    if (dataTypeSize > 1)
        sal(rAux, dataTypeShift); // multiply by source data type size.
    add(r64Pool[rGridIdx], rAux);

    L(lEnd);
}

template <>
void jitGridSampleKernel<x64::avx>::getTailCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
    Xbyak::Label lLoop2End, lEnd;

    Xbyak::Xmm xmmWCoord(vWCoord.getIdx());
    Xbyak::Xmm xmmHCoord(vHCoord.getIdx());

    auto rGridRest = r64Ref();
    mov(rGridRest, r64Pool[rWorkAmountIdx]);
    sal(rGridRest, 0x1); // multiply by gridShape[3] == 2

    for (int i = 0; i < dataElPerVec; i++) {
        cmp(rGridRest, 0);
        jle(lEnd, T_NEAR);

        if (i % 2 == 0)
            pinsrd(xmmWCoord, ptr[r64Pool[rGridIdx]], i / 2);
        else
            pinsrd(xmmHCoord, ptr[r64Pool[rGridIdx]], i / 2);

        add(r64Pool[rGridIdx], gridTypeSize);
        dec(rGridRest);
    }
    cmp(rGridRest, 0);
    jle(lEnd, T_NEAR);
    vperm2f128(vWCoord, vWCoord, vWCoord, 0x1);
    vperm2f128(vHCoord, vHCoord, vHCoord, 0x1);

    for (int i = 0; i < dataElPerVec; i++) {
        cmp(rGridRest, 0);
        jle(lLoop2End, T_NEAR);

        if (i % 2 == 0)
            pinsrd(xmmWCoord, ptr[r64Pool[rGridIdx]], i / 2);
        else
            pinsrd(xmmHCoord, ptr[r64Pool[rGridIdx]], i / 2);

        add(r64Pool[rGridIdx], gridTypeSize);
        dec(rGridRest);
    }

    L(lLoop2End);
    vperm2f128(vWCoord, vWCoord, vWCoord, 0x1);
    vperm2f128(vHCoord, vHCoord, vHCoord, 0x1);

    L(lEnd);
}

template <>
void jitGridSampleKernel<x64::sse41>::getTailCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
    Xbyak::Label lRest, lGridShift, lEnd;
    auto vAux = vmmRef();
    auto rAux = r64Ref();

    mov(rAux, r64Pool[rWorkAmountIdx]);
    sal(rAux, 0x1); // multiply by gridShape[3] == 2
    cmp(r64Pool[rWorkAmountIdx], gridElPerVec / 2);
    jl(lRest, T_NEAR);
    {
        pshufd(vWCoord, ptr[r64Pool[rGridIdx]], 0xD8);
        shufpd(vHCoord, vWCoord, 0x2);

        add(r64Pool[rGridIdx], vlen);
        sub(rAux, gridElPerVec);
        cmp(rAux, 0);
        jle(lEnd, T_NEAR);

        loadEl2vec32(vAux, r64Pool[rGridIdx], rAux);
        pshufd(vAux, vAux, 0xD8);
        shufpd(vWCoord, vAux, 0x0); // Extract X component
        shufpd(vHCoord, vAux, 0x3); // Extract Y component

        jmp(lGridShift, T_NEAR);
    }
    L(lRest);
    {
        loadEl2vec32(vWCoord, r64Pool[rGridIdx], rAux);
        pshufd(vWCoord, vWCoord, 0xD8);  // Extract X component
        shufpd(vHCoord, vWCoord, 0x2);   // Extract Y component
    }

    L(lGridShift);
    if (dataTypeSize > 1)
        sal(rAux, dataTypeShift); // multiply by source data type size.
    add(r64Pool[rGridIdx], rAux);

    L(lEnd);
}

template <>
void jitGridSampleKernel<x64::sse41>::denormalizeRawCoordinates(const Vmm& vWCoord, const Vmm& vHCoord) {
    auto vAux = vmmRef();
    auto rAux = r64Ref();

    if (jcp.alignCorners) {
        mov(rAux, ptr[regParams + GET_OFF(wDenormCoefF)]);
        uni_vmovups(vAux, ptr[(Xbyak::Reg64)rAux]);
        uni_vfmadd132ps(vWCoord, vAux, vAux);

        mov(rAux, ptr[regParams + GET_OFF(hDenormCoefF)]);
        uni_vmovups(vAux, ptr[(Xbyak::Reg64)rAux]);
        uni_vfmadd132ps(vHCoord, vAux, vAux);
    } else {
        static const float halfValues[4] = { 0.5f, 0.5f, 0.5f, 0.5f };
        mov(rAux, reinterpret_cast<uintptr_t>(halfValues));
        uni_vmovups(vAux, ptr[(Xbyak::Reg64)rAux]);

        uni_vfmadd132ps(vWCoord, vPool[srcWidthFIdx], vPool[srcWidthFIdx]);
        uni_vfmsub132ps(vWCoord, vAux, vAux);

        uni_vfmadd132ps(vHCoord, vPool[srcHeightFIdx], vPool[srcHeightFIdx]);
        uni_vfmsub132ps(vHCoord, vAux, vAux);
    }
}

template <x64::cpu_isa_t isa> // Works for AVX, AVX2, AVX512
void jitGridSampleKernel<isa>::denormalizeRawCoordinates(const Vmm& vWCoord, const Vmm& vHCoord) {
    if (jcp.alignCorners) {
        uni_vfmadd132ps(vWCoord, vPool[wDenormCoefFIdx], vPool[wDenormCoefFIdx]);
        uni_vfmadd132ps(vHCoord, vPool[hDenormCoefFIdx], vPool[hDenormCoefFIdx]);
    } else {
        uni_vfmadd132ps(vWCoord, vPool[srcWidthFIdx], vPool[srcWidthFIdx]);
        uni_vfmsub132ps(vWCoord, vPool[halfFIdx], vPool[halfFIdx]);

        uni_vfmadd132ps(vHCoord, vPool[srcHeightFIdx], vPool[srcHeightFIdx]);
        uni_vfmsub132ps(vHCoord, vPool[halfFIdx], vPool[halfFIdx]);
    }
}

template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::interpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
    if (jcp.interpolationMode == InterpolationMode::BILINEAR) {
        bilinearInterpolation(vWCoord, vHCoord, tail);
    } else if (jcp.interpolationMode == InterpolationMode::BICUBIC) {
        bicubicInterpolation(vWCoord, vHCoord, tail);
    } else if (jcp.interpolationMode == InterpolationMode::NEAREST) {
        nearestInterpolation(vWCoord, vHCoord, tail);
    }
}

template <>
void jitGridSampleKernel<x64::avx512_core>::zerosPadding0(const Vmask& kDst, const Vmm& vCoord, const Vmm& vUpperBound, const Vmask& kAux) {
    vcmpps(kAux, vCoord, vUpperBound, 0x1);            // vCoord < vUpperBound
    vcmpps(kDst | kAux, vPool[zerosIdx], vCoord, 0x2); // vCoord >= vZeros
}

template <>
void jitGridSampleKernel<x64::avx512_core>::zerosPadding1(const Vmask& kDst, const Vmm& vCoord, const Vmm& vUpperBound, const Vmask& kAux) {
    vcmpps(kDst | kAux, vCoord, vUpperBound, 0x1);     // vCoord < vUpperBound
    vcmpps(kDst | kDst, vPool[zerosIdx], vCoord, 0x2); // vCoord >= vZeros
}

template <>
void jitGridSampleKernel<x64::avx512_core>::zerosPadding(const Vmask& kDst, const Vmm& vHCoord, const Vmm& vWCoord) {
    zerosPadding0(kDst, vWCoord, vPool[srcWidthFIdx], kDst);
    zerosPadding1(kDst, vHCoord, vPool[srcHeightFIdx], kDst);
}

template <>
void jitGridSampleKernel<x64::sse41>::zerosPadding(const Vmask& kDst, const Vmm& vHCoord, const Vmm& vWCoord) {
    auto vAux = vmmRef();

    uni_vmovups(vAux, vWCoord);
    uni_vcmpps(vAux, vAux, vPool[srcWidthFIdx], 0x1); // vWCoord < vSrcWidthF
    uni_vpxor(kDst, kDst, kDst);
    uni_vcmpps(kDst, kDst, vWCoord, 0x2);    // vWCoord >= vZeros
    uni_vpand(kDst, kDst, vAux);

    uni_vmovups(vAux, vHCoord);
    uni_vcmpps(vAux, vAux, vPool[srcHeightFIdx], 0x1); // vHCoord < vSrcHeightF
    uni_vpand(kDst, kDst, vAux);
    uni_vpxor(vAux, vAux, vAux);
    uni_vcmpps(vAux, vAux, vHCoord, 0x2);     // vHCoord >= vZeros
    uni_vpand(kDst, kDst, vAux);
}

template <x64::cpu_isa_t isa> // Works for AVX, AVX2
void jitGridSampleKernel<isa>::zerosPadding(const Vmask& kDst, const Vmm& vHCoord, const Vmm& vWCoord) {
    auto vAux = vmmRef();

    uni_vcmpps(Xbyak::Xmm(vAux), vWCoord, vPool[srcWidthFIdx], 0x1);  // vWCoord < vSrcWidthF
    uni_vcmpps(kDst, vPool[zerosIdx], vWCoord, 0x2);      // vWCoord >= vZeros
    uni_vandps(kDst, kDst, vAux);

    uni_vcmpps(vAux, vHCoord, vPool[srcHeightFIdx], 0x1); // vHCoord < vSrcHeightF
    uni_vandps(kDst, kDst, vAux);
    uni_vcmpps(vAux, vPool[zerosIdx], vHCoord, 0x2);      // vHCoord >= vZeros
    uni_vandps(kDst, kDst, vAux);
}

template <>
void jitGridSampleKernel<x64::avx512_core>::borderPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const coord dim) {
    vrangeps(vCoordDst, vCoordOrigin, vPool[dim == coord::w ? srcWidthSub1FIdx : srcHeightSub1FIdx], 0x0); // vWCoord >= vSrcWidthF
    vrangeps(vCoordDst, vCoordDst, vPool[zerosIdx], 0x1); // vWCoord < vZeros
}

template <>
void jitGridSampleKernel<x64::sse41>::borderPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const coord dim) {
    auto rAux  = r64Ref();
    if (dim == coord::w) {
        mov(rAux, ptr[regParams + GET_OFF(srcWidthSub1F)]);
    } else if (dim == coord::h) {
        mov(rAux, ptr[regParams + GET_OFF(srcHeightSub1F)]);
    }
    auto vAux = vmmRef();

    uni_vmovups(vAux, vCoordOrigin);
    uni_vcmpps(vAux, vAux, ptr[(Xbyak::Reg64)rAux], 0x2); // vCoord < vUpperBound
    if (vCoordDst.getIdx() != vCoordOrigin.getIdx())
        uni_vmovups(vCoordDst, vCoordOrigin);
    uni_vandps(vCoordDst, vCoordDst, vAux);
    uni_vandps(vAux, vAux, ptr[(Xbyak::Reg64)rAux]);
    uni_vaddps(vCoordDst, vCoordDst, vAux);

    uni_vpxor(vAux, vAux, vAux);
    uni_vcmpps(vAux, vAux, vCoordDst, 0x2);    // vCoord >= vZeros
    uni_vandps(vCoordDst, vCoordDst, vAux);
}

template <x64::cpu_isa_t isa> // Works for AVX, AVX2
void jitGridSampleKernel<isa>::borderPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const coord dim) {
    auto rAux = r64Ref();
    auto vAux = vmmRef();

    if (dim == coord::w) {
        mov(rAux, ptr[regParams + GET_OFF(srcWidthSub1F)]);
    } else if (dim == coord::h) {
        mov(rAux, ptr[regParams + GET_OFF(srcHeightSub1F)]);
    }

    uni_vcmpps(vAux, vCoordOrigin, ptr[(Xbyak::Reg64)rAux], 0x2); // vCoord <= vUpperBound
    uni_vandps(vCoordDst, vCoordOrigin, vAux);
    vandnps(vAux, vAux, ptr[(Xbyak::Reg64)rAux]);
    uni_vaddps(vCoordDst, vCoordDst, vAux);

    uni_vcmpps(vAux, vPool[zerosIdx], vCoordDst, 0x2); // vCoord >= vZeros
    uni_vandps(vCoordDst, vCoordDst, vAux);
}

template <>
void jitGridSampleKernel<x64::avx512_core>::reflectionPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const Vmask& kAux, const coord dim) {
    auto vAux = vmmRef();
    const auto& vSrcDimMul2Sub1F = vPool[dim == coord::w ? srcWidthMul2Sub1FIdx : srcHeightMul2Sub1FIdx];

    if (jcp.alignCorners) {
        // abs(x) % D21
        uni_vandps(vCoordDst, vCoordOrigin, vPool[absMaskIdx]); // abs(x)
        uni_vdivps(vAux, vCoordDst, vSrcDimMul2Sub1F);
        uni_vroundps(vAux, vAux, 0x3);                          // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, vSrcDimMul2Sub1F);    // abs(x) % D21
    } else {
        const auto& vSrcDimMul2F = vPool[dim == coord::w ? srcWidthMul2FIdx : srcHeightMul2FIdx];
        // (x % D2 + D2) % D2
        if (vCoordDst.getIdx() != vCoordOrigin.getIdx())
            uni_vmovups(vCoordDst, vCoordOrigin);
        uni_vdivps(vAux, vCoordDst, vSrcDimMul2F);
        uni_vroundps(vAux, vAux, 0x3);                   // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, vSrcDimMul2F); // x % D2
        uni_vaddps(vCoordDst, vCoordDst, vSrcDimMul2F);  // x % D2 + D2
        uni_vdivps(vAux, vCoordDst, vSrcDimMul2F);
        uni_vroundps(vAux, vAux, 0x3);                   // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, vSrcDimMul2F); // (x % D2 + D2) % D2
    }

    uni_vsubps(vAux, vSrcDimMul2Sub1F, vCoordDst);
    vcmpps(kAux, vPool[dim == coord::w ? srcWidthFIdx : srcHeightFIdx], vCoordDst, 0x2); // vCoordDst >= vSrcDimF
    vmovups(vCoordDst | kAux, vAux);
}

template <>
void jitGridSampleKernel<x64::sse41>::reflectionPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const Vmask& kAux, const coord dim) {
    auto vAux = vmmRef();
    auto rAux = r64Ref();

    if (jcp.alignCorners) {
        // abs(x) % D21
        if (vCoordDst.getIdx() != vCoordOrigin.getIdx())
            uni_vmovups(vCoordDst, vCoordOrigin);
        static const unsigned absMask[4] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
        mov(rAux, reinterpret_cast<uintptr_t>(absMask)); // TODO: use PSIGND
        uni_vandps(vCoordDst, vCoordDst, ptr[(Xbyak::Reg64)rAux]); // abs(x)
        if (dim == coord::w) {
            mov(rAux, ptr[regParams + GET_OFF(srcWidthMul2Sub1F)]);
        } else if (coord::h) {
            mov(rAux, ptr[regParams + GET_OFF(srcHeightMul2Sub1F)]);
        }
        uni_vmovups(vAux, vCoordDst);
        uni_vdivps(vAux, vAux, ptr[(Xbyak::Reg64)rAux]);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, ptr[(Xbyak::Reg64)rAux]); // abs(x) % D21
    } else {
        // (x % D2 + D2) % D2
        if (vCoordDst.getIdx() != vCoordOrigin.getIdx())
            uni_vmovups(vCoordDst, vCoordOrigin);
        uni_vmovups(vAux, vCoordOrigin);
        if (dim == coord::w) {
            mov(rAux, ptr[regParams + GET_OFF(srcWidthMul2F)]);
        } else if (coord::h) {
            mov(rAux, ptr[regParams + GET_OFF(srcHeightMul2F)]);
        }
        uni_vdivps(vAux, vAux, ptr[(Xbyak::Reg64)rAux]);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, ptr[(Xbyak::Reg64)rAux]); // x % D2
        uni_vaddps(vCoordDst, vCoordDst, ptr[(Xbyak::Reg64)rAux]); // x % D2 + D2
        uni_vmovups(vAux, vCoordDst);
        uni_vdivps(vAux, vAux, ptr[(Xbyak::Reg64)rAux]);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, ptr[(Xbyak::Reg64)rAux]); // (x % D2 + D2) % D2
    }

    if (dim == coord::w) {
        mov(rAux, ptr[regParams + GET_OFF(srcWidthMul2Sub1F)]);
    } else if (coord::h) {
        mov(rAux, ptr[regParams + GET_OFF(srcHeightMul2Sub1F)]);
    }
    uni_vmovups(vAux, ptr[(Xbyak::Reg64)rAux]);
    uni_vsubps(vAux, vAux, vCoordDst);
    uni_vmovups(kAux, vPool[dim == coord::w ? srcWidthFIdx : srcHeightFIdx]);
    vcmpps(kAux, kAux, vCoordDst, 0x2); // vCoordDst >= vSrcDimF
    uni_vpand(vCoordDst, vCoordDst, kAux);
    uni_vpand(kAux, kAux, vAux);
    uni_vaddps(vCoordDst, vCoordDst, kAux);
}

template <x64::cpu_isa_t isa> // Works for AVX2, AVX
void jitGridSampleKernel<isa>::reflectionPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const Vmask& kAux, const coord dim) {
    auto vAux = vmmRef();
    auto rAux = r64Ref();

    if (jcp.alignCorners) {
        // abs(x) % D21
        if (vCoordDst.getIdx() != vCoordOrigin.getIdx())
            uni_vmovups(vCoordDst, vCoordOrigin);
        static const unsigned absMask[8] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
        mov(rAux, reinterpret_cast<uintptr_t>(absMask)); // TODO: use PSIGND or vpabsd
        uni_vandps(vCoordDst, vCoordDst, ptr[(Xbyak::Reg64)rAux]); // abs(x)
        if (dim == coord::w) {
            mov(rAux, ptr[regParams + GET_OFF(srcWidthMul2Sub1F)]);
        } else if (coord::h) {
            mov(rAux, ptr[regParams + GET_OFF(srcHeightMul2Sub1F)]);
        }
        uni_vdivps(vAux, vCoordDst, ptr[(Xbyak::Reg64)rAux]);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, ptr[(Xbyak::Reg64)rAux]); // abs(x) % D21
    } else {
        // (x % D2 + D2) % D2
        if (vCoordDst.getIdx() != vCoordOrigin.getIdx())
            uni_vmovups(vCoordDst, vCoordOrigin);
        if (dim == coord::w) {
            mov(rAux, ptr[regParams + GET_OFF(srcWidthMul2F)]);
        } else if (coord::h) {
            mov(rAux, ptr[regParams + GET_OFF(srcHeightMul2F)]);
        }
        uni_vdivps(vAux, vCoordOrigin, ptr[(Xbyak::Reg64)rAux]);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, ptr[(Xbyak::Reg64)rAux]); // x % D2
        uni_vaddps(vCoordDst, vCoordDst, ptr[(Xbyak::Reg64)rAux]);  // x % D2 + D2
        uni_vdivps(vAux, vCoordDst, ptr[(Xbyak::Reg64)rAux]);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, ptr[(Xbyak::Reg64)rAux]); // (x % D2 + D2) % D2
    }

    if (dim == coord::w) {
        mov(rAux, ptr[regParams + GET_OFF(srcWidthMul2Sub1F)]);
    } else if (coord::h) {
        mov(rAux, ptr[regParams + GET_OFF(srcHeightMul2Sub1F)]);
    }
    uni_vsubps(vAux, vCoordDst, ptr[(Xbyak::Reg64)rAux]);
    const auto& vUpperBound = vPool[dim == coord::w ? srcWidthFIdx : srcHeightFIdx];
    uni_vcmpps(kAux, vCoordDst, vUpperBound, 0x1); // vCoordDst < vUpperBound
    uni_vandps(vCoordDst, vCoordDst, kAux);
    vandnps(vAux, kAux, vAux);
    uni_vsubps(vCoordDst, vCoordDst, vAux);
}

template <>
void jitGridSampleKernel<x64::avx512_core>::bicubicCoefficients(const Vmm& vCoef, const Vmm& vDDim, const uint8_t idx) {
    if (idx == 0) {
        uni_vmovups(vCoef, vDDim);
        vfnmadd132ps(vCoef, vPool[onesFIdx], vPool[const_2_00_idx]);
        uni_vfmadd231ps(vCoef, vDDim, vDDim);
        uni_vmulps(vCoef, vCoef, vDDim);
        uni_vmulps(vCoef, vCoef, vPool[const_0_75_idx]);
    } else if (idx == 1) {
        uni_vmovups(vCoef, vDDim);
        vfmsub132ps(vCoef, vPool[const_2_25_idx], vPool[const_1_25_idx]);
        uni_vmulps(vCoef, vCoef, vDDim);
        uni_vfmadd132ps(vCoef, vPool[onesFIdx], vDDim);
    } else if (idx == 2) {
        uni_vmulps(vCoef, vDDim, vDDim);
        uni_vfmadd132ps(vCoef, vPool[const_0_75_idx], vPool[const_1_25_idx]);
        uni_vfmsub231ps(vCoef, vDDim, vPool[const_1_50_idx]);
        uni_vmulps(vCoef, vCoef, vDDim);
    } else if (idx == 3) {
        uni_vmulps(vCoef, vPool[const_0_75_idx], vDDim);
        uni_vmulps(vCoef, vCoef, vDDim);
        vfnmadd132ps(vCoef, vCoef, vDDim);
    }
}

template <>
void jitGridSampleKernel<x64::sse41>::bicubicCoefficients(const Vmm& vCoef, const Vmm& vDDim, const uint8_t idx) {
    auto vAux = vmmRef();
    auto rAux = r64Ref();
    static const float const_0_75[4] = { -0.75f, -0.75f, -0.75f, -0.75f };
    static const float const_1_25[4] = { 1.25f, 1.25f, 1.25f, 1.25f };
    static const float const_1_50[4] = { 1.5f, 1.5f, 1.5f, 1.5f };
    static const float const_2_00[4] = { 2.f, 2.f, 2.f, 2.f };
    static const float const_2_25[4] = { 2.25f, 2.25f, 2.25f, 2.25f };

    if (idx == 0) {
        uni_vmovups(vAux, vDDim);
        mov(rAux, reinterpret_cast<uintptr_t>(const_2_00));
        uni_vmulps(vAux, vAux, ptr[(Xbyak::Reg64)rAux]);
        uni_vsubps(vAux, vAux, vPool[onesFIdx]);
        uni_vmovups(vCoef, vDDim);
        uni_vmulps(vCoef, vCoef, vCoef);
        uni_vsubps(vCoef, vCoef, vAux);
        uni_vmulps(vCoef, vCoef, vDDim);
        mov(rAux, reinterpret_cast<uintptr_t>(const_0_75));
        uni_vmulps(vCoef, vCoef, ptr[(Xbyak::Reg64)rAux]);
    } else if (idx == 1) {
        uni_vmovups(vCoef, vDDim);
        mov(rAux, reinterpret_cast<uintptr_t>(const_1_25));
        uni_vmulps(vCoef, vCoef, ptr[(Xbyak::Reg64)rAux]);
        mov(rAux, reinterpret_cast<uintptr_t>(const_2_25));
        uni_vsubps(vCoef, vCoef, ptr[(Xbyak::Reg64)rAux]);
        uni_vmulps(vCoef, vCoef, vDDim);
        uni_vfmadd132ps(vCoef, vPool[onesFIdx], vDDim);
    } else if (idx == 2) {
        uni_vmovups(vAux, vDDim);
        uni_vmulps(vAux, vDDim, vDDim);
        mov(rAux, reinterpret_cast<uintptr_t>(const_1_25));
        uni_vmulps(vAux, vAux, ptr[(Xbyak::Reg64)rAux]);
        mov(rAux, reinterpret_cast<uintptr_t>(const_0_75));
        uni_vaddps(vAux, vAux, ptr[(Xbyak::Reg64)rAux]);
        uni_vmovups(vCoef, vDDim);
        mov(rAux, reinterpret_cast<uintptr_t>(const_1_50));
        uni_vmulps(vCoef, vCoef, ptr[(Xbyak::Reg64)rAux]);
        uni_vsubps(vCoef, vCoef, vAux);
        uni_vmulps(vCoef, vCoef, vDDim);
    } else if (idx == 3) {
        uni_vmovups(vCoef, vDDim);
        mov(rAux, reinterpret_cast<uintptr_t>(const_0_75));
        uni_vmulps(vCoef, vCoef, ptr[(Xbyak::Reg64)rAux]);
        uni_vmulps(vCoef, vCoef, vDDim);
        uni_vmovups(vAux, vCoef);
        uni_vmulps(vAux, vAux, vDDim);
        uni_vsubps(vCoef, vCoef, vAux);
    }
}

template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::bicubicCoefficients(const Vmm& vCoef, const Vmm& vDDim, const uint8_t idx) {
    if (idx == 0) {
        uni_vmovups(vCoef, vDDim);
        vfnmadd132ps(vCoef, vPool[onesFIdx], vPool[const_2_00_idx]);
        uni_vfmadd231ps(vCoef, vDDim, vDDim);
        uni_vmulps(vCoef, vCoef, vDDim);
        uni_vmulps(vCoef, vCoef, vPool[const_0_75_idx]);
    } else if (idx == 1) {
        uni_vmovups(vCoef, vDDim);
        vfmsub132ps(vCoef, vPool[const_2_25_idx], vPool[const_1_25_idx]);
        uni_vmulps(vCoef, vCoef, vDDim);
        uni_vfmadd132ps(vCoef, vPool[onesFIdx], vDDim);
    } else if (idx == 2) {
        uni_vmulps(vCoef, vDDim, vDDim);
        vfmadd132ps(vCoef, vPool[const_0_75_idx], vPool[const_1_25_idx]);
        uni_vfmsub231ps(vCoef, vDDim, vPool[const_1_50_idx]);
        uni_vmulps(vCoef, vCoef, vDDim);
    } else if (idx == 3) {
        uni_vmulps(vCoef, vPool[const_0_75_idx], vDDim);
        uni_vmulps(vCoef, vCoef, vDDim);
        vfnmadd132ps(vCoef, vCoef, vDDim);
    }
}

template <x64::cpu_isa_t isa> // Works for AVX512, AVX2, AVX, SSE41
void jitGridSampleKernel<isa>::nearestInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
    const auto& vSrcShift   = vWCoord;
    const auto& vAux        = vHCoord;
    const Vmask kGatherMask = Vmask(isa == x64::avx512_core ? 1 : getVecIdx());
    const Vmask kAuxMask    = Vmask(isa == x64::avx512_core ? 2 : getVecIdx());

    uni_vroundps(vWCoord, vWCoord, 0x0); // Round near
    uni_vroundps(vHCoord, vHCoord, 0x0); // Round near

    bool useMask = false, zeroFill = false;
    if (jcp.paddingMode == PaddingMode::ZEROS) {
        useMask = zeroFill = true;
        zerosPadding(kGatherMask, vHCoord, vWCoord);
    } else if (jcp.paddingMode == PaddingMode::BORDER) {
        borderPadding(vWCoord, vWCoord, coord::w);
        borderPadding(vHCoord, vHCoord, coord::h);
    } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
        reflectionPadding(vWCoord, vWCoord, kAuxMask, coord::w);
        reflectionPadding(vHCoord, vHCoord, kAuxMask, coord::h);
    }

    hwShiftPs2dq(vSrcShift, vHCoord, vWCoord, vPool[srcWidthFIdx]);

    // PER CHANNEL LOOP
    Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
    auto rChannel = r64Ref();
    auto rSrcTmp  = r64Ref();
    auto rDstTmp  = r64Ref();
    mov(rChannel, ptr[regParams + GET_OFF(channelsNum)]);
    mov(rSrcTmp, r64Pool[rSrcIdx]);
    mov(rDstTmp, r64Pool[rDstIdx]);
    L(lChannelLoopBegin);
    {
        cmp(rChannel, 0);
        jle(lChannelLoopEnd, T_NEAR);

        if (jcp.paddingMode == PaddingMode::ZEROS) {
            uni_kmovd(kAuxMask, kGatherMask);
        }

        if (!tail) {
            uni_vpgatherdd(vAux, rSrcTmp, vSrcShift, kAuxMask, useMask, zeroFill);
            uni_vmovups(ptr[(Xbyak::Reg64)rDstTmp], vAux);
        } else {
            if (isa == x64::avx512_core) {
                uni_vpgatherdd(vAux, rSrcTmp, vSrcShift, kAuxMask, useMask, zeroFill);
                uni_vmovups(ptr[(Xbyak::Reg64) rDstTmp] | kTailMask, vAux);
            } else {
                auto rWrkTmp = r64Ref();
                mov(rWrkTmp, r64Pool[rWorkAmountIdx]);
                maskMov32(rDstTmp, rSrcTmp, vPool[kAuxMask.getIdx()], vSrcShift, rWrkTmp, useMask, zeroFill);
            }
        }

        add(rSrcTmp, r64Pool[rSrcChannelStepBIdx]);
        add(rDstTmp, r64Pool[rDstChannelStepBIdx]);
        dec(rChannel);

        jmp(lChannelLoopBegin, T_NEAR);
        L(lChannelLoopEnd);
    }

    if (isa != x64::avx512_core) {
        releaseVecIdx(kGatherMask.getIdx());
        releaseVecIdx(kAuxMask.getIdx());
    }
}

template <>
void jitGridSampleKernel<x64::avx512_core>::bilinearInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
    auto vDX     = vmmRef();
    auto vDY     = vmmRef();
    const auto &shift00 = vWCoord;
    const auto &shift01 = vHCoord;
    auto shift10 = vmmRef();
    auto shift11 = vmmRef();
    auto vAux   = vmmRef();
    const auto &kMask00  = k1;
    const auto &kMask01  = k2;
    const auto &kMask10  = k3;
    const auto &kMask11  = k4;
    const auto &kAuxMask = k5;

    uni_vmovups(vDX, vWCoord);
    uni_vmovups(vDY, vHCoord);
    uni_vroundps(vWCoord, vWCoord, 0x1); // Round floor
    uni_vroundps(vHCoord, vHCoord, 0x1); // Round floor
    uni_vsubps(vDX, vDX, vWCoord);
    uni_vsubps(vDY, vDY, vHCoord);

    uni_vaddps(shift10, vWCoord, vPool[onesFIdx]);
    uni_vaddps(shift11, vHCoord, vPool[onesFIdx]);
    bool useMask = false, zeroFill = false;
    if (jcp.paddingMode == PaddingMode::ZEROS) {
        useMask = zeroFill = true;
        zerosPadding(kMask00, vHCoord, vWCoord); // (y; x)
        zerosPadding(kMask01, vHCoord, shift10); // (y; x + 1)
        zerosPadding(kMask11, shift11, shift10); // (y + 1; x + 1)
        zerosPadding(kMask10, shift11, vWCoord); // (y + 1; x)

        hwShiftPs2dq(shift00, vHCoord, vWCoord, vPool[srcWidthFIdx]);
        uni_vpaddd(shift01, shift00, vPool[dataTypeSizeIdx]);
        uni_vpaddd(shift10, shift00, vPool[srcWidthBIdx]); // shift11??
        uni_vpaddd(shift11, shift10, vPool[dataTypeSizeIdx]); // sub??
    } else if (jcp.paddingMode == PaddingMode::BORDER) {
        borderPadding(vWCoord, vWCoord, coord::w);
        borderPadding(vHCoord, vHCoord, coord::h);
        borderPadding(shift10, shift10, coord::w);
        borderPadding(shift11, shift11, coord::h);
    } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
        reflectionPadding(vWCoord, vWCoord, kAuxMask, coord::w);
        reflectionPadding(vHCoord, vHCoord, kAuxMask, coord::h);
        reflectionPadding(shift10, shift10, kAuxMask, coord::w);
        reflectionPadding(shift11, shift11, kAuxMask, coord::h);
    }
    if (jcp.paddingMode == PaddingMode::BORDER || jcp.paddingMode == PaddingMode::REFLECTION) {
        // W * y + x
        hwShiftPs2dq(vAux, shift11, vWCoord, vPool[srcWidthFIdx]);
        hwShiftPs2dq(vWCoord, vHCoord, vWCoord, vPool[srcWidthFIdx]);
        hwShiftPs2dq(vHCoord, vHCoord, shift10, vPool[srcWidthFIdx]);
        hwShiftPs2dq(shift11, shift11, shift10, vPool[srcWidthFIdx]);
        uni_vmovups(shift10, vAux);
    }

    auto vQ0 = vmmRef();
    auto vQ1 = vmmRef();

    // PER CHANNEL LOOP
    Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
    auto rChannel = r64Ref();
    auto rSrcTmp  = r64Ref();
    auto rDstTmp  = r64Ref();
    mov(rChannel, 0);
    mov(rSrcTmp, r64Pool[rSrcIdx]);
    mov(rDstTmp, r64Pool[rDstIdx]);
    L(lChannelLoopBegin);
    {
        cmp(rChannel, r64Pool[rChannelNumIdx]);
        jge(lChannelLoopEnd, T_NEAR);

        // (y; x)
        if (jcp.paddingMode == PaddingMode::ZEROS) {
            kmovw(kAuxMask, kMask00);
        }
        uni_vpgatherdd(vQ0, rSrcTmp, shift00, kAuxMask, useMask, zeroFill); // v00 -> vQ0
        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtdq2ps(vQ0, vQ0);
        }
        uni_vfmsub213ps(vQ0, vDX, vQ0); // q0 = -(v00 - dx * v00)

        // (y; x + 1)
        if (jcp.paddingMode == PaddingMode::ZEROS) {
            kmovw(kAuxMask, kMask01);
        }
        uni_vpgatherdd(vAux, rSrcTmp, shift01, kAuxMask, useMask, zeroFill);
        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtdq2ps(vAux, vAux);
        }
        uni_vfmsub231ps(vQ0, vAux, vDX); // q0 = -q0 + dx * v01

        // (y + 1; x + 1)
        if (jcp.paddingMode == PaddingMode::ZEROS) {
            kmovw(kAuxMask, kMask11);
        }
        uni_vpgatherdd(vAux, rSrcTmp, shift11, kAuxMask, useMask, zeroFill);
        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtdq2ps(vAux, vAux);
        }

        // (y + 1; x)
        if (jcp.paddingMode == PaddingMode::ZEROS) {
            kmovw(kAuxMask, kMask10);
        }
        uni_vpgatherdd(vQ1, rSrcTmp, shift10, kAuxMask, useMask, zeroFill);
        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtdq2ps(vQ1, vQ1);
        }

        uni_vfmsub213ps(vQ1, vDX, vQ1);  // q1 = -(v10 - dx * v10)
        uni_vfmsub231ps(vQ1, vAux, vDX); // q1 = -q1 + dx * v11
        // Res = q0 + dy * (q1 - q0)
        uni_vsubps(vQ1, vQ1, vQ0);
        uni_vfmadd132ps(vQ1, vQ0, vDY);

        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtps2dq(vQ1, vQ1);
        }

        if (!tail) {
            uni_vmovups(ptr[(Xbyak::Reg64)rDstTmp], vQ1);
        } else {
            uni_vmovups(ptr[(Xbyak::Reg64)rDstTmp] | kTailMask, vQ1);
        }
        add(rSrcTmp, r64Pool[rSrcChannelStepBIdx]);
        add(rDstTmp, r64Pool[rDstChannelStepBIdx]);
        add(rChannel, 1);

        jmp(lChannelLoopBegin, T_NEAR);
        L(lChannelLoopEnd);
    }
}

template <>
void jitGridSampleKernel<x64::sse41>::bilinearInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
    const auto& vDX = vPool[0];
    const auto& vDY = vPool[1];
    const auto& vGatherShift = vPool[2];
    const auto& vAux = vPool[3];
    const auto& vQ0 = vPool[4];
    const auto& vQ1 = vPool[5];
    const auto& kMask0 = masksContainer[4];
    const auto& kMask1 = masksContainer[5];

    uni_vmovups(vDX, vWCoord);
    uni_vmovups(vDY, vHCoord);
    uni_vroundps(vWCoord, vWCoord, 0x1); // Round floor
    uni_vroundps(vHCoord, vHCoord, 0x1); // Round floor
    uni_vsubps(vDX, vDX, vWCoord);
    uni_vsubps(vDY, vDY, vHCoord);

    if (jcp.paddingMode == PaddingMode::ZEROS) {
        zerosPadding(kMask0, vHCoord, vWCoord);
    }

    // PER CHANNEL LOOP
    Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
    auto rChannel = r64Ref();
    auto rSrcTmp  = r64Ref();
    auto rDstTmp  = r64Ref();
    mov(rChannel, 0);
    mov(rSrcTmp, r64Pool[rSrcIdx]);
    mov(rDstTmp, r64Pool[rDstIdx]);
    L(lChannelLoopBegin);
    {
        cmp(rChannel, r64Pool[rChannelNumIdx]);
        jge(lChannelLoopEnd, T_NEAR);

        if (jcp.paddingMode == PaddingMode::ZEROS) {
            // (x; y)
            zerosPadding(kMask0, vHCoord, vWCoord);
            uni_vmovups(vGatherShift, vHCoord);
            hwShiftPs2dq(vGatherShift, vGatherShift, vWCoord, vPool[srcWidthFIdx]);
            uni_vpxor(vQ0, vQ0, vQ0);
            uni_vpgatherdd(vQ0, rSrcTmp, vGatherShift, kMask0); // v00 -> vQ0
            if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                uni_vcvtdq2ps(vQ0, vQ0);
            }
            uni_vfmsub213ps(vQ0, vDX, vQ0); // q0 = -(v00 - dx * v00)

            // (x + 1; y)
            uni_vaddps(vWCoord, vWCoord, vPool[onesFIdx]);
            zerosPadding(kMask0, vHCoord, vWCoord);
            uni_vmovups(vGatherShift, vHCoord);
            hwShiftPs2dq(vGatherShift, vGatherShift, vWCoord, vPool[srcWidthFIdx]);
            uni_vpxor(vAux, vAux, vAux);
            uni_vpgatherdd(vAux, rSrcTmp, vGatherShift, kMask0);

            // q0 = -q0 + dx * v01
            if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                uni_vcvtdq2ps(vAux, vAux);
            }
            uni_vfmsub231ps(vQ0, vAux, vDX);

            // (x + 1; y + 1)
            uni_vaddps(vHCoord, vHCoord, vPool[onesFIdx]);
            zerosPadding(kMask0, vHCoord, vWCoord);
            uni_vmovups(vGatherShift, vHCoord);
            hwShiftPs2dq(vGatherShift, vGatherShift, vWCoord, vPool[srcWidthFIdx]);
            uni_vpxor(vAux, vAux, vAux);
            uni_vpgatherdd(vAux, rSrcTmp, vGatherShift, kMask0);

            // (x; y + 1)
            uni_vsubps(vWCoord, vWCoord, vPool[onesFIdx]);
            zerosPadding(kMask0, vHCoord, vWCoord);
            uni_vmovups(vGatherShift, vHCoord);
            hwShiftPs2dq(vGatherShift, vGatherShift, vWCoord, vPool[srcWidthFIdx]);
            uni_vpxor(vQ1, vQ1, vQ1);
            uni_vpgatherdd(vQ1, rSrcTmp, vGatherShift, kMask0);
            uni_vsubps(vHCoord, vHCoord, vPool[onesFIdx]);
            if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                uni_vcvtdq2ps(vQ1, vQ1);
            }

            uni_vfmsub213ps(vQ1, vDX, vQ1);  // q1 = -(v10 - dx * v10)
            uni_vfmsub231ps(vQ1, vAux, vDX); // q1 = -q1 + dx * v11
            // Res = q0 + dy * (q1 - q0)
            uni_vsubps(vQ1, vQ1, vQ0);
            uni_vfmadd132ps(vQ1, vQ0, vDY);

            if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                uni_vcvtps2dq(vQ1, vQ1);
            }
        }

        uni_vmovups(ptr[(Xbyak::Reg64)rDstTmp], vQ1);
        add(rSrcTmp, r64Pool[rSrcChannelStepBIdx]);
        add(rDstTmp, r64Pool[rDstChannelStepBIdx]);
        add(rChannel, 1);

        jmp(lChannelLoopBegin, T_NEAR);
        L(lChannelLoopEnd);
    }
}

template <x64::cpu_isa_t isa> // Works for AVX2, AVX
void jitGridSampleKernel<isa>::bilinearInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
    auto vDX         = vmmRef();
    auto vDY         = vmmRef();
    auto vAux        = vmmRef();
    auto kGatherMask = vmmRef();
    const auto& shift00 = vWCoord;
    const auto& shift01 = vHCoord;
    auto shift10     = vmmRef();
    auto shift11     = vmmRef();

    uni_vmovups(vDX, vWCoord);
    uni_vmovups(vDY, vHCoord);
    uni_vroundps(vWCoord, vWCoord, 0x1); // Round floor
    uni_vroundps(vHCoord, vHCoord, 0x1); // Round floor
    uni_vsubps(vDX, vDX, vWCoord);
    uni_vsubps(vDY, vDY, vHCoord);
    uni_vaddps(shift10, vWCoord, vPool[onesFIdx]);
    uni_vaddps(shift11, vHCoord, vPool[onesFIdx]);

    bool useMask = false, zeroFill = false;
    if (jcp.paddingMode == PaddingMode::ZEROS) {
        useMask = zeroFill = true;

        hwShiftPs2dq(shift00, vHCoord, vWCoord, vPool[srcWidthFIdx]);
        uni_vpaddd(shift01, shift00, vPool[dataTypeSizeIdx]);
        uni_vpaddd(shift10, shift00, vPool[srcWidthBIdx]); // shift11??
        uni_vpaddd(shift11, shift10, vPool[dataTypeSizeIdx]); // sub??
    } else if (jcp.paddingMode == PaddingMode::BORDER) {
        borderPadding(vWCoord, vWCoord, coord::w);
        borderPadding(vHCoord, vHCoord, coord::h);
        borderPadding(shift10, shift10, coord::w);
        borderPadding(shift11, shift11, coord::h);
    } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
        reflectionPadding(vWCoord, vWCoord, kGatherMask, coord::w);
        reflectionPadding(vHCoord, vHCoord, kGatherMask, coord::h);
        reflectionPadding(shift10, shift10, kGatherMask, coord::w);
        reflectionPadding(shift11, shift11, kGatherMask, coord::h);
    }
    if (jcp.paddingMode == PaddingMode::BORDER || jcp.paddingMode == PaddingMode::REFLECTION) {
        // W * y + x
        hwShiftPs2dq(vAux, shift11, vWCoord, vPool[srcWidthFIdx]);
        hwShiftPs2dq(vWCoord, vHCoord, vWCoord, vPool[srcWidthFIdx]);
        hwShiftPs2dq(vHCoord, vHCoord, shift10, vPool[srcWidthFIdx]);
        hwShiftPs2dq(shift11, shift11, shift10, vPool[srcWidthFIdx]);
        uni_vmovups(shift10, vAux);
    }

    auto vQ0 = vmmRef();
    auto vQ1 = vmmRef();

    // PER CHANNEL LOOP
    Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
    auto rChannel = r64Ref();
    auto rSrcTmp  = r64Ref();
    auto rDstTmp  = r64Ref();
    mov(rChannel, ptr[regParams + GET_OFF(channelsNum)]);
    mov(rSrcTmp, r64Pool[rSrcIdx]);
    mov(rDstTmp, r64Pool[rDstIdx]);
//    auto rAux = r64Ref();
    L(lChannelLoopBegin);
    {
        cmp(rChannel, 0);
        jle(lChannelLoopEnd, T_NEAR);

        // (y; x)
        // v00 -> vQ0
        if (jcp.paddingMode == PaddingMode::ZEROS) {
//            if (isa == x64::avx2)
//                uni_vmovups(kGatherMask, kMask00);
        }
        uni_vpgatherdd(vQ0, rSrcTmp, shift00, kGatherMask, useMask, zeroFill);
        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtdq2ps(vQ0, vQ0);
        }
        // q0 = -(v00 - dx * v00)
        if (isa == x64::avx2) {
            uni_vfmsub213ps(vQ0, vDX, vQ0);
        } else {
            uni_vmulps(kGatherMask, vQ0, vDX);
            uni_vsubps(vQ0, kGatherMask, vQ0);
        }

        // (y; x + 1)
        if (jcp.paddingMode == PaddingMode::ZEROS) {
//            if (isa == x64::avx2)
//                uni_vmovups(kGatherMask, kMask01);
        }
        uni_vpgatherdd(vAux, rSrcTmp, shift01, kGatherMask, useMask, zeroFill);
        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtdq2ps(vAux, vAux);
        }
        uni_vfmsub231ps(vQ0, vAux, vDX); // q0 = -q0 + dx * v01

        // (y + 1; x + 1)
        if (jcp.paddingMode == PaddingMode::ZEROS) {
//            if (isa == x64::avx2)
//                uni_vmovups(kGatherMask, kMask11);
        }
        uni_vpgatherdd(vAux, rSrcTmp, (Vmm)shift11, kGatherMask, useMask, zeroFill);
        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtdq2ps(vAux, vAux);
        }

        // (y + 1; x)
        if (jcp.paddingMode == PaddingMode::ZEROS) {
//            if (isa == x64::avx2)
//                uni_vmovups(kGatherMask, kMask10);
        }
        uni_vpgatherdd(vQ1, rSrcTmp, (Vmm)shift10, kGatherMask, useMask, zeroFill);
        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtdq2ps(vQ1, vQ1);
        }

        // q1 = -(v10 - dx * v10)
        if (isa == x64::avx2) {
            uni_vfmsub213ps(vQ1, vDX, vQ1);
        } else {
            uni_vmulps(kGatherMask, vQ1, vDX);
            uni_vsubps(vQ1, kGatherMask, vQ1);
        }
        uni_vfmsub231ps(vQ1, vAux, vDX); // q1 = -q1 + dx * v11
        // Res = q0 + dy * (q1 - q0)
        uni_vsubps(vQ1, vQ1, vQ0);
        uni_vfmadd132ps(vQ1, vQ0, vDY);

        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtps2dq(vQ1, vQ1);
        }

        if (!tail) {
            uni_vmovups(ptr[(Xbyak::Reg64)rDstTmp], vQ1);
        } else {
            storeVectorPart(rDstTmp, r64Pool[rWorkAmountIdx], vQ1, dataTypeSize);
        }

        add(rSrcTmp, r64Pool[rSrcChannelStepBIdx]);
        add(rDstTmp, r64Pool[rDstChannelStepBIdx]);
        dec(rChannel);

        jmp(lChannelLoopBegin, T_NEAR);
        L(lChannelLoopEnd);
    }
}

template <>
void jitGridSampleKernel<x64::avx512_core>::bicubicInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
    auto vHTop       = vmmRef();
    auto vWLeft      = vmmRef();
    auto vDX         = vmmRef();
    auto vDY         = vmmRef();
    auto vXDotProd   = vmmRef();
    auto& vYDotProd  = vDX;
    auto vSrcShift0  = vmmRef();
    auto vSrcShift   = vmmRef();
    auto vAux        = vmmRef();
    const auto& kMask0   = masksContainer[1];
    const auto& kMask1   = masksContainer[2];
    const auto& kMask2   = masksContainer[3];
    const auto& kMask3   = masksContainer[4];
    const auto& kAuxMask = masksContainer[5];
    const auto& kMaskH   = masksContainer[6];

    uni_vroundps(vHTop, vHCoord, 0x1);  // Round floor
    uni_vroundps(vWLeft, vWCoord, 0x1); // Round floor
    uni_vsubps(vDY, vHCoord, vHTop);
    uni_vsubps(vDX, vWCoord, vWLeft);
    uni_vsubps(vHTop, vHTop, vPool[onesFIdx]);
    uni_vsubps(vWLeft, vWLeft, vPool[onesFIdx]);

    vRefWrap<Vmm> vCX[4] = { vmmRef(), vmmRef(), vmmRef(), vmmRef() };
    for (int i = 0; i < 4; i++) {
        bicubicCoefficients(vCX[i], vDX, i);
    }

    if (jcp.paddingMode == PaddingMode::ZEROS) {
        zerosPadding0(kMask0, vWLeft, vPool[srcWidthFIdx], kAuxMask);
        uni_vaddps(vWCoord, vWLeft, vPool[onesFIdx]);
        zerosPadding0(kMask1, vWCoord, vPool[srcWidthFIdx], kAuxMask);
        uni_vaddps(vWCoord, vWCoord, vPool[onesFIdx]);
        zerosPadding0(kMask2, vWCoord, vPool[srcWidthFIdx], kAuxMask);
        uni_vaddps(vWCoord, vWCoord, vPool[onesFIdx]);
        zerosPadding0(kMask3, vWCoord, vPool[srcWidthFIdx], kAuxMask);
    }

    // PER CHANNEL LOOP
    Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
    auto rChannel = r64Ref();
    auto rSrcTmp  = r64Ref();
    auto rDstTmp  = r64Ref();
    mov(rChannel, 0);
    mov(rSrcTmp, r64Pool[rSrcIdx]);
    mov(rDstTmp, r64Pool[rDstIdx]);
    L(lChannelLoopBegin);
    {
        cmp(rChannel, r64Pool[rChannelNumIdx]);
        jge(lChannelLoopEnd, T_NEAR);

        uni_vmovups(vHCoord, vHTop);
        uni_vpxor(vYDotProd, vYDotProd, vYDotProd);
        for (int h = 0; h < 4; h++) {
            // (y - 1 + h; x - 1)
            if (jcp.paddingMode == PaddingMode::ZEROS) {
                zerosPadding0(kMaskH, vHCoord, vPool[srcHeightFIdx], kMaskH);
                kandw(kAuxMask, kMaskH, kMask0);
                uni_vmulps(vSrcShift0, vHCoord, vPool[srcWidthFIdx]);
                uni_vmovups(vWCoord, vWLeft);
                uni_vaddps(vSrcShift, vSrcShift0, vWCoord);
                uni_vpxor(vAux, vAux, vAux);
            } else if (jcp.paddingMode == PaddingMode::BORDER) {
                borderPadding(vSrcShift0, vHCoord, coord::h);
                uni_vmulps(vSrcShift0, vSrcShift0, vPool[srcWidthFIdx]);
                uni_vmovups(vWCoord, vWLeft);
                borderPadding(vSrcShift, vWCoord, coord::w);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                kxnorw(kAuxMask, kAuxMask, kAuxMask);
            } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
                reflectionPadding(vSrcShift0, vHCoord, kAuxMask, coord::h);
                uni_vmulps(vSrcShift0, vSrcShift0, vPool[srcWidthFIdx]);
                uni_vmovups(vWCoord, vWLeft);
                reflectionPadding(vSrcShift, vWCoord, kAuxMask, coord::w);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                kxnorw(kAuxMask, kAuxMask, kAuxMask);
            }
            uni_vcvtps2dq(vSrcShift, vSrcShift);
            if (dataTypeSize > 1)
                uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
            uni_vpgatherdd(vAux, rSrcTmp, vSrcShift, kAuxMask);
            if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                uni_vcvtdq2ps(vAux, vAux);
            }
            uni_vmulps(vXDotProd, vAux, vCX[0]);

            // (y - 1 + h; x)
            // (y - 1 + h; x + 1)
            // (y - 1 + h; x + 2)
            for (int w = 1; w < 4; w++) {
                uni_vaddps(vWCoord, vWCoord, vPool[onesFIdx]);
                if (jcp.paddingMode == PaddingMode::ZEROS) {
                    uni_vaddps(vSrcShift, vSrcShift0, vWCoord);
                    kandw(kAuxMask, kMaskH, (&kMask0)[w]);
                    uni_vpxor(vAux, vAux, vAux);
                } else if (jcp.paddingMode == PaddingMode::BORDER) {
                    borderPadding(vSrcShift, vWCoord, coord::w);
                    uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                    kxnorw(kAuxMask, kAuxMask, kAuxMask);
                } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
                    reflectionPadding(vSrcShift, vWCoord, kAuxMask, coord::w);
                    uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                    kxnorw(kAuxMask, kAuxMask, kAuxMask);
                }
                uni_vcvtps2dq(vSrcShift, vSrcShift);
                if (dataTypeSize > 1)
                    uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
                uni_vpgatherdd(vAux, rSrcTmp, vSrcShift, kAuxMask);
                if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                    uni_vcvtdq2ps(vAux, vAux);
                }
                uni_vfmadd231ps(vXDotProd, vAux, vCX[w]);
            }

            if (h != 3) {
                uni_vaddps(vHCoord, vHCoord, vPool[onesFIdx]);
            }

            bicubicCoefficients(vAux, vDY, h);
            uni_vfmadd231ps(vYDotProd, vXDotProd, vAux);
        }

        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtps2dq(vYDotProd, vYDotProd);
        }

        if (tail) {
            uni_vmovups(ptr[(Xbyak::Reg64)rDstTmp] | kTailMask, vYDotProd);
        } else {
            uni_vmovups(ptr[(Xbyak::Reg64)rDstTmp], vYDotProd);
        }
        add(rSrcTmp, r64Pool[rSrcChannelStepBIdx]);
        add(rDstTmp, r64Pool[rDstChannelStepBIdx]);
        add(rChannel, 1);

        jmp(lChannelLoopBegin, T_NEAR);
        L(lChannelLoopEnd);
    }
}

template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::bicubicInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
    const auto& vHTop      = vPool[0];
    const auto& vWLeft     = vPool[1];
    const auto& vDX        = vPool[2];
    const auto& vDY        = vPool[3];
    const auto& vXDotProd  = vPool[4];
    const auto& vYDotProd  = vDX;
    const auto& vSrcShift0 = vPool[5];
    const auto& vSrcShift  = vPool[6];
    const auto& vCX0       = vPool[7];
    const auto& vCX1       = vPool[8];
    const auto& vCX2       = vPool[9];
    const auto& vCX3       = vPool[10];
    const auto& vAux       = vPool[11]; // &vWLeft
    const auto& kMask0   = vCX0;
    const auto& kMask1   = vCX1;
    const auto& kMask2   = vCX2;
    const auto& kMask3   = vCX3;
    const auto& kAuxMask = vAux;
    const auto& kMaskH   = vSrcShift;

    uni_vroundps(vWLeft, vWCoord, 0x1); // Round floor
    uni_vroundps(vHTop, vHCoord, 0x1);  // Round floor
    uni_vsubps(vDX, vWCoord, vWLeft);
    uni_vsubps(vDY, vHCoord, vHTop);
    uni_vsubps(vWLeft, vWLeft, vPool[onesFIdx]);
    uni_vsubps(vHTop, vHTop, vPool[onesFIdx]);

    bicubicCoefficients(vCX0, vDX, 0); // TODO: for
    bicubicCoefficients(vCX1, vDX, 1);
    bicubicCoefficients(vCX2, vDX, 2);
    bicubicCoefficients(vCX3, vDX, 3);

    if (jcp.paddingMode == PaddingMode::ZEROS) {
        hwShiftPs2dq(vSrcShift0, vHTop, vWLeft, vPool[srcWidthFIdx]);
    }

    // PER CHANNEL LOOP
    Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
    auto rChannel = r64Ref();
    auto rSrcTmp = r64Ref();
    auto rDstTmp = r64Ref();
    mov(rChannel, 0);
    mov(rSrcTmp, r64Pool[rSrcIdx]);
    mov(rDstTmp, r64Pool[rDstIdx]);
    L(lChannelLoopBegin);
    {
        cmp(rChannel, r64Pool[rChannelNumIdx]);
        jge(lChannelLoopEnd, T_NEAR);

        if (jcp.paddingMode == PaddingMode::ZEROS) {
            uni_vmovups(vSrcShift, vSrcShift0);
        }
        uni_vmovups(vHCoord, vHTop);
        uni_vpxor(vYDotProd, vYDotProd, vYDotProd);
        for (int i = 0; i < 4; i++) {
            // (y - 1 + i; x - 1)
            if (jcp.paddingMode == PaddingMode::ZEROS) {
                zerosPadding(kMaskH, vPool[srcHeightFIdx], vHCoord);
                uni_vpxor(vAux, vAux, vAux);
            } else if (jcp.paddingMode == PaddingMode::BORDER) {
                borderPadding(vSrcShift0, vHCoord, coord::h);
                uni_vmulps(vSrcShift0, vSrcShift0, vPool[srcWidthFIdx]);
                uni_vmovups(vWCoord, vWLeft);
                borderPadding(vSrcShift, vWCoord, coord::w);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                uni_vcvtps2dq(vSrcShift, vSrcShift);
                if (dataTypeSize > 1)
                    uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
            } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
                reflectionPadding(vSrcShift0, vHCoord, kAuxMask, coord::h);
                uni_vmulps(vSrcShift0, vSrcShift0, vPool[srcWidthFIdx]);
                uni_vmovups(vWCoord, vWLeft);
                reflectionPadding(vSrcShift, vWCoord, kAuxMask, coord::w);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                uni_vcvtps2dq(vSrcShift, vSrcShift);
                if (dataTypeSize > 1)
                    uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
            }
            if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                uni_vcvtdq2ps(vAux, vAux);
            }
            uni_vmulps(vXDotProd, vAux, vCX0);
// TODO: cycle 3?
            // (y - 1 + i; x)
            if (jcp.paddingMode == PaddingMode::ZEROS) {
                uni_vpxor(vAux, vAux, vAux);
                uni_vpaddd(vSrcShift, vSrcShift, vPool[dataTypeSizeIdx]);
            } else if (jcp.paddingMode == PaddingMode::BORDER) {
                uni_vaddps(vWCoord, vWCoord, vPool[onesFIdx]);
                borderPadding(vSrcShift, vWCoord, coord::w);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                uni_vcvtps2dq(vSrcShift, vSrcShift);
                if (dataTypeSize > 1)
                    uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
            } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
                uni_vaddps(vWCoord, vWCoord, vPool[onesFIdx]);
                reflectionPadding(vSrcShift, vWCoord, kAuxMask, coord::w);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                uni_vcvtps2dq(vSrcShift, vSrcShift);
                if (dataTypeSize > 1)
                    uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
            }
            if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                uni_vcvtdq2ps(vAux, vAux);
            }
            uni_vfmadd231ps(vXDotProd, vAux, vCX1);

            // (y - 1 + i; x + 1)
            if (jcp.paddingMode == PaddingMode::ZEROS) {
                uni_vpxor(vAux, vAux, vAux);
                uni_vpaddd(vSrcShift, vSrcShift, vPool[dataTypeSizeIdx]);
            } else if (jcp.paddingMode == PaddingMode::BORDER) {
                uni_vaddps(vWCoord, vWCoord, vPool[onesFIdx]);
                borderPadding(vSrcShift, vWCoord, coord::w);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                uni_vcvtps2dq(vSrcShift, vSrcShift);
                if (dataTypeSize > 1)
                    uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
            } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
                uni_vaddps(vWCoord, vWCoord, vPool[onesFIdx]);
                reflectionPadding(vSrcShift, vWCoord, kAuxMask, coord::w);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                uni_vcvtps2dq(vSrcShift, vSrcShift);
                if (dataTypeSize > 1)
                    uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
            }
            if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                uni_vcvtdq2ps(vAux, vAux);
            }
            uni_vfmadd231ps(vXDotProd, vAux, vCX2);

            // (y - 1 + i; x + 2)
            if (jcp.paddingMode == PaddingMode::ZEROS) {
                uni_vpxor(vAux, vAux, vAux);
                uni_vpaddd(vSrcShift, vSrcShift, vPool[dataTypeSizeIdx]);
            } else if (jcp.paddingMode == PaddingMode::BORDER) {
                uni_vaddps(vWCoord, vWCoord, vPool[onesFIdx]);
                borderPadding(vSrcShift, vWCoord, coord::w);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                uni_vcvtps2dq(vSrcShift, vSrcShift);
                if (dataTypeSize > 1)
                    uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
            } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
                uni_vaddps(vWCoord, vWCoord, vPool[onesFIdx]);
                reflectionPadding(vSrcShift, vWCoord, kAuxMask, coord::w);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                uni_vcvtps2dq(vSrcShift, vSrcShift);
                if (dataTypeSize > 1)
                    uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
            }
            if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                uni_vcvtdq2ps(vAux, vAux);
            }
            uni_vfmadd231ps(vXDotProd, vAux, vCX3);

            if (i != 3) {
                uni_vaddps(vHCoord, vHCoord, vPool[onesFIdx]);
                if (jcp.paddingMode == PaddingMode::ZEROS) {
                    uni_vpaddd(vSrcShift, vSrcShift, vPool[srcWidthBIdx]);
                }
            }

            bicubicCoefficients(vAux, vDY, i);
            uni_vfmadd231ps(vYDotProd, vXDotProd, vAux);
        }

        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtps2dq(vYDotProd, vYDotProd);
        }

        uni_vmovups(ptr[(Xbyak::Reg64)rDstTmp], vYDotProd);
        add(rSrcTmp, r64Pool[rSrcChannelStepBIdx]);
        add(rDstTmp, r64Pool[rDstChannelStepBIdx]);
        add(rChannel, 1);

        jmp(lChannelLoopBegin, T_NEAR);
        L(lChannelLoopEnd);
    }
}

template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::tail() {
    Xbyak::Label lEnd;
    cmp(r64Pool[rWorkAmountIdx], 0);
    jle(lEnd, T_NEAR);

    const Vmm& vHCoord = vPool[getVecIdx()];
    const Vmm& vWCoord = vPool[getVecIdx()];

    getTailCoordinates(vHCoord, vWCoord);
    denormalizeRawCoordinates(vWCoord, vHCoord);
    interpolation(vWCoord, vHCoord, true);

    releaseVecIdx(vHCoord.getIdx());
    releaseVecIdx(vWCoord.getIdx());

    if (dataTypeSize > 1)
        sal(r64Pool[rWorkAmountIdx], dataTypeShift); // multiply by source data type size.
    add(r64Pool[rDstIdx], r64Pool[rWorkAmountIdx]);

    L(lEnd);
}

template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::hwShiftPs2dq(const Vmm& vDst, const Vmm& vHCoord,const Vmm& vWCoord, const Vmm& vWidth) {
    if (vDst.getIdx() == vWCoord.getIdx()) {
        if (one_of(isa, x64::avx2, x64::avx512_core)) {
            uni_vfmadd231ps(vDst, vHCoord, vWidth);
        } else {
            auto vTmp = vmmRef();
            uni_vmulps(vTmp, vHCoord, vWidth);
            uni_vaddps(vDst, vTmp, vWCoord);
        }
    } else if (vDst.getIdx() == vHCoord.getIdx()) {
        uni_vfmadd132ps(vDst, vWCoord, vWidth);
    } else if (vDst.getIdx() == vWidth.getIdx()) {
        uni_vfmadd132ps(vDst, vWCoord, vHCoord);
    } else {
        if (one_of(isa, x64::avx2, x64::avx512_core)) {
            uni_vmovups(vDst, vWCoord);
            uni_vfmadd231ps(vDst, vHCoord, vWidth);
        } else {
            uni_vmulps(vDst, vHCoord, vWidth);
            uni_vaddps(vDst, vDst, vWCoord);
        }
    }

    if (isa == x64::avx) { // vpslld works just with XMM for AVX, so use vmulps for YMM
        if (dataTypeSize > 1) {
            auto rAux = r64Ref();
            const float val = dataTypeSize;
            static const float dataTypeSizeArr[8] = {val, val, val, val, val, val, val, val};
            mov(rAux, reinterpret_cast<uintptr_t>(dataTypeSizeArr));
            uni_vmulps(vDst, vDst, ptr[(Xbyak::Reg64)rAux]);
        }
        uni_vcvtps2dq(vDst, vDst);
    } else {
        uni_vcvtps2dq(vDst, vDst);
        if (dataTypeSize > 1)
            uni_vpslld(vDst, vDst, dataTypeShift); // multiply by source data type size.
    }
}

template struct jitGridSampleKernel<x64::avx512_core>;
template struct jitGridSampleKernel<x64::avx2>;
template struct jitGridSampleKernel<x64::avx>;
template struct jitGridSampleKernel<x64::sse41>;

}   // namespace intel_cpu
}   // namespace ov
