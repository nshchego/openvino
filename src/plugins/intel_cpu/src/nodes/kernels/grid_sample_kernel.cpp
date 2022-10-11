// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "grid_sample_kernel.hpp"

using namespace dnnl::impl::cpu;

namespace ov {
namespace intel_cpu {

#define GET_OFF(field) offsetof(jGridSamplesExecArgs, field)

template <x64::cpu_isa_t isa>
JitGridSampleKernel<isa>::JitGridSampleKernel(const jGridSampleConfParams& jcp) :
        JitGridSampleKernelBase(jcp) {
    vlen = x64::cpu_isa_traits<isa>::vlen;
    dataTypeSize = jcp.inDataPrc.size();
    gridTypeSize = jcp.gridPrc.size();
    dataElPerVec = vlen / dataTypeSize;
    gridElPerVec = vlen / gridTypeSize;
    if (dataTypeSize == 2)
        dataTypeShift = 1;
    else if (dataTypeSize == 4)
        dataTypeShift = 2;
}

template <x64::cpu_isa_t isa>
void JitGridSampleKernel<isa>::create_ker() {
    auto code = x64::jit_generator::create_kernel();
    if (code != dnnl::impl::status::success)
        IE_THROW() << "Could not create GridSample kernel. Error code: " << std::to_string(code);
    ker_ = (decltype(ker_))jit_ker();
}

template <x64::cpu_isa_t isa>
void JitGridSampleKernel<isa>::generate() {
    this->preamble();
    registersPool = RegistersPool::create(isa, {rax, rcx, rsp, rdi, k0});

    regSrc  = getReg64();
    regGrid = getReg64();
    regDst  = getReg64();
    regSrcChannelStepB = getReg64();
    regDstChannelStepB = getReg64();

    mov(regSrc,  ptr[regParams + GET_OFF(src)]);
    mov(regGrid, ptr[regParams + GET_OFF(grid)]);
    mov(regDst,  ptr[regParams + GET_OFF(dst)]);
    mov(regSrcChannelStepB, ptr[regParams + GET_OFF(srcChannelStepB)]);
    mov(regDstChannelStepB, ptr[regParams + GET_OFF(dstChannelStepB)]);

    auto rAux = getReg64();

    if (isa == x64::avx512_core || jcp.interpolationMode != InterpolationMode::BICUBIC) {
        vSrcWidthF = getVmm();
        mov(rAux, ptr[regParams + GET_OFF(srcWidthF)]);
        uni_vpbroadcastd(vSrcWidthF, ptr[rAux]);
        vSrcHeightF = getVmm();
        mov(rAux, ptr[regParams + GET_OFF(srcHeightF)]);
        uni_vpbroadcastd(vSrcHeightF, ptr[rAux]);
    }

    if (one_of(jcp.paddingMode, PaddingMode::ZEROS, PaddingMode::BORDER) &&
            (isa == x64::avx512_core ||
             isa == x64::avx2 && !one_of(jcp.interpolationMode, InterpolationMode::BILINEAR, InterpolationMode::BICUBIC) ||
             one_of(isa, x64::avx, x64::sse41) && jcp.interpolationMode != InterpolationMode::BICUBIC)) {
        vZeros = getVmm();
        uni_vpxor(vZeros, vZeros, vZeros);
    }

    if (one_of(isa, x64::avx512_core, x64::avx2, x64::avx)) {
        if (isa == x64::avx512_core && (one_of(jcp.interpolationMode, InterpolationMode::BICUBIC, InterpolationMode::BILINEAR)) ||
                (jcp.interpolationMode == InterpolationMode::BILINEAR && jcp.paddingMode != PaddingMode::ZEROS)) {
            static const float onesVal = 1.f;
            mov(rAux, reinterpret_cast<uintptr_t>(&onesVal));
            vOnesF = getVmm();
            uni_vpbroadcastd(vOnesF, ptr[rAux]);
        }

        if (isa == x64::avx512_core || jcp.interpolationMode != InterpolationMode::BICUBIC) {
            if (jcp.alignCorners) {
                mov(rAux, ptr[regParams + GET_OFF(wDenormCoefF)]);
                vWDenormCoefF = getVmm();
                uni_vpbroadcastd(vWDenormCoefF, ptr[rAux]);
                if (isa == x64::avx512_core || !(jcp.interpolationMode == InterpolationMode::BILINEAR &&
                                                 jcp.paddingMode == PaddingMode::ZEROS)) {
                    mov(rAux, ptr[regParams + GET_OFF(hDenormCoefF)]);
                    vHDenormCoefF = getVmm();
                    uni_vpbroadcastd(vHDenormCoefF, ptr[rAux]);
                }
            } else {
                static const float halfVal = 0.5f;
                mov(rAux, reinterpret_cast<uintptr_t>(&halfVal));
                vHalfF = getVmm();
                uni_vpbroadcastd(vHalfF, ptr[rAux]);
            }
        }

        if (isa == x64::avx512_core) {
            static const unsigned gridPermMask[16]  = { 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15 };
            mov(rAux, reinterpret_cast<uintptr_t>(gridPermMask));
            vGridPermMask = getVmm();
            uni_vmovups(vGridPermMask, ptr[rAux]);
        } else if (isa == x64::avx2 && !one_of(jcp.interpolationMode, InterpolationMode::BICUBIC, InterpolationMode::BILINEAR)) {// TODO: check reg num for BILINEAR
            static const unsigned gridPermMask[8]  = { 0, 2, 4, 6, 1, 3, 5, 7 };
            mov(rAux, reinterpret_cast<uintptr_t>(gridPermMask));
            vGridPermMask = getVmm();
            uni_vmovups(vGridPermMask, ptr[rAux]);
        }

        Xbyak::Reg32 r32Aux(rAux.getIdx());
        if (isa == x64::avx512_core) {
            regChannelNum = getReg64();
            mov(regChannelNum, ptr[regParams + GET_OFF(channelsNum)]);
            kTailMask = getMask();

            if (jcp.paddingMode == PaddingMode::ZEROS) {
                vDataTypeSize = getVmm();
                mov(rAux, dataTypeSize);
                vpbroadcastd(vDataTypeSize, r32Aux);
                vSrcWidthB = getVmm();
                mov(rAux, ptr[regParams + GET_OFF(srcWidthB)]);
                uni_vpbroadcastd(vSrcWidthB, ptr[rAux]);
            } else if (jcp.paddingMode == PaddingMode::BORDER) {
                vSrcHeightSub1F = getVmm();
                mov(rAux, ptr[regParams + GET_OFF(srcHeightSub1F)]);
                uni_vpbroadcastd(vSrcHeightSub1F, ptr[rAux]);
                vSrcWidthSub1F = getVmm();
                mov(rAux, ptr[regParams + GET_OFF(srcWidthSub1F)]);
                uni_vpbroadcastd(vSrcWidthSub1F, ptr[rAux]);
            } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
                vSrcHeightMul2F = getVmm();
                mov(rAux, ptr[regParams + GET_OFF(srcHeightMul2F)]);
                uni_vpbroadcastd(vSrcHeightMul2F, ptr[rAux]);
                vSrcWidthMul2F = getVmm();
                mov(rAux, ptr[regParams + GET_OFF(srcWidthMul2F)]);
                uni_vpbroadcastd(vSrcWidthMul2F, ptr[rAux]);
                vSrcHeightMul2Sub1F = getVmm();
                mov(rAux, ptr[regParams + GET_OFF(srcHeightMul2Sub1F)]);
                uni_vpbroadcastd(vSrcHeightMul2Sub1F, ptr[rAux]);
                vSrcWidthMul2Sub1F = getVmm();
                mov(rAux, ptr[regParams + GET_OFF(srcWidthMul2Sub1F)]);
                uni_vpbroadcastd(vSrcWidthMul2Sub1F, ptr[rAux]);
                if (jcp.alignCorners) {
                    vAbsMask = getVmm();
                    mov(r32Aux, 0x7fffffff);
                    vpbroadcastd(vAbsMask, r32Aux);
                }
            }

            if (jcp.interpolationMode == InterpolationMode::BICUBIC) {
                vConst_0_75 = getVmm();
                mov(r32Aux, 0xbf400000); // -0.75f
                vpbroadcastd(vConst_0_75, r32Aux);
                vConst_1_25 = getVmm();
                mov(r32Aux, 0x3fa00000); // 1.25f
                vpbroadcastd(vConst_1_25, r32Aux);
                vConst_1_50 = getVmm();
                mov(r32Aux, 0x3fc00000); // 1.5f
                vpbroadcastd(vConst_1_50, r32Aux);
                vConst_2_00 = getVmm();
                mov(r32Aux, 0x40000000); // 2.0f
                vpbroadcastd(vConst_2_00, r32Aux);
                vConst_2_25 = getVmm();
                mov(r32Aux, 0x40100000); // 2.25f
                vpbroadcastd(vConst_2_25, r32Aux);
            }
        }
    } else if (isa == x64::sse41) {
        if (one_of(jcp.interpolationMode, InterpolationMode::BICUBIC, InterpolationMode::BILINEAR)) {
            static const float onesArr[4] = { 1.f, 1.f, 1.f, 1.f };
            mov(rAux, reinterpret_cast<uintptr_t>(onesArr));
            vOnesF = getVmm();
            uni_vmovups(vOnesF, ptr[rAux]);
        }
    }

    rAux.release();

    process();

    registersPool.reset();
    this->postamble();
}

template <x64::cpu_isa_t isa>
void JitGridSampleKernel<isa>::process() {
    regWorkAmount = getReg64();

    // Batch loop
    if (jcp.dynamicShapes) {
        Xbyak::Label lBatchLoop, lEnd;
        auto regBatch = getReg64();
        mov(regBatch, ptr[regParams + GET_OFF(batchNum)]);
        L(lBatchLoop);
        {
            cmp(regBatch, 0);
            jle(lEnd, T_NEAR);

            mov(regWorkAmount, ptr[regParams + GET_OFF(workAmount)]);
            spatialLoop();

            add(regSrc,  ptr[regParams + GET_OFF(srcBatchStepB)]);
            add(regGrid, ptr[regParams + GET_OFF(gridBatchStepB)]);
            add(regDst,  ptr[regParams + GET_OFF(dstBatchStepB)]);

            dec(regBatch);
            jmp(lBatchLoop, T_NEAR);
        }
        L(lEnd);
    } else {
        for (uint64_t i = 0lu; i < jcp.batchNum; i++) {
            mov(regWorkAmount, ptr[regParams + GET_OFF(workAmount)]);
            spatialLoop();

            add(regSrc,  jcp.srcBatchStepB);
            add(regGrid, ptr[regParams + GET_OFF(gridBatchStepB)]);
            add(regDst,  ptr[regParams + GET_OFF(dstBatchStepB)]);
        }
    }
}

template <x64::cpu_isa_t isa>
void JitGridSampleKernel<isa>::spatialLoop() {
    auto vHCoord = getVmm();
    auto vWCoord = getVmm();

    Xbyak::Label lSpacialLoop, lTail;
    L(lSpacialLoop);
    {
        cmp(regWorkAmount, dataElPerVec);
        jl(lTail, T_NEAR);

        getCoordinates(vHCoord, vWCoord);
        denormalizeRawCoordinates(vWCoord, vHCoord);
        interpolation(vWCoord, vHCoord);

        sub(regWorkAmount, dataElPerVec);
        add(regDst, vlen);

        jmp(lSpacialLoop, T_NEAR);
    }

    L(lTail);
    vHCoord.release();
    vWCoord.release();
    tail();
}

template <x64::cpu_isa_t isa>
void JitGridSampleKernel<isa>::tail() {
    Xbyak::Label lEnd;
    cmp(regWorkAmount, 0);
    jle(lEnd, T_NEAR);

    auto vHCoord = getVmm();
    auto vWCoord = getVmm();

    getTailCoordinates(vHCoord, vWCoord);
    denormalizeRawCoordinates(vWCoord, vHCoord);
    interpolation(vWCoord, vHCoord, true);

    vHCoord.release();
    vWCoord.release();

    if (dataTypeSize > 1)
        sal(regWorkAmount, dataTypeShift); // Multiply by source data type size.
    add(regDst, regWorkAmount);

    L(lEnd);
}

template <>
void JitGridSampleKernel<x64::avx512_core>::getCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
    auto vAux = getVmm();

    uni_vpermd(vWCoord, vGridPermMask, ptr[regGrid]); // Permute to XXXX.XXXX.YYYY.YYYY
    Xbyak::Ymm ymmH(vHCoord.getIdx());
    vextractf64x4(ymmH, vWCoord, 1); // Extract Y component

    add(regGrid, vlen);

    uni_vpermd(vAux, vGridPermMask, ptr[regGrid]); // Permute to XXXX.XXXX.YYYY.YYYY
    Xbyak::Ymm ymmAux(vAux.getIdx());
    vinsertf64x4(vWCoord, vWCoord, ymmAux, 1); // Extract X component
    vextractf64x4(ymmAux, vAux, 1);            // Extract Y component
    vinsertf64x4(vHCoord, vHCoord, ymmAux, 1);

    add(regGrid, vlen);
}

template <>
void JitGridSampleKernel<x64::avx2>::getCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
    auto vAux = getVmm();
    Vmm vPermMask;
    RegistersPool::Reg<Vmm> permMaskHolder;

    if (vGridPermMask.isInitialized()) {
        vPermMask = vGridPermMask;
    } else {
        static const unsigned gridPermMask[8] = {0, 2, 4, 6, 1, 3, 5, 7};
        auto rAux = getReg64();
        permMaskHolder = getVmm();
        vPermMask = permMaskHolder;
        mov(rAux, reinterpret_cast<uintptr_t>(gridPermMask));
        uni_vmovups(vPermMask, ptr[rAux]);
    }

    uni_vpermd(vWCoord, vPermMask, ptr[regGrid]); // Permute to XXXX.YYYY
    vperm2f128(vHCoord, vHCoord, vWCoord, 0B00000011);      // Extract Y component

    add(regGrid, vlen);

    uni_vpermd(vAux, vPermMask, ptr[regGrid]);    // Permute to XXXX.YYYY
    vperm2f128(vWCoord, vWCoord, vAux, 0B00100000);         // Extract X component
    vperm2f128(vHCoord, vHCoord, vAux, 0B00110000);         // Extract Y component

    add(regGrid, vlen);
}

template <x64::cpu_isa_t isa> // Works for AVX, SSE41
void JitGridSampleKernel<isa>::getCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
    auto vAux = getVmm();
    Xbyak::Xmm xmmWCoord(vWCoord.getIdx());
    Xbyak::Xmm xmmHCoord(vHCoord.getIdx());
    Xbyak::Xmm xmmAux(vAux.getIdx());
    const uint64_t xmmVlen = x64::cpu_isa_traits<x64::sse41>::vlen;

    uni_vpshufd(xmmWCoord, ptr[regGrid], 0xD8);
    shufpd(xmmHCoord, xmmWCoord, 0x2);

    add(regGrid, xmmVlen);

    uni_vpshufd(xmmAux, ptr[regGrid], 0xD8);
    shufpd(xmmWCoord, xmmAux, 0x0);
    shufpd(xmmHCoord, xmmAux, 0x3);

    add(regGrid, xmmVlen);

    if (isa == x64::avx) {
        Xbyak::Ymm ymmWCoord(vWCoord.getIdx());
        Xbyak::Ymm ymmHCoord(vHCoord.getIdx());

        vperm2f128(ymmWCoord, ymmWCoord, ymmWCoord, 0x1);
        vperm2f128(ymmHCoord, ymmHCoord, ymmHCoord, 0x1);

        // Here is movups + pshufd instead of vpshufd for two reasons:
        // 1. vpshufd zeroes the rest ov YMM.
        // 2. pshufd does not work with not aligned to YMM Address.
        movups(xmmWCoord, ptr[regGrid]);
        pshufd(xmmWCoord, xmmWCoord, 0xD8);
        shufpd(xmmHCoord, xmmWCoord, 0x2);

        add(regGrid, xmmVlen);

        uni_vpshufd(xmmAux, ptr[regGrid], 0xD8);
        shufpd(xmmWCoord, xmmAux, 0x0);
        shufpd(xmmHCoord, xmmAux, 0x3);

        add(regGrid, xmmVlen);

        vperm2f128(ymmWCoord, ymmWCoord, ymmWCoord, 0x1);
        vperm2f128(ymmHCoord, ymmHCoord, ymmHCoord, 0x1);
    }
}

template <>
void JitGridSampleKernel<x64::avx512_core>::getTailCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
    Xbyak::Label lEnd, lGridShift, lRest;

    auto vAux = getVmm();
    auto rAux = getReg64();
    Xbyak::Ymm ymmH(vHCoord.getIdx());

    mov(rAux, regWorkAmount);
    sal(rAux, 0x1); // multiply by gridShape[3]
    cmp(regWorkAmount, dataElPerVec / 2);
    jl(lRest, T_NEAR);
    {
        uni_vpermd(vWCoord, vGridPermMask, ptr[regGrid]);
        vextractf64x4(ymmH, vWCoord, 1); // Extract Y component

        add(regGrid, vlen);
        sub(rAux, dataElPerVec);
        cmp(rAux, 0);
        jle(lEnd, T_NEAR);

        fillRestWorkMask(kTailMask, vAux, rAux);
        uni_vmovups((Vmm)vAux | kTailMask, ptr[regGrid]);
        uni_vpermd(vAux, vGridPermMask, vAux);
        Xbyak::Ymm ymmAux(vAux.getIdx());
        vinsertf64x4(vWCoord, vWCoord, ymmAux, 1); // Extract X component
        vextractf64x4(ymmAux, vAux, 1); // Extract Y component
        vinsertf64x4(vHCoord, vHCoord, ymmAux, 1);

        jmp(lGridShift, T_NEAR);
    }
    L(lRest);
    {
        fillRestWorkMask(kTailMask, vAux, rAux);
        uni_vmovups(vWCoord | kTailMask, ptr[regGrid]);
        uni_vpermd(vWCoord, vGridPermMask, vWCoord);
        vextractf64x4(ymmH, vWCoord, 1); // Extract Y component
    }

    L(lGridShift);
    if (dataTypeSize > 1)
        sal(rAux, dataTypeShift); // multiply by source data type size.
    add(regGrid, rAux);

    L(lEnd);

    fillRestWorkMask(kTailMask, vAux, regWorkAmount);
}

template <>
void JitGridSampleKernel<x64::avx2>::getTailCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
    Xbyak::Label lRest, lGridShift, lEnd;

    auto rAux = getReg64();
    Vmm vPermMask;
    RegistersPool::Reg<Vmm> permMaskHolder;

    if (vGridPermMask.isInitialized()) {
        vPermMask = vGridPermMask;
    } else {
        static const unsigned gridPermMask[8] = {0, 2, 4, 6, 1, 3, 5, 7};
        permMaskHolder = getVmm();
        vPermMask = permMaskHolder;
        mov(rAux, reinterpret_cast<uintptr_t>(gridPermMask));
        uni_vmovups(vPermMask, ptr[rAux]);
    }

    mov(rAux, regWorkAmount);
    sal(rAux, 0x1); // multiply by gridShape[3] == 2
    cmp(regWorkAmount, dataElPerVec / 2);
    jl(lRest, T_NEAR);
    {
        uni_vpermd(vWCoord, vPermMask, ptr[regGrid]);      // Permute to XXXX.YYYY
        vperm2f128(vHCoord, vHCoord, vWCoord, 0B00000011); // Extract Y component

        add(regGrid, vlen);
        sub(rAux, dataElPerVec);
        cmp(rAux, 0);
        jle(lEnd, T_NEAR);

        auto vAux  = getVmm();
        load(vAux, regGrid, rAux, dataTypeSize);
        uni_vpermd(vAux, vPermMask, vAux);
        vperm2f128(vWCoord, vWCoord, vAux, 0B00100000); // Extract X component
        vperm2f128(vHCoord, vHCoord, vAux, 0B00110000); // Extract Y component

        jmp(lGridShift, T_NEAR);
    }
    L(lRest);
    {
        load(vWCoord, regGrid, rAux, dataTypeSize);
        uni_vpermd(vWCoord, vPermMask, vWCoord);           // Permute to XXXX.YYYY
        vperm2f128(vHCoord, vHCoord, vWCoord, 0B00000011); // Extract Y component
    }

    L(lGridShift);
    if (dataTypeSize > 1)
        sal(rAux, dataTypeShift); // Multiply by source data type size.
    add(regGrid, rAux);

    L(lEnd);
}

template <>
void JitGridSampleKernel<x64::avx>::getTailCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
    Xbyak::Label lLoop2End, lEnd;

    Xbyak::Xmm xmmWCoord(vWCoord.getIdx());
    Xbyak::Xmm xmmHCoord(vHCoord.getIdx());

    auto rGridRest = getReg64();
    mov(rGridRest, regWorkAmount);
    sal(rGridRest, 0x1); // multiply by gridShape[3] == 2

    for (int i = 0; i < dataElPerVec; i++) {
        cmp(rGridRest, 0);
        jle(lEnd, T_NEAR);

        if (gridTypeSize == 4)
            pinsrd(i % 2 == 0 ? xmmWCoord : xmmHCoord, ptr[regGrid], i / 2);
        else if (gridTypeSize == 2)
            pinsrw(i % 2 == 0 ? xmmWCoord : xmmHCoord, ptr[regGrid], i / 2);

        add(regGrid, gridTypeSize);
        dec(rGridRest);
    }

    cmp(rGridRest, 0);
    jle(lEnd, T_NEAR);

    vperm2f128(vWCoord, vWCoord, vWCoord, 0x1);
    vperm2f128(vHCoord, vHCoord, vHCoord, 0x1);

    for (int i = 0; i < dataElPerVec; i++) {
        cmp(rGridRest, 0);
        jle(lLoop2End, T_NEAR);

        if (gridTypeSize == 4)
            pinsrd(i % 2 == 0 ? xmmWCoord : xmmHCoord, ptr[regGrid], i / 2);
        else if (gridTypeSize == 2)
            pinsrw(i % 2 == 0 ? xmmWCoord : xmmHCoord, ptr[regGrid], i / 2);

        add(regGrid, gridTypeSize);
        dec(rGridRest);
    }

    L(lLoop2End);
    vperm2f128(vWCoord, vWCoord, vWCoord, 0x1);
    vperm2f128(vHCoord, vHCoord, vHCoord, 0x1);

    L(lEnd);
}

template <>
void JitGridSampleKernel<x64::sse41>::getTailCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
    Xbyak::Label lRest, lHShuf, lGridShift, lEnd;
    auto rAux = getReg64();

    mov(rAux, regWorkAmount);
    sal(rAux, 0x1); // multiply by gridShape[3] == 2
    cmp(regWorkAmount, gridElPerVec / 2);
    jl(lRest, T_NEAR);
    {
        pshufd(vWCoord, ptr[regGrid], 0B11011000);
        shufpd(vHCoord, vWCoord, 0B00000010);

        add(regGrid, vlen);
        sub(rAux, gridElPerVec);
        cmp(rAux, 0);
        jle(lHShuf, T_NEAR);

        auto vAux = getVmm();
        load(vAux, regGrid, rAux, dataTypeSize);
        pshufd(vAux, vAux, 0B11011000);
        shufpd(vWCoord, vAux, 0);          // Extract X component
        shufpd(vHCoord, vAux, 0B00000011); // Extract Y component

        jmp(lGridShift, T_NEAR);
        L(lHShuf);
        shufpd(vHCoord, vHCoord, 0B00000001); // Extract Y component
        jmp(lEnd, T_NEAR);
    }
    L(lRest);
    {
        load(vWCoord, regGrid, rAux, dataTypeSize);
        pshufd(vWCoord, vWCoord, 0B11011000); // Extract X component
        shufpd(vHCoord, vWCoord, 0B00000010); // Extract Y component
    }

    L(lGridShift);
    if (dataTypeSize > 1)
        sal(rAux, dataTypeShift); // multiply by source data type size.
    add(regGrid, rAux);

    L(lEnd);
}

template <>
void JitGridSampleKernel<x64::sse41>::denormalizeRawCoordinates(const Vmm& vWCoord, const Vmm& vHCoord) {
    auto vAux = getVmm();
    auto rAux = getReg64();

    if (jcp.alignCorners) {
        mov(rAux, ptr[regParams + GET_OFF(wDenormCoefF)]);
        uni_vmovups(vAux, ptr[rAux]);
        uni_vfmadd132ps(vWCoord, vAux, vAux);

        mov(rAux, ptr[regParams + GET_OFF(hDenormCoefF)]);
        uni_vmovups(vAux, ptr[rAux]);
        uni_vfmadd132ps(vHCoord, vAux, vAux);
    } else {
        static const float halfValues[4] = { 0.5f, 0.5f, 0.5f, 0.5f };
        mov(rAux, reinterpret_cast<uintptr_t>(halfValues));
        uni_vmovups(vAux, ptr[rAux]);

        uni_vfmadd132ps(vWCoord, vSrcWidthF, vSrcWidthF);
        uni_vfmsub132ps(vWCoord, vAux, vAux);

        uni_vfmadd132ps(vHCoord, vSrcHeightF, vSrcHeightF);
        uni_vfmsub132ps(vHCoord, vAux, vAux);
    }
}

template <x64::cpu_isa_t isa> // Works for AVX512, AVX2, AVX, SSE41
void JitGridSampleKernel<isa>::denormalizeRawCoordinates(const Vmm& vWCoord, const Vmm& vHCoord) {
    if (jcp.alignCorners) {
        if (vWDenormCoefF.isInitialized()) {
            uni_vfmadd132ps(vWCoord, vWDenormCoefF, vWDenormCoefF);
        } else {
            auto rAux = getReg64();
            auto vAux = getVmm();
            mov(rAux, ptr[regParams + GET_OFF(wDenormCoefF)]);
            uni_vmovups(vAux, ptr[rAux]);
            uni_vfmadd132ps(vWCoord, vAux, vAux);
        }

        if (vHDenormCoefF.isInitialized()) {
            uni_vfmadd132ps(vHCoord, vHDenormCoefF, vHDenormCoefF);
        } else {
            auto rAux = getReg64();
            auto vAux = getVmm();
            mov(rAux, ptr[regParams + GET_OFF(hDenormCoefF)]);
            uni_vmovups(vAux, ptr[rAux]);
            uni_vfmadd132ps(vHCoord, vAux, vAux);
        }
    } else {
        Vmm vHalfTmp;
        RegistersPool::Reg<Vmm> halfHolder;
        if (vHalfF.isInitialized()) {
            vHalfTmp = vHalfF;
        } else {
            auto rAux = getReg64();
            halfHolder = getVmm();
            vHalfTmp = halfHolder;
            static const float halfValues[x64::cpu_isa_traits<isa>::vlen / sizeof(float)] = { 0.5f };
            mov(rAux, reinterpret_cast<uintptr_t>(halfValues));
            uni_vmovups(vHalfTmp, ptr[rAux]);
        }

        if (vSrcWidthF.isInitialized()) {
            uni_vfmadd132ps(vWCoord, vSrcWidthF, vSrcWidthF);
        } else {
            auto rAux = getReg64();
            auto vAux = getVmm();
            mov(rAux, ptr[regParams + GET_OFF(srcWidthF)]);
            uni_vpbroadcastd(vAux, ptr[rAux]);
            uni_vfmadd132ps(vWCoord, vAux, vAux);
        }
        uni_vfmsub132ps(vWCoord, vHalfTmp, vHalfTmp);

        if (vSrcHeightF.isInitialized()) {
            uni_vfmadd132ps(vHCoord, vSrcHeightF, vSrcHeightF);
        } else {
            auto rAux = getReg64();
            auto vAux = getVmm();
            mov(rAux, ptr[regParams + GET_OFF(srcHeightF)]);
            uni_vpbroadcastd(vAux, ptr[rAux]);
            uni_vfmadd132ps(vHCoord, vAux, vAux);
        }
        uni_vfmsub132ps(vHCoord, vHalfTmp, vHalfTmp);
    }
}

template <x64::cpu_isa_t isa>
void JitGridSampleKernel<isa>::interpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
    if (jcp.interpolationMode == InterpolationMode::BILINEAR) {
        bilinearInterpolation(vWCoord, vHCoord, tail);
    } else if (jcp.interpolationMode == InterpolationMode::BICUBIC) {
        bicubicInterpolation(vWCoord, vHCoord, tail);
    } else if (jcp.interpolationMode == InterpolationMode::NEAREST) {
        nearestInterpolation(vWCoord, vHCoord, tail);
    }
}

template <>
void JitGridSampleKernel<x64::avx512_core>::zerosPadding0(const Vmask& kDst, const Vmm& vCoord, const Vmm& vUpperBound, const Vmask& kAux) {
    vcmpps(kAux, vCoord, vUpperBound, 0x1);            // vCoord < vUpperBound
    vcmpps(kDst | kAux, vZeros, vCoord, 0x2); // vCoord >= vZeros
}

template <>
void JitGridSampleKernel<x64::avx512_core>::zerosPadding1(const Vmask& kDst, const Vmm& vCoord, const Vmm& vUpperBound, const Vmask& kAux) {
    vcmpps(kDst | kAux, vCoord, vUpperBound, 0x1);     // vCoord < vUpperBound
    vcmpps(kDst | kDst, vZeros, vCoord, 0x2); // vCoord >= vZeros
}

template <>
void JitGridSampleKernel<x64::avx512_core>::zerosPadding(const Vmask& kDst, const Vmm& vHCoord, const Vmm& vWCoord) {
    zerosPadding0(kDst, vWCoord, vSrcWidthF, kDst);
    zerosPadding1(kDst, vHCoord, vSrcHeightF, kDst);
}

template <>
void JitGridSampleKernel<x64::sse41>::zerosPadding(const Vmask& kDst, const Vmm& vHCoord, const Vmm& vWCoord) {
    auto vAux = getVmm();

    uni_vmovups(vAux, vWCoord);
    uni_vcmpps(vAux, vAux, vSrcWidthF, 0x1); // vWCoord < vSrcWidthF
    uni_vpxor(kDst, kDst, kDst);
    uni_vcmpps(kDst, kDst, vWCoord, 0x2);    // vWCoord >= vZeros
    uni_vpand(kDst, kDst, vAux);

    uni_vmovups(vAux, vHCoord);
    uni_vcmpps(vAux, vAux, vSrcHeightF, 0x1); // vHCoord < vSrcHeightF
    uni_vpand(kDst, kDst, vAux);
    uni_vpxor(vAux, vAux, vAux);
    uni_vcmpps(vAux, vAux, vHCoord, 0x2);     // vHCoord >= vZeros
    uni_vpand(kDst, kDst, vAux);
}

template <x64::cpu_isa_t isa> // Works for AVX2, AVX
void JitGridSampleKernel<isa>::zerosPadding(const Vmask& kDst, const Vmm& vHCoord, const Vmm& vWCoord) {
    auto vAux = getVmm();
    Vmm vZerosTmp;
    RegistersPool::Reg<Vmm> zerosHolder;
    if (vZeros.isInitialized()) {
        vZerosTmp = vZeros;
    } else {
        zerosHolder = getVmm();
        vZerosTmp = zerosHolder;
        uni_vpxor(vZerosTmp, vZerosTmp, vZerosTmp);
    }
    if (vSrcWidthF.isInitialized()) {
        uni_vcmpps(vAux, vWCoord, vSrcWidthF, 0x1);  // vWCoord < vSrcWidthF
    } else {
        auto rAux = getReg64();
        mov(rAux, ptr[regParams + GET_OFF(srcWidthF)]);
        uni_vcmpps(vAux, vWCoord, ptr[rAux], 0x1);  // vWCoord < vSrcWidthF
    }

    uni_vcmpps(kDst, vZerosTmp, vWCoord, 0x2);      // vWCoord >= vZeros
    uni_vandps(kDst, kDst, vAux);

    if (vSrcHeightF.isInitialized()) {
        uni_vcmpps(vAux, vHCoord, vSrcHeightF, 0x1); // vHCoord < vSrcHeightF
    } else {
        auto rAux = getReg64();
        mov(rAux, ptr[regParams + GET_OFF(srcHeightF)]);
        uni_vcmpps(vAux, vHCoord, ptr[rAux], 0x1); // vHCoord < vSrcHeightF
    }
    uni_vandps(kDst, kDst, vAux);
    uni_vcmpps(vAux, vZerosTmp, vHCoord, 0x2);      // vHCoord >= vZeros
    uni_vandps(kDst, kDst, vAux);
}

template <>
void JitGridSampleKernel<x64::avx512_core>::borderPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const coord dim) {
    vrangeps(vCoordDst, vCoordOrigin, dim == coord::w ? vSrcWidthSub1F : vSrcHeightSub1F, 0x0); // vWCoord >= vSrcWidthF
    vrangeps(vCoordDst, vCoordDst, vZeros, 0x1); // vWCoord < vZeros
}

template <>
void JitGridSampleKernel<x64::sse41>::borderPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const coord dim) {
    auto rAux  = getReg64();
    if (dim == coord::w) {
        mov(rAux, ptr[regParams + GET_OFF(srcWidthSub1F)]);
    } else if (dim == coord::h) {
        mov(rAux, ptr[regParams + GET_OFF(srcHeightSub1F)]);
    }
    auto vAux = getVmm();

    uni_vmovups(vAux, vCoordOrigin);
    uni_vcmpps(vAux, vAux, ptr[rAux], 0x2); // vCoord < vUpperBound
    if (vCoordDst.getIdx() != vCoordOrigin.getIdx())
        uni_vmovups(vCoordDst, vCoordOrigin);
    uni_vandps(vCoordDst, vCoordDst, vAux);
    uni_vandps(vAux, vAux, ptr[rAux]);
    uni_vaddps(vCoordDst, vCoordDst, vAux);

    uni_vpxor(vAux, vAux, vAux);
    uni_vcmpps(vAux, vAux, vCoordDst, 0x2);    // vCoord >= vZeros
    uni_vandps(vCoordDst, vCoordDst, vAux);
}

template <x64::cpu_isa_t isa> // Works for AVX, AVX2
void JitGridSampleKernel<isa>::borderPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const coord dim) {
    auto rAux = getReg64();
    auto vAux = getVmm();

    if (dim == coord::w) {
        mov(rAux, ptr[regParams + GET_OFF(srcWidthSub1F)]); // TODO: move to vec for NEAREST
    } else if (dim == coord::h) {
        mov(rAux, ptr[regParams + GET_OFF(srcHeightSub1F)]);
    }

    uni_vcmpps(vAux, vCoordOrigin, ptr[rAux], 0x2); // vCoord <= vUpperBound
    uni_vandps(vCoordDst, vCoordOrigin, vAux);
    vandnps(vAux, vAux, ptr[rAux]);
    uni_vaddps(vCoordDst, vCoordDst, vAux);

    if (vZeros.isInitialized()) {
        uni_vcmpps(vAux, vCoordDst, vZeros, 0x6); // vCoord >= vZeros
    } else {
        uni_vpxor(vAux, vAux, vAux);
        uni_vcmpps(vAux, vCoordDst, vAux, 0x6); // vCoord >= vZeros
    }
    uni_vandps(vCoordDst, vCoordDst, vAux);
}

template <>
void JitGridSampleKernel<x64::avx512_core>::reflectionPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const coord dim) {
    auto vAux = getVmm();
    const auto& vSrcDimMul2Sub1F = dim == coord::w ? vSrcWidthMul2Sub1F : vSrcHeightMul2Sub1F;

    if (jcp.alignCorners) {
        // abs(x) % D21
        uni_vandps(vCoordDst, vCoordOrigin, vAbsMask); // abs(x)
        uni_vdivps(vAux, vCoordDst, vSrcDimMul2Sub1F);
        uni_vroundps(vAux, vAux, 0x3);                          // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, vSrcDimMul2Sub1F);    // abs(x) % D21
    } else {
        const auto& vSrcDimMul2F = dim == coord::w ? vSrcWidthMul2F : vSrcHeightMul2F;
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

    auto kAux = getMask();
    uni_vsubps(vAux, vSrcDimMul2Sub1F, vCoordDst);
    vcmpps(kAux, dim == coord::w ? vSrcWidthF : vSrcHeightF, vCoordDst, 0x2); // vCoordDst >= vSrcDimF
    vmovups(vCoordDst | kAux, vAux);
}

template <>
void JitGridSampleKernel<x64::sse41>::reflectionPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const coord dim) {
    auto vAux = getVmm();
    auto rAux = getReg64();

    if (jcp.alignCorners) {
        // abs(x) % D21
        if (vCoordDst.getIdx() != vCoordOrigin.getIdx())
            uni_vmovups(vCoordDst, vCoordOrigin);
        uni_vpsignd(vCoordDst, vCoordDst, vCoordDst); // abs(x)
        if (dim == coord::w) {
            mov(rAux, ptr[regParams + GET_OFF(srcWidthMul2Sub1F)]);
        } else if (coord::h) {
            mov(rAux, ptr[regParams + GET_OFF(srcHeightMul2Sub1F)]);
        }
        uni_vmovups(vAux, vCoordDst);
        uni_vdivps(vAux, vAux, ptr[rAux]);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, ptr[rAux]); // abs(x) % D21
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
        uni_vdivps(vAux, vAux, ptr[rAux]);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, ptr[rAux]); // x % D2
        uni_vaddps(vCoordDst, vCoordDst, ptr[rAux]); // x % D2 + D2
        uni_vmovups(vAux, vCoordDst);
        uni_vdivps(vAux, vAux, ptr[rAux]);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, ptr[rAux]); // (x % D2 + D2) % D2
    }

    if (dim == coord::w) {
        mov(rAux, ptr[regParams + GET_OFF(srcWidthMul2Sub1F)]);
    } else if (coord::h) {
        mov(rAux, ptr[regParams + GET_OFF(srcHeightMul2Sub1F)]);
    }
    uni_vmovups(vAux, ptr[rAux]);
    uni_vsubps(vAux, vAux, vCoordDst);

    auto vAux1 = getVmm();
    uni_vmovups(vAux1, dim == coord::w ? vSrcWidthF : vSrcHeightF);
    uni_vcmpps(vAux1, vAux1, vCoordDst, 0x2); // vCoordDst >= vSrcDimF
    uni_vpand(vCoordDst, vCoordDst, vAux1);
    uni_vpand(vAux1, vAux1, vAux);
    uni_vaddps(vCoordDst, vCoordDst, vAux1);
}

template <x64::cpu_isa_t isa> // Works for AVX2, AVX
void JitGridSampleKernel<isa>::reflectionPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const coord dim) {
    auto vAux = getVmm();
    auto rAux = getReg64();

    if (jcp.alignCorners) {
        // abs(x) % D21
        if (vCoordDst.getIdx() != vCoordOrigin.getIdx())
            uni_vmovups(vCoordDst, vCoordOrigin);
        if (isa == x64::avx2) { // abs(x)
            uni_vpsignd(vCoordDst, vCoordDst, vCoordDst);
        } else {
            static const unsigned absMask[8] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
            mov(rAux, reinterpret_cast<uintptr_t>(absMask));
            uni_vandps(vCoordDst, vCoordDst, ptr[rAux]);
        }
        if (dim == coord::w) {
            mov(rAux, ptr[regParams + GET_OFF(srcWidthMul2Sub1F)]);
        } else if (coord::h) {
            mov(rAux, ptr[regParams + GET_OFF(srcHeightMul2Sub1F)]);
        }
        uni_vdivps(vAux, vCoordDst, ptr[rAux]);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, ptr[rAux]); // abs(x) % D21
    } else {
        // (x % D2 + D2) % D2
        if (vCoordDst.getIdx() != vCoordOrigin.getIdx())
            uni_vmovups(vCoordDst, vCoordOrigin);
        if (dim == coord::w) {
            mov(rAux, ptr[regParams + GET_OFF(srcWidthMul2F)]);
        } else if (coord::h) {
            mov(rAux, ptr[regParams + GET_OFF(srcHeightMul2F)]);
        }
        uni_vdivps(vAux, vCoordOrigin, ptr[rAux]);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, ptr[rAux]); // x % D2
        uni_vaddps(vCoordDst, vCoordDst, ptr[rAux]);  // x % D2 + D2
        uni_vdivps(vAux, vCoordDst, ptr[rAux]);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, ptr[rAux]); // (x % D2 + D2) % D2
    }

    if (dim == coord::w) {
        mov(rAux, ptr[regParams + GET_OFF(srcWidthMul2Sub1F)]);
    } else if (coord::h) {
        mov(rAux, ptr[regParams + GET_OFF(srcHeightMul2Sub1F)]);
    }
    uni_vsubps(vAux, vCoordDst, ptr[rAux]);
    const auto& vUpperBound = dim == coord::w ? vSrcWidthF : vSrcHeightF;

    auto vAux1 = getVmm();
    uni_vcmpps(vAux1, vCoordDst, vUpperBound, 0x1); // vCoordDst < vUpperBound
    uni_vandps(vCoordDst, vCoordDst, vAux1);
    vandnps(vAux, vAux1, vAux);
    uni_vsubps(vCoordDst, vCoordDst, vAux);
}

template <>
void JitGridSampleKernel<x64::avx512_core>::bicubicCoefficients(const Vmm& vCoef, const Vmm& vDDim, const uint8_t idx) {
    if (idx == 0) {
        uni_vmovups(vCoef, vDDim);
        vfnmadd132ps(vCoef, vOnesF, vConst_2_00);
        uni_vfmadd231ps(vCoef, vDDim, vDDim);
        uni_vmulps(vCoef, vCoef, vDDim);
        uni_vmulps(vCoef, vCoef, vConst_0_75);
    } else if (idx == 1) {
        uni_vmovups(vCoef, vDDim);
        vfmsub132ps(vCoef, vConst_2_25, vConst_1_25);
        uni_vmulps(vCoef, vCoef, vDDim);
        uni_vfmadd132ps(vCoef, vOnesF, vDDim);
    } else if (idx == 2) {
        uni_vmovups(vCoef, vDDim);
        vfnmadd132ps(vCoef, vConst_1_50, vConst_1_25);
        uni_vfmsub132ps(vCoef, vConst_0_75, vDDim);
        uni_vmulps(vCoef, vCoef, vDDim);
    } else if (idx == 3) {
        uni_vmulps(vCoef, vConst_0_75, vDDim);
        uni_vmulps(vCoef, vCoef, vDDim);
        vfnmadd132ps(vCoef, vCoef, vDDim);
    }
}

template <>
void JitGridSampleKernel<x64::sse41>::bicubicCoefficients(const Vmm& vCoef, const Vmm& vDDim, const uint8_t idx) {
    auto vAux = getVmm();
    auto rAux = getReg64();
    static const size_t elPerVec = x64::cpu_isa_traits<x64::sse41>::vlen / sizeof(float);
    static const float const_0_75[elPerVec] = { -0.75f, -0.75f, -0.75f, -0.75f };
    static const float const_1_25[elPerVec] = { 1.25f, 1.25f, 1.25f, 1.25f };
    static const float const_1_50[elPerVec] = { 1.5f, 1.5f, 1.5f, 1.5f };
    static const float const_2_00[elPerVec] = { 2.f, 2.f, 2.f, 2.f };
    static const float const_2_25[elPerVec] = { 2.25f, 2.25f, 2.25f, 2.25f };

    if (idx == 0) {
        uni_vmovups(vAux, vDDim);
        mov(rAux, reinterpret_cast<uintptr_t>(const_2_00));
        uni_vmulps(vAux, vAux, ptr[rAux]);
        uni_vsubps(vAux, vAux, vOnesF);
        uni_vmovups(vCoef, vDDim);
        uni_vmulps(vCoef, vCoef, vCoef);
        uni_vsubps(vCoef, vCoef, vAux);
        uni_vmulps(vCoef, vCoef, vDDim);
        mov(rAux, reinterpret_cast<uintptr_t>(const_0_75));
        uni_vmulps(vCoef, vCoef, ptr[rAux]);
    } else if (idx == 1) {
        uni_vmovups(vCoef, vDDim);
        mov(rAux, reinterpret_cast<uintptr_t>(const_1_25));
        uni_vmulps(vCoef, vCoef, ptr[rAux]);
        mov(rAux, reinterpret_cast<uintptr_t>(const_2_25));
        uni_vsubps(vCoef, vCoef, ptr[rAux]);
        uni_vmulps(vCoef, vCoef, vDDim);
        uni_vfmadd132ps(vCoef, vOnesF, vDDim);
    } else if (idx == 2) {
        uni_vmovups(vAux, vDDim);
        uni_vmulps(vAux, vDDim, vDDim);
        mov(rAux, reinterpret_cast<uintptr_t>(const_1_25));
        uni_vmulps(vAux, vAux, ptr[rAux]);
        mov(rAux, reinterpret_cast<uintptr_t>(const_0_75));
        uni_vaddps(vAux, vAux, ptr[rAux]);
        uni_vmovups(vCoef, vDDim);
        mov(rAux, reinterpret_cast<uintptr_t>(const_1_50));
        uni_vmulps(vCoef, vCoef, ptr[rAux]);
        uni_vsubps(vCoef, vCoef, vAux);
        uni_vmulps(vCoef, vCoef, vDDim);
    } else if (idx == 3) {
        uni_vmovups(vCoef, vDDim);
        mov(rAux, reinterpret_cast<uintptr_t>(const_0_75));
        uni_vmulps(vCoef, vCoef, ptr[rAux]);
        uni_vmulps(vCoef, vCoef, vDDim);
        uni_vmovups(vAux, vCoef);
        uni_vmulps(vAux, vAux, vDDim);
        uni_vsubps(vCoef, vCoef, vAux);
    }
}

template <x64::cpu_isa_t isa> // Works for AVX2, AVX
void JitGridSampleKernel<isa>::bicubicCoefficients(const Vmm& vCoef, const Vmm& vDDim, const uint8_t idx) {
    auto vAux = getVmm();
    auto rAux = getReg64();
    static const size_t elPerVec = x64::cpu_isa_traits<isa>::vlen / sizeof(float);
    static const float const_0_75[elPerVec] = { -0.75f };
    static const float const_1_25[elPerVec] = { 1.25f };
    static const float const_1_50[elPerVec] = { 1.5f };
    static const float const_2_00[elPerVec] = { 2.f };
    static const float const_2_25[elPerVec] = { 2.25f };

    if (idx == 0) {
        uni_vmovups(vCoef, vDDim);
        vfnmadd132ps(vCoef, vOnesF, vConst_2_00);
        uni_vfmadd231ps(vCoef, vDDim, vDDim);
        uni_vmulps(vCoef, vCoef, vDDim);
        uni_vmulps(vCoef, vCoef, vConst_0_75);
    } else if (idx == 1) {
        uni_vmovups(vCoef, vDDim);
        vfmsub132ps(vCoef, vConst_2_25, vConst_1_25);
        uni_vmulps(vCoef, vCoef, vDDim);
        uni_vfmadd132ps(vCoef, vOnesF, vDDim);
    } else if (idx == 2) {
        uni_vmulps(vCoef, vDDim, vDDim);
        vfmadd132ps(vCoef, vConst_0_75, vConst_1_25);
        uni_vfmsub231ps(vCoef, vDDim, vConst_1_50);
        uni_vmulps(vCoef, vCoef, vDDim);
    } else if (idx == 3) {
        uni_vmulps(vCoef, vConst_0_75, vDDim);
        uni_vmulps(vCoef, vCoef, vDDim);
        vfnmadd132ps(vCoef, vCoef, vDDim);
    }
}

template <x64::cpu_isa_t isa> // Works for AVX512, AVX2, AVX, SSE41
void JitGridSampleKernel<isa>::nearestInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
    const auto& vSrcShift = vWCoord;
    const auto& vAux      = vHCoord;
    auto kGatherMask      = getMask();
    auto kAuxMask         = getMask();

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
        reflectionPadding(vWCoord, vWCoord, coord::w);
        reflectionPadding(vHCoord, vHCoord, coord::h);
    }

    hwShiftPs2dq(vSrcShift, vHCoord, vWCoord, vSrcWidthF);

    // PER CHANNEL LOOP
    Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
    auto rChannel = getReg64();
    auto rSrcTmp  = getReg64();
    auto rDstTmp  = getReg64();
    mov(rChannel, ptr[regParams + GET_OFF(channelsNum)]);
    mov(rSrcTmp, regSrc);
    mov(rDstTmp, regDst);
    L(lChannelLoopBegin);
    {
        cmp(rChannel, 0);
        jle(lChannelLoopEnd, T_NEAR);

        if (jcp.paddingMode == PaddingMode::ZEROS) {
            if (isa == x64::avx512_core && tail)
                uni_kandd(kAuxMask, kTailMask, kGatherMask);
            else
                uni_kmovd(kAuxMask, kGatherMask);
        }

        if (!tail) {
            gatherdd(vAux, rSrcTmp, vSrcShift, kAuxMask, useMask, zeroFill);
            uni_vmovups(ptr[rDstTmp], vAux);
        } else {
            if (isa == x64::avx512_core) {
                if (jcp.paddingMode != PaddingMode::ZEROS) {
                    uni_kmovd(kAuxMask, kTailMask);
                }
                gatherdd(vAux, rSrcTmp, vSrcShift, kAuxMask, tail, zeroFill);
                uni_vmovups(ptr[rDstTmp] | Xbyak::Opmask(kTailMask.getIdx()), vAux);
            } else {
                memMovDD(rDstTmp, rSrcTmp, Vmm(kAuxMask.getIdx()), vSrcShift, regWorkAmount, useMask, zeroFill);
            }
        }

        add(rSrcTmp, regSrcChannelStepB);
        add(rDstTmp, regDstChannelStepB);
        dec(rChannel);

        jmp(lChannelLoopBegin, T_NEAR);
        L(lChannelLoopEnd);
    }
}

template <>
void JitGridSampleKernel<x64::avx512_core>::bilinearInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
    auto vDX     = getVmm();
    auto vDY     = getVmm();
    const auto& shift00 = vWCoord;
    const auto& shift01 = vHCoord;
    auto shift10 = getVmm();
    auto shift11 = getVmm();
    auto vAux    = getVmm();
    RegistersPool::Reg<Vmask> kMask00, kMask01, kMask10, kMask11;

    uni_vmovups(vDX, vWCoord);
    uni_vmovups(vDY, vHCoord);
    uni_vroundps(vWCoord, vWCoord, 0x1); // Round floor
    uni_vroundps(vHCoord, vHCoord, 0x1); // Round floor
    uni_vsubps(vDX, vDX, vWCoord);
    uni_vsubps(vDY, vDY, vHCoord);
    uni_vaddps(shift10, vWCoord, vOnesF);
    uni_vaddps(shift11, vHCoord, vOnesF);

    bool useMask = false, zeroFill = false;
    if (jcp.paddingMode == PaddingMode::ZEROS) {
        useMask = zeroFill = true;
        kMask00 = getMask();
        kMask01 = getMask();
        kMask10 = getMask();
        kMask11 = getMask();

        zerosPadding(kMask00, vHCoord, vWCoord); // (y; x)
        zerosPadding(kMask01, vHCoord, shift10); // (y; x + 1)
        zerosPadding(kMask11, shift11, shift10); // (y + 1; x + 1)
        zerosPadding(kMask10, shift11, vWCoord); // (y + 1; x)

        hwShiftPs2dq(shift00, vHCoord, vWCoord, vSrcWidthF);
        uni_vpaddd(shift01, shift00, vDataTypeSize);
        uni_vpaddd(shift10, shift00, vSrcWidthB); // shift11??
        uni_vpaddd(shift11, shift10, vDataTypeSize); // sub??
    } else if (jcp.paddingMode == PaddingMode::BORDER) {
        borderPadding(vWCoord, vWCoord, coord::w);
        borderPadding(vHCoord, vHCoord, coord::h);
        borderPadding(shift10, shift10, coord::w);
        borderPadding(shift11, shift11, coord::h);
    } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
        reflectionPadding(vWCoord, vWCoord, coord::w);
        reflectionPadding(vHCoord, vHCoord, coord::h);
        reflectionPadding(shift10, shift10, coord::w);
        reflectionPadding(shift11, shift11, coord::h);
    }
    if (jcp.paddingMode == PaddingMode::BORDER || jcp.paddingMode == PaddingMode::REFLECTION) {
        // W * y + x
        hwShiftPs2dq(vAux, shift11, vWCoord, vSrcWidthF);
        hwShiftPs2dq(vWCoord, vHCoord, vWCoord, vSrcWidthF);
        hwShiftPs2dq(vHCoord, vHCoord, shift10, vSrcWidthF);
        hwShiftPs2dq(shift11, shift11, shift10, vSrcWidthF);
        uni_vmovups(shift10, vAux);
    }

    auto kAuxMask = getMask();
    auto vQ0 = getVmm();
    auto vQ1 = getVmm();

    // PER CHANNEL LOOP
    Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
    auto rChannel = getReg64();
    auto rSrcTmp  = getReg64();
    auto rDstTmp  = getReg64();
    mov(rChannel, 0);
    mov(rSrcTmp, regSrc);
    mov(rDstTmp, regDst);
    L(lChannelLoopBegin);
    {
        cmp(rChannel, regChannelNum);
        jge(lChannelLoopEnd, T_NEAR);

        // (y; x)
        if (jcp.paddingMode == PaddingMode::ZEROS) {
            kmovw(kAuxMask, kMask00);
        }
        gatherdd(vQ0, rSrcTmp, shift00, kAuxMask, useMask, zeroFill); // v00 -> vQ0
        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtdq2ps(vQ0, vQ0);
        }
        uni_vfmsub213ps(vQ0, vDX, vQ0); // q0 = -(v00 - dx * v00)

        // (y; x + 1)
        if (jcp.paddingMode == PaddingMode::ZEROS) {
            kmovw(kAuxMask, kMask01);
        }
        gatherdd(vAux, rSrcTmp, shift01, kAuxMask, useMask, zeroFill);
        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtdq2ps(vAux, vAux);
        }
        uni_vfmsub231ps(vQ0, vAux, vDX); // q0 = -q0 + dx * v01

        // (y + 1; x + 1)
        if (jcp.paddingMode == PaddingMode::ZEROS) {
            kmovw(kAuxMask, kMask11);
        }
        gatherdd(vAux, rSrcTmp, shift11, kAuxMask, useMask, zeroFill);
        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtdq2ps(vAux, vAux);
        }

        // (y + 1; x)
        if (jcp.paddingMode == PaddingMode::ZEROS) {
            kmovw(kAuxMask, kMask10);
        }
        gatherdd(vQ1, rSrcTmp, shift10, kAuxMask, useMask, zeroFill);
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
            uni_vmovups(ptr[rDstTmp], vQ1);
        } else {
            uni_vmovups(ptr[rDstTmp] | kTailMask, vQ1);
        }
        add(rSrcTmp, regSrcChannelStepB);
        add(rDstTmp, regDstChannelStepB);
        add(rChannel, 1);

        jmp(lChannelLoopBegin, T_NEAR);
        L(lChannelLoopEnd);
    }
}

template <>
void JitGridSampleKernel<x64::sse41>::bilinearInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
    const auto& vDX    = getVmm();
    const auto& vDY    = getVmm();
    const auto& vGatherShift = getVmm();
    const auto& vAux   = getVmm();
    const auto& vQ0    = getVmm();
    const auto& vQ1    = getVmm();
    const auto& kMask0 = getMask();
    const auto& kMask1 = getMask();

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
    auto rChannel = getReg64();
    auto rSrcTmp  = getReg64();
    auto rDstTmp  = getReg64();
    mov(rChannel, 0);
    mov(rSrcTmp, regSrc);
    mov(rDstTmp, regDst);
    L(lChannelLoopBegin);
    {
        cmp(rChannel, regChannelNum);
        jge(lChannelLoopEnd, T_NEAR);

        if (jcp.paddingMode == PaddingMode::ZEROS) {
            // (x; y)
            zerosPadding(kMask0, vHCoord, vWCoord);
            uni_vmovups(vGatherShift, vHCoord);
            hwShiftPs2dq(vGatherShift, vGatherShift, vWCoord, vSrcWidthF);
            uni_vpxor(vQ0, vQ0, vQ0);
            gatherdd(vQ0, rSrcTmp, vGatherShift, kMask0); // v00 -> vQ0
            if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                uni_vcvtdq2ps(vQ0, vQ0);
            }
            uni_vfmsub213ps(vQ0, vDX, vQ0); // q0 = -(v00 - dx * v00)

            // (x + 1; y)
            uni_vaddps(vWCoord, vWCoord, vOnesF);
            zerosPadding(kMask0, vHCoord, vWCoord);
            uni_vmovups(vGatherShift, vHCoord);
            hwShiftPs2dq(vGatherShift, vGatherShift, vWCoord, vSrcWidthF);
            uni_vpxor(vAux, vAux, vAux);
            gatherdd(vAux, rSrcTmp, vGatherShift, kMask0);

            // q0 = -q0 + dx * v01
            if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                uni_vcvtdq2ps(vAux, vAux);
            }
            uni_vfmsub231ps(vQ0, vAux, vDX);

            // (x + 1; y + 1)
            uni_vaddps(vHCoord, vHCoord, vOnesF);
            zerosPadding(kMask0, vHCoord, vWCoord);
            uni_vmovups(vGatherShift, vHCoord);
            hwShiftPs2dq(vGatherShift, vGatherShift, vWCoord, vSrcWidthF);
            uni_vpxor(vAux, vAux, vAux);
            gatherdd(vAux, rSrcTmp, vGatherShift, kMask0);

            // (x; y + 1)
            uni_vsubps(vWCoord, vWCoord, vOnesF);
            zerosPadding(kMask0, vHCoord, vWCoord);
            uni_vmovups(vGatherShift, vHCoord);
            hwShiftPs2dq(vGatherShift, vGatherShift, vWCoord, vSrcWidthF);
            uni_vpxor(vQ1, vQ1, vQ1);
            gatherdd(vQ1, rSrcTmp, vGatherShift, kMask0);
            uni_vsubps(vHCoord, vHCoord, vOnesF);
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

        uni_vmovups(ptr[rDstTmp], vQ1);
        add(rSrcTmp, regSrcChannelStepB);
        add(rDstTmp, regDstChannelStepB);
        add(rChannel, 1);

        jmp(lChannelLoopBegin, T_NEAR);
        L(lChannelLoopEnd);
    }
}

template <x64::cpu_isa_t isa> // Works for AVX2, AVX
void JitGridSampleKernel<isa>::bilinearInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
    if (!one_of(isa, x64::avx2, x64::avx)) {
        IE_THROW() << "This instance of the bilinearInterpolation supports only AVX2 and AVX instruction sets.";
    }

    auto vWRound = getVmm();
    auto vHRound = getVmm();
    auto& vDX    = vWCoord;
    auto& vDY    = vHCoord;
    auto vAux    = getVmm();
    Vmm shift00, shift01, shift10, shift11;
    RegistersPool::Reg<Vmm> shift10Holder, shift11Holder;
    // For ZEROS padding only.
    RegistersPool::Reg<Vmm> vMask00, vMask01, vMask10, vMask11;

    uni_vroundps(vWRound, vWCoord, 0x1); // Round floor
    uni_vroundps(vHRound, vHCoord, 0x1); // Round floor
    uni_vsubps(vDX, vDX, vWRound);
    uni_vsubps(vDY, vDY, vHRound);

    if (jcp.paddingMode != PaddingMode::ZEROS) {
        shift00 = vWRound;
        shift01 = vHRound;
        shift10Holder = getVmm();
        shift10 = shift10Holder;
        shift11Holder = getVmm();
        shift11 = shift11Holder;

        uni_vaddps(shift10, vWRound, vOnesF);
        uni_vaddps(shift11, vHRound, vOnesF);
    }

    bool useMask = false, zeroFill = false;
    if (jcp.paddingMode == PaddingMode::ZEROS) {
        useMask = zeroFill = true;
        {
            auto rAux = getReg64();
            static const float onesVal = 1.f;
            mov(rAux, reinterpret_cast<uintptr_t>(&onesVal));
            uni_vpbroadcastd(vAux, ptr[rAux]);
        }
        shift00 = vWRound;
        shift10 = vHRound;
        vMask00 = getVmm();
        vMask01 = getVmm();
        vMask10 = getVmm();
        vMask11 = getVmm();

        uni_vaddps(vMask00, vWRound, vAux);
        uni_vaddps(vAux, vHRound, vAux);

        zerosPadding(vMask01, vHRound, vMask00); // (y; x + 1)
        zerosPadding(vMask10, vAux, vWRound);    // (y + 1; x)
        zerosPadding(vMask11, vAux, vMask00);    // (y + 1; x + 1)
        zerosPadding(vMask00, vHRound, vWRound); // (y; x)

        hwShiftPs2dq(shift00, vHRound, vWRound, vSrcWidthF);
    } else if (jcp.paddingMode == PaddingMode::BORDER) {
        borderPadding(vWRound, vWRound, coord::w);
        borderPadding(vHRound, vHRound, coord::h);
        borderPadding(shift10, shift10, coord::w);
        borderPadding(shift11, shift11, coord::h);
    } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
        reflectionPadding(vWRound, vWRound, coord::w);
        reflectionPadding(vHRound, vHRound, coord::h);
        reflectionPadding(shift10, shift10, coord::w);
        reflectionPadding(shift11, shift11, coord::h);
    }
    if (one_of(jcp.paddingMode, PaddingMode::BORDER, PaddingMode::REFLECTION)) {
        // W * y + x
        hwShiftPs2dq(vAux, shift11, vWRound, vSrcWidthF);
        hwShiftPs2dq(vWRound, vHRound, vWRound, vSrcWidthF);
        hwShiftPs2dq(vHRound, vHRound, shift10, vSrcWidthF);
        hwShiftPs2dq(shift11, shift11, shift10, vSrcWidthF);
        uni_vmovups(shift10, vAux);
    }

    auto vGatherMask = getVmm();
    auto vQ0         = getVmm();
    auto vQ1         = getVmm();

    // PER CHANNEL LOOP
    Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
    auto rChannel  = getReg64();
    auto rSrcTmp   = getReg64();
    auto rDstTmp   = getReg64();
    auto rTypeSize = getReg64();
    mov(rChannel,  ptr[regParams + GET_OFF(channelsNum)]);
    mov(rSrcTmp,   regSrc);
    mov(rDstTmp,   regDst);
    mov(rTypeSize, ptr[regParams + GET_OFF(dataTypeSize)]);

    L(lChannelLoopBegin);
    {
        cmp(rChannel, 0);
        jle(lChannelLoopEnd, T_NEAR);

        // (y; x)
        if (jcp.paddingMode == PaddingMode::ZEROS && isa == x64::avx2) {
            uni_vmovups(vGatherMask, vMask00);
        }
        gatherdd(vQ0, rSrcTmp, shift00, (isa == x64::avx2 || !vMask00.isInitialized()) ? vGatherMask : vMask00, useMask, zeroFill); // v00 -> vQ0
        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtdq2ps(vQ0, vQ0);
        }
        if (isa == x64::avx2) {
            uni_vfmsub213ps(vQ0, vDX, vQ0); // q0 = -(v00 - dx * v00)
        } else {
            uni_vmulps(vGatherMask, vQ0, vDX);
            uni_vsubps(vQ0, vGatherMask, vQ0);
        }

        // (y; x + 1)
        if (jcp.paddingMode == PaddingMode::ZEROS) {
            uni_vpaddd(shift10, shift00, ptr[rTypeSize]);
            if (isa == x64::avx2)
                uni_vmovups(vGatherMask, vMask01);
        }
        gatherdd(vAux, rSrcTmp, jcp.paddingMode != PaddingMode::ZEROS ? shift01 : shift10,
                 (isa == x64::avx2 || !vMask01.isInitialized()) ? vGatherMask : vMask01, useMask, zeroFill);
        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtdq2ps(vAux, vAux);
        }
        uni_vfmsub231ps(vQ0, vAux, vDX); // q0 = -q0 + dx * v01

        // (y + 1; x + 1)
        if (jcp.paddingMode == PaddingMode::ZEROS) {
            {
                auto rSrcWidth = getReg64();
                mov(rSrcWidth, ptr[regParams + GET_OFF(srcWidthB)]);
                uni_vpaddd(shift10, shift10, ptr[rSrcWidth]);
            }
            if (isa == x64::avx2)
                uni_vmovups(vGatherMask, vMask11);
        }
        gatherdd(vAux, rSrcTmp, jcp.paddingMode != PaddingMode::ZEROS ? shift11 : shift10,
                 (isa == x64::avx2 || !vMask11.isInitialized()) ? vGatherMask : vMask11, useMask, zeroFill);
        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtdq2ps(vAux, vAux);
        }

        // (y + 1; x)
        if (jcp.paddingMode == PaddingMode::ZEROS) {
            uni_vpsubd(shift10, shift10, ptr[rTypeSize]);
            if (isa == x64::avx2)
                uni_vmovups(vGatherMask, vMask10);
        }
        gatherdd(vQ1, rSrcTmp, shift10, (isa == x64::avx2 || !vMask10.isInitialized()) ? vGatherMask : vMask10, useMask, zeroFill);
        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtdq2ps(vQ1, vQ1);
        }

        // q1 = -(v10 - dx * v10)
        if (isa == x64::avx2) {
            uni_vfmsub213ps(vQ1, vDX, vQ1);
        } else {
            uni_vmulps(vGatherMask, vQ1, vDX);
            uni_vsubps(vQ1, vGatherMask, vQ1);
        }
        uni_vfmsub231ps(vQ1, vAux, vDX); // q1 = -q1 + dx * v11
        // Res = q0 + dy * (q1 - q0)
        uni_vsubps(vQ1, vQ1, vQ0);
        uni_vfmadd132ps(vQ1, vQ0, vDY);

        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtps2dq(vQ1, vQ1);
        }

        if (!tail) {
            uni_vmovups(ptr[rDstTmp], vQ1);
        } else {
            store(rDstTmp, regWorkAmount, vQ1, dataTypeSize);
        }

        add(rSrcTmp, regSrcChannelStepB);
        add(rDstTmp, regDstChannelStepB);
        dec(rChannel);

        jmp(lChannelLoopBegin, T_NEAR);
        L(lChannelLoopEnd);
    }
}

template <>
void JitGridSampleKernel<x64::avx512_core>::bicubicInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
    auto vHTop      = getVmm();
    auto vWLeft     = getVmm();
    auto vDX        = getVmm();
    auto vDY        = getVmm();
    auto vXDotProd  = getVmm();
    auto& vYDotProd = vDX;
    auto vSrcShift0 = getVmm();
    auto vSrcShift  = getVmm();
    auto vAux       = getVmm();
    auto kAuxMask   = getMask();
    RegistersPool::Reg<Vmask> kMaskH;
    std::vector<RegistersPool::Reg<Vmask>> wMasks;

    uni_vroundps(vHTop, vHCoord, 0x1);  // Round floor
    uni_vroundps(vWLeft, vWCoord, 0x1); // Round floor
    uni_vsubps(vDY, vHCoord, vHTop);
    uni_vsubps(vDX, vWCoord, vWLeft);
    uni_vsubps(vHTop, vHTop, vOnesF);
    uni_vsubps(vWLeft, vWLeft, vOnesF);

    RegistersPool::Reg<Vmm> vCX[4] = {getVmm(), getVmm(), getVmm(), getVmm() };
    for (int i = 0; i < 4; i++) {
        bicubicCoefficients(vCX[i], vDX, i);
    }

    bool useMask = false, zeroFill = false;
    if (jcp.paddingMode == PaddingMode::ZEROS) {
        useMask = zeroFill = true;
        wMasks.resize(4);
        for (auto& mask : wMasks) {
            mask = getMask();
        }
        zerosPadding0(wMasks[0], vWLeft, vSrcWidthF, kAuxMask);
        uni_vaddps(vWCoord, vWLeft, vOnesF);
        zerosPadding0(wMasks[1], vWCoord, vSrcWidthF, kAuxMask);
        uni_vaddps(vWCoord, vWCoord, vOnesF);
        zerosPadding0(wMasks[2], vWCoord, vSrcWidthF, kAuxMask);
        uni_vaddps(vWCoord, vWCoord, vOnesF);
        zerosPadding0(wMasks[3], vWCoord, vSrcWidthF, kAuxMask);
        kMaskH = getMask();
    }

    // PER CHANNEL LOOP
    Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
    auto rChannel = getReg64();
    auto rSrcTmp  = getReg64();
    auto rDstTmp  = getReg64();
    mov(rChannel, 0);
    mov(rSrcTmp, regSrc);
    mov(rDstTmp, regDst);
    L(lChannelLoopBegin);
    {
        cmp(rChannel, regChannelNum);
        jge(lChannelLoopEnd, T_NEAR);

        uni_vmovups(vHCoord, vHTop);
        uni_vpxor(vYDotProd, vYDotProd, vYDotProd);
        for (int h = 0; h < 4; h++) {
            // (y - 1 + h; x - 1)
            if (jcp.paddingMode == PaddingMode::ZEROS) {
                zerosPadding0(kMaskH, vHCoord, vSrcHeightF, kMaskH);
                kandw(kAuxMask, kMaskH, wMasks[0]);
                uni_vmulps(vSrcShift0, vHCoord, vSrcWidthF);
                uni_vmovups(vWCoord, vWLeft);
                uni_vaddps(vSrcShift, vSrcShift0, vWCoord);
            } else if (jcp.paddingMode == PaddingMode::BORDER) {
                borderPadding(vSrcShift0, vHCoord, coord::h);
                uni_vmulps(vSrcShift0, vSrcShift0, vSrcWidthF);
                uni_vmovups(vWCoord, vWLeft);
                borderPadding(vSrcShift, vWCoord, coord::w);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
            } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
                reflectionPadding(vSrcShift0, vHCoord, coord::h);
                uni_vmulps(vSrcShift0, vSrcShift0, vSrcWidthF);
                uni_vmovups(vWCoord, vWLeft);
                reflectionPadding(vSrcShift, vWCoord, coord::w);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
            }
            uni_vcvtps2dq(vSrcShift, vSrcShift);
            if (dataTypeSize > 1)
                uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
            gatherdd(vAux, rSrcTmp, vSrcShift, kAuxMask, useMask, zeroFill);
            if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                uni_vcvtdq2ps(vAux, vAux);
            }
            uni_vmulps(vXDotProd, vAux, vCX[0]);

            // (y - 1 + h; x)
            // (y - 1 + h; x + 1)
            // (y - 1 + h; x + 2)
            for (int w = 1; w < 4; w++) {
                uni_vaddps(vWCoord, vWCoord, vOnesF);
                if (jcp.paddingMode == PaddingMode::ZEROS) {
                    uni_vaddps(vSrcShift, vSrcShift0, vWCoord);
                    kandw(kAuxMask, kMaskH, wMasks[w]);
                } else if (jcp.paddingMode == PaddingMode::BORDER) {
                    borderPadding(vSrcShift, vWCoord, coord::w);
                    uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
                    reflectionPadding(vSrcShift, vWCoord, coord::w);
                    uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                }
                uni_vcvtps2dq(vSrcShift, vSrcShift);
                if (dataTypeSize > 1)
                    uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
                gatherdd(vAux, rSrcTmp, vSrcShift, kAuxMask, useMask, zeroFill);
                if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                    uni_vcvtdq2ps(vAux, vAux);
                }
                uni_vfmadd231ps(vXDotProd, vAux, vCX[w]);
            }

            if (h != 3) {
                uni_vaddps(vHCoord, vHCoord, vOnesF);
            }

            bicubicCoefficients(vAux, vDY, h);
            uni_vfmadd231ps(vYDotProd, vXDotProd, vAux);
        }

        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtps2dq(vYDotProd, vYDotProd);
        }

        if (!tail) {
            uni_vmovups(ptr[rDstTmp], vYDotProd);
        } else {
            uni_vmovups(ptr[rDstTmp] | kTailMask, vYDotProd);
        }
        add(rSrcTmp, regSrcChannelStepB);
        add(rDstTmp, regDstChannelStepB);
        add(rChannel, 1);

        jmp(lChannelLoopBegin, T_NEAR);
        L(lChannelLoopEnd);
    }
}

template <>
void JitGridSampleKernel<x64::sse41>::bicubicInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
    const auto& vHTop      = getVmm();
    const auto& vWLeft     = getVmm();
    const auto& vDX        = getVmm();
    const auto& vDY        = getVmm();
    const auto& vXDotProd  = getVmm();
    const auto& vYDotProd  = vDX;
    const auto& vSrcShift0 = getVmm();
    const auto& vSrcShift  = getVmm();
    const auto& vCX0       = getVmm();
    const auto& vCX1       = getVmm();
    const auto& vCX2       = getVmm();
    const auto& vCX3       = getVmm();
    const auto& vAux       = getVmm(); // &vWLeft
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
    uni_vsubps(vWLeft, vWLeft, vOnesF);
    uni_vsubps(vHTop, vHTop, vOnesF);

    bicubicCoefficients(vCX0, vDX, 0); // TODO: for
    bicubicCoefficients(vCX1, vDX, 1);
    bicubicCoefficients(vCX2, vDX, 2);
    bicubicCoefficients(vCX3, vDX, 3);

    if (jcp.paddingMode == PaddingMode::ZEROS) {
        hwShiftPs2dq(vSrcShift0, vHTop, vWLeft, vSrcWidthF);
    }

    // PER CHANNEL LOOP
    Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
    auto rChannel = getReg64();
    auto rSrcTmp = getReg64();
    auto rDstTmp = getReg64();
    mov(rChannel, 0);
    mov(rSrcTmp, regSrc);
    mov(rDstTmp, regDst);
    L(lChannelLoopBegin);
    {
        cmp(rChannel, regChannelNum);
        jge(lChannelLoopEnd, T_NEAR);

        if (jcp.paddingMode == PaddingMode::ZEROS) {
            uni_vmovups(vSrcShift, vSrcShift0);
        }
        uni_vmovups(vHCoord, vHTop);
        uni_vpxor(vYDotProd, vYDotProd, vYDotProd);
        for (int i = 0; i < 4; i++) {
            // (y - 1 + i; x - 1)
            if (jcp.paddingMode == PaddingMode::ZEROS) {
                zerosPadding(kMaskH, vSrcHeightF, vHCoord);
                uni_vpxor(vAux, vAux, vAux);
            } else if (jcp.paddingMode == PaddingMode::BORDER) {
                borderPadding(vSrcShift0, vHCoord, coord::h);
                uni_vmulps(vSrcShift0, vSrcShift0, vSrcWidthF);
                uni_vmovups(vWCoord, vWLeft);
                borderPadding(vSrcShift, vWCoord, coord::w);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                uni_vcvtps2dq(vSrcShift, vSrcShift);
                if (dataTypeSize > 1)
                    uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
            } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
                reflectionPadding(vSrcShift0, vHCoord, coord::h);
                uni_vmulps(vSrcShift0, vSrcShift0, vSrcWidthF);
                uni_vmovups(vWCoord, vWLeft);
                reflectionPadding(vSrcShift, vWCoord, coord::w);
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
//                uni_vpaddd(vSrcShift, vSrcShift, vDataTypeSize);
            } else if (jcp.paddingMode == PaddingMode::BORDER) {
                uni_vaddps(vWCoord, vWCoord, vOnesF);
                borderPadding(vSrcShift, vWCoord, coord::w);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                uni_vcvtps2dq(vSrcShift, vSrcShift);
                if (dataTypeSize > 1)
                    uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
            } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
                uni_vaddps(vWCoord, vWCoord, vOnesF);
                reflectionPadding(vSrcShift, vWCoord, coord::w);
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
//                uni_vpaddd(vSrcShift, vSrcShift, vDataTypeSize);
            } else if (jcp.paddingMode == PaddingMode::BORDER) {
                uni_vaddps(vWCoord, vWCoord, vOnesF);
                borderPadding(vSrcShift, vWCoord, coord::w);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                uni_vcvtps2dq(vSrcShift, vSrcShift);
                if (dataTypeSize > 1)
                    uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
            } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
                uni_vaddps(vWCoord, vWCoord, vOnesF);
                reflectionPadding(vSrcShift, vWCoord, coord::w);
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
//                uni_vpaddd(vSrcShift, vSrcShift, vDataTypeSize);
            } else if (jcp.paddingMode == PaddingMode::BORDER) {
                uni_vaddps(vWCoord, vWCoord, vOnesF);
                borderPadding(vSrcShift, vWCoord, coord::w);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                uni_vcvtps2dq(vSrcShift, vSrcShift);
                if (dataTypeSize > 1)
                    uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
            } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
                uni_vaddps(vWCoord, vWCoord, vOnesF);
                reflectionPadding(vSrcShift, vWCoord, coord::w);
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
                uni_vaddps(vHCoord, vHCoord, vOnesF);
                if (jcp.paddingMode == PaddingMode::ZEROS) {
//                    uni_vpaddd(vSrcShift, vSrcShift, vSrcWidthB);
                }
            }

            bicubicCoefficients(vAux, vDY, i);
            uni_vfmadd231ps(vYDotProd, vXDotProd, vAux);
        }

        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtps2dq(vYDotProd, vYDotProd);
        }

        uni_vmovups(ptr[rDstTmp], vYDotProd);
        add(rSrcTmp, regSrcChannelStepB);
        add(rDstTmp, regDstChannelStepB);
        add(rChannel, 1);

        jmp(lChannelLoopBegin, T_NEAR);
        L(lChannelLoopEnd);
    }
}

template <x64::cpu_isa_t isa> // Works for AVX2 and AVX
void JitGridSampleKernel<isa>::bicubicInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
    auto vHTop      = getVmm();
    auto vWLeft     = getVmm();
    auto vDX        = getVmm();
    auto vDY        = getVmm();
    auto vXDotProd  = getVmm();
    auto& vYDotProd = vDX;
    auto vSrcShift0 = getVmm();
    auto vSrcShift  = getVmm();
    auto vAux       = getVmm();
    auto kMask0     = getVmm();
    auto kMask1     = getVmm();
    auto kMask2     = getVmm();
    auto kMask3     = getVmm();
    auto kAuxMask   = getVmm();
    auto kMaskH     = getVmm();

    uni_vroundps(vHTop,  vHCoord, 0x1); // Round floor
    uni_vroundps(vWLeft, vWCoord, 0x1); // Round floor
    uni_vsubps(vDY, vHCoord, vHTop);
    uni_vsubps(vDX, vWCoord, vWLeft);
    uni_vsubps(vHTop, vHTop, vOnesF);
    uni_vsubps(vWLeft, vWLeft, vOnesF);

    RegistersPool::Reg<Vmm> vCX[4] = {getVmm(), getVmm(), getVmm(), getVmm() };
    for (int i = 0; i < 4; i++) {
        bicubicCoefficients(vCX[i], vDX, i);
    }

    if (jcp.paddingMode == PaddingMode::ZEROS) {
//        zerosPadding0(kMask0, vWLeft, vSrcWidthF, kAuxMask);
//        uni_vaddps(vWCoord, vWLeft, vOnesF);
//        zerosPadding0(kMask1, vWCoord, vSrcWidthF, kAuxMask);
//        uni_vaddps(vWCoord, vWCoord, vOnesF);
//        zerosPadding0(kMask2, vWCoord, vSrcWidthF, kAuxMask);
//        uni_vaddps(vWCoord, vWCoord, vOnesF);
//        zerosPadding0(kMask3, vWCoord, vSrcWidthF, kAuxMask);
    }

    // PER CHANNEL LOOP
    Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
    auto rChannel = getReg64();
    auto rSrcTmp  = getReg64();
    auto rDstTmp  = getReg64();
    mov(rChannel, ptr[regParams + GET_OFF(channelsNum)]);
    mov(rSrcTmp, regSrc);
    mov(rDstTmp, regDst);
    L(lChannelLoopBegin);
    {
        cmp(rChannel, 0);
        jge(lChannelLoopEnd, T_NEAR);

        uni_vmovups(vHCoord, vHTop);
        uni_vpxor(vYDotProd, vYDotProd, vYDotProd);
        for (int h = 0; h < 4; h++) {
            // (y - 1 + h; x - 1)
            if (jcp.paddingMode == PaddingMode::ZEROS) {
//                zerosPadding0(kMaskH, vHCoord, vSrcHeightF, kMaskH);
//                kandw(kAuxMask, kMaskH, kMask0);
                uni_vmulps(vSrcShift0, vHCoord, vSrcWidthF);
                uni_vmovups(vWCoord, vWLeft);
                uni_vaddps(vSrcShift, vSrcShift0, vWCoord);
                uni_vpxor(vAux, vAux, vAux);
            } else if (jcp.paddingMode == PaddingMode::BORDER) {
                borderPadding(vSrcShift0, vHCoord, coord::h);
                uni_vmulps(vSrcShift0, vSrcShift0, vSrcWidthF);
                uni_vmovups(vWCoord, vWLeft);
                borderPadding(vSrcShift, vWCoord, coord::w);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
//                kxnorw(kAuxMask, kAuxMask, kAuxMask);
            } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
                reflectionPadding(vSrcShift0, vHCoord, coord::h);
                uni_vmulps(vSrcShift0, vSrcShift0, vSrcWidthF);
                uni_vmovups(vWCoord, vWLeft);
                reflectionPadding(vSrcShift, vWCoord, coord::w);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
//                kxnorw(kAuxMask, kAuxMask, kAuxMask);
            }
            uni_vcvtps2dq(vSrcShift, vSrcShift);
            if (dataTypeSize > 1)
                uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
            gatherdd(vAux, rSrcTmp, vSrcShift, kAuxMask);
            if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                uni_vcvtdq2ps(vAux, vAux);
            }
            uni_vmulps(vXDotProd, vAux, vCX[0]);

            // (y - 1 + h; x)
            // (y - 1 + h; x + 1)
            // (y - 1 + h; x + 2)
            for (int w = 1; w < 4; w++) {
                uni_vaddps(vWCoord, vWCoord, vOnesF);
                if (jcp.paddingMode == PaddingMode::ZEROS) {
                    uni_vaddps(vSrcShift, vSrcShift0, vWCoord);
//                    kandw(kAuxMask, kMaskH, (&kMask0)[w]);
                    uni_vpxor(vAux, vAux, vAux);
                } else if (jcp.paddingMode == PaddingMode::BORDER) {
                    borderPadding(vSrcShift, vWCoord, coord::w);
                    uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
//                    kxnorw(kAuxMask, kAuxMask, kAuxMask);
                } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
                    reflectionPadding(vSrcShift, vWCoord, coord::w);
                    uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
//                    kxnorw(kAuxMask, kAuxMask, kAuxMask);
                }
                uni_vcvtps2dq(vSrcShift, vSrcShift);
                if (dataTypeSize > 1)
                    uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
                gatherdd(vAux, rSrcTmp, vSrcShift, kAuxMask);
                if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                    uni_vcvtdq2ps(vAux, vAux);
                }
                uni_vfmadd231ps(vXDotProd, vAux, vCX[w]);
            }

            if (h != 3) {
                uni_vaddps(vHCoord, vHCoord, vOnesF);
            }

            bicubicCoefficients(vAux, vDY, h);
            uni_vfmadd231ps(vYDotProd, vXDotProd, vAux);
        }

        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtps2dq(vYDotProd, vYDotProd);
        }

        if (tail) {
//            uni_vmovups(ptr[rDstTmp] | kTailMask, vYDotProd);
        } else {
            uni_vmovups(ptr[rDstTmp], vYDotProd);
        }
        add(rSrcTmp, regSrcChannelStepB);
        add(rDstTmp, regDstChannelStepB);
        dec(rChannel);

        jmp(lChannelLoopBegin, T_NEAR);
        L(lChannelLoopEnd);
    }
}

template <x64::cpu_isa_t isa>
void JitGridSampleKernel<isa>::hwShiftPs2dq(const Vmm& vDst, const Vmm& vHCoord,const Vmm& vWCoord, const Vmm& vWidth) {
    if (vDst.getIdx() == vWCoord.getIdx()) {
        if (one_of(isa, x64::avx512_core, x64::avx2)) {
            uni_vfmadd231ps(vDst, vHCoord, vWidth);
        } else {
            auto vTmp = getVmm();
            uni_vmulps(vTmp, vHCoord, vWidth);
            uni_vaddps(vDst, vWCoord, vTmp);
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
            auto rAux = getReg64();
            const float val = dataTypeSize;
            static const float dataTypeSizeArr[8] = {val, val, val, val, val, val, val, val};
            mov(rAux, reinterpret_cast<uintptr_t>(dataTypeSizeArr));
            uni_vmulps(vDst, vDst, ptr[rAux]);
        }
        uni_vcvtps2dq(vDst, vDst);
    } else {
        uni_vcvtps2dq(vDst, vDst);
        if (dataTypeSize > 1)
            uni_vpslld(vDst, vDst, dataTypeShift); // multiply by source data type size.
    }
}

template struct JitGridSampleKernel<x64::avx512_core>;
template struct JitGridSampleKernel<x64::avx2>;
template struct JitGridSampleKernel<x64::avx>;
template struct JitGridSampleKernel<x64::sse41>;

}   // namespace intel_cpu
}   // namespace ov
