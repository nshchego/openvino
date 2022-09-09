// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "grid_sample_kernel.hpp"
#include "utils/general_utils.h"

using namespace dnnl::impl::cpu;

namespace ov {
namespace intel_cpu {

#define GET_OFF(field) offsetof(jGridSamplesExecArgs, field)
#define vmmRef(idx) vRefWrap<Vmm>(this, vPool[idx])
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
    vecNum = isa == dnnl::impl::cpu::x64::avx512_core ? 32 : isa == dnnl::impl::cpu::x64::sse41 ? 8 : 16;
    for (int i = 0; i < vecNum; i++) {
        vPool.push_back(Vmm(i));
        vecSet.insert(i);
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

    mov(regSrc,  ptr[regParams + GET_OFF(src)]);
    mov(regDst,  ptr[regParams + GET_OFF(dst)]);
    mov(regGrid, ptr[regParams + GET_OFF(grid)]);

    mov(regAux1, ptr[regParams + GET_OFF(srcWidthF)]);
    uni_vpbroadcastd(vPool[getVecIdx(srcWidthFIdx)], ptr[regAux1]);
    mov(regAux1, ptr[regParams + GET_OFF(srcHeightF)]);
    uni_vpbroadcastd(vPool[getVecIdx(srcHeightFIdx)], ptr[regAux1]);

    mov(regSrcChannelStepB, ptr[regParams + GET_OFF(srcChannelStepB)]);
    mov(regDstChannelStepB, ptr[regParams + GET_OFF(dstChannelStepB)]);

    if (one_of(jcp.paddingMode, PaddingMode::ZEROS, PaddingMode::BORDER)) {
        zerosIdx = getVecIdx();
        uni_vpxor(vPool[zerosIdx], vPool[zerosIdx], vPool[zerosIdx]);
    }

    if (one_of(isa, x64::avx512_core, x64::avx2, x64::avx)) {
        mov(regChannelsNum, ptr[regParams + GET_OFF(channelsNum)]);

        if (one_of(jcp.interpolationMode, InterpolationMode::BICUBIC, InterpolationMode::BILINEAR)) {
            static const float onesVal = 1.f;
            mov(regAux1, reinterpret_cast<uintptr_t>(&onesVal));
            uni_vpbroadcastd(vPool[getVecIdx(onesFIdx)], ptr[regAux1]);
        }

        if (jcp.alignCorners) {
            mov(regAux1, ptr[regParams + GET_OFF(wDenormCoefF)]);
            uni_vpbroadcastd(vPool[getVecIdx(wDenormCoefFIdx)], ptr[regAux1]);
            mov(regAux1, ptr[regParams + GET_OFF(hDenormCoefF)]);
            uni_vpbroadcastd(vPool[getVecIdx(hDenormCoefFIdx)], ptr[regAux1]);
        } else {
            static const float halfVal = 0.5f;
            mov(regAux1, reinterpret_cast<uintptr_t>(&halfVal));
            uni_vpbroadcastd(vPool[getVecIdx(halfFIdx)], ptr[regAux1]);
        }

        if (isa == x64::avx512_core) {
            static const unsigned gridPermMask[16]  = { 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15 };
            mov(regAux1, reinterpret_cast<uintptr_t>(gridPermMask));
            uni_vmovups(vPool[getVecIdx(gridPermMaskIdx)], ptr[regAux1]);
        } else if (isa == x64::avx2) {
            static const unsigned gridPermMask[8]  = { 0, 2, 4, 6, 1, 3, 5, 7 };
            mov(regAux1, reinterpret_cast<uintptr_t>(gridPermMask));
            uni_vmovups(vPool[getVecIdx(gridPermMaskIdx)], ptr[regAux1]);
        }

        if (isa == x64::avx512_core) {
            if (jcp.paddingMode == PaddingMode::ZEROS) {
                mov(regAux1, dataTypeSize);
                vpbroadcastd(vPool[getVecIdx(dataTypeSizeIdx)], reg32Aux1);
                mov(regAux1, ptr[regParams + GET_OFF(srcWidthB)]);
                uni_vpbroadcastd(vPool[getVecIdx(srcWidthBIdx)], ptr[regAux1]);
            } else if (jcp.paddingMode == PaddingMode::BORDER) {
                mov(regAux1, ptr[regParams + GET_OFF(srcHeightSub1F)]);
                uni_vpbroadcastd(vPool[getVecIdx(srcHeightSub1FIdx)], ptr[regAux1]);
                mov(regAux1, ptr[regParams + GET_OFF(srcWidthSub1F)]);
                uni_vpbroadcastd(vPool[getVecIdx(srcWidthSub1FIdx)], ptr[regAux1]);
            } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
                mov(regAux1, ptr[regParams + GET_OFF(srcHeightMul2F)]);
                uni_vpbroadcastd(vPool[getVecIdx(srcHeightMul2FIdx)], ptr[regAux1]);
                mov(regAux1, ptr[regParams + GET_OFF(srcWidthMul2F)]);
                uni_vpbroadcastd(vPool[getVecIdx(srcWidthMul2FIdx)], ptr[regAux1]);
                mov(regAux1, ptr[regParams + GET_OFF(srcHeightMul2Sub1F)]);
                uni_vpbroadcastd(vPool[getVecIdx(srcHeightMul2Sub1FIdx)], ptr[regAux1]);
                mov(regAux1, ptr[regParams + GET_OFF(srcWidthMul2Sub1F)]);
                uni_vpbroadcastd(vPool[getVecIdx(srcWidthMul2Sub1FIdx)], ptr[regAux1]);
                if (jcp.alignCorners) {
                    mov(reg32Aux1, 0x7fffffff);
                    vpbroadcastd(vPool[getVecIdx(absMaskIdx)], reg32Aux1);
                }
            }

            if (jcp.interpolationMode == InterpolationMode::BICUBIC) {
                mov(reg32Aux1, 0xbf400000); // -0.75f
                vpbroadcastd(vPool[getVecIdx(const_0_75_idx)], reg32Aux1);
                mov(reg32Aux1, 0x3fa00000); // 1.25f
                vpbroadcastd(vPool[getVecIdx(const_1_25_idx)], reg32Aux1);
                mov(reg32Aux1, 0x3fc00000); // 1.5f
                vpbroadcastd(vPool[getVecIdx(const_1_50_idx)], reg32Aux1);
                mov(reg32Aux1, 0x40000000); // 2.0f
                vpbroadcastd(vPool[getVecIdx(const_2_00_idx)], reg32Aux1);
                mov(reg32Aux1, 0x40100000); // 2.25f
                vpbroadcastd(vPool[getVecIdx(const_2_25_idx)], reg32Aux1);
            }
        }
    } else if (isa == x64::sse41) {
        if (one_of(jcp.interpolationMode, InterpolationMode::BICUBIC, InterpolationMode::BILINEAR)) {
            static const float onesArr[4] = { 1.f, 1.f, 1.f, 1.f };
            mov(regAux1, reinterpret_cast<uintptr_t>(onesArr));
            uni_vmovups(vPool[getVecIdx(onesFIdx)], ptr[regAux1]);
        }
    }

    process();

    this->postamble();
}

template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::process() {
    if (jcp.dynamicShapes) {
        Xbyak::Label lBatchLoop, lEnd;
        mov(regBatch, ptr[regParams + GET_OFF(batchNum)]);
        L(lBatchLoop);
        {
;            cmp(regBatch, 0);
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
void jitGridSampleKernel<isa>::spatialLoop() {
    const Vmm& vHCoord = vPool[getVecIdx()];
    const Vmm& vWCoord = vPool[getVecIdx()];

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

    releaseVecIdx(vHCoord.getIdx());
    releaseVecIdx(vWCoord.getIdx());

    L(lTail);
    tail();
}

template <>
void jitGridSampleKernel<x64::avx512_core>::getCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
    auto vAux = vmmRef();

    uni_vpermd(vWCoord, vPool[gridPermMaskIdx], ptr[regGrid]); // Permute to XXXX.XXXX.YYYY.YYYY
    Xbyak::Ymm ymmH = Xbyak::Ymm(vHCoord.getIdx());
    vextractf64x4(ymmH, vWCoord, 1); // Extract Y component

    add(regGrid, vlen);

    uni_vpermd(vAux, vPool[gridPermMaskIdx], ptr[regGrid]); // Permute to XXXX.XXXX.YYYY.YYYY
    Xbyak::Ymm ymmAux0 = Xbyak::Ymm(vAux.getIdx());
    vinsertf64x4(vWCoord, vWCoord, ymmAux0, 1); // Extract X component
    vextractf64x4(ymmAux0, vAux, 1);               // Extract Y component
    vinsertf64x4(vHCoord, vHCoord, ymmAux0, 1);

    add(regGrid, vlen);
}

template <>
void jitGridSampleKernel<x64::avx2>::getCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
    auto vAux = vmmRef();
    uni_vpermd(vWCoord, vPool[gridPermMaskIdx], ptr[regGrid]); // Permute to XXXX.YYYY
    Xbyak::Xmm xmmH = Xbyak::Xmm(vHCoord.getIdx());
    vextracti128(xmmH, vWCoord, 1);

    add(regGrid, vlen);

    uni_vpermd(vAux, vPool[gridPermMaskIdx], ptr[regGrid]); // Permute to XXXX.YYYY
    Xbyak::Xmm xmmAux0 = Xbyak::Xmm(vAux.getIdx());
    vinserti128(vWCoord, vWCoord, xmmAux0, 1); // Extract X component
    vextracti128(xmmAux0, vAux, 1);               // Extract Y component
    vinserti128(vHCoord, vHCoord, xmmAux0, 1);

    add(regGrid, vlen);
}

template <x64::cpu_isa_t isa> // Works for AVX, SSE41
void jitGridSampleKernel<isa>::getCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
    auto vAux = vmmRef();
    Xbyak::Xmm xmmWCoord = Xbyak::Xmm(vWCoord.getIdx());
    Xbyak::Xmm xmmHCoord = Xbyak::Xmm(vHCoord.getIdx());
    Xbyak::Xmm xmmAux    = Xbyak::Xmm(vAux.getIdx());
    const uint64_t xmmVlen = x64::cpu_isa_traits<x64::sse41>::vlen;

    uni_vpshufd(xmmWCoord, ptr[regGrid], 0xD8);
    shufpd(xmmHCoord, xmmWCoord, 0x2);

    add(regGrid, xmmVlen);

    uni_vpshufd(xmmAux, ptr[regGrid], 0xD8);
    shufpd(xmmWCoord, xmmAux, 0x0);
    shufpd(xmmHCoord, xmmAux, 0x3);

    add(regGrid, xmmVlen);

    if (isa == x64::avx) {
        Xbyak::Ymm ymmWCoord = Xbyak::Ymm(vWCoord.getIdx());
        Xbyak::Ymm ymmHCoord = Xbyak::Ymm(vHCoord.getIdx());

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
void jitGridSampleKernel<x64::avx512_core>::getTailCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
    Xbyak::Label lEnd, lGridShift, lRest;

    auto vAux = vmmRef();
    Xbyak::Ymm ymmH = Xbyak::Ymm(vHCoord.getIdx());

    mov(regAux3, regWorkAmount);
    sal(regAux3, 0x1); // multiply by gridShape[3]
    cmp(regWorkAmount, dataElPerVec / 2);
    jl(lRest, T_NEAR);
    {
        uni_vpermd(vWCoord, vPool[gridPermMaskIdx], ptr[regGrid]);
        vextractf64x4(ymmH, vWCoord, 1); // Extract Y component

        add(regGrid, vlen);
        sub(regAux3, dataElPerVec);
        cmp(regAux3, 0);
        jle(lEnd, T_NEAR);

        fillRestWorkMask(kTailMask, vAux, regAux3, regAux1, regAux2);
        uni_vmovups((Vmm)vAux | kTailMask, ptr[regGrid]);
        uni_vpermd(vAux, vPool[gridPermMaskIdx], vAux);
        Xbyak::Ymm ymmAux0 = Xbyak::Ymm(vAux.getIdx());
        vinsertf64x4(vWCoord, vWCoord, ymmAux0, 1); // Extract X component
        vextractf64x4(ymmAux0, vAux, 1); // Extract Y component
        vinsertf64x4(vHCoord, vHCoord, ymmAux0, 1);

        jmp(lGridShift, T_NEAR);
    }
    L(lRest);
    {
        fillRestWorkMask(kTailMask, vAux, regAux3, regAux1, regAux2);
        uni_vmovups(vWCoord | kTailMask, ptr[regGrid]);
        uni_vpermd(vWCoord, vPool[gridPermMaskIdx], vWCoord);
        vextractf64x4(ymmH, vWCoord, 1); // Extract Y component
    }

    L(lGridShift);
    if (dataTypeSize > 1)
        sal(regAux3, dataTypeShift); // multiply by source data type size.
    add(regGrid, regAux3);

    L(lEnd);

    fillRestWorkMask(kTailMask, vAux, regWorkAmount, regAux1, regAux2);
}

template <>
void jitGridSampleKernel<x64::avx2>::getTailCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
    Xbyak::Label lRest, lGridShift, lEnd;

    auto vAux  = vmmRef();
    auto vAux1 = vmmRef();
    Xbyak::Xmm xmmH = Xbyak::Xmm(vHCoord.getIdx());

    mov(regAux3, regWorkAmount);
    sal(regAux3, 0x1); // multiply by gridShape[3] == 2
    cmp(regWorkAmount, dataElPerVec / 2);
    jl(lRest, T_NEAR);
    {
        uni_vpermd(vWCoord, vPool[gridPermMaskIdx], ptr[regGrid]); // Permute to XXXX.YYYY
        vextractf128(xmmH, vWCoord, 1); // Extract Y component

        add(regGrid, vlen);
        sub(regAux3, dataElPerVec);
        cmp(regAux3, 0);
        jle(lEnd, T_NEAR);

        loadEl2vec32(vAux, regGrid, vAux1, regAux3, regAux2);
        vpermilps(vAux, vAux, 0xD8);
        Xbyak::Xmm xmmAux = Xbyak::Xmm(vAux.getIdx());
        vinsertf128(vWCoord, vWCoord, xmmAux, 1); // Extract X component
        vextractf128(xmmAux, vAux, 1); // Extract Y component
        vinsertf128(vHCoord, vHCoord, xmmAux, 1);

        jmp(lGridShift, T_NEAR);
    }
    L(lRest);
    {
        loadEl2vec32(vWCoord, regGrid, vAux1, regAux3, regAux2);
        uni_vpermd(vWCoord, vPool[gridPermMaskIdx], vWCoord); // Permute to XXXX.YYYY
        vextractf128(xmmH, vWCoord, 1); // Extract Y component
    }

    L(lGridShift);
    if (dataTypeSize > 1)
        sal(regAux3, dataTypeShift); // multiply by source data type size.
    add(regGrid, regAux3);

    L(lEnd);
}

template <>
void jitGridSampleKernel<x64::avx>::getTailCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
    Xbyak::Label lLoop2End, lEnd;

    Xbyak::Xmm xmmWCoord = Xbyak::Xmm(vWCoord.getIdx());
    Xbyak::Xmm xmmHCoord = Xbyak::Xmm(vHCoord.getIdx());

    const auto& rGridRest = regAux3;
    mov(rGridRest, regWorkAmount);
    sal(rGridRest, 0x1); // multiply by gridShape[3] == 2

    for (int i = 0; i < dataElPerVec; i++) {
        cmp(rGridRest, 0);
        jle(lEnd, T_NEAR);

        if (i % 2 == 0)
//            uni_vpinsrd(xmmWCoord, xmmWCoord, ptr[regGrid], i / 2);
            pinsrd(xmmWCoord, ptr[regGrid], i / 2);
        else
//            uni_vpinsrd(xmmHCoord, xmmHCoord, ptr[regGrid], i / 2);
            pinsrd(xmmHCoord, ptr[regGrid], i / 2);

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

        if (i % 2 == 0)
//            uni_vpinsrd(xmmWCoord, xmmWCoord, ptr[regGrid], i / 2);
            pinsrd(xmmWCoord, ptr[regGrid], i / 2);
        else
//            uni_vpinsrd(xmmHCoord, xmmHCoord, ptr[regGrid], i / 2);
            pinsrd(xmmHCoord, ptr[regGrid], i / 2);

        add(regGrid, gridTypeSize);
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

    mov(regAux3, regWorkAmount);
    sal(regAux3, 0x1); // multiply by gridShape[3] == 2
    cmp(regWorkAmount, gridElPerVec / 2);
    jl(lRest, T_NEAR);
    {
        pshufd(vWCoord, ptr[regGrid], 0xD8);
        shufpd(vHCoord, vWCoord, 0x2);

        add(regGrid, vlen);
        sub(regAux3, gridElPerVec);
        cmp(regAux3, 0);
        jle(lEnd, T_NEAR);

        loadEl2vec32(vAux, regGrid, regAux3, regAux2);
        pshufd(vAux, vAux, 0xD8);
        shufpd(vWCoord, vAux, 0x0); // Extract X component
        shufpd(vHCoord, vAux, 0x3); // Extract Y component

        jmp(lGridShift, T_NEAR);
    }
    L(lRest);
    {
        loadEl2vec32(vWCoord, regGrid, regAux3, regAux2);
        pshufd(vWCoord, vWCoord, 0xD8);  // Extract X component
        shufpd(vHCoord, vWCoord, 0x2);   // Extract Y component
    }

    L(lGridShift);
    if (dataTypeSize > 1)
        sal(regAux3, dataTypeShift); // multiply by source data type size.
    add(regGrid, regAux3);

    L(lEnd);
}

template <>
void jitGridSampleKernel<x64::sse41>::denormalizeRawCoordinates(const Vmm& vWCoord, const Vmm& vHCoord) {
    auto vAux = vmmRef();

    if (jcp.alignCorners) {
        mov(regAux4, ptr[regParams + GET_OFF(wDenormCoefF)]);
        uni_vmovups(vAux, ptr[regAux4]);
        uni_vfmadd132ps(vWCoord, vAux, vAux);

        mov(regAux4, ptr[regParams + GET_OFF(hDenormCoefF)]);
        uni_vmovups(vAux, ptr[regAux4]);
        uni_vfmadd132ps(vHCoord, vAux, vAux);
    } else {
        static const float halfValues[4] = { 0.5f, 0.5f, 0.5f, 0.5f };
        mov(regAux4, reinterpret_cast<uintptr_t>(halfValues));
        uni_vmovups(vAux, ptr[regAux4]);

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
    vcmpps(kAux, vCoord, vUpperBound, 0x1);   // vCoord < vUpperBound
    vcmpps(kDst | kAux, vPool[zerosIdx], vCoord, 0x2); // vCoord >= vZeros
}

template <>
void jitGridSampleKernel<x64::avx512_core>::zerosPadding1(const Vmask& kDst, const Vmm& vCoord, const Vmm& vUpperBound, const Vmask& kAux) {
    vcmpps(kDst | kAux, vCoord, vUpperBound, 0x1); // vCoord < vUpperBound
    vcmpps(kDst | kDst, vPool[zerosIdx], vCoord, 0x2);      // vCoord >= vZeros
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
    if (dim == coord::w) {
        vrangeps(vCoordDst, vCoordOrigin, vPool[srcWidthSub1FIdx], 0x0);   // vWCoord >= vSrcWidthF
    } else if (dim == coord::h) {
        vrangeps(vCoordDst, vCoordOrigin, vPool[srcHeightSub1FIdx], 0x0);  // vWCoord >= vSrcWidthF
    }
    vrangeps(vCoordDst, vCoordDst, vPool[zerosIdx], 0x1); // vWCoord < vZeros
}

template <>
void jitGridSampleKernel<x64::sse41>::borderPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const coord dim) {
    if (dim == coord::w) {
        mov(regAux4, ptr[regParams + GET_OFF(srcWidthSub1F)]);
    } else if (dim == coord::h) {
        mov(regAux4, ptr[regParams + GET_OFF(srcHeightSub1F)]);
    }
    auto vAux = vmmRef(getVecIdx());

    uni_vmovups(vAux, vCoordOrigin);
    uni_vcmpps(vAux, vAux, ptr[regAux4], 0x2); // vCoord < vUpperBound
    if (vCoordDst.getIdx() != vCoordOrigin.getIdx())
        uni_vmovups(vCoordDst, vCoordOrigin);
    uni_vandps(vCoordDst, vCoordDst, vAux);
    uni_vandps(vAux, vAux, ptr[regAux4]);
    uni_vaddps(vCoordDst, vCoordDst, vAux);

    uni_vpxor(vAux, vAux, vAux);
    uni_vcmpps(vAux, vAux, vCoordDst, 0x2);    // vCoord >= vZeros
    uni_vandps(vCoordDst, vCoordDst, vAux);
}

template <x64::cpu_isa_t isa> // Works for AVX, AVX2
void jitGridSampleKernel<isa>::borderPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const coord dim) {
    if (dim == coord::w) {
        mov(regAux4, ptr[regParams + GET_OFF(srcWidthSub1F)]);
    } else if (dim == coord::h) {
        mov(regAux4, ptr[regParams + GET_OFF(srcHeightSub1F)]);
    }
    auto vAux = vmmRef();

    uni_vcmpps(vAux, vCoordOrigin, ptr[regAux4], 0x2); // vCoord <= vUpperBound
    uni_vandps(vCoordDst, vCoordOrigin, vAux);
    vandnps(vAux, vAux, ptr[regAux4]);
    uni_vaddps(vCoordDst, vCoordDst, vAux);

    uni_vcmpps(vAux, vPool[zerosIdx], vCoordDst, 0x2); // vCoord >= vZeros
    uni_vandps(vCoordDst, vCoordDst, vAux);
}

template <>
void jitGridSampleKernel<x64::avx512_core>::reflectionPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const Vmm& vAux, const Vmask& kAux, const coord dim) {
    Vmm vSrcDimF, vSrcDimMul2F, vSrcDimMul2Sub1F;
    if (dim == coord::w) {
        vSrcDimF = vPool[srcWidthFIdx];
        vSrcDimMul2F = vPool[srcWidthMul2FIdx];
        vSrcDimMul2Sub1F = vPool[srcWidthMul2Sub1FIdx];
    } else if (coord::h) {
        vSrcDimF = vPool[srcHeightFIdx];
        vSrcDimMul2F = vPool[srcHeightMul2FIdx];
        vSrcDimMul2Sub1F = vPool[srcHeightMul2Sub1FIdx];
    }

    if (jcp.alignCorners) {
        // abs(x) % D21
        uni_vandps(vCoordDst, vCoordOrigin, vPool[absMaskIdx]); // abs(x)
        uni_vdivps(vAux, vCoordDst, vSrcDimMul2Sub1F);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, vSrcDimMul2Sub1F); // abs(x) % D21
    } else {
        // (x % D2 + D2) % D2
        if (vCoordDst.getIdx() != vCoordOrigin.getIdx())
            uni_vmovups(vCoordDst, vCoordOrigin);
        uni_vdivps(vAux, vCoordDst, vSrcDimMul2F);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, vSrcDimMul2F); // x % D2
        uni_vaddps(vCoordDst, vCoordDst, vSrcDimMul2F); // x % D2 + D2
        uni_vdivps(vAux, vCoordDst, vSrcDimMul2F);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, vSrcDimMul2F); // (x % D2 + D2) % D2
    }

    uni_vsubps(vAux, vSrcDimMul2Sub1F, vCoordDst);
    vcmpps(kAux, vSrcDimF, vCoordDst, 0x2); // vCoordDst >= vSrcDimF
    vmovups(vCoordDst | kAux, vAux);
}

template <>
void jitGridSampleKernel<x64::sse41>::reflectionPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const Vmm& vAux, const Vmask& kAux, const coord dim) {
    if (jcp.alignCorners) {
        // abs(x) % D21
        if (vCoordDst.getIdx() != vCoordOrigin.getIdx())
            uni_vmovups(vCoordDst, vCoordOrigin);
        static const unsigned absMask[4] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
        mov(regAux4, reinterpret_cast<uintptr_t>(absMask)); // TODO: use PSIGND
        uni_vandps(vCoordDst, vCoordDst, ptr[regAux4]); // abs(x)
        if (dim == coord::w) {
            mov(regAux4, ptr[regParams + GET_OFF(srcWidthMul2Sub1F)]);
        } else if (coord::h) {
            mov(regAux4, ptr[regParams + GET_OFF(srcHeightMul2Sub1F)]);
        }
        uni_vmovups(vAux, vCoordDst);
        uni_vdivps(vAux, vAux, ptr[regAux4]);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, ptr[regAux4]); // abs(x) % D21
    } else {
        // (x % D2 + D2) % D2
        if (vCoordDst.getIdx() != vCoordOrigin.getIdx())
            uni_vmovups(vCoordDst, vCoordOrigin);
        uni_vmovups(vAux, vCoordOrigin);
        if (dim == coord::w) {
            mov(regAux4, ptr[regParams + GET_OFF(srcWidthMul2F)]);
        } else if (coord::h) {
            mov(regAux4, ptr[regParams + GET_OFF(srcHeightMul2F)]);
        }
        uni_vdivps(vAux, vAux, ptr[regAux4]);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, ptr[regAux4]); // x % D2
        uni_vaddps(vCoordDst, vCoordDst, ptr[regAux4]); // x % D2 + D2
        uni_vmovups(vAux, vCoordDst);
        uni_vdivps(vAux, vAux, ptr[regAux4]);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, ptr[regAux4]); // (x % D2 + D2) % D2
    }

    if (dim == coord::w) {
        mov(regAux4, ptr[regParams + GET_OFF(srcWidthMul2Sub1F)]);
    } else if (coord::h) {
        mov(regAux4, ptr[regParams + GET_OFF(srcHeightMul2Sub1F)]);
    }
    uni_vmovups(vAux, ptr[regAux4]);
    uni_vsubps(vAux, vAux, vCoordDst);
    if (dim == coord::w) {
        uni_vmovups(kAux, vPool[srcWidthFIdx]);
    } else if (coord::h) {
        uni_vmovups(kAux, vPool[srcHeightFIdx]);
    }
    vcmpps(kAux, kAux, vCoordDst, 0x2); // vCoordDst >= vSrcDimF
    uni_vpand(vCoordDst, vCoordDst, kAux);
    uni_vpand(kAux, kAux, vAux);
    uni_vaddps(vCoordDst, vCoordDst, kAux);
}

template <x64::cpu_isa_t isa> // Works for AVX2, AVX
void jitGridSampleKernel<isa>::reflectionPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const Vmm& vAux, const Vmask& kAux, const coord dim) {
    if (jcp.alignCorners) {
        // abs(x) % D21
        if (vCoordDst.getIdx() != vCoordOrigin.getIdx())
            uni_vmovups(vCoordDst, vCoordOrigin);
        static const unsigned absMask[8] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
        mov(regAux4, reinterpret_cast<uintptr_t>(absMask)); // TODO: use PSIGND or vpabsd
        uni_vandps(vCoordDst, vCoordDst, ptr[regAux4]); // abs(x)
        if (dim == coord::w) {
            mov(regAux4, ptr[regParams + GET_OFF(srcWidthMul2Sub1F)]);
        } else if (coord::h) {
            mov(regAux4, ptr[regParams + GET_OFF(srcHeightMul2Sub1F)]);
        }
        uni_vdivps(vAux, vCoordDst, ptr[regAux4]);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, ptr[regAux4]); // abs(x) % D21
    } else {
        // (x % D2 + D2) % D2
        if (vCoordDst.getIdx() != vCoordOrigin.getIdx())
            uni_vmovups(vCoordDst, vCoordOrigin);
        if (dim == coord::w) {
            mov(regAux4, ptr[regParams + GET_OFF(srcWidthMul2F)]);
        } else if (coord::h) {
            mov(regAux4, ptr[regParams + GET_OFF(srcHeightMul2F)]);
        }
        uni_vdivps(vAux, vCoordOrigin, ptr[regAux4]);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, ptr[regAux4]); // x % D2
        uni_vaddps(vCoordDst, vCoordDst, ptr[regAux4]);  // x % D2 + D2
        uni_vdivps(vAux, vCoordDst, ptr[regAux4]);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, ptr[regAux4]); // (x % D2 + D2) % D2
    }

    if (dim == coord::w) {
        mov(regAux4, ptr[regParams + GET_OFF(srcWidthMul2Sub1F)]);
    } else if (coord::h) {
        mov(regAux4, ptr[regParams + GET_OFF(srcHeightMul2Sub1F)]);
    }
    uni_vsubps(vAux, vCoordDst, ptr[regAux4]);
    Vmm vUpperBound;
    if (dim == coord::w) {
        vUpperBound = vPool[srcWidthFIdx];
    } else if (coord::h) {
        vUpperBound = vPool[srcHeightFIdx];
    }
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
        vfmadd132ps(vCoef, vPool[const_0_75_idx], vPool[const_1_25_idx]);
        vfmsub231ps(vCoef, vDDim, vPool[const_1_50_idx]);
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
    static const float const_0_75[4] = { -0.75f, -0.75f, -0.75f, -0.75f };
    static const float const_1_25[4] = { 1.25f, 1.25f, 1.25f, 1.25f };
    static const float const_2_00[4] = { 2.f, 2.f, 2.f, 2.f };
    static const float const_2_25[4] = { 2.25f, 2.25f, 2.25f, 2.25f };
    if (idx == 0) {
        uni_vmovups(vAux, vDDim);
        mov(regAux1, reinterpret_cast<uintptr_t>(const_2_00));
        uni_vmulps(vAux, vAux, ptr[regAux1]);
        uni_vsubps(vAux, vAux, vPool[onesFIdx]);
        uni_vmovups(vCoef, vDDim);
        uni_vmulps(vCoef, vCoef, vCoef);
        uni_vsubps(vCoef, vCoef, vAux);
        uni_vmulps(vCoef, vCoef, vDDim);
        mov(regAux1, reinterpret_cast<uintptr_t>(const_0_75));
        uni_vmulps(vCoef, vCoef, ptr[regAux1]);
    } else if (idx == 1) {
        uni_vmovups(vCoef, vDDim);
        mov(regAux1, reinterpret_cast<uintptr_t>(const_1_25));
        uni_vmulps(vCoef, vCoef, ptr[regAux1]);
        mov(regAux1, reinterpret_cast<uintptr_t>(const_2_25));
        uni_vsubps(vCoef, vCoef, ptr[regAux1]);
        uni_vmulps(vCoef, vCoef, vDDim);
        uni_vfmadd132ps(vCoef, vPool[onesFIdx], vDDim);
    } else if (idx == 2) {
        uni_vmulps(vCoef, vDDim, vDDim);
        vfmadd132ps(vCoef, vPool[const_0_75_idx], vPool[const_1_25_idx]);
        vfmsub231ps(vCoef, vDDim, vPool[const_1_50_idx]);
        uni_vmulps(vCoef, vCoef, vDDim);
    } else if (idx == 3) {
        uni_vmulps(vCoef, vPool[const_0_75_idx], vDDim);
        uni_vmulps(vCoef, vCoef, vDDim);
        vfnmadd132ps(vCoef, vCoef, vDDim);
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
        vfmsub231ps(vCoef, vDDim, vPool[const_1_50_idx]);
        uni_vmulps(vCoef, vCoef, vDDim);
    } else if (idx == 3) {
        uni_vmulps(vCoef, vPool[const_0_75_idx], vDDim);
        uni_vmulps(vCoef, vCoef, vDDim);
        vfnmadd132ps(vCoef, vCoef, vDDim);
    }
}

template <>
void jitGridSampleKernel<x64::avx512_core>::nearestInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
    const auto& vSrcShift  = vWCoord;
    auto vAux = vmmRef();
    const auto& kMask0   = k1;
    const auto& kAuxMask = k2;

    uni_vroundps(vWCoord, vWCoord, 0x0); // Round near
    uni_vroundps(vHCoord, vHCoord, 0x0); // Round near

    if (jcp.paddingMode == PaddingMode::ZEROS) {
        zerosPadding(kMask0, vHCoord, vWCoord);
    } else if (jcp.paddingMode == PaddingMode::BORDER) {
        borderPadding(vWCoord, vWCoord, coord::w);
        borderPadding(vHCoord, vHCoord, coord::h);
    } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
        reflectionPadding(vWCoord, vWCoord, vAux, kAuxMask, coord::w);
        reflectionPadding(vHCoord, vHCoord, vAux, kAuxMask, coord::h);
    }

    hwShiftPs2dq(vSrcShift, vHCoord, vWCoord, vPool[srcWidthFIdx], regAux1);

    // PER CHANNEL LOOP
    Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
    const Xbyak::Reg64& rChannel = regAux1;
    const Xbyak::Reg64& rSrcTmp  = regAux2;
    const Xbyak::Reg64& rDstTmp  = regAux3;
    mov(rChannel, 0);
    mov(rSrcTmp, regSrc);
    mov(rDstTmp, regDst);
    L(lChannelLoopBegin);
    {
        cmp(rChannel, regChannelsNum);
        jge(lChannelLoopEnd, T_NEAR);

        if (jcp.paddingMode == PaddingMode::ZEROS) {
            kmovw(kAuxMask, kMask0);
            uni_vpxor(vAux, vAux, vAux);
        } else {
            kxnorw(kAuxMask, kAuxMask, kAuxMask);
        }

        uni_vpgatherdd(vAux, ptr[rSrcTmp + vSrcShift], kAuxMask);

        if (tail) {
            uni_vmovups(ptr[rDstTmp] | kTailMask, vAux);
        } else {
            uni_vmovups(ptr[rDstTmp], vAux);
        }
        add(rSrcTmp, regSrcChannelStepB);
        add(rDstTmp, regDstChannelStepB);
        inc(rChannel);

        jmp(lChannelLoopBegin, T_NEAR);
        L(lChannelLoopEnd);
    }
}

template <x64::cpu_isa_t isa> // Works for AVX2, AVX, SSE41
void jitGridSampleKernel<isa>::nearestInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
    const auto& vSrcShift = vWCoord;
//    const auto& vAux      = vPool[getVecIdx()];
    const auto& vAux1     = vHCoord;
//    const auto& kMask     = vPool[getVecIdx()];
    auto vAux  = vmmRef();
    auto kMask = vmmRef();

//storeVectorPart(regDst, regWorkAmount, vWCoord, 4);
//uni_vmovups(ptr[regDst], vWCoord);
    uni_vroundps(vWCoord, vWCoord, 0x0); // Round near
    uni_vroundps(vHCoord, vHCoord, 0x0); // Round near
//uni_vmovups(ptr[regDst], vHCoord);
//Xbyak::Xmm xmm = Xbyak::Xmm(vHCoord.getIdx());
//for (uint8_t i = 0; i < 1; i++) {
//    uni_vpextrd(ptr[regDst + i * 4], xmm, i);
//}

    bool useMask = false, zeroMask = false;
    if (jcp.paddingMode == PaddingMode::ZEROS) {
        zerosPadding(kMask, vHCoord, vWCoord);
        useMask  = true;
        zeroMask = true;
    } else if (jcp.paddingMode == PaddingMode::BORDER) {
        borderPadding(vWCoord, vWCoord, coord::w);
        borderPadding(vHCoord, vHCoord, coord::h);
    } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
        reflectionPadding(vWCoord, vWCoord, vAux, kMask, coord::w);
        reflectionPadding(vHCoord, vHCoord, vAux, kMask, coord::h);
    }

//uni_vmovups(ptr[regDst], kMask);
//Xbyak::Xmm xmm = Xbyak::Xmm(vWCoord.getIdx());
//for (uint8_t i = 0; i < 3; i++) {
//    uni_vpextrd(ptr[regDst + i * 4], xmm, i);
//}
    hwShiftPs2dq(vSrcShift, vHCoord, vWCoord, vPool[srcWidthFIdx], regAux1);
//Xbyak::Xmm xmm = Xbyak::Xmm(vSrcShift.getIdx());
//uni_vpextrd(ptr[regDst], xmm, 0);
//uni_vpextrd(ptr[regDst + 4], xmm, 1);

//Xbyak::Ymm ymmMask = Xbyak::Ymm(kMask.getIdx());
//vperm2f128(ymmMask, ymmMask, ymmMask, 0x1);
//
//Xbyak::Reg32 r32Aux = Xbyak::Reg32(regAux4.getIdx());
//Xbyak::Xmm xmmMask = Xbyak::Xmm(kMask.getIdx());
//for (uint8_t i = 0; i < 4; i++) {
//    uni_vpextrd(r32Aux, xmmMask, i);
//    cmp(r32Aux, 0);
//    mov(ptr[regDst + i * 4], r32Aux);
//}

    // PER CHANNEL LOOP
    Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
    const Xbyak::Reg64& rChannel = regAux1;
    const Xbyak::Reg64& rSrcTmp  = regAux2;
    const Xbyak::Reg64& rDstTmp  = regAux3;
    const Xbyak::Reg64& rWrkTmp  = regAux5;
    mov(rChannel, ptr[regParams + GET_OFF(channelsNum)]);
    mov(rSrcTmp, regSrc);
    mov(rDstTmp, regDst);
    L(lChannelLoopBegin);
    {
        cmp(rChannel, 0);
        jle(lChannelLoopEnd, T_NEAR);

//Xbyak::Ymm ymmMask = Xbyak::Ymm(kMask.getIdx());
//vperm2f128(ymmMask, ymmMask, ymmMask, 0x1);
//
//Xbyak::Reg32 r32Aux = Xbyak::Reg32(regAux4.getIdx());
//Xbyak::Xmm xmmMask = Xbyak::Xmm(kMask.getIdx());
//for (uint8_t i = 0; i < 4; i++) {
//    uni_vpextrd(r32Aux, xmmMask, i);
//    cmp(r32Aux, 0);
//    mov(ptr[rDstTmp + i * 4], r32Aux);
//}
//uni_vmovups(ptr[rDstTmp], kMask);
        if (!tail) {
            if (isa == x64::avx2) {
                if (jcp.paddingMode == PaddingMode::ZEROS)
                    uni_vpxor(vAux1, vAux1, vAux1);
                uni_vpgatherdd(vAux1, ptr[rSrcTmp + vSrcShift], kMask);
                uni_vmovups(ptr[rDstTmp], vAux1);
            } else {
                mov(rWrkTmp, regWorkAmount);
                maskMov32(rDstTmp, rSrcTmp, kMask, vSrcShift, vAux, rWrkTmp, regAux4, useMask, zeroMask);
            }
//            uni_vmovups(ptr[rDstTmp], vAux1);
        } else {
            mov(rWrkTmp, regWorkAmount);
            maskMov32(rDstTmp, rSrcTmp, kMask, vSrcShift, vAux, rWrkTmp, regAux4, useMask, zeroMask);
        }

        add(rSrcTmp, regSrcChannelStepB);
        add(rDstTmp, regDstChannelStepB);
        dec(rChannel);

        jmp(lChannelLoopBegin, T_NEAR);
        L(lChannelLoopEnd);
    }
}

template <>
void jitGridSampleKernel<x64::avx512_core>::bilinearInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
    auto vDX     = vmmRef();
    auto vDY     = vmmRef();
    auto vQ0     = vmmRef();
    auto vQ1     = vmmRef();
    const auto &shift00 = vWCoord;
    const auto &shift01 = vHCoord;
    auto shift10 = vmmRef();
    auto shift11 = vmmRef();
    auto vAux3   = vmmRef();
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
    if (jcp.paddingMode == PaddingMode::ZEROS) {
        zerosPadding(kMask00, vHCoord, vWCoord); // (y; x)
        zerosPadding(kMask01, vHCoord, shift10); // (y; x + 1)
        zerosPadding(kMask11, shift11, shift10); // (y + 1; x + 1)
        zerosPadding(kMask10, shift11, vWCoord); // (y + 1; x)

        hwShiftPs2dq(shift00, vHCoord, vWCoord, vPool[srcWidthFIdx], regAux1);
        uni_vpaddd(shift01, shift00, vPool[dataTypeSizeIdx]);
        uni_vpaddd(shift10, shift00, vPool[srcWidthBIdx]); // shift11??
        uni_vpaddd(shift11, shift10, vPool[dataTypeSizeIdx]); // sub??
    } else if (jcp.paddingMode == PaddingMode::BORDER) {
        borderPadding(vWCoord, vWCoord, coord::w);
        borderPadding(vHCoord, vHCoord, coord::h);
        borderPadding(shift10, shift10, coord::w);
        borderPadding(shift11, shift11, coord::h);
    } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
        reflectionPadding(vWCoord, vWCoord, vQ0, kAuxMask, coord::w);
        reflectionPadding(vHCoord, vHCoord, vQ0, kAuxMask, coord::h);
        reflectionPadding(shift10, shift10, vQ0, kAuxMask, coord::w);
        reflectionPadding(shift11, shift11, vQ0, kAuxMask, coord::h);
    }
    if (jcp.paddingMode == PaddingMode::BORDER || jcp.paddingMode == PaddingMode::REFLECTION) {
        uni_vmovups(vAux3, shift11);
        // W * y + x
        uni_vfmadd132ps(vAux3, vWCoord, vPool[srcWidthFIdx]);   // (y + 1; x)
        uni_vfmadd231ps(vWCoord, vHCoord, vPool[srcWidthFIdx]); // (y; x)
        uni_vfmadd132ps(vHCoord, shift10, vPool[srcWidthFIdx]); // (y; x + 1)
        uni_vfmadd132ps(shift11, shift10, vPool[srcWidthFIdx]); // (y + 1; x + 1)
        uni_vcvtps2dq(shift00, vWCoord);
        uni_vcvtps2dq(shift01, vHCoord);
        uni_vcvtps2dq(shift10, vAux3);
        uni_vcvtps2dq(shift11, shift11);
        if (dataTypeSize > 1) {
            uni_vpslld(shift00, shift00, dataTypeShift);
            uni_vpslld(shift01, shift01, dataTypeShift);
            uni_vpslld(shift10, shift10, dataTypeShift);
            uni_vpslld(shift11, shift11, dataTypeShift);
        }
    }

    // PER CHANNEL LOOP
    Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
    const Xbyak::Reg64 &rChannel = regAux1;
    const Xbyak::Reg64 &rSrcTmp  = regAux2;
    const Xbyak::Reg64 &rDstTmp  = regAux3;
    mov(rChannel, 0);
    mov(rSrcTmp, regSrc);
    mov(rDstTmp, regDst);
    L(lChannelLoopBegin);
    {
        cmp(rChannel, regChannelsNum);
        jge(lChannelLoopEnd, T_NEAR);

        // (y; x)
        if (jcp.paddingMode == PaddingMode::ZEROS) {
            kmovw(kAuxMask, kMask00);
            uni_vpxor(vQ0, vQ0, vQ0);
        } else {
            kxnorw(kAuxMask, kAuxMask, kAuxMask);
        }
        uni_vpgatherdd(vQ0, ptr[rSrcTmp + shift00], kAuxMask); // v00 -> vQ0
        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtdq2ps(vQ0, vQ0);
        }
        uni_vfmsub213ps(vQ0, vDX, vQ0); // q0 = -(v00 - dx * v00)

        // (y; x + 1)
        if (jcp.paddingMode == PaddingMode::ZEROS) {
            kmovw(kAuxMask, kMask01);
            uni_vpxor(vAux3, vAux3, vAux3);
        } else {
            kxnorw(kAuxMask, kAuxMask, kAuxMask);
        }
        uni_vpgatherdd(vAux3, ptr[rSrcTmp + shift01], kAuxMask);
        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtdq2ps(vAux3, vAux3);
        }
        uni_vfmsub231ps(vQ0, vAux3, vDX); // q0 = -q0 + dx * v01

        // (y + 1; x + 1)
        if (jcp.paddingMode == PaddingMode::ZEROS) {
            kmovw(kAuxMask, kMask11);
            uni_vpxor(vAux3, vAux3, vAux3);
        } else {
            kxnorw(kAuxMask, kAuxMask, kAuxMask);
        }
        uni_vpgatherdd(vAux3, ptr[rSrcTmp + (Vmm)shift11], kAuxMask);
        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtdq2ps(vAux3, vAux3);
        }

        // (y + 1; x)
        if (jcp.paddingMode == PaddingMode::ZEROS) {
            kmovw(kAuxMask, kMask10);
            uni_vpxor(vQ1, vQ1, vQ1);
        } else {
            kxnorw(kAuxMask, kAuxMask, kAuxMask);
        }
        uni_vpgatherdd(vQ1, ptr[rSrcTmp + (Vmm)shift10], kAuxMask);
        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtdq2ps(vQ1, vQ1);
        }

        uni_vfmsub213ps(vQ1, vDX, vQ1); // q1 = -(v10 - dx * v10)
        uni_vfmsub231ps(vQ1, vAux3, vDX); // q1 = -q1 + dx * v11
        // Res = q0 + dy * (q1 - q0)
        uni_vsubps(vQ1, vQ1, vQ0);
        uni_vfmadd132ps(vQ1, vQ0, vDY);

        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtps2dq(vQ1, vQ1);
        }

        if (tail) {
            uni_vmovups(ptr[rDstTmp] | kTailMask, vQ1);
        } else {
            uni_vmovups(ptr[rDstTmp], vQ1);
        }
        add(rSrcTmp, regSrcChannelStepB);
        add(rDstTmp, regDstChannelStepB);
        add(rChannel, 1);

        jmp(lChannelLoopBegin, T_NEAR);
        L(lChannelLoopEnd);
    }
}

template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::bilinearInterpolation(const Vmm& vWCoord, const Vmm& vHCoord, bool tail) {
    const auto& vDX = vPool[0];
    const auto& vDY = vPool[1];
    const auto& vGatherShift = vPool[2];
    const auto& vAux3 = vPool[3];
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
    const Xbyak::Reg64& rChannel = regAux1;
    const Xbyak::Reg64& rSrcTmp = regAux2;
    const Xbyak::Reg64& rDstTmp = regAux3;
    mov(rChannel, 0);
    mov(rSrcTmp, regSrc);
    mov(rDstTmp, regDst);
    L(lChannelLoopBegin);
    {
        cmp(rChannel, regChannelsNum);
        jge(lChannelLoopEnd, T_NEAR);

        if (jcp.paddingMode == PaddingMode::ZEROS) {
            // (x; y)
            zerosPadding(kMask0, vHCoord, vWCoord);
            uni_vmovups(vGatherShift, vHCoord);
            hwShiftPs2dq(vGatherShift, vGatherShift, vWCoord, vPool[srcWidthFIdx], regAux5);
            uni_vpxor(vQ0, vQ0, vQ0);
            uni_vpgatherdd(vQ0, ptr[rSrcTmp + vGatherShift], kMask0); // v00 -> vQ0
            if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                uni_vcvtdq2ps(vQ0, vQ0);
            }
            uni_vfmsub213ps(vQ0, vDX, vQ0); // q0 = -(v00 - dx * v00)

            // (x + 1; y)
            uni_vaddps(vWCoord, vWCoord, vPool[onesFIdx]);
            zerosPadding(kMask0, vHCoord, vWCoord);
            uni_vmovups(vGatherShift, vHCoord);
            hwShiftPs2dq(vGatherShift, vGatherShift, vWCoord, vPool[srcWidthFIdx], regAux5);
            uni_vpxor(vAux3, vAux3, vAux3);
            uni_vpgatherdd(vAux3, ptr[rSrcTmp + vGatherShift], kMask0);

            // q0 = -q0 + dx * v01
            if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                uni_vcvtdq2ps(vAux3, vAux3);
            }
            uni_vfmsub231ps(vQ0, vAux3, vDX);

            // (x + 1; y + 1)
            uni_vaddps(vHCoord, vHCoord, vPool[onesFIdx]);
            zerosPadding(kMask0, vHCoord, vWCoord);
            uni_vmovups(vGatherShift, vHCoord);
            hwShiftPs2dq(vGatherShift, vGatherShift, vWCoord, vPool[srcWidthFIdx], regAux5);
            uni_vpxor(vAux3, vAux3, vAux3);
            uni_vpgatherdd(vAux3, ptr[rSrcTmp + vGatherShift], kMask0);

            // (x; y + 1)
            uni_vsubps(vWCoord, vWCoord, vPool[onesFIdx]);
            zerosPadding(kMask0, vHCoord, vWCoord);
            uni_vmovups(vGatherShift, vHCoord);
            hwShiftPs2dq(vGatherShift, vGatherShift, vWCoord, vPool[srcWidthFIdx], regAux5);
            uni_vpxor(vQ1, vQ1, vQ1);
            uni_vpgatherdd(vQ1, ptr[rSrcTmp + vGatherShift], kMask0);
            uni_vsubps(vHCoord, vHCoord, vPool[onesFIdx]);
            if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                uni_vcvtdq2ps(vQ1, vQ1);
            }

            uni_vfmsub213ps(vQ1, vDX, vQ1); // q1 = -(v10 - dx * v10)
            uni_vfmsub231ps(vQ1, vAux3, vDX); // q1 = -q1 + dx * v11
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
    auto vCX0        = vmmRef();
    auto vCX1        = vmmRef();
    auto vCX2        = vmmRef();
    auto vCX3        = vmmRef();
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

    for (int i = 0; i < 4; i++) {
        bicubicCoefficients((&vCX0)[i], vDX, i);
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
    const Xbyak::Reg64& rChannel = regAux1;
    const Xbyak::Reg64& rSrcTmp  = regAux2;
    const Xbyak::Reg64& rDstTmp  = regAux3;
    mov(rChannel, 0);
    mov(rSrcTmp, regSrc);
    mov(rDstTmp, regDst);
    L(lChannelLoopBegin);
    {
        cmp(rChannel, regChannelsNum);
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
                reflectionPadding(vSrcShift0, vHCoord, vAux, kAuxMask, coord::h);
                uni_vmulps(vSrcShift0, vSrcShift0, vPool[srcWidthFIdx]);
                uni_vmovups(vWCoord, vWLeft);
                reflectionPadding(vSrcShift, vWCoord, vAux, kAuxMask, coord::w);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                kxnorw(kAuxMask, kAuxMask, kAuxMask);
            }
            uni_vcvtps2dq(vSrcShift, vSrcShift);
            if (dataTypeSize > 1)
                uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
            uni_vpgatherdd(vAux, ptr[rSrcTmp + (Vmm)vSrcShift], kAuxMask);
            if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                uni_vcvtdq2ps(vAux, vAux);
            }
            uni_vmulps(vXDotProd, vAux, vCX0);

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
                    reflectionPadding(vSrcShift, vWCoord, vAux, kAuxMask, coord::w);
                    uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                    kxnorw(kAuxMask, kAuxMask, kAuxMask);
                }
                uni_vcvtps2dq(vSrcShift, vSrcShift);
                if (dataTypeSize > 1)
                    uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
                uni_vpgatherdd(vAux, ptr[rSrcTmp + (Vmm)vSrcShift], kAuxMask);
                if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                    uni_vcvtdq2ps(vAux, vAux);
                }
                uni_vfmadd231ps(vXDotProd, vAux, (&vCX0)[w]);
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
            uni_vmovups(ptr[rDstTmp] | kTailMask, vYDotProd);
        } else {
            uni_vmovups(ptr[rDstTmp], vYDotProd);
        }
        add(rSrcTmp, regSrcChannelStepB);
        add(rDstTmp, regDstChannelStepB);
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
        hwShiftPs2dq(vSrcShift0, vHTop, vWLeft, vPool[srcWidthFIdx], regAux1);
    }

    // PER CHANNEL LOOP
    Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
    const Xbyak::Reg64& rChannel = regAux1;
    const Xbyak::Reg64& rSrcTmp = regAux2;
    const Xbyak::Reg64& rDstTmp = regAux3;
    mov(rChannel, 0);
    mov(rSrcTmp, regSrc);
    mov(rDstTmp, regDst);
    L(lChannelLoopBegin);
    {
        cmp(rChannel, regChannelsNum);
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
                reflectionPadding(vSrcShift0, vHCoord, vAux, kAuxMask, coord::h);
                uni_vmulps(vSrcShift0, vSrcShift0, vPool[srcWidthFIdx]);
                uni_vmovups(vWCoord, vWLeft);
                reflectionPadding(vSrcShift, vWCoord, vAux, kAuxMask, coord::w);
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
                reflectionPadding(vSrcShift, vWCoord, vAux, kAuxMask, coord::w);
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
                reflectionPadding(vSrcShift, vWCoord, vAux, kAuxMask, coord::w);
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
                reflectionPadding(vSrcShift, vWCoord, vAux, kAuxMask, coord::w);
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

        uni_vmovups(ptr[rDstTmp], vYDotProd);
        add(rSrcTmp, regSrcChannelStepB);
        add(rDstTmp, regDstChannelStepB);
        add(rChannel, 1);

        jmp(lChannelLoopBegin, T_NEAR);
        L(lChannelLoopEnd);
    }
}

template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::tail() {
    Xbyak::Label lEnd;
    cmp(regWorkAmount, 0);
    jle(lEnd, T_NEAR);

    const Vmm& vHCoord = vPool[getVecIdx()];
    const Vmm& vWCoord = vPool[getVecIdx()];

    getTailCoordinates(vHCoord, vWCoord);
    denormalizeRawCoordinates(vWCoord, vHCoord);
    interpolation(vWCoord, vHCoord, true);

    releaseVecIdx(vHCoord.getIdx());
    releaseVecIdx(vWCoord.getIdx());

    if (dataTypeSize > 1)
        sal(regWorkAmount, dataTypeShift); // multiply by source data type size.
    add(regDst, regWorkAmount);

    L(lEnd);
}

template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::hwShiftPs2dq(const Vmm& vDst, const Vmm& vHCoord,const Vmm& vWCoord, const Vmm& vWidth, const Xbyak::Reg64& rAux) {
    if (vDst.getIdx() == vWCoord.getIdx()) {
        uni_vfmadd231ps(vDst, vHCoord, vWidth);
    } else if (vDst.getIdx() == vHCoord.getIdx()) {
        uni_vfmadd132ps(vDst, vWCoord, vWidth);
    } else if (vDst.getIdx() == vWidth.getIdx()) {
        uni_vfmadd132ps(vDst, vWCoord, vHCoord);
    } else {
        uni_vmovups(vDst, vWCoord);
        uni_vfmadd231ps(vDst, vHCoord, vWidth);
    }

    if (isa == x64::avx) { // vpslld works just with XMM for AVX, so use vmulps for YMM
        if (dataTypeSize > 1) {
            const float val = dataTypeSize;
            static const float dataTypeSizeArr[8] = {val, val, val, val, val, val, val, val};
            mov(rAux, reinterpret_cast<uintptr_t>(dataTypeSizeArr));
            uni_vmulps(vDst, vDst, ptr[rAux]);
        }
        uni_vcvtps2dq(vDst, vDst);
    } else {
        uni_vcvtps2dq(vDst, vWCoord);
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
