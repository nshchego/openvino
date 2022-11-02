// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unique.hpp"

using namespace dnnl::impl::cpu;
using InferenceEngine::Precision;

namespace ov {
namespace intel_cpu {

#define GET_OFF(field) offsetof(UniqueKernelExecArgs, field)

template <x64::cpu_isa_t isa>
UniqueKernel<isa>::UniqueKernel(const UniqueKernelConfParams& jcp) :
        UniqueKernelBase(jcp) {
    vlen = x64::cpu_isa_traits<isa>::vlen;
    dataTypeSize = jcp.dataPrc.size();
    dataElPerVec = vlen / dataTypeSize;
    if (dataTypeSize == 2)
        dataTypeShift = 1;
    else if (dataTypeSize == 4)
        dataTypeShift = 2;
}

template <x64::cpu_isa_t isa>
void UniqueKernel<isa>::create_ker() {
    auto code = x64::jit_generator::create_kernel();
    if (code != dnnl::impl::status::success)
        IE_THROW() << "Could not create GridSample kernel. Error code: " << std::to_string(code);
    ker_ = (decltype(ker_))jit_ker();
}

template <x64::cpu_isa_t isa>
void UniqueKernel<isa>::generate() {
    this->preamble();
    registersPool = RegistersPool::create(isa, {rax, rcx, rsp, rdi, k0});

    regSrc = getReg64();
    mov(regSrc,  ptr[regParams + GET_OFF(srcPtr)]);

    for (int i = 0; i < 4; i++) {
        if (jcp.definedOutputs[i]) {
            regDst[i] = getReg64();
            mov(regDst[i],  ptr[regParams + GET_OFF(dstPtr[i])]);
        }
    }

    initVectors();
    process();

    registersPool.reset();
    this->postamble();
}

template <>
void UniqueKernel<x64::avx512_core>::initVectors() {
    auto rAux = getReg64();

//    vInc = getVmm();
//    mov(rAux, dataTypeSize);
//    vpbroadcastd(vInc, rAux);

//    vSteps = getVmm();
//    static const unsigned val = dataTypeSize;
//    static const unsigned steps[16]  = { 0 * val, 1 * val, 2 * val,   3 * val,   4 * val,   5 * val,   6 * val,   7 * val,
//                                         8 * val, 9 * val, 10 * val, 11 * val, 12 * val, 13 * val, 14 * val, 14 * val };
//    mov(rAux, reinterpret_cast<uintptr_t>(steps));
//    uni_vmovups(vSteps, ptr[rAux]);

    Xbyak::Reg32 rMask(rAux.getIdx());

    kMaxMask0 = getMask();
    mov(rMask, 0B1010101010101010);
    kmovw(kMaxMask0, rMask);

    kMaxMask1 = getMask();
    mov(rMask, 0B1101010101010100);
    kmovw(kMaxMask1, rMask);

//    kMaxMask0 = getMask();
//    mov(rMask, 0B1010101010101010);
//    kmovw(kMaxMask0, rMask);

//    kMaxMask1 = getMask();
//    mov(rMask, 0B1100110011001100);
//    kmovw(kMaxMask1, rMask);

    kMaxMask2 = getMask();
    mov(rMask, 0B1101010011010100);
    kmovw(kMaxMask2, rMask);

    kMaxMask3 = getMask();
    mov(rMask, 0B1111000101110000);
    kmovw(kMaxMask3, rMask);

    kMaxMask4 = getMask();
    mov(rMask, 0B1010101010101010);
    kmovw(kMaxMask4, rMask);

    static const unsigned permMask[16]  = { 15, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 0 };
    mov(rAux, reinterpret_cast<uintptr_t>(permMask));
    vPermMask = getVmm();
    uni_vmovups(vPermMask, ptr[rAux]);

    static const unsigned permMask2[16]  = { 4, 2, 1, 7, 0, 6, 5, 3, 12, 10, 9, 15, 8, 14, 13, 11 };
    mov(rAux, reinterpret_cast<uintptr_t>(permMask2));
    vPermMask2 = getVmm();
    uni_vmovups(vPermMask2, ptr[rAux]);

    static const unsigned permMask3[16]  = { 8, 5, 6, 4, 3, 1, 2, 15, 0, 13, 14, 12, 11, 9, 10, 7 };
    mov(rAux, reinterpret_cast<uintptr_t>(permMask3));
    vPermMask3 = getVmm();
    uni_vmovups(vPermMask3, ptr[rAux]);

    static const unsigned permMask4[16]  = { 4, 2, 1, 7, 0, 6, 5, 3, 12, 10, 9, 15, 8, 14, 13, 11 };
    mov(rAux, reinterpret_cast<uintptr_t>(permMask4));
    vPermMask4 = getVmm();
    uni_vmovups(vPermMask4, ptr[rAux]);

//    Xbyak::Reg32 r32Aux(rAux.getIdx());
//
//    if (jcp.dynamicShapes) {
//        regChannelNum = getReg64();
//        mov(regChannelNum, ptr[regParams + GET_OFF(channelsNum)]);
//    }
//    kTailMask = getMask();
//
//    vSrcWidthF = getVmm();
//    mov(rAux, ptr[regParams + GET_OFF(srcWidthF)]);
//    uni_vpbroadcastd(vSrcWidthF, ptr[rAux]);
//
//    vSrcHeightF = getVmm();
//    mov(rAux, ptr[regParams + GET_OFF(srcHeightF)]);
//    uni_vpbroadcastd(vSrcHeightF, ptr[rAux]);
//
//    if (one_of(jcp.paddingMode, PaddingMode::ZEROS, PaddingMode::BORDER)) {
//        vZeros = getVmm();
//        uni_vpxor(vZeros, vZeros, vZeros);
//    }
//
//    if (one_of(jcp.interpolationMode, InterpolationMode::BICUBIC, InterpolationMode::BILINEAR)) {
//        vOnesF = getVmm();
//        mov(r32Aux, 0x3f800000); // 1.f
//        vpbroadcastd(vOnesF, r32Aux);
//    }
//
//    if (jcp.alignCorners) {
//        vWDenormCoefF = getVmm();
//        mov(rAux, ptr[regParams + GET_OFF(wDenormCoefF)]);
//        uni_vpbroadcastd(vWDenormCoefF, ptr[rAux]);
//
//        vHDenormCoefF = getVmm();
//        mov(rAux, ptr[regParams + GET_OFF(hDenormCoefF)]);
//        uni_vpbroadcastd(vHDenormCoefF, ptr[rAux]);
//    } else {
//        vHalfF = getVmm();
//        mov(r32Aux, 0x3f000000); // 0.5f
//        vpbroadcastd(vHalfF, r32Aux);
//    }
//
//    static const unsigned gridPermMask[16]  = { 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15 };
//    mov(rAux, reinterpret_cast<uintptr_t>(gridPermMask));
//    vGridPermMask = getVmm();
//    uni_vmovups(vGridPermMask, ptr[rAux]);
//
//    if (jcp.paddingMode == PaddingMode::ZEROS) {
//        vDataTypeSizeB = getVmm();
//        mov(rAux, dataTypeSize);
//        vpbroadcastd(vDataTypeSizeB, r32Aux);
//        vSrcWidthB = getVmm();
//        mov(rAux, ptr[regParams + GET_OFF(srcWidthB)]);
//        uni_vpbroadcastd(vSrcWidthB, ptr[rAux]);
//    } else if (jcp.paddingMode == PaddingMode::BORDER) {
//        vSrcHeightSub1F = getVmm();
//        mov(rAux, ptr[regParams + GET_OFF(srcHeightSub1F)]);
//        uni_vpbroadcastd(vSrcHeightSub1F, ptr[rAux]);
//        vSrcWidthSub1F = getVmm();
//        mov(rAux, ptr[regParams + GET_OFF(srcWidthSub1F)]);
//        uni_vpbroadcastd(vSrcWidthSub1F, ptr[rAux]);
//    } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
//        vSrcHeightMul2F = getVmm();
//        mov(rAux, ptr[regParams + GET_OFF(srcHeightMul2F)]);
//        uni_vpbroadcastd(vSrcHeightMul2F, ptr[rAux]);
//        vSrcWidthMul2F = getVmm();
//        mov(rAux, ptr[regParams + GET_OFF(srcWidthMul2F)]);
//        uni_vpbroadcastd(vSrcWidthMul2F, ptr[rAux]);
//        vSrcHeightMul2Sub1F = getVmm();
//        mov(rAux, ptr[regParams + GET_OFF(srcHeightMul2Sub1F)]);
//        uni_vpbroadcastd(vSrcHeightMul2Sub1F, ptr[rAux]);
//        vSrcWidthMul2Sub1F = getVmm();
//        mov(rAux, ptr[regParams + GET_OFF(srcWidthMul2Sub1F)]);
//        uni_vpbroadcastd(vSrcWidthMul2Sub1F, ptr[rAux]);
//        if (jcp.alignCorners) {
//            vAbsMask = getVmm();
//            mov(r32Aux, 0x7fffffff);
//            vpbroadcastd(vAbsMask, r32Aux);
//        }
//    }
//
//    if (jcp.interpolationMode == InterpolationMode::BICUBIC) {
//        vConst_0_75 = getVmm();
//        mov(r32Aux, 0xbf400000); // -0.75f
//        vpbroadcastd(vConst_0_75, r32Aux);
//        vConst_1_25 = getVmm();
//        mov(r32Aux, 0x3fa00000); // 1.25f
//        vpbroadcastd(vConst_1_25, r32Aux);
//        vConst_1_50 = getVmm();
//        mov(r32Aux, 0x3fc00000); // 1.5f
//        vpbroadcastd(vConst_1_50, r32Aux);
//        vConst_2_00 = getVmm();
//        mov(r32Aux, 0x40000000); // 2.0f
//        vpbroadcastd(vConst_2_00, r32Aux);
//        vConst_2_25 = getVmm();
//        mov(r32Aux, 0x40100000); // 2.25f
//        vpbroadcastd(vConst_2_25, r32Aux);
//    }
}

template <x64::cpu_isa_t isa>
void UniqueKernel<isa>::initVectors() {
//    auto rAux = getReg64();
//
//    vSrcWidthF = getVmm();
//    mov(rAux, ptr[regParams + GET_OFF(srcWidthF)]);
//    uni_vmovups(vSrcWidthF, ptr[rAux]);
//
//    if (one_of(jcp.interpolationMode, InterpolationMode::BILINEAR, InterpolationMode::NEAREST) ||
//            jcp.interpolationMode == InterpolationMode::BICUBIC && (jcp.paddingMode == PaddingMode::REFLECTION ||
//                    jcp.paddingMode == PaddingMode::BORDER && !jcp.alignCorners ||
//                    jcp.paddingMode == PaddingMode::ZEROS)) {
//        vSrcHeightF = getVmm();
//        mov(rAux, ptr[regParams + GET_OFF(srcHeightF)]);
//        uni_vmovups(vSrcHeightF, ptr[rAux]);
//    }
//
//    if (jcp.interpolationMode == InterpolationMode::BICUBIC &&
//            jcp.paddingMode == PaddingMode::BORDER && jcp.alignCorners) {
//        vHDenormCoefF = getVmm();
//        mov(rAux, ptr[regParams + GET_OFF(hDenormCoefF)]);
//        uni_vmovups(vHDenormCoefF, ptr[rAux]);
//    }
//
//    if (jcp.interpolationMode != InterpolationMode::BICUBIC) {
//        if (one_of(jcp.paddingMode, PaddingMode::BORDER, PaddingMode::ZEROS) &&
//                (isa == x64::avx2 && jcp.interpolationMode == InterpolationMode::NEAREST || one_of(isa, x64::avx, x64::sse41))) {
//            vZeros = getVmm();
//            uni_vpxor(vZeros, vZeros, vZeros);
//        }
//
//        if (jcp.alignCorners) {
//            mov(rAux, ptr[regParams + GET_OFF(wDenormCoefF)]);
//            vWDenormCoefF = getVmm();
//            uni_vmovups(vWDenormCoefF, ptr[rAux]);
//            if (!(jcp.interpolationMode == InterpolationMode::BILINEAR && jcp.paddingMode == PaddingMode::ZEROS)) {
//                mov(rAux, ptr[regParams + GET_OFF(hDenormCoefF)]);
//                vHDenormCoefF = getVmm();
//                uni_vmovups(vHDenormCoefF, ptr[rAux]);
//            }
//        } else {
//            static const float halfArr[8] = { 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f };
//            mov(rAux, reinterpret_cast<uintptr_t>(halfArr));
//            vHalfF = getVmm();
//            uni_vmovups(vHalfF, ptr[rAux]);
//        }
//
//        if (isa == x64::avx2 && jcp.interpolationMode == InterpolationMode::NEAREST) {// TODO: check reg num for BILINEAR
//            static const unsigned gridPermMask[8]  = { 0, 2, 4, 6, 1, 3, 5, 7 };
//            mov(rAux, reinterpret_cast<uintptr_t>(gridPermMask));
//            vGridPermMask = getVmm();
//            uni_vmovups(vGridPermMask, ptr[rAux]);
//        }
//    }
//
//    if (jcp.interpolationMode == InterpolationMode::BICUBIC ||
//            (jcp.interpolationMode == InterpolationMode::BILINEAR && jcp.paddingMode != PaddingMode::ZEROS)) {
//        static const float onesArr[8] = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };
//        mov(rAux, reinterpret_cast<uintptr_t>(onesArr));
//        vOnesF = getVmm();
//        uni_vmovups(vOnesF, ptr[rAux]);
//    }
}

template <>
void UniqueKernel<x64::avx512_core>::quickSort(const Xbyak::Reg64& rSrc) {
//    Xbyak::Label lEnd;
//    cmp(regLeft, regRight);
//    jge(lEnd, T_NEAR);

//    auto vData = getVmm();
//    auto vPivot = getVmm();
//    auto vLeft = getVmm();
//    auto vRight = getVmm();
//    auto vAux   = getVmm();
//    auto kAux   = getMask();
//
//    uni_vmovups(vData, ptr[regSrc]);
//    Xbyak::Xmm xmmLow(vData.getIdx());
//    for (int i = 0; i < dataElPerVec; i++) {
//        vbroadcastss(vPivot, xmmLow);
//        vminps(vLeft, vData, vPivot);
//        vmaxps(vRight, vData, vPivot);
//        vcmpps(kAux, vLeft, vPivot, 0x1);
//        vcompressps(Vmm(vLeft.getIdx()) | kAux, vLeft);
//        vcmpps(kAux, vRight, vPivot, 0x5);
//        vcompressps(Vmm(vRight.getIdx()) | kAux, vRight);
//    }

//    L(lEnd);
}

template <x64::cpu_isa_t isa>
void UniqueKernel<isa>::quickSort(const Xbyak::Reg64& rSrc) {

}

template <>
void UniqueKernel<x64::avx512_core>::partition() {
//    auto vData = getVmm();
//    auto vPivot = getVmm();
//    auto vLeft = getVmm();
//    auto vRight = getVmm();
//    auto vAux   = getVmm();
//    auto kAux   = getMask();
//
//    uni_vmovups(vData, ptr[regSrc]);
//    Xbyak::Xmm xmmLow(vData.getIdx());
//    for (int i = 0; i < dataElPerVec; i++) {
//        vbroadcastss(vPivot, xmmLow);
//        vminps(vLeft, vData, vPivot);
//        vmaxps(vRight, vData, vPivot);
//    }
//
//    vpbroadcastd(vRight, regRight);
//    uni_vpaddd(vRight, vRight, vSteps);
//    gatherdd(vPivot, regSrc, vRight, kAux, false);
//
//    Xbyak::Label lLoop, lEnd;
//    auto rIter = getReg64();
//    auto vIter = getVmm();
//    auto vCurr = getVmm();
//    mov(rIter, regLeft);
//    vpbroadcastd(vIter, rIter);
//    L(lLoop);
//    {
//        cmp(rIter, regRight);
//        jge(lEnd, T_NEAR);
//
//        gatherdd(vCurr, regSrc, vIter, kAux, false);
////        if (jcp.inDataPrc == Precision::FP32) {
////            vcmpps(kAux, vCurr, vPivot, 0x2); // vCurr <= vPivot
////        } else if (jcp.inDataPrc == Precision::I32) {
////            vpcmpd(kAux, vCurr, vPivot, 0x2); // vCurr <= vPivot
////        }
//
//
//        add(rIter, dataTypeSize);
//        uni_vpaddd(vIter, vIter, vInc);
//        jmp(lLoop, T_NEAR);
//    }
//    L(lEnd);
}

//template <>
//void UniqueKernel<x64::avx512_core>::partition() {
//    auto vPivot = getVmm();
//    auto vRight = getVmm();
//    auto vAux   = getVmm();
//    auto kAux   = getMask();
//
//    vpbroadcastd(vRight, regRight);
//    uni_vpaddd(vRight, vRight, vSteps);
//    gatherdd(vPivot, regSrc, vRight, kAux, false);
//
//    Xbyak::Label lLoop, lEnd;
//    auto rIter = getReg64();
//    auto vIter = getVmm();
//    auto vCurr = getVmm();
//    mov(rIter, regLeft);
//    vpbroadcastd(vIter, rIter);
//    L(lLoop);
//    {
//        cmp(rIter, regRight);
//        jge(lEnd, T_NEAR);
//
//        gatherdd(vCurr, regSrc, vIter, kAux, false);
//        if (jcp.inDataPrc == Precision::FP32) {
//            vcmpps(kAux, vCurr, vPivot, 0x2); // vCurr <= vPivot
//        } else if (jcp.inDataPrc == Precision::I32) {
//            vpcmpd(kAux, vCurr, vPivot, 0x2); // vCurr <= vPivot
//        }
//
//
//        add(rIter, dataTypeSize);
//        uni_vpaddd(vIter, vIter, vInc);
//        jmp(lLoop, T_NEAR);
//    }
//    L(lEnd);
//}

template <x64::cpu_isa_t isa>
void UniqueKernel<isa>::partition() {

}

template <x64::cpu_isa_t isa>
void UniqueKernel<isa>::process() {
    Xbyak::Label lLoop, lEnd;
    auto rSrcTmp = getReg64();
    auto rDstTmp = getReg64();
    mov(rSrcTmp, regSrc);
    mov(rDstTmp, regDst[UNIQUE_DATA]);
    auto vSrc = getVmm();

    // Per vector sort
    regWorkAmount = getReg64();
    mov(regWorkAmount, ptr[regParams + GET_OFF(workAmount)]);

    L(lLoop);
    {
        cmp(regWorkAmount, dataElPerVec);
        jl(lEnd, T_NEAR);

        uni_vmovups(vSrc, ptr[rSrcTmp]);
        sortVector(vSrc);
        uni_vmovups(ptr[rDstTmp], vSrc);

        add(rSrcTmp, vlen);
        add(rDstTmp, vlen);

        sub(regWorkAmount, dataElPerVec);
        jmp(lLoop, T_NEAR);
    }
    L(lEnd);

    tail();
}

template <>
void UniqueKernel<x64::avx512_core>::cmpPerm(const Vmm& vDst, const Vmm& vSrc1, const Vmm& vSrc2, const Vmask& kMaxMask) {
    if (jcp.dataPrc == Precision::FP32) {
        vminps(vDst, vSrc1, vSrc2);
        vmaxps(vDst | kMaxMask, vSrc1, vSrc2);
    } else if (jcp.dataPrc == Precision::I32) {
        vpminsd(vDst, vSrc1, vSrc2);
        vpmaxsd(vDst | kMaxMask, vSrc1, vSrc2);
    }  else if (jcp.dataPrc == Precision::U8) {
        vpminub(vDst, vSrc1, vSrc2);
        vpmaxub(vDst | kMaxMask, vSrc1, vSrc2);
    } else if (jcp.dataPrc == Precision::I8) {
        vpminsb(vDst, vSrc1, vSrc2);
        vpmaxsb(vDst | kMaxMask, vSrc1, vSrc2);
    }
}

template <x64::cpu_isa_t isa>
void UniqueKernel<isa>::cmpPerm(const Vmm& vDst, const Vmm& vSrc1, const Vmm& vSrc2, const Vmask& kMaxMask) {

}

template <>
void UniqueKernel<x64::avx512_core>::sortVector(const Vmm& vToSort) {
    auto vAux1   = getVmm();
    auto vSrcTmp = getVmm();

    const int iterNum = dataElPerVec / 2 - 1;
    for (int i = 0; i < iterNum; i++) {
        vpshufd(vAux1, vToSort, 0B10110001);
        cmpPerm(vSrcTmp, vToSort, vAux1, kMaxMask0);

        vpermd(vAux1, vPermMask, vSrcTmp);
        cmpPerm(vToSort, vSrcTmp, vAux1, kMaxMask1);
    }
    vpshufd(vAux1, vToSort, 0B10110001);
    cmpPerm(vSrcTmp, vToSort, vAux1, kMaxMask0);

    uni_vmovups(vToSort, vSrcTmp);
}

template <x64::cpu_isa_t isa>
void UniqueKernel<isa>::sortVector(const Vmm& vToSort) {

}

//template <x64::cpu_isa_t isa>
//void UniqueKernel<isa>::spatialLoop() {
//    auto vHCoord = getVmm();
//    auto vWCoord = getVmm();
//
//    Xbyak::Label lSpacialLoop, lTail;
//    L(lSpacialLoop);
//    {
//        cmp(regWorkAmount, dataElPerVec);
//        jl(lTail, T_NEAR);
//
//        getCoordinates(vHCoord, vWCoord);
////        interpolation(vWCoord, vHCoord);
//
//        sub(regWorkAmount, dataElPerVec);
//        add(regDst, vlen);
//
//        jmp(lSpacialLoop, T_NEAR);
//    }
//
//    L(lTail);
//    vHCoord.release();
//    vWCoord.release();
//    tail();
//}

template <x64::cpu_isa_t isa>
void UniqueKernel<isa>::tail() {
//    Xbyak::Label lEnd;
//    cmp(regWorkAmount, 0);
//    jle(lEnd, T_NEAR);
//
//    auto vHCoord = getVmm();
//    auto vWCoord = getVmm();
//
////    getTailCoordinates(vHCoord, vWCoord);
//
//    if (dataTypeSize > 1)
//        sal(regWorkAmount, dataTypeShift); // Multiply by source data type size.
//    add(regDst, regWorkAmount);
//
//    L(lEnd);
}

template <>
void UniqueKernel<x64::avx512_core>::getCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
//    auto vAux = getVmm();
//
//    vpermd(vWCoord, vGridPermMask, ptr[regGrid]); // Permute to XXXX.XXXX.YYYY.YYYY
//    Xbyak::Ymm ymmH(vHCoord.getIdx());
//    vextractf64x4(ymmH, vWCoord, 1); // Extract Y component
//
//    add(regGrid, vlen);
//
//    vpermd(vAux, vGridPermMask, ptr[regGrid]); // Permute to XXXX.XXXX.YYYY.YYYY
//    Xbyak::Ymm ymmAux(vAux.getIdx());
//    vinsertf64x4(vWCoord, vWCoord, ymmAux, 1); // Extract X component
//    vextractf64x4(ymmAux, vAux, 1);            // Extract Y component
//    vinsertf64x4(vHCoord, vHCoord, ymmAux, 1);
//
//    add(regGrid, vlen);
}

template <>
void UniqueKernel<x64::avx2>::getCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
//    auto vAux = getVmm();
//    Vmm vPermMask;
//    RegistersPool::Reg<Vmm> permMaskHolder;
//
//    if (vGridPermMask.isInitialized()) {
//        vPermMask = vGridPermMask;
//    } else {
//        static const unsigned gridPermMask[8] = {0, 2, 4, 6, 1, 3, 5, 7};
//        auto rAux = getReg64();
//        permMaskHolder = getVmm();
//        vPermMask = permMaskHolder;
//        mov(rAux, reinterpret_cast<uintptr_t>(gridPermMask));
//        uni_vmovups(vPermMask, ptr[rAux]);
//    }
//
//    vpermd(vWCoord, vPermMask, ptr[regGrid]); // Permute to XXXX.YYYY
//    vperm2f128(vHCoord, vHCoord, vWCoord, 0B00000011);      // Extract Y component
//
//    add(regGrid, vlen);
//
//    vpermd(vAux, vPermMask, ptr[regGrid]);    // Permute to XXXX.YYYY
//    vperm2f128(vWCoord, vWCoord, vAux, 0B00100000);         // Extract X component
//    vperm2f128(vHCoord, vHCoord, vAux, 0B00110000);         // Extract Y component
//
//    add(regGrid, vlen);
}

template <x64::cpu_isa_t isa> // Works for AVX, SSE41
void UniqueKernel<isa>::getCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
//    auto vAux = getVmm();
//    Xbyak::Xmm xmmWCoord(vWCoord.getIdx());
//    Xbyak::Xmm xmmHCoord(vHCoord.getIdx());
//    Xbyak::Xmm xmmAux(vAux.getIdx());
//    const uint64_t xmmVlen = x64::cpu_isa_traits<x64::sse41>::vlen;
//
//    uni_vmovups(xmmWCoord, ptr[regGrid]);
//    uni_vpshufd(xmmWCoord, xmmWCoord, 0xD8);
//    shufpd(xmmHCoord, xmmWCoord, 0x2);
//
//    add(regGrid, xmmVlen);
//
//    uni_vmovups(xmmAux, ptr[regGrid]);
//    uni_vpshufd(xmmAux, xmmAux, 0xD8);
//    shufpd(xmmWCoord, xmmAux, 0x0);
//    shufpd(xmmHCoord, xmmAux, 0x3);
//
//    add(regGrid, xmmVlen);
//
//    if (isa == x64::avx) {
//        Xbyak::Ymm ymmWCoord(vWCoord.getIdx());
//        Xbyak::Ymm ymmHCoord(vHCoord.getIdx());
//
//        vperm2f128(ymmWCoord, ymmWCoord, ymmWCoord, 0x1);
//        vperm2f128(ymmHCoord, ymmHCoord, ymmHCoord, 0x1);
//
//        // Here is movups + pshufd instead of vpshufd for two reasons:
//        // 1. vpshufd zeroes the rest ov YMM.
//        // 2. pshufd does not work with not aligned address.
//        movups(xmmWCoord, ptr[regGrid]);
//        pshufd(xmmWCoord, xmmWCoord, 0xD8);
//        shufpd(xmmHCoord, xmmWCoord, 0x2);
//
//        add(regGrid, xmmVlen);
//
//        uni_vpshufd(xmmAux, ptr[regGrid], 0xD8);
//        shufpd(xmmWCoord, xmmAux, 0x0);
//        shufpd(xmmHCoord, xmmAux, 0x3);
//
//        add(regGrid, xmmVlen);
//
//        vperm2f128(ymmWCoord, ymmWCoord, ymmWCoord, 0x1);
//        vperm2f128(ymmHCoord, ymmHCoord, ymmHCoord, 0x1);
//    }
}

template <>
void UniqueKernel<x64::avx512_core>::getTailCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
//    Xbyak::Label lEnd, lGridShift, lRest;
//
//    auto vAux = getVmm();
//    auto rAux = getReg64();
//    Xbyak::Ymm ymmH(vHCoord.getIdx());
//
//    mov(rAux, regWorkAmount);
//    sal(rAux, 0x1); // multiply by gridShape[3]
//    cmp(regWorkAmount, dataElPerVec / 2);
//    jl(lRest, T_NEAR);
//    {
//        vpermd(vWCoord, vGridPermMask, ptr[regGrid]);
//        vextractf64x4(ymmH, vWCoord, 1); // Extract Y component
//
//        add(regGrid, vlen);
//        sub(rAux, dataElPerVec);
//        cmp(rAux, 0);
//        jle(lEnd, T_NEAR);
//
//        fillRestWorkMask(kTailMask, vAux, rAux);
//        uni_vmovups((Vmm)vAux | kTailMask, ptr[regGrid]);
//        vpermd(vAux, vGridPermMask, vAux);
//        Xbyak::Ymm ymmAux(vAux.getIdx());
//        vinsertf64x4(vWCoord, vWCoord, ymmAux, 1); // Extract X component
//        vextractf64x4(ymmAux, vAux, 1); // Extract Y component
//        vinsertf64x4(vHCoord, vHCoord, ymmAux, 1);
//
//        jmp(lGridShift, T_NEAR);
//    }
//    L(lRest);
//    {
//        fillRestWorkMask(kTailMask, vAux, rAux);
//        uni_vmovups(vWCoord | kTailMask, ptr[regGrid]);
//        vpermd(vWCoord, vGridPermMask, vWCoord);
//        vextractf64x4(ymmH, vWCoord, 1); // Extract Y component
//    }
//
//    L(lGridShift);
//    if (dataTypeSize > 1)
//        sal(rAux, dataTypeShift); // multiply by source data type size.
//    add(regGrid, rAux);
//
//    L(lEnd);
//
//    fillRestWorkMask(kTailMask, vAux, regWorkAmount);
}

template <>
void UniqueKernel<x64::avx2>::getTailCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
//    Xbyak::Label lRest, lGridShift, lEnd;
//
//    auto rAux = getReg64();
//    Vmm vPermMask;
//    RegistersPool::Reg<Vmm> permMaskHolder;
//
//    if (vGridPermMask.isInitialized()) {
//        vPermMask = vGridPermMask;
//    } else {
//        static const unsigned gridPermMask[8] = {0, 2, 4, 6, 1, 3, 5, 7};
//        permMaskHolder = getVmm();
//        vPermMask = permMaskHolder;
//        mov(rAux, reinterpret_cast<uintptr_t>(gridPermMask));
//        uni_vmovups(vPermMask, ptr[rAux]);
//    }
//
//    mov(rAux, regWorkAmount);
//    sal(rAux, 0x1); // multiply by gridShape[3] == 2
//    cmp(regWorkAmount, dataElPerVec / 2);
//    jl(lRest, T_NEAR);
//    {
//        vpermd(vWCoord, vPermMask, ptr[regGrid]);      // Permute to XXXX.YYYY
//        vperm2f128(vHCoord, vHCoord, vWCoord, 0B00000011); // Extract Y component
//
//        add(regGrid, vlen);
//        sub(rAux, dataElPerVec);
//        cmp(rAux, 0);
//        jle(lEnd, T_NEAR);
//
//        auto vAux  = getVmm();
//        load(vAux, ptr[regGrid], rAux, dataTypeSize);
//        vpermd(vAux, vPermMask, vAux);
//        vperm2f128(vWCoord, vWCoord, vAux, 0B00100000); // Extract X component
//        vperm2f128(vHCoord, vHCoord, vAux, 0B00110000); // Extract Y component
//
//        jmp(lGridShift, T_NEAR);
//    }
//    L(lRest);
//    {
//        load(vWCoord, ptr[regGrid], rAux, dataTypeSize);
//        vpermd(vWCoord, vPermMask, vWCoord);               // Permute to XXXX.YYYY
//        vperm2f128(vHCoord, vHCoord, vWCoord, 0B00000011); // Extract Y component
//    }
//
//    L(lGridShift);
//    if (dataTypeSize > 1)
//        sal(rAux, dataTypeShift); // Multiply by source data type size.
//    add(regGrid, rAux);
//
//    L(lEnd);
}

template <>
void UniqueKernel<x64::avx>::getTailCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
//    Xbyak::Label lLoop2End, lEnd;
//
//    Xbyak::Xmm xmmWCoord(vWCoord.getIdx());
//    Xbyak::Xmm xmmHCoord(vHCoord.getIdx());
//
//    auto rGridRest = getReg64();
//    mov(rGridRest, regWorkAmount);
//    sal(rGridRest, 0x1); // multiply by gridShape[3] == 2
//
//    for (int i = 0; i < dataElPerVec; i++) {
//        cmp(rGridRest, 0);
//        jle(lEnd, T_NEAR);
//
//        if (gridTypeSize == 4)
//            pinsrd(i % 2 == 0 ? xmmWCoord : xmmHCoord, ptr[regGrid], i / 2);
//        else if (gridTypeSize == 2)
//            pinsrw(i % 2 == 0 ? xmmWCoord : xmmHCoord, ptr[regGrid], i / 2);
//
//        add(regGrid, gridTypeSize);
//        dec(rGridRest);
//    }
//
//    cmp(rGridRest, 0);
//    jle(lEnd, T_NEAR);
//
//    vperm2f128(vWCoord, vWCoord, vWCoord, 0x1);
//    vperm2f128(vHCoord, vHCoord, vHCoord, 0x1);
//
//    for (int i = 0; i < dataElPerVec; i++) {
//        cmp(rGridRest, 0);
//        jle(lLoop2End, T_NEAR);
//
//        if (gridTypeSize == 4)
//            pinsrd(i % 2 == 0 ? xmmWCoord : xmmHCoord, ptr[regGrid], i / 2);
//        else if (gridTypeSize == 2)
//            pinsrw(i % 2 == 0 ? xmmWCoord : xmmHCoord, ptr[regGrid], i / 2);
//
//        add(regGrid, gridTypeSize);
//        dec(rGridRest);
//    }
//
//    L(lLoop2End);
//    vperm2f128(vWCoord, vWCoord, vWCoord, 0x1);
//    vperm2f128(vHCoord, vHCoord, vHCoord, 0x1);
//
//    L(lEnd);
}

template <>
void UniqueKernel<x64::sse41>::getTailCoordinates(const Vmm& vHCoord, const Vmm& vWCoord) {
//    Xbyak::Label lRest, lHShuf, lGridShift, lEnd;
//    auto rAux = getReg64();
//
//    mov(rAux, regWorkAmount);
//    sal(rAux, 0x1); // Multiply by gridShape[3] == 2
//    cmp(regWorkAmount, dataElPerVec / 2);
//    jl(lRest, T_NEAR);
//    {
//        // Here is movups + pshufd instead of pshufd due to
//        // pshufd does not work with not aligned address.
//        movups(vWCoord, ptr[regGrid]);
//        pshufd(vWCoord, vWCoord, 0B11011000);
//        shufpd(vHCoord, vWCoord, 0B00000010);
//
//        add(regGrid, vlen);
//        sub(rAux, dataElPerVec);
//        cmp(rAux, 0);
//        jle(lHShuf, T_NEAR);
//
//        auto vAux = getVmm();
//        load(vAux, ptr[regGrid], rAux, dataTypeSize);
//        pshufd(vAux, vAux, 0B11011000);
//        shufpd(vWCoord, vAux, 0x0);        // Extract X component
//        shufpd(vHCoord, vAux, 0B00000011); // Extract Y component
//
//        jmp(lGridShift, T_NEAR);
//        L(lHShuf);
//        shufpd(vHCoord, vHCoord, 0B00000001); // Extract Y component
//        jmp(lEnd, T_NEAR);
//    }
//    L(lRest);
//    {
//        load(vWCoord, ptr[regGrid], rAux, dataTypeSize);
//        pshufd(vWCoord, vWCoord, 0B11011000); // Extract X component
//        shufpd(vHCoord, vWCoord, 0B00000010); // Extract Y component
//        shufpd(vHCoord, vHCoord, 0B00000001);
//    }
//
//    L(lGridShift);
//    if (dataTypeSize > 1)
//        sal(rAux, dataTypeShift); // Multiply by source data type size.
//    add(regGrid, rAux);
//
//    L(lEnd);
}

template <x64::cpu_isa_t isa>
void UniqueKernel<isa>::dataTypeShiftPs2Dq(const Vmm& vDst, const Vmm& vSrc) {
    if (dataTypeSize == 1)
        return;

    if (isa == x64::avx) { // vpslld works just with XMM for AVX, so use vmulps for YMM
        auto rAux = getReg64();
        static const float val = dataTypeSize;
        static const float dataTypeSizeArr[8] = {val, val, val, val, val, val, val, val};
        mov(rAux, reinterpret_cast<uintptr_t>(dataTypeSizeArr));
        uni_vmulps(vDst, vSrc, ptr[rAux]);
        uni_vcvtps2dq(vDst, vDst);
    } else {
        uni_vcvtps2dq(vDst, vSrc);
        if (dataTypeSize > 1)
            uni_vpslld(vDst, vDst, dataTypeShift); // multiply by source data type size.
    }
}

template <x64::cpu_isa_t isa>
void UniqueKernel<isa>::hwShiftPs2dq(const Vmm& vDst, const Vmm& vHCoord, const Vmm& vWCoord, const Vmm& vWidth) {
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

template class UniqueKernel<x64::avx512_core>;
template class UniqueKernel<x64::avx2>;
template class UniqueKernel<x64::avx>;
template class UniqueKernel<x64::sse41>;

}   // namespace intel_cpu
}   // namespace ov
