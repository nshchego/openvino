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

//    for (int i = 0; i < 3; i++) {
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
    Xbyak::Reg32 rMask(rAux.getIdx()); // TODO use r64

    kMask0 = getMask();
    mov(rMask, 0B0101010101010101);
    kmovw(kMask0, rMask);

    kMask1 = getMask();
    mov(rMask, 0B1010101010101010);
    kmovw(kMask1, rMask);

    kMaskMinLast = getMask();
    mov(rMask, 0B0010101010101010);
    kmovw(kMaskMinLast, rMask);

    kMaskMaxFirst = getMask();
    mov(rMask, 0B0101010101010100);
    kmovw(kMaskMaxFirst, rMask);

    kFirstElMask = getMask();
    mov(rMask, 0B0000000000000001);
    kmovw(kFirstElMask, rMask);

    kLastElMask = getMask();
    mov(rMask, 0B1000000000000000);
    kmovw(kLastElMask, rMask);

    kTailMask = getMask();
    mov(rMask, 0xFFFFFFFF);
    kmovw(kTailMask, rMask);

    static const unsigned permElem[16]  = { 15, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 0 };
    mov(rAux, reinterpret_cast<uintptr_t>(permElem));
    vPermElem = getVmm();
    uni_vmovups(vPermElem, ptr[rAux]);

    const auto vecNum = registersPool->countFree<Vmm>() - 3;//2;
    for (int i = 0; i < vecNum; i++) {
        contiguousVec.push_back(getVmm());
    }

//    if (one_of(jcp.interpolationMode, InterpolationMode::BICUBIC, InterpolationMode::BILINEAR)) {
//        vOnesF = getVmm();
//        mov(r32Aux, 0x3f800000); // 1.f
//        vpbroadcastd(vOnesF, r32Aux);
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
    Xbyak::Label lLoop, lFinishLoad, lFinishStore, lTail;
    const auto& rDstSorted = regDst[UNIQUE_DATA];

    auto rBlocksNum   = getReg64();
    auto rBlockLenPtr = getReg64();
    auto rBlockLen    = getReg64();

    mov(rBlocksNum,   ptr[regParams + GET_OFF(blocksNum)]);
    mov(rBlockLenPtr, ptr[regParams + GET_OFF(blockLen)]);

    // SORT IN BLOCKS
    // Loop over contiguous vectors.
    L(lLoop);
    {
        cmp(rBlocksNum, 0);
        jle(lTail, T_NEAR);

        // Load to contiguous vector.
        mov(rBlockLen, ptr[rBlockLenPtr]);
        for (int v = 0; v < contiguousVec.size(); v++) {
            Xbyak::Label lLoadNext, lLoadTail;
            cmp(rBlockLen, dataElPerVec * (v + 1));
            jl(lLoadTail, T_NEAR);

            const auto& vec = contiguousVec[v];
            uni_vmovups(vec, ptr[regSrc]);

            add(regSrc, vlen);
            jmp(lLoadNext, T_NEAR);

            L(lLoadTail);
            {
                cmp(rBlockLen, dataElPerVec * v);
                je(lFinishLoad, T_NEAR);

                auto rRest = getReg64();
                mov(rRest, rBlockLen);
                sub(rRest, dataElPerVec * v);
                fillRestWorkMask(Xbyak::Opmask(kTailMask.getIdx()), rRest, dataTypeSize);
                uni_vmovups((Vmm) vec | Xbyak::Opmask(kTailMask.getIdx()), ptr[regSrc]);
                imul(rRest, rRest, dataTypeSize);
                add(regSrc, rRest);
                jmp(lFinishLoad, T_NEAR);
            }

            L(lLoadNext);
        }
        L(lFinishLoad);

        sortContiguousVec(rBlockLen);

        // Store from contiguous vector.
        // TODO: Optimization. Do unique search without storing if all work is fitted into one block.
        for (int v = 0; v < contiguousVec.size(); v++) {
            Xbyak::Label lStoreNext, lStoreTail;
            cmp(rBlockLen, dataElPerVec * (v + 1));
            jl(lStoreTail, T_NEAR);

            const auto& vec = contiguousVec[v];
            uni_vmovups(ptr[rDstSorted], vec);

            add(rDstSorted, vlen);
            jmp(lStoreNext, T_NEAR);

            L(lStoreTail);
            {
                cmp(rBlockLen, dataElPerVec * v);
                je(lFinishStore, T_NEAR);

                uni_vmovups(ptr[rDstSorted] | Xbyak::Opmask(kTailMask.getIdx()), vec);
                sub(rBlockLen, dataElPerVec * v);
                imul(rBlockLen, rBlockLen, dataTypeSize);
                add(rDstSorted, rBlockLen);
                jmp(lFinishStore, T_NEAR);
            }

            L(lStoreNext);
        }
        L(lFinishStore);
        // ALL BLOCKS SORTED

        // Gather samples { w/Nb + j*W, 2*w/Nb + j*w, ... , (Np - 1)*w/Np + j*w },
        // where W - total kernel's work, Nb - blocks num, w = W/Nb, 0 <= j < Nb.


        // Sort samples.

        // Gather samples { Nb/2, Nb + Nb/2, ... , (Nb - 2)*p + p/2 },

        dec(rBlocksNum);
        add(rBlockLenPtr, sizeof(int64_t));
        jmp(lLoop, T_NEAR);
    }

    L(lTail);
    tail();
}

template <>
void UniqueKernel<x64::avx512_core>::alignTailMask(const Vmask& kDst, const Vmask& kSrc, bool even) {
    Xbyak::Label lEnd, lCopy;
    auto rAux = getReg64();

    kmovq(rAux, kSrc);
    popcnt(rAux, rAux);
    and_(rAux, 0x1);
    cmp(rAux, 0);
    je(kDst.getIdx() != kSrc.getIdx() ? lCopy : lEnd, T_NEAR);

    kshiftrq(kDst, kSrc, 0x1);
    jmp(lEnd, T_NEAR);
    L(lCopy);
    kmovq(kDst, kSrc);
    L(lEnd);
}

template <x64::cpu_isa_t isa>
void UniqueKernel<isa>::alignTailMask(const Vmask& kDst, const Vmask& kSrc, bool even) {

        }

template <>
void UniqueKernel<x64::avx512_core>::cmpPerm(const Vmm& vDst, const Vmm& vSrc1, const Vmm& vSrc2, const Vmask& kMinMask, const Vmask& kMaxMask, bool tail) {
    if (jcp.dataPrc == Precision::FP32) {
        vminps(vDst | kMinMask, vSrc1, vSrc2);
        vmaxps(vDst | kMaxMask, vSrc1, vSrc2);
    } else if (jcp.dataPrc == Precision::I32) {
        vpminsd(vDst | kMinMask, vSrc1, vSrc2);
        vpmaxsd(vDst | kMaxMask, vSrc1, vSrc2);
    }  else if (jcp.dataPrc == Precision::U8) {
        vpminub(vDst | kMinMask, vSrc1, vSrc2);
        vpmaxub(vDst | kMaxMask, vSrc1, vSrc2);
    } else if (jcp.dataPrc == Precision::I8) {
        vpminsb(vDst | kMinMask, vSrc1, vSrc2);
        vpmaxsb(vDst | kMaxMask, vSrc1, vSrc2);
    }
}

template <x64::cpu_isa_t isa>
void UniqueKernel<isa>::cmpPerm(const Vmm& vDst, const Vmm& vSrc1, const Vmm& vSrc2, const Vmask& kMinMask, const Vmask& kMaxMask, bool tail) {

}

template <>
void UniqueKernel<x64::avx512_core>::permOnEdge(const Vmm& vSrc1, const Vmm& vSrc2, const Vmm& vOrigin1) {
    if (jcp.dataPrc.size() == 4) {
        uni_vmovups(vSrc1 | kLastElMask, vSrc2);
        vpermd(vSrc2 | kFirstElMask, vPermElem, vOrigin1);
    } else if (jcp.dataPrc.size() == 1) {
        vmovdqu8(vSrc1 | kLastElMask, vSrc2);
        vmovdqu8(vSrc2 | kFirstElMask, vSrc1);
    }
}

template <x64::cpu_isa_t isa>
void UniqueKernel<isa>::permOnEdge(const Vmm& vSrc1, const Vmm& vSrc2, const Vmm& vOrigin1) {

}

template <>
void UniqueKernel<x64::avx512_core>::sortContiguousVec(const Xbyak::Reg64& rBlockLen) {
    Xbyak::Label lFew, lEnd;
    auto vAux1 = getVmm();

    cmp(rBlockLen, dataElPerVec);
    jg(lFew, T_NEAR);
    {
        const int iterNum = dataElPerVec / 2 - 1;
        const auto &vToSort = contiguousVec[0];
        for (int i = 0; i < iterNum; i++) {
            vpshufd(vAux1, vToSort, 0B10110001);
            cmpPerm(vToSort, vToSort, vAux1, kMask0, kMask1);

            vpermd(vAux1, vPermElem, vToSort);
            cmpPerm(vToSort, vToSort, vAux1, kMaskMinLast, kMaskMaxFirst);
        }
        vpshufd(vAux1, vToSort, 0B10110001);
        cmpPerm(vToSort, vToSort, vAux1, kMask0, kMask1);

        jmp(lEnd, T_NEAR);
    }
    L(lFew);
    {
        Xbyak::Label lLastCmp;
        const int iterNum = (contiguousVec.size() * dataElPerVec) / 2 - 1;
        auto vAux2 = getVmm();

        for (int i = 0; i < iterNum; i++) {
            cmp(rBlockLen, 2 * (i + 1));
            jle(lLastCmp, T_NEAR);
            Xbyak::Label lSecond, lNext;

            // Compare and permute pairs {0;1}{2;3}...{14;15}
            for (int v = 0; v < contiguousVec.size(); v++) {
                Xbyak::Label lNext1, lTail1;
                const auto &vToSort = contiguousVec[v];
                cmp(rBlockLen, dataElPerVec * (v + 1));
                jl(lTail1, T_NEAR);

                vpshufd(vAux1, vToSort, 0B10110001);
                cmpPerm(vToSort, vToSort, vAux1, kMask0, kMask1);
                jmp(lNext1, T_NEAR);

                L(lTail1);
                {
                    cmp(rBlockLen, dataElPerVec * v);
                    je(lSecond, T_NEAR);

                    vpshufd(vAux1, vToSort, 0B10110001);
                    kmovw(k0, kTailMask);
                    alignTailMask(kTailMask, k0, true);
                    kandw(kTailMask, kTailMask, kMask0);
                    cmpPerm(vToSort, vToSort, vAux1, kTailMask, kMask1, false);
                    kmovw(kTailMask, k0);
                    jmp(lSecond, T_NEAR);
                }

                L(lNext1);
            }

            L(lSecond);
            // Compare and permute pairs {15';0}{1;2}...{13;14}{15;0'}, where n' are values form neighbor vectors.
            const auto& vFirst = contiguousVec[0];
            vpermd(vAux1, vPermElem, vFirst);
            for (int v = 0; v < contiguousVec.size() - 1; v++) {
                Xbyak::Label lLast, lNext2;
                cmp(rBlockLen, dataElPerVec * (v + 1));
                jle(lLast, T_NEAR);

                const auto& vCurr = contiguousVec[v];
                const auto& vNext = contiguousVec[v + 1];
                const auto& vCurrAux = v % 2 == 0 ? vAux1 : vAux2;
                const auto& vNextAux = v % 2 == 0 ? vAux2 : vAux1;

                vpermd(vNextAux, vPermElem, vNext);
                permOnEdge(vCurrAux, vNextAux, vCurr);
                cmpPerm(vCurr, vCurr, vCurrAux, kMask1, v == 0 ? kMaskMaxFirst : kMask0);
                jmp(lNext2, T_NEAR);

                L(lLast);
                {
                    kmovw(k0, kTailMask);
                    alignTailMask(kTailMask, k0, false);
                    kandw(kTailMask, kTailMask, kMaskMinLast);
                    cmpPerm(vCurr, vCurr, vCurrAux, kTailMask, kMask0);
                    kmovw(kTailMask, k0);
                    jmp(lNext, T_NEAR);
                }

                L(lNext2);
            }

            L(lNext);
        }

        L(lLastCmp);
        for (int v = 0; v < contiguousVec.size(); v++) {
            Xbyak::Label lNext3, lTail3;
            const auto &vToSort = contiguousVec[v];
            cmp(rBlockLen, dataElPerVec * (v + 1));
            jl(lTail3, T_NEAR);

            vpshufd(vAux1, vToSort, 0B10110001);
            cmpPerm(vToSort, vToSort, vAux1, kMask0, kMask1);
            jmp(lNext3, T_NEAR);

            L(lTail3);
            {
                cmp(rBlockLen, dataElPerVec * v);
                je(lEnd, T_NEAR);

                vpshufd(vAux1, vToSort, 0B10110001);
                cmpPerm(vToSort, vToSort, vAux1, kMask0, kMask1, true);
                jmp(lEnd, T_NEAR);
            }

            L(lNext3);
        }
    }

    L(lEnd);
}

template <x64::cpu_isa_t isa>
void UniqueKernel<isa>::sortContiguousVec(const Xbyak::Reg64& regVecNum) {

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


template class UniqueKernel<x64::avx512_core>;
template class UniqueKernel<x64::avx2>;
template class UniqueKernel<x64::avx>;
template class UniqueKernel<x64::sse41>;

}   // namespace intel_cpu
}   // namespace ov
