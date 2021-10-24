// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_uni_kernel.hpp"

using namespace mkldnn::impl::cpu;

namespace MKLDNNPlugin {

const unsigned jitGatherKernelBase::shufMask8bitUni[16]  = {0x0C080400, 0x80808080, 0x80808080, 0x80808080, 0x0C080400, 0x80808080, 0x80808080, 0x80808080,
                                                            0x0C080400, 0x80808080, 0x80808080, 0x80808080, 0x0C080400, 0x80808080, 0x80808080, 0x80808080};
const unsigned jitGatherKernelBase::permMask8bitA2[8]    = {0, 4, 1, 5, 2, 6, 3, 7};
const unsigned jitGatherKernelBase::permMask8bitA5[16]   = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};

const unsigned jitGatherKernelBase::shufMask16bitUni[16] = {0x05040100, 0x0D0C0908, 0x80808080, 0x80808080, 0x05040100, 0x0D0C0908, 0x80808080, 0x80808080,
                                                            0x05040100, 0x0D0C0908, 0x80808080, 0x80808080, 0x05040100, 0x0D0C0908, 0x80808080, 0x80808080};
const unsigned jitGatherKernelBase::permMask16bitA2[8]   = {0, 1, 4, 5, 2, 3, 6, 7};
const unsigned jitGatherKernelBase::permMask16bitA5[16]  = {0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15};

const unsigned jitGatherKernelBase::incVec[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

#define GET_OFF(field) offsetof(gatherJitExecArgs, field)

template <x64::cpu_isa_t isa>
jitUniGatherKernel<isa>::jitUniGatherKernel(jGatherConfParams jcp) :
        jitGatherKernelBase(jcp), x64::jit_generator() {
    vlen = x64::cpu_isa_traits<isa>::vlen;
    dataElPerVec = vlen / jcp.dataTypeSize;
    idxElPerVec = vlen / indicesTypeSize;
    if (jcp.dataTypeSize == 2)
        dataTypeShift = 1;
    else if (jcp.dataTypeSize == 4)
        dataTypeShift = 2;

    if (isa == x64::avx2) {
        permMask8bitUni = permMask8bitA2;
        permMask16bitUni = permMask16bitA2;
    } else if (isa == x64::avx512_common) {
        permMask8bitUni = permMask8bitA5;
        permMask16bitUni = permMask16bitA5;
    }
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::create_ker() {
    x64::jit_generator::create_kernel();
    ker_ = (decltype(ker_))jit_ker();
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::generate() {
    this->preamble();

    mov(regSrc, ptr[regParams + GET_OFF(src)]);
    mov(regDst, ptr[regParams + GET_OFF(dst)]);
    mov(regIndices, ptr[regParams + GET_OFF(indices)]);

    auto& vmmAux0 = vmmAuxContainer[0];
    auto& vmmAux1 = vmmAuxContainer[1];
    auto& xmmAux0 = xmmAuxContainer[0];
    auto& xmmAux1 = xmmAuxContainer[1];

//    if (jcp.dynamicShape) {
        mov(regAux1, ptr[regParams + GET_OFF(start)]);
        uni_vpbroadcastd(vmmSpecIndicesInBytes, ptr[regAux1]);
        mov(regAux1, reinterpret_cast<uintptr_t>(incVec));
        uni_vpaddd(vmmSpecIndicesInBytes, vmmSpecIndicesInBytes, ptr[regAux1]);
        uni_vcvtdq2ps(vmmSpecIndicesInBytes, vmmSpecIndicesInBytes);

        // specIndices = (start % specIndicesSize) * idxTypeSize
        mov(regAux1, ptr[regParams + GET_OFF(specIndicesSize)]);
        uni_vpbroadcastd(vmmSpecIdxSizeInBytes, ptr[regAux1]);
        uni_vcvtdq2ps(vmmAux1, vmmSpecIdxSizeInBytes);
        uni_vdivps(vmmSrcBeforeAxisSum, vmmSpecIndicesInBytes, vmmAux1);
        uni_vroundps(vmmSrcBeforeAxisSum, vmmSrcBeforeAxisSum, 0x1);
        vfnmadd231ps(vmmSpecIndicesInBytes, vmmSrcBeforeAxisSum, vmmAux1);
        uni_vcvtps2dq(vmmSpecIndicesInBytes, vmmSpecIndicesInBytes);
        uni_vpslld(vmmSpecIndicesInBytes, vmmSpecIndicesInBytes, 2); // multiply by indices type size.
        uni_vpslld(vmmSpecIdxSizeInBytes, vmmSpecIdxSizeInBytes, 2); // multiply by indices type size.
        vmovd(reg32SpecIdxSizeInBytes, xmmSpecIdxSizeInBytes);

        mov(regAux1, ptr[regParams + GET_OFF(betweenBatchAndAxisSize)]);
        uni_vpbroadcastd(vmmAux1, ptr[regAux1]);
        vmovd(reg32BetweenBatchAndAxisSize, xmmAux1);
        uni_vcvtdq2ps(vmmAux1, vmmAux1);
        uni_vdivps(vmmIdxBatchSumInBytes, vmmSrcBeforeAxisSum, vmmAux1);
        uni_vroundps(vmmIdxBatchSumInBytes, vmmIdxBatchSumInBytes, 0x1);
        vfnmadd231ps(vmmSrcBeforeAxisSum, vmmIdxBatchSumInBytes, vmmAux1);
        uni_vcvtps2dq(vmmSrcBeforeAxisSum, vmmSrcBeforeAxisSum);
        vmovd(reg32BetweenBatchAndAxisIter, xmmSrcBeforeAxisSum);
        uni_vcvtps2dq(vmmIdxBatchSumInBytes, vmmIdxBatchSumInBytes);
uni_vmovups(ptr[regDst], vmmIdxBatchSumInBytes);

        mov(regAux1, ptr[regParams + GET_OFF(axisAndAfterAxisSizeInBytes)]);
        uni_vpbroadcastd(vmmAxisAndAfterAxisSize, ptr[regAux1]);
        uni_vpmulld(vmmSrcBeforeAxisSum, vmmSrcBeforeAxisSum, vmmAxisAndAfterAxisSize);
        mov(regAux1, ptr[regParams + GET_OFF(srcAfterBatchSizeInBytes)]);
        uni_vpbroadcastd(vmmAux0, ptr[regAux1]);
        uni_vpmulld(vmmAux0, vmmAux0, vmmIdxBatchSumInBytes);
        // srcBeforeAxisSum = ((start / specIndicesSize) % betweenBatchAndAxis) * axisAndAfterAxisSize + srcAfterBatchSize * idxBatchSum
        uni_vpaddd(vmmSrcBeforeAxisSum, vmmSrcBeforeAxisSum, vmmAux0);

        // idxBatchSum = specIdxSize * (start / afterBatchSize)
        uni_vpmulld(vmmIdxBatchSumInBytes, vmmIdxBatchSumInBytes, vmmSpecIdxSizeInBytes);

        mov(regAux1, ptr[regParams + GET_OFF(axisDim)]);
        uni_vpbroadcastd(vmmAxisDim, ptr[regAux1]);
//    } else {
//        mov(regAux1, ptr[regParams + GET_OFF(specIndicesSize)]);
//        uni_vpbroadcastd(vmmSpecIdxSizeInBytes, ptr[regAux1]);
//        uni_vpslld(vmmSpecIdxSizeInBytes, vmmSpecIdxSizeInBytes, 2); // multiply by indices type size.
//        mov(regAux1, ptr[regParams + GET_OFF(axisAndAfterAxisSizeInBytes)]);
//        uni_vpbroadcastd(vmmAxisAndAfterAxisSize, ptr[regAux1]);
//        mov(regAux1, ptr[regParams + GET_OFF(axisDim)]);
//        uni_vpbroadcastd(vmmAxisDim, ptr[regAux1]);
//
//        mov(regAux1, ptr[regParams + GET_OFF(specIndicesInBytes)]);
//        uni_vmovups(vmmSpecIndicesInBytes, ptr[regAux1]);
//        mov(regAux1, ptr[regParams + GET_OFF(idxBatchSumInBytes)]);
//        uni_vmovups(vmmIdxBatchSumInBytes, ptr[regAux1]);
//        mov(regAux1, ptr[regParams + GET_OFF(srcBeforeAxisSum)]);
//        uni_vmovups(vmmSrcBeforeAxisSum, ptr[regAux1]);
//    }

    mov(regWorkAmount, ptr[regParams + GET_OFF(workAmount)]);

    uni_vpxor(vmmZeros, vmmZeros, vmmZeros);

    Xbyak::Label lBlock, lEnd;
    mov(regAux2, ptr[regParams + GET_OFF(afterAxSize)]);
    cmp(regAux2, 1);
    jg(lBlock, T_NEAR);
    {
        Xbyak::Label lLessThanVector1, lTail1, lTail2, lE1;

        cmp(regSpecIdxSizeInBytes, vlen);
        jl(lLessThanVector1, T_NEAR);
            vmovd(reg32IdxIter, xmmSpecIndicesInBytes);
            mov(regAux1, reinterpret_cast<uintptr_t>(&vlen));
            uni_vpbroadcastd(vmmVecLen, ptr[regAux1]); // TODO: uni_vpslld ?
            cmp(regWorkAmount, dataElPerVec);
            jl(lTail1, T_NEAR);

            if (jcp.dataTypeSize == 4)
                process32b(false, false);
            else if (jcp.dataTypeSize == 2)
                process16b(false, false);
            else if (jcp.dataTypeSize == 1)
                process8b(false, false);
            jmp(lE1, T_NEAR);

            L(lTail1);
            tail(false, false);
            jmp(lE1, T_NEAR);
        L(lLessThanVector1);
            mov(regAux1, ptr[regParams + GET_OFF(permIdx)]);
            uni_vmovups(vmmPermIdxMask, ptr[regAux1]);
            mov(regAux1, ptr[regParams + GET_OFF(beforeAxisDiff)]);
            uni_vmovups(vmmBeforeAxisDiff, ptr[regAux1]);
            if (jcp.dataTypeSize != 1)
                vpslld(vmmBeforeAxisDiff, vmmBeforeAxisDiff, dataTypeShift); // multiply by data type size
            mov(regAux1, ptr[regParams + GET_OFF(srcAfterBatchSizeInBytes)]);
            uni_vpbroadcastd(vmmSrcAfterBatchSize, ptr[regAux1]);
            cmp(regWorkAmount, dataElPerVec);
            jl(lTail2, T_NEAR);

            if (jcp.dataTypeSize == 4)
                process32b(true, false);
            else if (jcp.dataTypeSize == 2)
                process16b(true, false);
            else if (jcp.dataTypeSize == 1)
                process8b(true, false);
            jmp(lE1, T_NEAR);
            L(lTail2);
            tail(true, false);
        L(lE1);
        jmp(lEnd, T_NEAR);
    }
    L(lBlock);
    {
        mov(regAux1, ptr[regParams + GET_OFF(start)]);
        uni_vpbroadcastd(vmmBlockIdxInBytes, ptr[regAux1]);
        mov(regAux1, reinterpret_cast<uintptr_t>(incVec));
        uni_vpaddd(vmmBlockIdxInBytes, vmmBlockIdxInBytes, ptr[regAux1]);
        uni_vcvtdq2ps(vmmBlockIdxInBytes, vmmBlockIdxInBytes);

        // afterAxIdxInBytes = (start % afterAxSize) * idxTypeSize
        movd(xmmAux0, reg32Aux1);
        uni_vpbroadcastd(vmmAux1, xmmAux0); // A5 broadcast from reg32Aux1
        uni_vcvtdq2ps(vmmAux1, vmmAux1);
        uni_vdivps(vmmSrcBeforeAxisSum, vmmBlockIdxInBytes, vmmAux1);
        uni_vroundps(vmmSrcBeforeAxisSum, vmmSrcBeforeAxisSum, 0x1);
        vfnmadd231ps(vmmBlockIdxInBytes, vmmSrcBeforeAxisSum, vmmAux1);
        uni_vcvtps2dq(vmmBlockIdxInBytes, vmmBlockIdxInBytes);
vmovups(ptr[regDst], vmmBlockIdxInBytes);
        uni_vpslld(vmmBlockIdxInBytes, vmmBlockIdxInBytes, 2); // multiply by indices type size.

        Xbyak::Label lLessThanVector2, lTail3, lTail4, lE2;

        cmp(regAux2, dataElPerVec);
        jl(lLessThanVector2, T_NEAR);
            vmovd(reg32IdxIter, xmmSpecIndicesInBytes);
            mov(regAux1, reinterpret_cast<uintptr_t>(&vlen));
            uni_vpbroadcastd(vmmVecLen, ptr[regAux1]);
            cmp(regWorkAmount, dataElPerVec);
            jl(lTail3, T_NEAR);

//            if (jcp.dataTypeSize == 4)
//                processBlock32b(false, true);
//            else if (jcp.dataTypeSize == 2)
//                process16b(false);
//            else if (jcp.dataTypeSize == 1)
//                process8b(false);
//            jmp(lE2, T_NEAR);

            L(lTail3);
//            tail(false, false);
            jmp(lE2, T_NEAR);
        L(lLessThanVector2);
            auto& vmmAux2 = vmmAuxContainer[2];
            // Calculate permute mask
            vmovd(xmmAux0, reg32Aux2);
            uni_vpbroadcastd(vmmAux1, xmmAux0);
            mov(regAux1, reinterpret_cast<uintptr_t>(&idxElPerVec));
            uni_vpbroadcastd(vmmAux0, ptr[regAux1]);
            uni_vpsubd(vmmPermIdxMask, vmmAux0, vmmAux1);
            mov(regAux1, reinterpret_cast<uintptr_t>(incVec));
            uni_vpaddd(vmmPermIdxMask, vmmPermIdxMask, ptr[regAux1]);
            for (int i = 0; i < 2; i++) { // two cycles is enough
                vpcmpgtd(vmmAux2, vmmAux0, vmmPermIdxMask); // TODO A5
                vpandn(vmmAux2, vmmAux2, vmmAux1);
                uni_vpsubd(vmmPermIdxMask, vmmPermIdxMask, vmmAux2);
            }

//            mov(regAux1, ptr[regParams + GET_OFF(beforeAxisDiff)]);
//            uni_vmovups(vmmBeforeBlockDiff, ptr[regAux1]);
//uni_vmovups(ptr[regDst], vmmBeforeBlockDiff);
//            if (jcp.dataTypeSize != 1)
//                vpslld(vmmBeforeBlockDiff, vmmBeforeBlockDiff, dataTypeShift); // multiply by data type size
//            uni_vpmulld(vmmBeforeAxisDiff, vmmBeforeBlockDiff, vmmAxisDim);
//            mov(regAux1, ptr[regParams + GET_OFF(srcAfterBatchSizeInBytes)]);
//            uni_vpbroadcastd(vmmSrcAfterBatchSize, ptr[regAux1]);
            cmp(regWorkAmount, dataElPerVec);
            jl(lTail4, T_NEAR);

//            if (jcp.dataTypeSize == 4)
//                process32b(true, true);
//            else if (jcp.dataTypeSize == 2)
//                process16b(true);
//            else if (jcp.dataTypeSize == 1)
//                process8b(true);
//            jmp(lE2, T_NEAR);
            L(lTail4);
//            tail(true, false);
        L(lE2);
//        jmp(lEnd, T_NEAR);
    }
    L(lEnd);

    this->postamble();
}

template <>
void jitUniGatherKernel<x64::avx2>::uni_vpgatherdd(Vmm& vDst, const Xbyak::Address& srcAddr, Vmask& vMask) {
    vpgatherdd(vDst, srcAddr, vMask);
}

template <>
void jitUniGatherKernel<x64::avx512_common>::uni_vpgatherdd(Vmm& vDst, const Xbyak::Address& srcAddr, Vmask& vMask) {
    vpgatherdd(vDst | vMask, srcAddr);
}

template <>
void jitUniGatherKernel<x64::avx2>::uni_vpcmpeqd(Vmask& vMask, Vmm& vOp0, Vmm& vOp2) {
    vpcmpeqd(vMask, vOp0, vOp2);
}

template <>
void jitUniGatherKernel<x64::avx512_common>::uni_vpcmpeqd(Vmask& vMask, Vmm& vOp0, Vmm& vOp2) {
    vpcmpd(vMask, vOp0, vOp2, 0);
}

template <>
void jitUniGatherKernel<x64::avx2>::normalizeRawIndices(Vmm& rawIndices, Vmask& dstMask, Vmask& auxMask) {
    // Compensate negative indices.
    if (jcp.reverseIndexing) {
        vpcmpgtd(auxMask, vmmZeros, rawIndices);
        vpand(auxMask, auxMask, vmmAxisDim);
        uni_vpaddd(rawIndices, rawIndices, auxMask);
    }
    // Check boundaries.
    vpcmpgtd(dstMask, vmmAxisDim, rawIndices);
    vpcmpgtd(auxMask, vmmZeros, rawIndices);
    vpandn(dstMask, auxMask, dstMask);
    // Multiply by type size.
    if (jcp.dataTypeSize != 1)
        vpslld(rawIndices, rawIndices, dataTypeShift);
}

template <>
void jitUniGatherKernel<x64::avx512_common>::normalizeRawIndices(Vmm& rawIndices, Vmask& dstMask, Vmask& auxMask) {
    // Compensate negative indices.
    if (jcp.reverseIndexing) {
        vpcmpgtd(auxMask, vmmZeros, rawIndices);
        vpaddd(rawIndices | auxMask, rawIndices, vmmAxisDim);
    }
    // Check boundaries.
    vpcmpgtd(auxMask, vmmAxisDim, rawIndices);
    vpcmpd(dstMask | auxMask, vmmZeros, rawIndices, 2); // 2 - LE
    // Multiply by type size.
    if (jcp.dataTypeSize != 1)
        vpslld(rawIndices, rawIndices, dataTypeShift);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::tail(bool isShortIdx, bool shiftFirst) {
    auto& vmmAux0 = vmmAuxContainer[0];
    auto& vmmAux1 = vmmAuxContainer[2];
    auto& vmmGatherMask = vmmAuxContainer[3];
    auto& vGatherMask = masksContainer[vmmGatherMask.getIdx()];
    auto  kGatherMask = Xbyak::Opmask(vmmGatherMask.getIdx());
    auto& vAuxMask0 = masksContainer[vmmAux0.getIdx()];
    auto  kAuxMask0 = Xbyak::Opmask(vmmAux0.getIdx());
    Xbyak::Label lEnd;

    const int secondStepCycles = jcp.dataTypeSize == 4 ? 1 : jcp.dataTypeSize == 2 ? 2 : 4;
    for (int p = 0; p < secondStepCycles; p++) {
        cmp(regWorkAmount, 0);
        jle(lEnd, T_NEAR);

        // calcSrcShift -> vmmAuxContainer[1]
        if (isShortIdx)
            calcSrcShiftShort(vmmAuxContainer, vGatherMask, p > 0 || shiftFirst);
        else
            calcSrcShiftLong(vmmAuxContainer, vGatherMask, p > 0 || shiftFirst);
        fillRestWorkMask(vmmAux1, vmmAux0, regWorkAmount, regAux1, rdx);
        if (isa == x64::avx2) {
            vpand(vmmGatherMask, vmmGatherMask, vmmAux1);
            if (jcp.dataTypeSize == 4)
                vmovups(vmmAux0, vmmGatherMask);
        } else if (isa == x64::avx512_common) {
            vpmovd2m(kAuxMask0, vmmAux1);
            kandd(kGatherMask, kGatherMask, kAuxMask0);
            if (jcp.dataTypeSize == 4)
                kmovd(kAuxMask0, kGatherMask);
        }

        uni_vmovups(vmmAux1, vmmZeros);
        uni_vpgatherdd(vmmAux1, ptr[regSrc + vmmAuxContainer[1]], vGatherMask);
        if (jcp.dataTypeSize == 4) {
            uni_vmovups_tail(ptr[regDst], vAuxMask0, vmmAux1);
            sub(regWorkAmount, dataElPerVec);
        } else {
            storeScalar(regDst, regWorkAmount, vmmAux1, vmmAux0);
        }
    }
    L(lEnd);
}

// Requires vAuxPool length 3.
// Returns gathered data in vAuxPool[1].
template <>
void jitUniGatherKernel<x64::avx2>::calcSrcShiftLong(Vmm* vAuxPool, Vmask& dstMask, bool shiftFirst) {
    auto& vmmDstShifts = vAuxPool[1];
    auto& vmmAux0 = vAuxPool[0];
    auto& vmmAux1 = vAuxPool[2];
    auto& vAuxMask0 = masksContainer[vmmAux1.getIdx()];

    Xbyak::Label lIdxStride, lExit;
    if (shiftFirst)
        uni_vpaddd(vmmSpecIndicesInBytes, vmmSpecIndicesInBytes, vmmVecLen);

    add(regIdxIter, vlen);
    cmp(regIdxIter, regSpecIdxSizeInBytes);
    jge(lIdxStride, T_NEAR);
        auto& xmmAux = xmmAuxContainer[vmmDstShifts.getIdx()];
        uni_vpaddd(vmmDstShifts, vmmIdxBatchSumInBytes, vmmSpecIndicesInBytes);
        vmovd(reg32Aux1, xmmAux);
        vmovdqu(vmmDstShifts, ptr[regIndices + regAux1]);
        normalizeRawIndices(vmmDstShifts, dstMask, vAuxMask0);
        uni_vpaddd(vmmDstShifts, vmmDstShifts, vmmSrcBeforeAxisSum);
    jmp(lExit, T_NEAR);
    L(lIdxStride);
        sub(regIdxIter, regSpecIdxSizeInBytes);
        uni_vpcmpeqd(dstMask, vmmAux0, vmmAux0);
        if (shiftFirst) {
            vpcmpgtd(vmmAux0, vmmSpecIdxSizeInBytes, vmmSpecIndicesInBytes);
            vpandn(vmmAux1, vmmAux0, vmmSpecIdxSizeInBytes);
            uni_vpsubd(vmmDstShifts, vmmSpecIndicesInBytes, vmmAux1);
            uni_vpaddd(vmmAux1, vmmIdxBatchSumInBytes, vmmDstShifts);
            uni_vpsubd(vmmSpecIndicesInBytes, vmmSpecIndicesInBytes, vmmSpecIdxSizeInBytes);
        } else {
            uni_vpaddd(vmmAux0, vmmIdxBatchSumInBytes, vmmSpecIndicesInBytes);
            uni_vpgatherdd(vmmDstShifts, ptr[regIndices + vmmAux0], dstMask);
            normalizeRawIndices(vmmDstShifts, dstMask, vAuxMask0);
            uni_vpaddd(vmmDstShifts, vmmDstShifts, vmmSrcBeforeAxisSum);

            uni_vpaddd(vmmAux0, vmmSpecIndicesInBytes, vmmVecLen);
            vpcmpgtd(vmmAux1, vmmSpecIdxSizeInBytes, vmmAux0);
            vpandn(vmmAux0, vmmAux1, vmmSpecIdxSizeInBytes);
            uni_vpsubd(vmmSpecIndicesInBytes, vmmSpecIndicesInBytes, vmmAux0);

            vpandn(vmmAux0, vmmAux1, vmmAxisAndAfterAxisSize);
            uni_vpaddd(vmmSrcBeforeAxisSum, vmmSrcBeforeAxisSum, vmmAux0);
        }

        Xbyak::Label l1;
        inc(regBetweenBatchAndAxisIter);
        cmp(regBetweenBatchAndAxisIter, regBetweenBatchAndAxisSize);
        jl(l1, T_NEAR);
            mov(regBetweenBatchAndAxisIter, 0);
            uni_vpaddd(vmmIdxBatchSumInBytes, vmmIdxBatchSumInBytes, vmmSpecIdxSizeInBytes);
            if (shiftFirst) {
                vpandn(vmmDstShifts, vmmAux0, vmmSpecIdxSizeInBytes);
                uni_vpaddd(vmmAux1, vmmAux1, vmmDstShifts);
            }
        L(l1);

        if (shiftFirst) {
            uni_vpgatherdd(vmmDstShifts, ptr[regIndices + vmmAux1], dstMask);
            normalizeRawIndices(vmmDstShifts, dstMask, vAuxMask0);

            vpandn(vmmAux0, vmmAux0, vmmAxisAndAfterAxisSize);
            uni_vpaddd(vmmAux0, vmmAux0, vmmSrcBeforeAxisSum);
            uni_vpaddd(vmmSrcBeforeAxisSum, vmmSrcBeforeAxisSum, vmmAxisAndAfterAxisSize);

            uni_vpaddd(vmmDstShifts, vmmDstShifts, vmmAux0);
        }
    L(lExit);
}

//template <>
//void jitUniGatherKernel<x64::avx512_common>::calcSrcShiftLong(Vmm& dstIndices, Vmask& dstMask, bool shiftFirst) {
//    Xbyak::Label lIdxStride, lExit;
//    if (shiftFirst)
//        vpaddd(vmmSpecIndicesInBytes, vmmSpecIndicesInBytes, vmmVecLen);
//
//    add(regIdxIter, vlen);
//    cmp(regIdxIter, regSpecIdxSizeInBytes);
//    jge(lIdxStride, T_NEAR);
//        vpaddd(vmmAux0, vmmIdxBatchSumInBytes, vmmSpecIndicesInBytes);
//        vmovd(reg32Aux1, xmmAux0);
//        vmovdqu64(dstIndices, ptr[regIndices + regAux1]);
//        normalizeRawIndices(dstIndices, dstMask, vAuxMask0);
//        vpaddd(dstIndices, dstIndices, vmmSrcBeforeAxisSum);
//    jmp(lExit, T_NEAR);
//    L(lIdxStride);
//        sub(regIdxIter, regSpecIdxSizeInBytes);
//        vpcmpb(dstMask, dstIndices, dstIndices, 0);
//        if (shiftFirst) {
//            vpcmpd(kMaskAux1, vmmSpecIdxSizeInBytes, vmmSpecIndicesInBytes, 2); // 2 - LE
//            vpaddd(vmmAux1, vmmIdxBatchSumInBytes, vmmSpecIndicesInBytes);
//            vpsubd(vmmAux1 | kMaskAux1, vmmAux1, vmmSpecIdxSizeInBytes);
//            vpsubd(vmmSpecIndicesInBytes, vmmSpecIndicesInBytes, vmmSpecIdxSizeInBytes);
//        } else {
//            vpaddd(vmmAux0, vmmIdxBatchSumInBytes, vmmSpecIndicesInBytes);
//            uni_vpgatherdd(dstIndices, ptr[regIndices + vmmAux0], dstMask);
//            normalizeRawIndices(dstIndices, dstMask, vAuxMask0);
//            vpaddd(dstIndices, dstIndices, vmmSrcBeforeAxisSum);
//
//            vpaddd(vmmAux0, vmmSpecIndicesInBytes, vmmVecLen);
//            vpcmpd(kMaskAux1, vmmSpecIdxSizeInBytes, vmmAux0, 2); // 2 - LE
//            vpsubd(vmmSpecIndicesInBytes | kMaskAux1, vmmSpecIndicesInBytes, vmmSpecIdxSizeInBytes);
//
//            vpaddd(vmmSrcBeforeAxisSum | kMaskAux1, vmmSrcBeforeAxisSum, vmmAxisAndAfterAxisSize);
//        }
//
//        Xbyak::Label l1;
//        inc(regBetweenBatchAndAxisIter);
//        cmp(regBetweenBatchAndAxisIter, regBetweenBatchAndAxisSize);
//        jl(l1, T_NEAR);
//            mov(regBetweenBatchAndAxisIter, 0);
//            vpaddd(vmmIdxBatchSumInBytes, vmmIdxBatchSumInBytes, vmmSpecIdxSizeInBytes);
//            if (shiftFirst)
//                vpaddd(vmmAux1 | kMaskAux1, vmmAux1, vmmSpecIdxSizeInBytes);
//        L(l1);
//
//        if (shiftFirst) {
//            uni_vpgatherdd(dstIndices, ptr[regIndices + vmmAux1], dstMask);
//            normalizeRawIndices(dstIndices, dstMask, vAuxMask0);
//
//            vpaddd(dstIndices, dstIndices, vmmSrcBeforeAxisSum);
//            vpaddd(dstIndices | kMaskAux1, dstIndices, vmmAxisAndAfterAxisSize);
//            vpaddd(vmmSrcBeforeAxisSum, vmmSrcBeforeAxisSum, vmmAxisAndAfterAxisSize);
//        }
//    L(lExit);
//}

template <>
void jitUniGatherKernel<x64::avx512_common>::calcSrcShiftLong(Vmm* vAuxPool, Vmask& dstMask, bool shiftFirst) {
}

// Requires vAuxPool length 2.
// Returns gathered data in vAuxPool[1].
template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::calcSrcShiftShort(Vmm* vAuxPool, Vmask& dstMask, bool shiftFirst) {
    auto& vmmDstShifts = vAuxPool[1];
    auto& vmmAux = vAuxPool[0];
    auto& vAuxMask0 = masksContainer[vAuxPool[0].getIdx()];

    if (shiftFirst) {
        vpermd(vmmSpecIndicesInBytes, vmmPermIdxMask, vmmSpecIndicesInBytes);
        uni_vpaddd(vmmSrcBeforeAxisSum, vmmSrcBeforeAxisSum, vmmBeforeAxisDiff);
        vpermd(vmmBeforeAxisDiff, vmmPermIdxMask, vmmBeforeAxisDiff);
    }
    // Calculate batch indices
    uni_vcvtdq2ps(vmmAux, vmmSrcBeforeAxisSum);
    uni_vcvtdq2ps(vmmDstShifts, vmmSrcAfterBatchSize);
    uni_vdivps(vmmAux, vmmAux, vmmDstShifts);
    uni_vroundps(vmmAux, vmmAux, 0x1);
    uni_vcvtps2dq(vmmAux, vmmAux);

    uni_vpmulld(vmmAux, vmmAux, vmmSpecIdxSizeInBytes);
    uni_vpaddd(vmmAux, vmmAux, vmmSpecIndicesInBytes);

    uni_vpcmpeqd(dstMask, vmmAux, vmmAux);
    uni_vpgatherdd(vmmDstShifts, ptr[regIndices + vmmAux], dstMask);
    normalizeRawIndices(vmmDstShifts, dstMask, vAuxMask0);
    uni_vpaddd(vmmDstShifts, vmmDstShifts, vmmSrcBeforeAxisSum);
}

// Requires vAuxPool length 2.
// Returns indices in vAuxPool[1].
template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::calcSrcShiftShortBlock(Vmm* vAuxPool, Vmask& dstMask, bool shiftFirst) {
    auto& vmmDstShifts = vAuxPool[1];
    auto& vmmAux0 = vAuxPool[0];
    auto& vAuxMask0 = masksContainer[vAuxPool[0].getIdx()];
//    auto& vmmAux1 = vAuxPool[2];

    if (shiftFirst) {
        vpermd(vmmBlockIdxInBytes, vmmPermIdxMask, vmmBlockIdxInBytes);

        uni_vpaddd(vmmSrcBeforeAxisSum, vmmSrcBeforeAxisSum, vmmBeforeAxisDiff);
        vpermd(vmmBeforeAxisDiff, vmmPermIdxMask, vmmBeforeAxisDiff);

        uni_vpaddd(vmmSpecIndicesInBytes, vmmSpecIndicesInBytes, vmmSpecIdxDiff);
        vpermd(vmmSpecIdxDiff, vmmPermIdxMask, vmmSpecIdxDiff);

        vpcmpgtd(vmmAux0, vmmSpecIdxSizeInBytes, vmmSpecIndicesInBytes); // TODO A5
        vpandn(vmmAux0, vmmAux0, vmmSpecIdxSizeInBytes);
        uni_vpsubd(vmmSpecIndicesInBytes, vmmSpecIndicesInBytes, vmmAux0);
    }
    // Calculate batch indices
    uni_vcvtdq2ps(vmmAux0, vmmSrcBeforeAxisSum);
    uni_vcvtdq2ps(vmmDstShifts, vmmSrcAfterBatchSize);
    uni_vdivps(vmmAux0, vmmAux0, vmmDstShifts);
    uni_vroundps(vmmAux0, vmmAux0, 0x1);
    uni_vcvtps2dq(vmmAux0, vmmAux0);

    uni_vpmulld(vmmAux0, vmmAux0, vmmSpecIdxSizeInBytes);
    uni_vpaddd(vmmAux0, vmmAux0, vmmSpecIndicesInBytes);

    uni_vpcmpeqd(dstMask, vmmAux0, vmmAux0);
    uni_vpgatherdd(vmmDstShifts, ptr[regIndices + vmmAux0], dstMask);
    normalizeRawIndices(vmmDstShifts, dstMask, vAuxMask0);

    uni_vpmulld(vmmDstShifts, vmmDstShifts, vmmAfterAxisSize);
    uni_vpaddd(vmmDstShifts, vmmDstShifts, vmmSrcBeforeAxisSum);
    uni_vpaddd(vmmDstShifts, vmmDstShifts, vmmBlockIdxInBytes);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::process32b(bool isShortIdx, bool blocked) {
//uni_vmovups(ptr[regDst], vmmIdxBatchSumInBytes);
//    Xbyak::Label lDstIdxLoop, lTail;
//
//    // First iteration
//    shiftIdxAndGather(vmmAuxContainer, isShortIdx, false, blocked);
//    uni_vmovups(ptr[regDst], vmmAuxContainer[1]);
//
//    // Main loop
//    L(lDstIdxLoop);
//    {
//        add(regDst, vlen);
//        sub(regWorkAmount, dataElPerVec);
//        cmp(regWorkAmount, dataElPerVec);
//        jl(lTail, T_NEAR);
//
//        shiftIdxAndGather(vmmAuxContainer, isShortIdx, true, blocked);
//        uni_vmovups(ptr[regDst], vmmAuxContainer[1]);
//
//        jmp(lDstIdxLoop, T_NEAR);
//    }
//
//    L(lTail);
//    tail(isShortIdx);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::process16b(bool isShortIdx, bool blocked) {
    Xbyak::Label lDstIdxLoop1, lTail;

    auto& vmmShufMask = vmmAuxContainer[5];
    mov(regAux1, reinterpret_cast<uintptr_t>(shufMask16bitUni));
    uni_vmovups(vmmShufMask, ptr[regAux1]);

    auto& vmmPermMask = vmmAuxContainer[6];
    mov(regAux1, reinterpret_cast<uintptr_t>(permMask16bitUni));
    uni_vmovups(vmmPermMask, ptr[regAux1]);

    // First iteration
    shiftIdxAndGather(vmmAuxContainer, isShortIdx, false, blocked);
    vpshufb(vmmAuxContainer[4], vmmAuxContainer[1], vmmShufMask);

    shiftIdxAndGather(vmmAuxContainer, isShortIdx, true, blocked);
    vpshufb(vmmAuxContainer[0], vmmAuxContainer[1], vmmShufMask);

    vshufps(vmmAuxContainer[0], vmmAuxContainer[4], vmmAuxContainer[0], 0x44);
    vpermd(vmmAuxContainer[0], vmmPermMask, vmmAuxContainer[0]);

    uni_vmovups(ptr[regDst], vmmAuxContainer[0]);

    // Main loop.
    L(lDstIdxLoop1);
    {
        add(regDst, vlen);
        sub(regWorkAmount, dataElPerVec);
        cmp(regWorkAmount, dataElPerVec);
        jl(lTail, T_NEAR);

        shiftIdxAndGather(vmmAuxContainer, isShortIdx, true, blocked);
        vpshufb(vmmAuxContainer[4], vmmAuxContainer[1], vmmShufMask);

        shiftIdxAndGather(vmmAuxContainer, isShortIdx, true, blocked);
        vpshufb(vmmAuxContainer[0], vmmAuxContainer[1], vmmShufMask);

        vshufps(vmmAuxContainer[0], vmmAuxContainer[4], vmmAuxContainer[0], 0x44);
        vpermd(vmmAuxContainer[0], vmmPermMask, vmmAuxContainer[0]);

        uni_vmovups(ptr[regDst], vmmAuxContainer[0]);

        jmp(lDstIdxLoop1, T_NEAR);
    }

    L(lTail);
    tail(isShortIdx);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::process8b(bool isShortIdx, bool blocked) {
    Xbyak::Label lDstIdxLoop1, lTail;

    auto& vmmShufMask = vmmAuxContainer[6];
    mov(regAux1, reinterpret_cast<uintptr_t>(shufMask8bitUni));
    uni_vmovups(vmmShufMask, ptr[regAux1]);

    Vmm vmmPermMask;
    if (isa == x64::avx512_common) {
        vmmPermMask = vmmAuxContainer[7];
        mov(regAux1, reinterpret_cast<uintptr_t>(permMask8bitUni));
        uni_vmovups(vmmPermMask, ptr[regAux1]);
    } else {
        vmmPermMask = vmmAuxContainer[1];
    }

    // First iteration
    shiftIdxAndGather(vmmAuxContainer, isShortIdx, false, blocked);
    vpshufb(vmmAuxContainer[4], vmmAuxContainer[1], vmmShufMask);

    shiftIdxAndGather(vmmAuxContainer, isShortIdx, true, blocked);
    vpshufb(vmmAuxContainer[0], vmmAuxContainer[1], vmmShufMask);

    vshufps(vmmAuxContainer[4], vmmAuxContainer[4], vmmAuxContainer[0], 0x0);

    shiftIdxAndGather(vmmAuxContainer, isShortIdx, true, blocked);
    vpshufb(vmmAuxContainer[5], vmmAuxContainer[1], vmmShufMask);

    shiftIdxAndGather(vmmAuxContainer, isShortIdx, true, blocked);
    vpshufb(vmmAuxContainer[0], vmmAuxContainer[1], vmmShufMask);

    vshufps(vmmAuxContainer[5], vmmAuxContainer[5], vmmAuxContainer[0], 0x0);
    vshufps(vmmAuxContainer[0], vmmAuxContainer[4], vmmAuxContainer[5], 0x88);

    if (isa == x64::avx2) {
        mov(regAux1, reinterpret_cast<uintptr_t>(permMask8bitUni));
        uni_vmovups(vmmPermMask, ptr[regAux1]);
    }
    vpermd(vmmAuxContainer[0], vmmPermMask, vmmAuxContainer[0]);

    uni_vmovups(ptr[regDst], vmmAuxContainer[0]);

    // Main loop.
    L(lDstIdxLoop1);
    {
        add(regDst, vlen);
        sub(regWorkAmount, dataElPerVec);
        cmp(regWorkAmount, dataElPerVec);
        jl(lTail, T_NEAR);

        shiftIdxAndGather(vmmAuxContainer, isShortIdx, true, blocked);
        vpshufb(vmmAuxContainer[4], vmmAuxContainer[1], vmmShufMask);

        shiftIdxAndGather(vmmAuxContainer, isShortIdx, true, blocked);
        vpshufb(vmmAuxContainer[0], vmmAuxContainer[1], vmmShufMask);

        vshufps(vmmAuxContainer[4], vmmAuxContainer[4], vmmAuxContainer[0], 0x0);

        shiftIdxAndGather(vmmAuxContainer, isShortIdx, true, blocked);
        vpshufb(vmmAuxContainer[5], vmmAuxContainer[1], vmmShufMask);

        shiftIdxAndGather(vmmAuxContainer, isShortIdx, true, blocked);
        vpshufb(vmmAuxContainer[0], vmmAuxContainer[1], vmmShufMask);

        vshufps(vmmAuxContainer[0], vmmAuxContainer[5], vmmAuxContainer[0], 0x0);
        vshufps(vmmAuxContainer[0], vmmAuxContainer[4], vmmAuxContainer[0], 0x88);

        if (isa == x64::avx2) {
            mov(regAux1, reinterpret_cast<uintptr_t>(permMask8bitUni));
            uni_vmovups(vmmPermMask, ptr[regAux1]);
        }
        vpermd(vmmAuxContainer[0], vmmPermMask, vmmAuxContainer[0]);

        uni_vmovups(ptr[regDst], vmmAuxContainer[0]);

        jmp(lDstIdxLoop1, T_NEAR);
    }

    L(lTail);
    tail(isShortIdx);
}

//template <x64::cpu_isa_t isa>
//void jitUniGatherKernel<isa>::processBlock32b(bool isShortIdx, bool shiftFirst, bool blocked) {
//    Xbyak::Label lDstIdxLoop, lTail;
//
//    // First iteration
//    shiftIdxAndGather(vmmAuxContainer, isShortIdx, false, blocked);
//    uni_vmovups(ptr[regDst], vmmAuxContainer[0]);
//
//    // Main loop
//    L(lDstIdxLoop);
//    {
//        add(regDst, vlen);
//        sub(regWorkAmount, dataElPerVec);
//        cmp(regWorkAmount, dataElPerVec);
//        jl(lTail, T_NEAR);
//
//        shiftIdxAndGather(vmmAuxContainer, isShortIdx, true, blocked);
//        uni_vmovups(ptr[regDst], vmmAuxContainer[0]);
//
//        jmp(lDstIdxLoop, T_NEAR);
//    }
//
//    L(lTail);
//    tail(isShortIdx);
//}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::fillRestWorkMask(Vmm& vmmDstMask, Vmm& vmmAux, const Xbyak::Reg64& rWorkRest,
        const Xbyak::Reg64& rAux0, const Xbyak::Reg64& rAux1) {
    Xbyak::Label lEnd;
    mov(rAux0, rWorkRest);
    Xbyak::Reg32 rOnes(rAux1.getIdx());
    mov(rOnes, 0xFFFFFFFF);
    Xbyak::Xmm xmmAux(vmmAux.getIdx());
    uni_vmovups(vmmDstMask, vmmZeros);
    for (uint8_t i = 0; i < idxElPerVec; i++) {
        cmp(rAux0, 0);
        je(lEnd, T_NEAR);

        if (i % 4 == 0)
            uni_vmovups(xmmAux, xmmZeros);

        vpinsrd(xmmAux, xmmAux, rOnes, i % 4);
        if (isa == x64::avx2) {
            vinserti128(vmmDstMask, vmmDstMask, xmmAux, i / 4);
        } else if (isa == x64::avx512_common) {
            vinserti64x2(vmmDstMask, vmmDstMask, xmmAux, i / 4);
        }
        sub(rAux0, 1);
    }
    L(lEnd);
}

// Requires vAuxPool length 4.
// Returns gathered data in vAuxPool[1].
template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::shiftIdxAndGather(Vmm* vAuxPool, bool isShortIdx, bool shiftFirst, bool blocked) {
    auto& vGatherMask = masksContainer[vAuxPool[0].getIdx()];
    if (blocked) {
        if (isShortIdx) {
            calcSrcShiftShortBlock(&vAuxPool[1], vGatherMask, shiftFirst);
        } else {
            calcSrcShiftLong(&vAuxPool[1], vGatherMask, shiftFirst);
        }
    } else {
        if (isShortIdx) {
            calcSrcShiftShort(&vAuxPool[1], vGatherMask, shiftFirst);
        } else {
            calcSrcShiftLong(&vAuxPool[1], vGatherMask, shiftFirst);
        }
    }
    uni_vmovups(vAuxPool[1], vmmZeros);
    uni_vpgatherdd(vAuxPool[1], ptr[regSrc + vAuxPool[2]], vGatherMask);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::storeScalar(const Xbyak::Reg64& rDst, const Xbyak::Reg64& rToStoreCounter, Vmm& vmmSrc, Vmm& vAux) {
    Xbyak::Label lEnd;
    Xbyak::Xmm xAux(vAux.getIdx());
    for (int j = 0; j < vlen / vlenXmm; j++) {
        if (isa == x64::avx2)
            vextracti128(xAux, vmmSrc, j);
        else if (isa == x64::avx512_common)
            vextracti64x2(xAux, vmmSrc, j);

        for (int k = 0; k < 4; k++) {
            cmp(rToStoreCounter, 0);
            jle(lEnd, T_NEAR);

            if (jcp.dataTypeSize == 4)
                uni_vpextrd(ptr[rDst], xAux, k);
            else if (jcp.dataTypeSize == 2)
                uni_vpextrw(ptr[rDst], xAux, k * 2);
            else if (jcp.dataTypeSize == 1)
                uni_vpextrb(ptr[rDst], xAux, k * 4);

            add(rDst, jcp.dataTypeSize);
            sub(rToStoreCounter, 1);
        }
    }
    L(lEnd);
}

template struct jitUniGatherKernel<x64::avx2>;
template struct jitUniGatherKernel<x64::avx512_common>;

}  // namespace MKLDNNPlugin
