// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_uni_kernel.hpp"

using namespace mkldnn::impl::cpu;

namespace MKLDNNPlugin {

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
        vGatherMask = Vmask(vmmGatherMask.getIdx());
        vAuxMask0 = Vmask(vmmAux0.getIdx());
        vAuxMask1 = Vmask(vmmAux1.getIdx());
    } else if (isa == x64::avx512_common) {
        permMask8bitUni = permMask8bitA5;
        permMask16bitUni = permMask16bitA5;
        vGatherMask = Vmask(kGatherMask.getIdx());
        vAuxMask0 = Vmask(kMaskAux0.getIdx());
        vAuxMask1 = Vmask(kMaskAux1.getIdx());
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
    uni_vdivps(vmmIdxBatchSum, vmmSrcBeforeAxisSum, vmmAux1);
    uni_vroundps(vmmIdxBatchSum, vmmIdxBatchSum, 0x1);
    vfnmadd231ps(vmmSrcBeforeAxisSum, vmmIdxBatchSum, vmmAux1);
    uni_vcvtps2dq(vmmSrcBeforeAxisSum, vmmSrcBeforeAxisSum);
    vmovd(reg32BetweenBatchAndAxisIter, xmmSrcBeforeAxisSum);
    uni_vcvtps2dq(vmmIdxBatchSum, vmmIdxBatchSum);

    mov(regAux1, ptr[regParams + GET_OFF(axisAndAfterAxisSizeInBytes)]);
    uni_vpbroadcastd(vmmAxisAndAfterAxisSize, ptr[regAux1]);
    uni_vpmulld(vmmSrcBeforeAxisSum, vmmSrcBeforeAxisSum, vmmAxisAndAfterAxisSize);
    mov(regAux1, ptr[regParams + GET_OFF(srcAfterBatchSizeInBytes)]);
    uni_vpbroadcastd(vmmAux0, ptr[regAux1]);
    uni_vpmulld(vmmAux0, vmmAux0, vmmIdxBatchSum);
    // srcBeforeAxisSum = ((start / specIndicesSize) % betweenBatchAndAxis) * axisAndAfterAxisSize + srcAfterBatchSize * idxBatchSum
    uni_vpaddd(vmmSrcBeforeAxisSum, vmmSrcBeforeAxisSum, vmmAux0);

    // idxBatchSum = specIdxSize * (start / afterBatchSize)
    uni_vpmulld(vmmIdxBatchSum, vmmIdxBatchSum, vmmSpecIdxSizeInBytes);

    mov(regAux1, ptr[regParams + GET_OFF(axisDim)]);
    uni_vpbroadcastd(vmmAxisDim, ptr[regAux1]);

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
                process32b(false);
            else if (jcp.dataTypeSize == 2)
                process16b(false);
            else if (jcp.dataTypeSize == 1)
                process8b(false);
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
                process32b(true);
            else if (jcp.dataTypeSize == 2)
                process16b(true);
            else if (jcp.dataTypeSize == 1)
                process8b(true);
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
//                processBlock32b(false);
//            else if (jcp.dataTypeSize == 2)
//                process16b(false);
//            else if (jcp.dataTypeSize == 1)
//                process8b(false);
//            jmp(lE2, T_NEAR);

            L(lTail3);
//            tail(false, false);
            jmp(lE2, T_NEAR);
        L(lLessThanVector2);
            // Calculate permute mask
            vmovd(xmmAux0, reg32Aux2);
            uni_vpbroadcastd(vmmAux1, xmmAux0);
            mov(regAux1, reinterpret_cast<uintptr_t>(&idxElPerVec));
            uni_vpbroadcastd(vmmAux0, ptr[regAux1]);
            uni_vpsubd(vmmPermIdxMask, vmmAux0, vmmAux1);
            mov(regAux1, reinterpret_cast<uintptr_t>(incVec));
            uni_vpaddd(vmmPermIdxMask, vmmPermIdxMask, ptr[regAux1]);
            for (int i = 0; i < 2; i++) { // two cycles is enough
                vpcmpgtd(vmmAux3, vmmAux0, vmmPermIdxMask); // TODO A5
                vpandn(vmmAux3, vmmAux3, vmmAux1);
                uni_vpsubd(vmmPermIdxMask, vmmPermIdxMask, vmmAux3);
            }

            mov(regAux1, ptr[regParams + GET_OFF(beforeAxisDiff)]);
            uni_vmovups(vmmBeforeBlockDiff, ptr[regAux1]);
//uni_vmovups(ptr[regDst], vmmBeforeBlockDiff);
            if (jcp.dataTypeSize != 1)
                vpslld(vmmBeforeBlockDiff, vmmBeforeBlockDiff, dataTypeShift); // multiply by data type size
            mov(regAux1, ptr[regParams + GET_OFF(srcAfterBatchSizeInBytes)]);
            uni_vpbroadcastd(vmmSrcAfterBatchSize, ptr[regAux1]);
            cmp(regWorkAmount, dataElPerVec);
            jl(lTail4, T_NEAR);

//            if (jcp.dataTypeSize == 4)
//                processBlock32b(true);
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
    Xbyak::Label lEnd;

    const int secondStepCycles = jcp.dataTypeSize == 4 ? 1 : jcp.dataTypeSize == 2 ? 2 : 4;
    for (int p = 0; p < secondStepCycles; p++) {
        cmp(regWorkAmount, 0);
        jle(lEnd, T_NEAR);

        if (isShortIdx)
            calcSrcShiftShort(vmmSrcShifts, vGatherMask, p > 0 || shiftFirst);
        else
            calcSrcShiftLong(vmmSrcShifts, vGatherMask, p > 0 || shiftFirst);
        fillRestWorkMask(vmmDst, vmmAux0, regWorkAmount, regAux1, rdx);
        if (isa == x64::avx2) {
            vpand(vmmGatherMask, vmmGatherMask, vmmDst);
            if (jcp.dataTypeSize == 4)
                vmovups(vmmAux0, vmmDst);
        } else if (isa == x64::avx512_common) {
            vpmovd2m(kMaskAux1, vmmDst);
            kandd(kGatherMask, kGatherMask, kMaskAux1);
            if (jcp.dataTypeSize == 4)
                kmovd(kMaskAux0, kGatherMask);
        }

        uni_vmovups(vmmDst, vmmZeros);
        uni_vpgatherdd(vmmDst, ptr[regSrc + vmmSrcShifts], vGatherMask);
        if (jcp.dataTypeSize == 4) {
            uni_vmovups_tail(ptr[regDst], vAuxMask0, vmmDst);
            sub(regWorkAmount, dataElPerVec);
        } else {
            storeScalar(regDst, regWorkAmount, vmmDst, vmmAux0);
        }
    }
    L(lEnd);
}

template <>
void jitUniGatherKernel<x64::avx2>::calcSrcShiftLong(Vmm& dstIndices, Vmask& dstMask, bool shiftFirst) {
    Xbyak::Label lIdxStride, lExit;
    if (shiftFirst)
        uni_vpaddd(vmmSpecIndicesInBytes, vmmSpecIndicesInBytes, vmmVecLen);

    add(regIdxIter, vlen);
    cmp(regIdxIter, regSpecIdxSizeInBytes);
    jge(lIdxStride, T_NEAR);
        uni_vpaddd(vmmAux0, vmmIdxBatchSum, vmmSpecIndicesInBytes);
        vmovd(reg32Aux1, xmmAux0);
        vmovdqu(dstIndices, ptr[regIndices + regAux1]);
        normalizeRawIndices(dstIndices, dstMask, vAuxMask0);
        uni_vpaddd(dstIndices, dstIndices, vmmSrcBeforeAxisSum);
    jmp(lExit, T_NEAR);
    L(lIdxStride);
        sub(regIdxIter, regSpecIdxSizeInBytes);
        uni_vpcmpeqd(dstMask, vmmAux0, vmmAux0);
        if (shiftFirst) {
            vpcmpgtd(vmmAux0, vmmSpecIdxSizeInBytes, vmmSpecIndicesInBytes);
            vpandn(vmmAux1, vmmAux0, vmmSpecIdxSizeInBytes);
            uni_vpsubd(dstIndices, vmmSpecIndicesInBytes, vmmAux1);
            uni_vpaddd(vmmAux1, vmmIdxBatchSum, dstIndices);
            uni_vpsubd(vmmSpecIndicesInBytes, vmmSpecIndicesInBytes, vmmSpecIdxSizeInBytes);
        } else {
            uni_vpaddd(vmmAux0, vmmIdxBatchSum, vmmSpecIndicesInBytes);
            uni_vpgatherdd(dstIndices, ptr[regIndices + vmmAux0], dstMask);
            normalizeRawIndices(dstIndices, dstMask, vAuxMask0);
            uni_vpaddd(dstIndices, dstIndices, vmmSrcBeforeAxisSum);

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
            uni_vpaddd(vmmIdxBatchSum, vmmIdxBatchSum, vmmSpecIdxSizeInBytes);
            if (shiftFirst) {
                vpandn(dstIndices, vmmAux0, vmmSpecIdxSizeInBytes);
                uni_vpaddd(vmmAux1, vmmAux1, dstIndices);
            }
        L(l1);

        if (shiftFirst) {
            uni_vpgatherdd(dstIndices, ptr[regIndices + vmmAux1], dstMask);
            normalizeRawIndices(dstIndices, dstMask, vAuxMask1);

            vpandn(vmmAux0, vmmAux0, vmmAxisAndAfterAxisSize);
            uni_vpaddd(vmmAux0, vmmAux0, vmmSrcBeforeAxisSum);
            uni_vpaddd(vmmSrcBeforeAxisSum, vmmSrcBeforeAxisSum, vmmAxisAndAfterAxisSize);

            uni_vpaddd(dstIndices, dstIndices, vmmAux0);
        }
    L(lExit);
}

template <>
void jitUniGatherKernel<x64::avx2>::calcSrcShiftLong(Vmm* vAuxPool, Vmask& dstMask, bool shiftFirst) {
    auto& vmmAux0 = vAuxPool[0];
    auto& vmmAux1 = vAuxPool[2];
    auto& vmmDstShifts = vAuxPool[1];

    Xbyak::Label lIdxStride, lExit;
    if (shiftFirst)
        uni_vpaddd(vmmSpecIndicesInBytes, vmmSpecIndicesInBytes, vmmVecLen);

    add(regIdxIter, vlen);
    cmp(regIdxIter, regSpecIdxSizeInBytes);
    jge(lIdxStride, T_NEAR);
        uni_vpaddd(vmmAux0, vmmIdxBatchSum, vmmSpecIndicesInBytes);
        vmovd(reg32Aux1, xmmAux0);
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
            uni_vpaddd(vmmAux1, vmmIdxBatchSum, vmmDstShifts);
            uni_vpsubd(vmmSpecIndicesInBytes, vmmSpecIndicesInBytes, vmmSpecIdxSizeInBytes);
        } else {
            uni_vpaddd(vmmAux0, vmmIdxBatchSum, vmmSpecIndicesInBytes);
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
            uni_vpaddd(vmmIdxBatchSum, vmmIdxBatchSum, vmmSpecIdxSizeInBytes);
            if (shiftFirst) {
                vpandn(vmmDstShifts, vmmAux0, vmmSpecIdxSizeInBytes);
                uni_vpaddd(vmmAux1, vmmAux1, vmmDstShifts);
            }
        L(l1);

        if (shiftFirst) {
            uni_vpgatherdd(vmmDstShifts, ptr[regIndices + vmmAux1], dstMask);
            normalizeRawIndices(vmmDstShifts, dstMask, vAuxMask1);

            vpandn(vmmAux0, vmmAux0, vmmAxisAndAfterAxisSize);
            uni_vpaddd(vmmAux0, vmmAux0, vmmSrcBeforeAxisSum);
            uni_vpaddd(vmmSrcBeforeAxisSum, vmmSrcBeforeAxisSum, vmmAxisAndAfterAxisSize);

            uni_vpaddd(vmmDstShifts, vmmDstShifts, vmmAux0);
        }
    L(lExit);
}

template <>
void jitUniGatherKernel<x64::avx512_common>::calcSrcShiftLong(Vmm& dstIndices, Vmask& dstMask, bool shiftFirst) {
    Xbyak::Label lIdxStride, lExit;
    if (shiftFirst)
        vpaddd(vmmSpecIndicesInBytes, vmmSpecIndicesInBytes, vmmVecLen);

    add(regIdxIter, vlen);
    cmp(regIdxIter, regSpecIdxSizeInBytes);
    jge(lIdxStride, T_NEAR);
        vpaddd(vmmAux0, vmmIdxBatchSum, vmmSpecIndicesInBytes);
        vmovd(reg32Aux1, xmmAux0);
        vmovdqu64(dstIndices, ptr[regIndices + regAux1]);
        normalizeRawIndices(dstIndices, dstMask, vAuxMask0);
        vpaddd(dstIndices, dstIndices, vmmSrcBeforeAxisSum);
    jmp(lExit, T_NEAR);
    L(lIdxStride);
        sub(regIdxIter, regSpecIdxSizeInBytes);
        vpcmpb(dstMask, dstIndices, dstIndices, 0);
        if (shiftFirst) {
            vpcmpd(kMaskAux1, vmmSpecIdxSizeInBytes, vmmSpecIndicesInBytes, 2); // 2 - LE
            vpaddd(vmmAux1, vmmIdxBatchSum, vmmSpecIndicesInBytes);
            vpsubd(vmmAux1 | kMaskAux1, vmmAux1, vmmSpecIdxSizeInBytes);
            vpsubd(vmmSpecIndicesInBytes, vmmSpecIndicesInBytes, vmmSpecIdxSizeInBytes);
        } else {
            vpaddd(vmmAux0, vmmIdxBatchSum, vmmSpecIndicesInBytes);
            uni_vpgatherdd(dstIndices, ptr[regIndices + vmmAux0], dstMask);
            normalizeRawIndices(dstIndices, dstMask, vAuxMask0);
            vpaddd(dstIndices, dstIndices, vmmSrcBeforeAxisSum);

            vpaddd(vmmAux0, vmmSpecIndicesInBytes, vmmVecLen);
            vpcmpd(kMaskAux1, vmmSpecIdxSizeInBytes, vmmAux0, 2); // 2 - LE
            vpsubd(vmmSpecIndicesInBytes | kMaskAux1, vmmSpecIndicesInBytes, vmmSpecIdxSizeInBytes);

            vpaddd(vmmSrcBeforeAxisSum | kMaskAux1, vmmSrcBeforeAxisSum, vmmAxisAndAfterAxisSize);
        }

        Xbyak::Label l1;
        inc(regBetweenBatchAndAxisIter);
        cmp(regBetweenBatchAndAxisIter, regBetweenBatchAndAxisSize);
        jl(l1, T_NEAR);
            mov(regBetweenBatchAndAxisIter, 0);
            vpaddd(vmmIdxBatchSum, vmmIdxBatchSum, vmmSpecIdxSizeInBytes);
            if (shiftFirst)
                vpaddd(vmmAux1 | kMaskAux1, vmmAux1, vmmSpecIdxSizeInBytes);
        L(l1);

        if (shiftFirst) {
            uni_vpgatherdd(dstIndices, ptr[regIndices + vmmAux1], dstMask);
            normalizeRawIndices(dstIndices, dstMask, vAuxMask0);

            vpaddd(dstIndices, dstIndices, vmmSrcBeforeAxisSum);
            vpaddd(dstIndices | kMaskAux1, dstIndices, vmmAxisAndAfterAxisSize);
            vpaddd(vmmSrcBeforeAxisSum, vmmSrcBeforeAxisSum, vmmAxisAndAfterAxisSize);
        }
    L(lExit);
}

template <>
void jitUniGatherKernel<x64::avx512_common>::calcSrcShiftLong(Vmm* vAuxPool, Vmask& dstMask, bool shiftFirst) {
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::calcSrcShiftShort(Vmm& dstIndices, Vmask& dstMask, bool shiftFirst) {
    if (shiftFirst) {
        vpermd(vmmSpecIndicesInBytes, vmmPermIdxMask, vmmSpecIndicesInBytes);
        uni_vpaddd(vmmSrcBeforeAxisSum, vmmSrcBeforeAxisSum, vmmBeforeAxisDiff);
        vpermd(vmmBeforeAxisDiff, vmmPermIdxMask, vmmBeforeAxisDiff);
    }
    // Calculate batch indices
    uni_vcvtdq2ps(vmmAux0, vmmSrcBeforeAxisSum);
    uni_vcvtdq2ps(dstIndices, vmmSrcAfterBatchSize);
    uni_vdivps(vmmAux0, vmmAux0, dstIndices);
    uni_vroundps(vmmAux0, vmmAux0, 0x1);
    uni_vcvtps2dq(vmmAux0, vmmAux0);

    uni_vpmulld(vmmAux0, vmmAux0, vmmSpecIdxSizeInBytes);
    uni_vpaddd(vmmAux0, vmmAux0, vmmSpecIndicesInBytes);

    uni_vpcmpeqd(dstMask, vmmAux0, vmmAux0);
    uni_vpgatherdd(dstIndices, ptr[regIndices + vmmAux0], dstMask);
    normalizeRawIndices(dstIndices, dstMask, vAuxMask0);
    uni_vpaddd(dstIndices, dstIndices, vmmSrcBeforeAxisSum);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::calcSrcShiftShort(Vmm* vAuxPool, Vmask& dstMask, bool shiftFirst) {
    auto& vmmAux = vAuxPool[0];
    auto& vmmDstShifts = vAuxPool[1];

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

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::calcSrcShiftShortBlock(Vmm& dstIndices, Vmask& dstMask, bool shiftFirst) {
    if (shiftFirst) {
        vpermd(vmmBlockIdxInBytes, vmmPermIdxMask, vmmBlockIdxInBytes);

        uni_vpaddd(vmmSrcBeforeAxisSum, vmmSrcBeforeAxisSum, vmmBeforeAxisDiff);
        vpermd(vmmBeforeAxisDiff, vmmPermIdxMask, vmmBeforeAxisDiff);

        uni_vpaddd(vmmSpecIndicesInBytes, vmmSpecIndicesInBytes, vmmSpecIdxDiff);
        vpermd(vmmSpecIdxDiff, vmmPermIdxMask, vmmSpecIdxDiff);

        vpcmpgtd(vmmAux3, vmmSpecIdxSizeInBytes, vmmSpecIndicesInBytes); // TODO A5
        vpandn(vmmAux3, vmmAux3, vmmSpecIdxSizeInBytes);
        uni_vpsubd(vmmSpecIndicesInBytes, vmmSpecIndicesInBytes, vmmAux3);
    }
    // Calculate batch indices
    uni_vcvtdq2ps(vmmAux0, vmmSrcBeforeAxisSum);
    uni_vcvtdq2ps(dstIndices, vmmSrcAfterBatchSize);
    uni_vdivps(vmmAux0, vmmAux0, dstIndices);
    uni_vroundps(vmmAux0, vmmAux0, 0x1);
    uni_vcvtps2dq(vmmAux0, vmmAux0);

    uni_vpmulld(vmmAux0, vmmAux0, vmmSpecIdxSizeInBytes);
    uni_vpaddd(vmmAux0, vmmAux0, vmmSpecIndicesInBytes);

    uni_vpcmpeqd(dstMask, vmmAux0, vmmAux0);
    uni_vpgatherdd(dstIndices, ptr[regIndices + vmmAux0], dstMask);
    normalizeRawIndices(dstIndices, dstMask, vAuxMask0);

    uni_vpmulld(dstIndices, dstIndices, vmmAfterAxisSize);
    uni_vpaddd(dstIndices, dstIndices, vmmSrcBeforeAxisSum);
    uni_vpaddd(dstIndices, dstIndices, vmmBlockIdxInBytes);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::process32b(bool isShortIdx) {
    Xbyak::Label lDstIdxLoop, lTail;

    // First iteration
    shiftIdxAndGather(vmmAuxContainer, vGatherMask, isShortIdx, false);
    uni_vmovups(ptr[regDst], vmmAuxContainer[0]);

    // Main loop
    L(lDstIdxLoop);
    {
        add(regDst, vlen);
        sub(regWorkAmount, dataElPerVec);
        cmp(regWorkAmount, dataElPerVec);
        jl(lTail, T_NEAR);

        shiftIdxAndGather(vmmAuxContainer, vGatherMask, isShortIdx);
        uni_vmovups(ptr[regDst], vmmAuxContainer[0]);

        jmp(lDstIdxLoop, T_NEAR);
    }

    L(lTail);
    tail(isShortIdx);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::process16b(bool isShortIdx) {
    Xbyak::Label lDstIdxLoop1, lTail;

    Vmm vmmShufMask;
    if (isShortIdx)
        vmmShufMask = vmmAux8;
    else
        vmmShufMask = vmmAux2;
    mov(regAux1, reinterpret_cast<uintptr_t>(shufMask16bitUni));
    uni_vmovups(vmmShufMask, ptr[regAux1]);

    auto& vmmPermMask = vmmAux3;
    mov(regAux1, reinterpret_cast<uintptr_t>(permMask16bitUni));
    uni_vmovups(vmmPermMask, ptr[regAux1]);

    // First iteration
    shiftIdxAndGather(vmmDst, vmmSrcShifts, vGatherMask, isShortIdx, false);
    vpshufb(vmmDst, vmmDst, vmmShufMask);

    shiftIdxAndGather(vmmAux0, vmmSrcShifts, vGatherMask, isShortIdx);
    vpshufb(vmmAux0, vmmAux0, vmmShufMask);

    vshufps(vmmDst, vmmDst, vmmAux0, 0x44);
    vpermd(vmmDst, vmmPermMask, vmmDst);

    uni_vmovups(ptr[regDst], vmmDst);

    // Main loop.
    L(lDstIdxLoop1);
    {
        add(regDst, vlen);
        sub(regWorkAmount, dataElPerVec);
        cmp(regWorkAmount, dataElPerVec);
        jl(lTail, T_NEAR);

        shiftIdxAndGather(vmmDst, vmmSrcShifts, vGatherMask, isShortIdx);
        vpshufb(vmmDst, vmmDst, vmmShufMask);

        shiftIdxAndGather(vmmAux0, vmmSrcShifts, vGatherMask, isShortIdx);
        vpshufb(vmmAux0, vmmAux0, vmmShufMask);

        vshufps(vmmDst, vmmDst, vmmAux0, 0x44);
        vpermd(vmmDst, vmmPermMask, vmmDst);

        uni_vmovups(ptr[regDst], vmmDst);

        jmp(lDstIdxLoop1, T_NEAR);
    }

    L(lTail);
    tail(isShortIdx);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::process8b(bool isShortIdx) {
    Xbyak::Label lDstIdxLoop1, lTail;

    auto& vmmShufMask = vmmAux3;
    mov(regAux1, reinterpret_cast<uintptr_t>(shufMask8bitUni));
    uni_vmovups(vmmShufMask, ptr[regAux1]);

    Vmm vmmPermMask;
    if (isa == x64::avx512_common) {
        vmmPermMask = vmmAux5;
        mov(regAux1, reinterpret_cast<uintptr_t>(permMask8bitUni));
        uni_vmovups(vmmPermMask, ptr[regAux1]);
    } else {
        vmmPermMask = vmmAux4;
    }

    // First iteration
    shiftIdxAndGather(vmmDst, vmmSrcShifts, vGatherMask, isShortIdx, false);
    vpshufb(vmmDst, vmmDst, vmmShufMask);

    shiftIdxAndGather(vmmAux0, vmmSrcShifts, vGatherMask, isShortIdx);
    vpshufb(vmmAux0, vmmAux0, vmmShufMask);

    vshufps(vmmDst, vmmDst, vmmAux0, 0x0);

    shiftIdxAndGather(vmmAux4, vmmSrcShifts, vGatherMask, isShortIdx);
    vpshufb(vmmAux4, vmmAux4, vmmShufMask);

    shiftIdxAndGather(vmmAux0, vmmSrcShifts, vGatherMask, isShortIdx);
    vpshufb(vmmAux0, vmmAux0, vmmShufMask);

    vshufps(vmmAux4, vmmAux4, vmmAux0, 0x0);
    vshufps(vmmDst, vmmDst, vmmAux4, 0x88);

    if (isa == x64::avx2) {
        mov(regAux1, reinterpret_cast<uintptr_t>(permMask8bitUni));
        uni_vmovups(vmmPermMask, ptr[regAux1]);
    }
    vpermd(vmmDst, vmmPermMask, vmmDst);

    uni_vmovups(ptr[regDst], vmmDst);

    // Main loop.
    L(lDstIdxLoop1);
    {
        add(regDst, vlen);
        sub(regWorkAmount, dataElPerVec);
        cmp(regWorkAmount, dataElPerVec);
        jl(lTail, T_NEAR);

        shiftIdxAndGather(vmmDst, vmmSrcShifts, vGatherMask, isShortIdx);
        vpshufb(vmmDst, vmmDst, vmmShufMask);

        shiftIdxAndGather(vmmAux0, vmmSrcShifts, vGatherMask, isShortIdx);
        vpshufb(vmmAux0, vmmAux0, vmmShufMask);

        vshufps(vmmDst, vmmDst, vmmAux0, 0x0);

        shiftIdxAndGather(vmmAux4, vmmSrcShifts, vGatherMask, isShortIdx);
        vpshufb(vmmAux4, vmmAux4, vmmShufMask);

        shiftIdxAndGather(vmmAux0, vmmSrcShifts, vGatherMask, isShortIdx);
        vpshufb(vmmAux0, vmmAux0, vmmShufMask);

        vshufps(vmmAux4, vmmAux4, vmmAux0, 0x0);

        vshufps(vmmDst, vmmDst, vmmAux4, 0x88);

        if (isa == x64::avx2) {
            mov(regAux1, reinterpret_cast<uintptr_t>(permMask8bitUni));
            uni_vmovups(vmmPermMask, ptr[regAux1]);
        }
        vpermd(vmmDst, vmmPermMask, vmmDst);

        uni_vmovups(ptr[regDst], vmmDst);

        jmp(lDstIdxLoop1, T_NEAR);
    }

    L(lTail);
    tail(isShortIdx);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::processBlock32b(bool isShortIdx, bool shiftFirst) {
    Xbyak::Label lDstIdxLoop, lTail;

    // First iteration
    shiftIdxAndGather(vmmDst, vmmSrcShifts, vGatherMask, isShortIdx, false);
//    uni_vpgatherdd(vmmDst, ptr[regSrc + vmmSrcShifts], vGatherMask);
    uni_vmovups(ptr[regDst], vmmDst);

    // Main loop
    L(lDstIdxLoop);
    {
        add(regDst, vlen);
        sub(regWorkAmount, dataElPerVec);
        cmp(regWorkAmount, dataElPerVec);
        jl(lTail, T_NEAR);

        shiftIdxAndGather(vmmDst, vmmSrcShifts, vGatherMask, isShortIdx);
        uni_vmovups(ptr[regDst], vmmDst);

        jmp(lDstIdxLoop, T_NEAR);
    }

    L(lTail);
    tail(isShortIdx);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::fillRestWorkMask(Vmm& vmmOnesMask, Vmm& vmmAux, Xbyak::Reg64& rWorkRest, Xbyak::Reg64& rAux0, const Xbyak::Reg64& rAux1) {
    Xbyak::Label lEnd;
    mov(rAux0, rWorkRest);
    Xbyak::Reg32 rOnes(rAux1.getIdx());
    mov(rOnes, 0xFFFFFFFF);
    Xbyak::Xmm xmmAux(vmmAux.getIdx());
    uni_vmovups(vmmOnesMask, vmmZeros);
    for (uint8_t i = 0; i < idxElPerVec; i++) {
        cmp(rAux0, 0);
        je(lEnd, T_NEAR);

        if (i % 4 == 0)
            uni_vmovups(xmmAux, xmmZeros);

        vpinsrd(xmmAux, xmmAux, rOnes, i % 4);
        if (isa == x64::avx2) {
            vinserti128(vmmOnesMask, vmmOnesMask, xmmAux, i / 4);
        } else if (isa == x64::avx512_common) {
            vinserti64x2(vmmOnesMask, vmmOnesMask, xmmAux, i / 4);
        }
        sub(rAux0, 1);
    }
    L(lEnd);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::shiftIdxAndGather(Vmm& vDst, Vmm& vAux, Vmask& mAux, bool isShortIdx, bool shiftFirst) {
    if (isShortIdx)
        calcSrcShiftShort(vAux, mAux, shiftFirst);
    else
        calcSrcShiftLong(vAux, mAux, shiftFirst);
    uni_vmovups(vDst, vmmZeros);
    uni_vpgatherdd(vDst, ptr[regSrc + vAux], mAux);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::shiftIdxAndGather(Vmm* vAuxPool, Vmask& mAux, bool isShortIdx, bool shiftFirst) {
    if (isShortIdx)
        calcSrcShiftShort(vAuxPool, mAux, shiftFirst);
    else
        calcSrcShiftLong(vAuxPool, mAux, shiftFirst);
    uni_vpgatherdd(vAuxPool[0], ptr[regSrc + vAuxPool[1]], mAux);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::storeScalar(Xbyak::Reg64& rDst, Xbyak::Reg64& rToStoreCounter, Vmm& vmmSrc, Vmm& vAux) {
    Xbyak::Label lEnd;
    Xbyak::Xmm xAux(vAux.getIdx());
    for (int j = 0; j < vlen / vlenXmm; j++) {
        if (isa == x64::avx2)
            vextracti128(xAux, vmmSrc, j);
        else if (isa == x64::avx512_common)
            vextracti64x2(xAux, vmmSrc, j);

        for (int k = 0; k < 4; k++) {
            cmp(rToStoreCounter, 0);
            je(lEnd, T_NEAR);

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
