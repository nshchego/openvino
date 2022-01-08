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
        vlenShift = 5;
        permMask8bitUni = permMask8bitA2;
        permMask16bitUni = permMask16bitA2;
    } else if (isa == x64::avx512_common) {
        vlenShift = 6;
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

    mov(regWorkAmount, ptr[regParams + GET_OFF(workAmount)]);
    uni_vpxor(vmmZeros, vmmZeros, vmmZeros);

    auto& vmmAux0 = vmmAuxContainer[0];
    auto& vmmAux1 = vmmAuxContainer[1];
    auto& xmmAux0 = xmmAuxContainer[0];
    auto& xmmAux1 = xmmAuxContainer[1];

    if (jcp.dynamicShapes) {
        mov(regAux1, ptr[regParams + GET_OFF(start)]);
        vpbroadcastd(vmmSpecIdxB, ptr[regAux1]);
        mov(regAux1, reinterpret_cast<uintptr_t>(incVec));
        vpaddd(vmmSpecIdxB, vmmSpecIdxB, ptr[regAux1]);
        vcvtdq2ps(vmmSpecIdxB, vmmSpecIdxB);

        // Formula: specIndices = (start % specIndicesSize) * idxTypeSize
        mov(regAux1, ptr[regParams + GET_OFF(specIndicesSize)]);
        uni_vpbroadcastd(vmmSpecIdxSizeB, ptr[regAux1]);
        uni_vcvtdq2ps(vmmAux1, vmmSpecIdxSizeB);
        uni_vdivps(vmmSrcBeforeAxisSumB, vmmSpecIdxB, vmmAux1);
        uni_vroundps(vmmSrcBeforeAxisSumB, vmmSrcBeforeAxisSumB, 0x1);
        vfnmadd231ps(vmmSpecIdxB, vmmSrcBeforeAxisSumB, vmmAux1);
        uni_vcvtps2dq(vmmSpecIdxB, vmmSpecIdxB);
        vpslld(vmmSpecIdxB, vmmSpecIdxB, idxTypeShift); // multiply by indices type size.
        vpslld(vmmSpecIdxSizeB, vmmSpecIdxSizeB, idxTypeShift); // multiply by indices type size.
        vmovd(reg32SpecIdxSizeB, xmmSpecIdxSizeB);

        mov(regAux1, ptr[regParams + GET_OFF(betweenBatchAndAxisSize)]);
        uni_vpbroadcastd(vmmAux1, ptr[regAux1]);
        vmovd(reg32BetweenBatchAndAxisSize, xmmAux1);
        uni_vcvtdq2ps(vmmAux1, vmmAux1);
        uni_vdivps(vmmIdxBatchSumB, vmmSrcBeforeAxisSumB, vmmAux1);
        uni_vroundps(vmmIdxBatchSumB, vmmIdxBatchSumB, 0x1);
        vfnmadd231ps(vmmSrcBeforeAxisSumB, vmmIdxBatchSumB, vmmAux1);
        uni_vcvtps2dq(vmmSrcBeforeAxisSumB, vmmSrcBeforeAxisSumB);
        vmovd(reg32BetweenBatchAndAxisIter, xmmSrcBeforeAxisSum);
        uni_vcvtps2dq(vmmIdxBatchSumB, vmmIdxBatchSumB);

        mov(regAux1, ptr[regParams + GET_OFF(axisAndAfterAxisSizeB)]);
        uni_vpbroadcastd(vmmAxisAndAfterAxisSizeB, ptr[regAux1]);
        uni_vpmulld(vmmSrcBeforeAxisSumB, vmmSrcBeforeAxisSumB, vmmAxisAndAfterAxisSizeB);
        mov(regAux1, ptr[regParams + GET_OFF(srcAfterBatchSizeB)]);
        uni_vpbroadcastd(vmmAux0, ptr[regAux1]);
        uni_vpmulld(vmmAux0, vmmAux0, vmmIdxBatchSumB);
        // Formula: srcBeforeAxisSum = ((start / specIndicesSize) % betweenBatchAndAxis) * axisAndAfterAxisSize + srcAfterBatchSize * idxBatchSum
        uni_vpaddd(vmmSrcBeforeAxisSumB, vmmSrcBeforeAxisSumB, vmmAux0);

        // Formula: idxBatchSum = specIdxSize * (start / afterBatchSize)
        uni_vpmulld(vmmIdxBatchSumB, vmmIdxBatchSumB, vmmSpecIdxSizeB);

        mov(regAux1, ptr[regParams + GET_OFF(axisDim)]);
        uni_vpbroadcastd(vmmAxisDim, ptr[regAux1]);
    } else {
        mov(regAux1, ptr[regParams + GET_OFF(axisDim)]);
        uni_vpbroadcastd(vmmAxisDim, ptr[regAux1]);

        if (jcp.afterAxisSize == 1lu) { // Elementwise case.
            mov(regAux1, ptr[regParams + GET_OFF(specIndicesSize)]);
            uni_vpbroadcastd(vmmSpecIdxSizeB, ptr[regAux1]);
            vpslld(vmmSpecIdxSizeB, vmmSpecIdxSizeB, idxTypeShift); // multiply by indices type size.
            vmovd(reg32SpecIdxSizeB, xmmSpecIdxSizeB);
            mov(regAux1, ptr[regParams + GET_OFF(axisAndAfterAxisSizeB)]);
            uni_vpbroadcastd(vmmAxisAndAfterAxisSizeB, ptr[regAux1]);

            mov(regAux1, ptr[regParams + GET_OFF(specIdxB)]);
            uni_vmovups(vmmSpecIdxB, ptr[regAux1]);
            mov(regAux1, ptr[regParams + GET_OFF(idxBatchSumB)]);
            uni_vmovups(vmmIdxBatchSumB, ptr[regAux1]);
            mov(regAux1, ptr[regParams + GET_OFF(dataBeforeAxisSumB)]);
            uni_vmovups(vmmSrcBeforeAxisSumB, ptr[regAux1]);

            mov(regAux1, ptr[regParams + GET_OFF(betweenBatchAndAxisSize)]);
            mov(regBetweenBatchAndAxisSize, ptr[regAux1]);
            mov(regBetweenBatchAndAxisIter, ptr[regParams + GET_OFF(betweenBatchAndAxisIter)]);

            if (jcp.specIdxSize < idxElPerVec) { // Short case.
                mov(regAux1, ptr[regParams + GET_OFF(permIdxMask)]);
                uni_vmovups(vmmPermIdxMask, ptr[regAux1]);
                mov(regAux1, ptr[regParams + GET_OFF(beforeAxisDiff)]);
                uni_vmovups(vmmBeforeAxDiffB, ptr[regAux1]);
                if (jcp.dataTypeSize != 1)
                    vpslld(vmmBeforeAxDiffB, vmmBeforeAxDiffB, dataTypeShift); // multiply by data type size
                if (jcp.batchDims > 0lu) {
                    mov(regAux1, ptr[regParams + GET_OFF(srcAfterBatchSizeB)]);
                    uni_vpbroadcastd(vmmSrcAfterBatchSizeB, ptr[regAux1]);
                }

                process(true, false);
            } else { // Long case.
                vmovd(reg32IdxIter, xmmSpecIdxB);
                vpcmpeqd(vmmVecLenB, vmmVecLenB, vmmVecLenB);
                vpsrld(vmmVecLenB, vmmVecLenB, 31);
                vpslld(vmmVecLenB, vmmVecLenB, vlenShift);

                process(false, false);
            }
        } else { // Blocked case.
            if (jcp.afterAxisSize < idxElPerVec) { // Short case.
                mov(regAux1, ptr[regParams + GET_OFF(afterAxIdxB)]);
                uni_vmovups(vmmAfterAxisIdxB, ptr[regAux1]);
                mov(regAux1, ptr[regParams + GET_OFF(afterAxisPermMask)]);
                uni_vmovups(vmmAfterAxisPermMask, ptr[regAux1]);
                mov(regAux1, ptr[regParams + GET_OFF(specIdxDiff)]);
                uni_vmovups(vmmSpecIdxDiff, ptr[regAux1]);
                mov(regAux1, ptr[regParams + GET_OFF(specIdxB)]);
                uni_vmovups(vmmSpecIdxB, ptr[regAux1]);
                mov(regAux1, ptr[regParams + GET_OFF(specIndicesSize)]);
                uni_vpbroadcastd(vmmSpecIdxSizeB, ptr[regAux1]);
                // TODO: move outside kernel?
                vpslld(vmmSpecIdxSizeB, vmmSpecIdxSizeB, idxTypeShift); // multiply by indices type size.
                mov(regAux1, ptr[regParams + GET_OFF(srcAfterBatchSizeB)]);
                uni_vpbroadcastd(vmmSrcAfterBatchSizeB, ptr[regAux1]);
                mov(regAux1, ptr[regParams + GET_OFF(afterAxSizePtr)]);
                uni_vpbroadcastd(vmmAfterAxisSize, ptr[regAux1]);

                if (jcp.beforeAxisSize != 1lu) {
                    mov(regAux1, ptr[regParams + GET_OFF(dataBeforeAxisSumB)]);
                    uni_vmovups(vmmSrcBeforeAxisSumB, ptr[regAux1]);
                    mov(rSpecIdxAndAfterAxIterB, ptr[regParams + GET_OFF(specIdxAndAfterAxIterB)]);
                    mov(rSpecIdxAndAfterAxSizeB, ptr[regParams + GET_OFF(specIdxAndAfterAxSizeB)]);
                    if (jcp.specIdxSize * jcp.afterAxisSize < idxElPerVec) {
                        mov(regAux1, ptr[regParams + GET_OFF(beforeAxisDiff)]);
                        uni_vmovups(vmmBeforeAxDiffB, ptr[regAux1]);
                    } else {
                        mov(regAux1, ptr[regParams + GET_OFF(axisAndAfterAxisSizeB)]);
                        uni_vpbroadcastd(vmmAxisAndAfterAxisSizeB, ptr[regAux1]);
                    }
                    mov(regAux1, ptr[regParams + GET_OFF(beforeAxisPermMask)]);
                    uni_vmovups(vmmBeforeAxPermMask, ptr[regAux1]);
                }

                process(true, true);
            } else { // Long case.
                throw std::invalid_argument("Unsupported case.");
            }
        }
    }

    if (jcp.dynamicShapes) {
        Xbyak::Label lBlock, lEnd;
        mov(regAux2, ptr[regParams + GET_OFF(afterAxSize)]);
        cmp(regAux2, 1);
        jg(lBlock, T_NEAR);
        {
            Xbyak::Label lLessThanVector1, lTail1, lTail2, lE1;

            cmp(regSpecIdxSizeB, vlen);
            jl(lLessThanVector1, T_NEAR);
                vmovd(reg32IdxIter, xmmSpecIdxB);
                vpcmpeqd(vmmVecLenB, vmmVecLenB, vmmVecLenB);
                vpsrld(vmmVecLenB, vmmVecLenB, 31);
                vpslld(vmmVecLenB, vmmVecLenB, vlenShift);

                process(false, false);
                jmp(lE1, T_NEAR);
            L(lLessThanVector1);
                mov(regAux1, ptr[regParams + GET_OFF(permIdxMask)]);
                uni_vmovups(vmmPermIdxMask, ptr[regAux1]);
                mov(regAux1, ptr[regParams + GET_OFF(beforeAxisDiff)]);
                uni_vmovups(vmmBeforeAxDiffB, ptr[regAux1]);
                if (jcp.dataTypeSize != 1)
                    vpslld(vmmBeforeAxDiffB, vmmBeforeAxDiffB, dataTypeShift); // multiply by data type size
                mov(regAux1, ptr[regParams + GET_OFF(srcAfterBatchSizeB)]);
                uni_vpbroadcastd(vmmSrcAfterBatchSizeB, ptr[regAux1]);

                process(true, false);
            L(lE1);
            jmp(lEnd, T_NEAR);
        }
        L(lBlock); {
            mov(regAux1, ptr[regParams + GET_OFF(start)]);
            uni_vpbroadcastd(vmmAfterAxisIdxB, ptr[regAux1]);
            mov(regAux1, reinterpret_cast<uintptr_t>(incVec));
            uni_vpaddd(vmmAfterAxisIdxB, vmmAfterAxisIdxB, ptr[regAux1]);
            uni_vcvtdq2ps(vmmAfterAxisIdxB, vmmAfterAxisIdxB);

            // afterAxIdxB = (start % afterAxSize) * idxTypeSize
            movd(xmmAux0, reg32Aux1); // afterAxSize ??
            uni_vpbroadcastd(vmmAux1, xmmAux0); // A5 broadcast from reg32Aux1
            uni_vcvtdq2ps(vmmAux1, vmmAux1);
            uni_vdivps(vmmSrcBeforeAxisSumB, vmmAfterAxisIdxB, vmmAux1);
            uni_vroundps(vmmSrcBeforeAxisSumB, vmmSrcBeforeAxisSumB, 0x1);
            vfnmadd231ps(vmmAfterAxisIdxB, vmmSrcBeforeAxisSumB, vmmAux1);
            uni_vcvtps2dq(vmmAfterAxisIdxB, vmmAfterAxisIdxB);
            vpslld(vmmAfterAxisIdxB, vmmAfterAxisIdxB, idxTypeShift); // multiply by indices type size.

            Xbyak::Label lLessThanVector2, lTail3, lTail4, lE2;

            cmp(regAux2, dataElPerVec);
            jl(lLessThanVector2, T_NEAR);
                vmovd(reg32IdxIter, xmmSpecIdxB);
                vpcmpeqd(vmmVecLenB, vmmVecLenB, vmmVecLenB);
                vpsrld(vmmVecLenB, vmmVecLenB, 31);
                vpslld(vmmVecLenB, vmmVecLenB, vlenShift);

//                process(false, true);
                jmp(lE2, T_NEAR);
            L(lLessThanVector2);
                auto& vmmAux2 = vmmAuxContainer[2];
                // Calculate permute mask
                vmovd(xmmAux0, reg32Aux2);
                uni_vpbroadcastd(vmmAux1, xmmAux0);
                mov(regAux1, reinterpret_cast<uintptr_t>(&idxElPerVec));
                uni_vpbroadcastd(vmmAux0, ptr[regAux1]);
                uni_vpsubd(vmmAfterAxisPermMask, vmmAux0, vmmAux1);
                mov(regAux1, reinterpret_cast<uintptr_t>(incVec));
                uni_vpaddd(vmmAfterAxisPermMask, vmmAfterAxisPermMask, ptr[regAux1]);
                for (int i = 0; i < 2; i++) { // two cycles is enough
                    vpcmpgtd(vmmAux2, vmmAux0, vmmAfterAxisPermMask); // TODO A5
                    vpandn(vmmAux2, vmmAux2, vmmAux1);
                    uni_vpsubd(vmmAfterAxisPermMask, vmmAfterAxisPermMask, vmmAux2);
                }

    //            mov(regAux1, ptr[regParams + GET_OFF(beforeAxisDiff)]);
    //            uni_vmovups(vmmBeforeBlockDiff, ptr[regAux1]);
    //uni_vmovups(ptr[regDst], vmmBeforeBlockDiff);
    //            if (jcp.dataTypeSize != 1)
    //                vpslld(vmmBeforeBlockDiff, vmmBeforeBlockDiff, dataTypeShift); // multiply by data type size
    //            uni_vpmulld(vmmBeforeAxDiffB, vmmBeforeBlockDiff, vmmAxisDim);
    //            mov(regAux1, ptr[regParams + GET_OFF(srcAfterBatchSizeB)]);
    //            uni_vpbroadcastd(vmmSrcAfterBatchSizeB, ptr[regAux1]);

                process(true, true);
            L(lE2);
    //        jmp(lEnd, T_NEAR);
        }
        L(lEnd);
    }

    this->postamble();
}

template <>
void jitUniGatherKernel<x64::avx2>::uni_vpgatherdd(Vmm& vDst, const Xbyak::Address& srcAddr, Vmask& kMask) {
    vpgatherdd(vDst, srcAddr, kMask);
}

template <>
void jitUniGatherKernel<x64::avx512_common>::uni_vpgatherdd(Vmm& vDst, const Xbyak::Address& srcAddr, Vmask& kMask) {
    vpgatherdd(vDst | kMask, srcAddr);
}

template <>
void jitUniGatherKernel<x64::avx2>::uni_vpcmpeqd(Vmask& kMask, Vmm& vOp0, Vmm& vOp2) {
    vpcmpeqd(kMask, vOp0, vOp2);
}

template <>
void jitUniGatherKernel<x64::avx512_common>::uni_vpcmpeqd(Vmask& kMask, Vmm& vOp0, Vmm& vOp2) {
    vpcmpd(kMask, vOp0, vOp2, 0);
}

template <>
void jitUniGatherKernel<x64::avx2>::normalizeRawIndices(Vmm& vRawIndices, Vmask& kDstMask, Vmask& kAuxMask) {
    // Compensate negative indices.
    if (jcp.reverseIndexing) {
        vpcmpgtd(kAuxMask, vmmZeros, vRawIndices);
        vpand(kAuxMask, kAuxMask, vmmAxisDim);
        uni_vpaddd(vRawIndices, vRawIndices, kAuxMask);
    }
    // Check boundaries.
    vpcmpgtd(kDstMask, vmmAxisDim, vRawIndices);
    vpcmpgtd(kAuxMask, vmmZeros, vRawIndices);
    vpandn(kDstMask, kAuxMask, kDstMask);
    // Multiply by type size.
    if (jcp.dataTypeSize > 1)
        vpslld(vRawIndices, vRawIndices, dataTypeShift);
}

template <>
void jitUniGatherKernel<x64::avx512_common>::normalizeRawIndices(Vmm& vRawIndices, Vmask& kDstMask, Vmask& kAuxMask) {
    // Compensate negative indices.
    if (jcp.reverseIndexing) {
        vpcmpgtd(kAuxMask, vmmZeros, vRawIndices);
        vpaddd(vRawIndices | kAuxMask, vRawIndices, vmmAxisDim);
    }
    // Check boundaries.
    vpcmpgtd(kAuxMask, vmmAxisDim, vRawIndices);
    vpcmpd(kDstMask | kAuxMask, vmmZeros, vRawIndices, 2); // 2 - LE
    // Multiply by type size.
    if (jcp.dataTypeSize > 1)
        vpslld(vRawIndices, vRawIndices, dataTypeShift);
}

template <>
void jitUniGatherKernel<x64::avx2>::normWithUpperBound(Vmm& vTarget, Vmm& vMax, Vmask& kAuxMask) {
    vpcmpgtd(kAuxMask, vMax, vTarget);
    vpandn(kAuxMask, kAuxMask, vMax);
    vpsubd(vTarget, vTarget, kAuxMask);
}

template <>
void jitUniGatherKernel<x64::avx512_common>::normWithUpperBound(Vmm& vTarget, Vmm& vMax, Vmask& kAuxMask) {
    vpcmpd(kAuxMask, vMax, vTarget, 2); // 2 -> LE
    vpsubd(vTarget | kAuxMask, vTarget, vMax);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::tail(bool isShortIdx, bool shiftFirst, bool blocked) {
    auto& vmmAux0 = vmmAuxContainer[0];
    auto& vmmAux1 = vmmAuxContainer[2];
    auto& kGatherMask = masksContainer[vmmAuxContainer[3].getIdx()];
    auto& kAuxMask0 = masksContainer[vmmAux0.getIdx()];
    Xbyak::Label lEnd;

    const int secondStepCycles = jcp.dataTypeSize == 4 ? 1 : jcp.dataTypeSize == 2 ? 2 : 4;
    for (int p = 0; p < secondStepCycles; p++) {
        cmp(regWorkAmount, 0);
        jle(lEnd, T_NEAR);

        // calcSrcShift -> vmmAuxContainer[1]
        if (isShortIdx) {
            if (blocked)
                calcSrcShiftShortBlock(vmmAuxContainer, kGatherMask, p > 0 || shiftFirst);
            else
                calcSrcShiftShort(vmmAuxContainer, kGatherMask, p > 0 || shiftFirst);
        } else {
            if (blocked)
                calcSrcShiftLongBlock(vmmAuxContainer, kGatherMask, p > 0 || shiftFirst);
            else
                calcSrcShiftLong(vmmAuxContainer, kGatherMask, p > 0 || shiftFirst);
        }

        fillRestWorkMask(vmmAux1, vmmAux0, regWorkAmount, regAux1, rdx);

        // Combining masks.
        if (isa == x64::avx512_common) {
            // Explicit Opmask declaration is required here.
            auto kMask0 = Xbyak::Opmask(kAuxMask0.getIdx());
            auto kMask1 = Xbyak::Opmask(kGatherMask.getIdx());

            vpmovd2m(kMask0, vmmAux1);
            kandd(kMask1, kMask1, kMask0);
            if (jcp.dataTypeSize == 4)
                kmovd(kMask0, kMask1);
        } else if (isa == x64::avx2) {
            auto& vGatherMask = vmmAuxContainer[kGatherMask.getIdx()];

            vpand(vGatherMask, vGatherMask, vmmAux1);
            if (jcp.dataTypeSize == 4)
                vmovups(vmmAux0, vGatherMask);
        }

        uni_vmovups(vmmAux1, vmmZeros);
        uni_vpgatherdd(vmmAux1, ptr[regSrc + vmmAuxContainer[1]], kGatherMask);
        if (jcp.dataTypeSize == 4) {
            uni_vmovups_tail(ptr[regDst], kAuxMask0, vmmAux1);
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
void jitUniGatherKernel<x64::avx2>::calcSrcShiftLong(Vmm* vAuxPool, Vmask& vDstMask, bool shiftFirst) {
    auto& vmmDstShifts = vAuxPool[1];
    auto& vmmAux0 = vAuxPool[0];
    auto& vmmAux1 = vAuxPool[2];
    auto& kAuxMask0 = masksContainer[vmmAux1.getIdx()];

    Xbyak::Label lIdxStride, lExit;
    if (shiftFirst)
        uni_vpaddd(vmmSpecIdxB, vmmSpecIdxB, vmmVecLenB);

    add(regIdxIter, vlen);
    cmp(regIdxIter, regSpecIdxSizeB);
    jge(lIdxStride, T_NEAR);
        if (jcp.batchDims > 0lu) {
            uni_vpaddd(vmmDstShifts, vmmIdxBatchSumB, vmmSpecIdxB);
            auto& xmmAux = xmmAuxContainer[vmmDstShifts.getIdx()];
            vmovd(reg32Aux1, xmmAux);
        } else {
            auto& xmmAux = xmmAuxContainer[vmmSpecIdxB.getIdx()];
            vmovd(reg32Aux1, xmmAux);
        }
        vmovdqu(vmmDstShifts, ptr[regIndices + regAux1]);
        normalizeRawIndices(vmmDstShifts, vDstMask, kAuxMask0);
        if (jcp.beforeAxisSize != 1lu)
            uni_vpaddd(vmmDstShifts, vmmDstShifts, vmmSrcBeforeAxisSumB);
    jmp(lExit, T_NEAR);
    L(lIdxStride);
        sub(regIdxIter, regSpecIdxSizeB);
        uni_vpcmpeqd(vDstMask, vmmAux0, vmmAux0);
        if (shiftFirst) {
            vpcmpgtd(vmmAux0, vmmSpecIdxSizeB, vmmSpecIdxB);
            vpandn(vmmAux1, vmmAux0, vmmSpecIdxSizeB);
            vpsubd(vmmAux1, vmmSpecIdxB, vmmAux1);
            if (jcp.batchDims > 0lu)
                uni_vpaddd(vmmAux1, vmmIdxBatchSumB, vmmAux1);
            vpsubd(vmmSpecIdxB, vmmSpecIdxB, vmmSpecIdxSizeB);
        } else {
            if (jcp.batchDims > 0lu) {
                uni_vpaddd(vmmAux0, vmmIdxBatchSumB, vmmSpecIdxB);
                uni_vpgatherdd(vmmDstShifts, ptr[regIndices + vmmAux0], vDstMask);
            } else {
                uni_vpgatherdd(vmmDstShifts, ptr[regIndices + vmmSpecIdxB], vDstMask);
            }
            normalizeRawIndices(vmmDstShifts, vDstMask, kAuxMask0);
            uni_vpaddd(vmmDstShifts, vmmDstShifts, vmmSrcBeforeAxisSumB);

            vpbroadcastd(vmmAux0, xmmSpecIdxB);
            vpcmpgtd(vmmAux1, vmmAux0, vmmSpecIdxB);
            vpandn(vmmAux0, vmmAux1, vmmSpecIdxSizeB);
            uni_vpsubd(vmmSpecIdxB, vmmSpecIdxB, vmmAux0);

            vpandn(vmmAux0, vmmAux1, vmmAxisAndAfterAxisSizeB);
            uni_vpaddd(vmmSrcBeforeAxisSumB, vmmSrcBeforeAxisSumB, vmmAux0);
        }

        if (jcp.batchDims > 0lu) {
            Xbyak::Label l1;
            inc(regBetweenBatchAndAxisIter);
            cmp(regBetweenBatchAndAxisIter, regBetweenBatchAndAxisSize);
            jl(l1, T_NEAR);
                mov(regBetweenBatchAndAxisIter, 0);
                if (shiftFirst) {
                    uni_vpaddd(vmmIdxBatchSumB, vmmIdxBatchSumB, vmmSpecIdxSizeB);
                    vpandn(vmmDstShifts, vmmAux0, vmmSpecIdxSizeB);
                    uni_vpaddd(vmmAux1, vmmAux1, vmmDstShifts);
                } else {
                    vpandn(vmmAux0, vmmAux1, vmmSpecIdxSizeB);
                    uni_vpaddd(vmmIdxBatchSumB, vmmIdxBatchSumB, vmmAux0);
                }
            L(l1);
        }

        if (shiftFirst) {
            uni_vpgatherdd(vmmDstShifts, ptr[regIndices + vmmAux1], vDstMask);
            normalizeRawIndices(vmmDstShifts, vDstMask, kAuxMask0);

            if (jcp.beforeAxisSize != 1lu) {
                vpandn(vmmAux0, vmmAux0, vmmAxisAndAfterAxisSizeB);
                uni_vpaddd(vmmAux0, vmmAux0, vmmSrcBeforeAxisSumB);
                uni_vpaddd(vmmSrcBeforeAxisSumB, vmmSrcBeforeAxisSumB, vmmAxisAndAfterAxisSizeB);

                uni_vpaddd(vmmDstShifts, vmmDstShifts, vmmAux0);
            }
        }
    L(lExit);
}

template <>
void jitUniGatherKernel<x64::avx512_common>::calcSrcShiftLong(Vmm* vAuxPool, Vmask& vDstMask, bool shiftFirst) {
    auto& vmmDstShifts = vAuxPool[1];
    auto& vmmAux0 = vAuxPool[0];
    auto& vmmAux1 = vAuxPool[2];
    auto& kAuxMask0 = masksContainer[vmmAux1.getIdx()];
    auto& vAuxMask1 = masksContainer[vmmAux1.getIdx() + 1];

    Xbyak::Label lIdxStride, lExit;
    if (shiftFirst)
        vpaddd(vmmSpecIdxB, vmmSpecIdxB, vmmVecLenB);

    add(regIdxIter, vlen);
    cmp(regIdxIter, regSpecIdxSizeB);
    jge(lIdxStride, T_NEAR);
        auto& xmmAux = xmmAuxContainer[vmmDstShifts.getIdx()];
        uni_vpaddd(vmmDstShifts, vmmIdxBatchSumB, vmmSpecIdxB);
        vmovd(reg32Aux1, xmmAux);
        vmovdqu64(vmmDstShifts, ptr[regIndices + regAux1]);
        normalizeRawIndices(vmmDstShifts, vDstMask, kAuxMask0);
        vpaddd(vmmDstShifts, vmmDstShifts, vmmSrcBeforeAxisSumB);
    jmp(lExit, T_NEAR);
    L(lIdxStride);
        sub(regIdxIter, regSpecIdxSizeB);
        vpcmpb(vDstMask, vmmDstShifts, vmmDstShifts, 0);
        if (shiftFirst) {
            vpcmpd(vAuxMask1, vmmSpecIdxSizeB, vmmSpecIdxB, 2); // 2 - LE
            vpaddd(vmmAux1, vmmIdxBatchSumB, vmmSpecIdxB);
            vpsubd(vmmAux1 | vAuxMask1, vmmAux1, vmmSpecIdxSizeB);
            vpsubd(vmmSpecIdxB, vmmSpecIdxB, vmmSpecIdxSizeB);
        } else {
            vpaddd(vmmAux0, vmmIdxBatchSumB, vmmSpecIdxB);
            uni_vpgatherdd(vmmDstShifts, ptr[regIndices + vmmAux0], vDstMask);
            normalizeRawIndices(vmmDstShifts, vDstMask, kAuxMask0);
            vpaddd(vmmDstShifts, vmmDstShifts, vmmSrcBeforeAxisSumB);

            vpbroadcastd(vmmAux0, xmmSpecIdxB);
            vpcmpd(vAuxMask1, vmmAux0, vmmSpecIdxB, 2); // 2 - LE
            vpsubd(vmmSpecIdxB | vAuxMask1, vmmSpecIdxB, vmmSpecIdxSizeB);

            vpaddd(vmmSrcBeforeAxisSumB | vAuxMask1, vmmSrcBeforeAxisSumB, vmmAxisAndAfterAxisSizeB);
        }

        Xbyak::Label l1;
        inc(regBetweenBatchAndAxisIter);
        cmp(regBetweenBatchAndAxisIter, regBetweenBatchAndAxisSize);
        jl(l1, T_NEAR);
            mov(regBetweenBatchAndAxisIter, 0);
            if (shiftFirst) {
                vpaddd(vmmIdxBatchSumB, vmmIdxBatchSumB, vmmSpecIdxSizeB);
                vpaddd(vmmAux1 | vAuxMask1, vmmAux1, vmmSpecIdxSizeB);
            } else {
                vpaddd(vmmIdxBatchSumB | vAuxMask1, vmmIdxBatchSumB, vmmSpecIdxSizeB);
            }
        L(l1);

        if (shiftFirst) {
            uni_vpgatherdd(vmmDstShifts, ptr[regIndices + vmmAux1], vDstMask);
            normalizeRawIndices(vmmDstShifts, vDstMask, kAuxMask0);

            vpaddd(vmmDstShifts, vmmDstShifts, vmmSrcBeforeAxisSumB);
            vpaddd(vmmDstShifts | vAuxMask1, vmmDstShifts, vmmAxisAndAfterAxisSizeB);
            vpaddd(vmmSrcBeforeAxisSumB, vmmSrcBeforeAxisSumB, vmmAxisAndAfterAxisSizeB);
        }
    L(lExit);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::calcSrcShiftLongBlock(Vmm* vAuxPool, Vmask& dstMask, bool shiftFirst) {
    // Most likely there will no significant performance gain vs memcpy in reference implementation on big blocks after axis,
    // therefore no time was invested to this case yet.
    throw std::invalid_argument("Unsupported case.");
}

// Requires vAuxPool length 2.
// Returns gathered data in vAuxPool[1].
template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::calcSrcShiftShort(Vmm* vAuxPool, Vmask& kDstMask, bool shiftFirst) {
    auto& vDstShifts = vAuxPool[1];
    auto& vAux0 = vAuxPool[0];
    auto& kAuxMask0 = masksContainer[vAuxPool[0].getIdx()];

    if (shiftFirst) {
        if (jcp.beforeAxisSize != 1lu)
            vpaddd(vmmSrcBeforeAxisSumB, vmmSrcBeforeAxisSumB, vmmBeforeAxDiffB);
        // No sense to permute if specIdxSize is one of {1, 2, 4, 8}.
        if (jcp.specIdxSize != 1 && jcp.specIdxSize != 2 && jcp.specIdxSize != 4 && jcp.specIdxSize != 8) {
            vpermd(vmmSpecIdxB, vmmPermIdxMask, vmmSpecIdxB);
            if (jcp.beforeAxisSize != 1lu)
                vpermd(vmmBeforeAxDiffB, vmmPermIdxMask, vmmBeforeAxDiffB);
        }
    }

    uni_vpcmpeqd(kDstMask, vAux0, vAux0);
    if (jcp.batchDims > 0lu) {
        // Calculate indices batch sum.
        uni_vcvtdq2ps(vAux0, vmmSrcBeforeAxisSumB);
        uni_vcvtdq2ps(vDstShifts, vmmSrcAfterBatchSizeB);
        uni_vdivps(vAux0, vAux0, vDstShifts);
        uni_vroundps(vAux0, vAux0, 0x1);
        uni_vcvtps2dq(vAux0, vAux0);

        uni_vpmulld(vAux0, vAux0, vmmSpecIdxSizeB);
        uni_vpaddd(vAux0, vAux0, vmmSpecIdxB);

        uni_vpgatherdd(vDstShifts, ptr[regIndices + vAux0], kDstMask);
    } else {
        uni_vpgatherdd(vDstShifts, ptr[regIndices + vmmSpecIdxB], kDstMask);
    }

    normalizeRawIndices(vDstShifts, kDstMask, kAuxMask0);
    if (jcp.beforeAxisSize != 1lu)
        vpaddd(vDstShifts, vDstShifts, vmmSrcBeforeAxisSumB);
}

// Requires vAuxPool length 3.
// Returns gathered data in vAuxPool[1].
template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::calcSrcShiftShortBlock(Vmm* vAuxPool, Vmask& kDstMask, bool shiftFirst) {
    auto& vDstShifts = vAuxPool[1];
    auto& vAux0 = vAuxPool[0];
    auto& vAux1 = vAuxPool[2];
    auto& kAuxMask0 = masksContainer[vAuxPool[0].getIdx()];
    const uint64_t specIdxAndAfterAxisSize = jcp.specIdxSize * jcp.afterAxisSize;

    if (shiftFirst) {
        if (jcp.specIdxSize != 1) {
            vpaddd(vmmSpecIdxB, vmmSpecIdxB, vmmSpecIdxDiff);
            normWithUpperBound(vmmSpecIdxB, vmmSpecIdxSizeB, kAuxMask0);
        }
        // No sense to permute if afterAxisSize is one of {1, 2, 4, 8}.
        if (jcp.afterAxisSize != 1 && jcp.afterAxisSize != 2 && jcp.afterAxisSize != 4 && jcp.afterAxisSize != 8) {
            vpermd(vmmAfterAxisIdxB, vmmAfterAxisPermMask, vmmAfterAxisIdxB);
            if (jcp.specIdxSize != 1)
                vpermd(vmmSpecIdxDiff, vmmAfterAxisPermMask, vmmSpecIdxDiff);
        }

        if (jcp.beforeAxisSize != 1lu) {
            if (!jcp.dynamicShapes) {
                if (specIdxAndAfterAxisSize > 0lu && specIdxAndAfterAxisSize <= idxElPerVec) {
                    vpaddd(vmmSrcBeforeAxisSumB, vmmSrcBeforeAxisSumB, vmmBeforeAxDiffB);
                    vmovups(vAux1, vmmSrcBeforeAxisSumB);
                    if (specIdxAndAfterAxisSize != 1 && specIdxAndAfterAxisSize != 2 && specIdxAndAfterAxisSize != 4 &&
                            specIdxAndAfterAxisSize % 8 != 0)
                        vpermd(vmmBeforeAxDiffB, vmmBeforeAxPermMask, vmmBeforeAxDiffB);
                } else {
                    Xbyak::Label lBeforeAxStep, lBeforeAxStepEnd;
                    add(rSpecIdxAndAfterAxIterB, vlen);
                    cmp(rSpecIdxAndAfterAxIterB, rSpecIdxAndAfterAxSizeB);
                    jl(lBeforeAxStep, T_NEAR);
                        sub(rSpecIdxAndAfterAxIterB, rSpecIdxAndAfterAxSizeB);

                        vpmulld(vAux0, vmmSpecIdxB, vmmAfterAxisSize);
                        vpaddd(vAux0, vAux0, vmmAfterAxisIdxB);
                        Xbyak::Xmm& xAux0 = xmmAuxContainer[vAux0.getIdx()];
                        vpbroadcastd(vAux1, xAux0);
                        vpcmpgtd(vAux1, vAux1, vAux0);
                        vpand(vAux1, vAux1, vmmAxisAndAfterAxisSizeB);
                        vpaddd(vAux1, vmmSrcBeforeAxisSumB, vAux1);
                        vpaddd(vmmSrcBeforeAxisSumB, vmmSrcBeforeAxisSumB, vmmAxisAndAfterAxisSizeB);
                        jmp(lBeforeAxStepEnd);
                    L(lBeforeAxStep);
                        uni_vmovups(vAux1, vmmSrcBeforeAxisSumB);
                    L(lBeforeAxStepEnd);
                }
            } else {
            }
        }
    } else {
        if (jcp.beforeAxisSize != 1lu) {
            uni_vmovups(vAux1, vmmSrcBeforeAxisSumB);
            if (specIdxAndAfterAxisSize > idxElPerVec) {
                // Broadcast the last element.
                if (isa == x64::avx512_common) { // TODO: A5
                } else {
                    vpermq(vmmSrcBeforeAxisSumB, vmmSrcBeforeAxisSumB, 0xFF);
                    vpshufd(vmmSrcBeforeAxisSumB, vmmSrcBeforeAxisSumB, 0x55);
                }

                Xbyak::Label lBeforeAxStepEnd1;
                add(rSpecIdxAndAfterAxIterB, vlen);
                cmp(rSpecIdxAndAfterAxIterB, rSpecIdxAndAfterAxSizeB);
                jl(lBeforeAxStepEnd1, T_NEAR);
                    sub(rSpecIdxAndAfterAxIterB, rSpecIdxAndAfterAxSizeB);
                cmp(rSpecIdxAndAfterAxIterB, 0);
                jne(lBeforeAxStepEnd1, T_NEAR);
                    vpaddd(vmmSrcBeforeAxisSumB, vmmSrcBeforeAxisSumB, vmmAxisAndAfterAxisSizeB);
                L(lBeforeAxStepEnd1);
            }
        }
    }
//uni_vmovups(ptr[regDst], vAux1);

    uni_vpcmpeqd(kDstMask, vAux0, vAux0);
    if (jcp.batchDims > 0lu) {
        // Calculate indices batch sum.
        uni_vcvtdq2ps(vAux0, vAux1);
        uni_vcvtdq2ps(vDstShifts, vmmSrcAfterBatchSizeB);
        uni_vdivps(vAux0, vAux0, vDstShifts);
        uni_vroundps(vAux0, vAux0, 0x1);
        uni_vcvtps2dq(vAux0, vAux0);

        uni_vpmulld(vAux0, vAux0, vmmSpecIdxSizeB);
        uni_vpaddd(vAux0, vAux0, vmmSpecIdxB);

        uni_vpgatherdd(vDstShifts, ptr[regIndices + vAux0], kDstMask);
    } else {
        uni_vpgatherdd(vDstShifts, ptr[regIndices + vmmSpecIdxB], kDstMask);
    }

    normalizeRawIndices(vDstShifts, kDstMask, kAuxMask0);

    if (jcp.afterAxisSize != 1lu) {
        vpmulld(vDstShifts, vDstShifts, vmmAfterAxisSize);
        vpaddd(vDstShifts, vDstShifts, vmmAfterAxisIdxB);
    }
    if (jcp.beforeAxisSize != 1lu)
        vpaddd(vDstShifts, vDstShifts, vAux1);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::process(bool isShortIdx, bool blocked) {
    Xbyak::Label lTailProc, lEndProc;
    cmp(regWorkAmount, dataElPerVec);
    jl(lTailProc, T_NEAR);
        if (jcp.dataTypeSize == 4)
            process32b(isShortIdx, blocked);
        else if (jcp.dataTypeSize == 2)
            process16b(isShortIdx, blocked);
        else if (jcp.dataTypeSize == 1)
            process8b(isShortIdx, blocked);
    jmp(lEndProc, T_NEAR);
    L(lTailProc);
        tail(isShortIdx, false, blocked);
    L(lEndProc);
}

template <x64::cpu_isa_t isa>
void jitUniGatherKernel<isa>::process32b(bool isShortIdx, bool blocked) {
    Xbyak::Label lDstIdxLoop, lTail;

    // First iteration
    shiftIdxAndGather(vmmAuxContainer, isShortIdx, false, blocked);
    uni_vmovups(ptr[regDst], vmmAuxContainer[1]);

    // Main loop
    L(lDstIdxLoop);
    {
        add(regDst, vlen);
        sub(regWorkAmount, dataElPerVec);
        cmp(regWorkAmount, dataElPerVec);
        jl(lTail, T_NEAR);

        shiftIdxAndGather(vmmAuxContainer, isShortIdx, true, blocked);
        uni_vmovups(ptr[regDst], vmmAuxContainer[1]);

        jmp(lDstIdxLoop, T_NEAR);
    }

    L(lTail);
    tail(isShortIdx, true, blocked);
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
    auto& kGatherMask = masksContainer[vAuxPool[0].getIdx()];
    if (blocked) {
        if (isShortIdx) {
            calcSrcShiftShortBlock(&vAuxPool[1], kGatherMask, shiftFirst);
        } else {
            calcSrcShiftLongBlock(&vAuxPool[1], kGatherMask, shiftFirst);
        }
    } else {
        if (isShortIdx) {
            calcSrcShiftShort(&vAuxPool[1], kGatherMask, shiftFirst);
        } else {
            calcSrcShiftLong(&vAuxPool[1], kGatherMask, shiftFirst);
        }
    }
    uni_vmovups(vAuxPool[1], vmmZeros);
    uni_vpgatherdd(vAuxPool[1], ptr[regSrc + vAuxPool[2]], kGatherMask);
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
