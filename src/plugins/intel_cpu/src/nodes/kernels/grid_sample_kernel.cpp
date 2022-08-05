// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "grid_sample_kernel.hpp"
#include <ie_common.h>

using namespace dnnl::impl::cpu;

namespace ov {
namespace intel_cpu {

const unsigned jitGridSampleKernelBase::permGridMask32bA2[8]  = {0, 2, 4, 6, 1, 3, 5, 7};
const unsigned jitGridSampleKernelBase::permGridMask32bA5[16]  = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};

//const unsigned jitGridSmapleKernelBase::shufMask8bitUni[16]  = {0x0C080400, 0x80808080, 0x80808080, 0x80808080, 0x0C080400, 0x80808080, 0x80808080, 0x80808080,
//                                                            0x0C080400, 0x80808080, 0x80808080, 0x80808080, 0x0C080400, 0x80808080, 0x80808080, 0x80808080};
//const unsigned jitGridSmapleKernelBase::permMask8bitA2[8]    = {0, 4, 1, 5, 2, 6, 3, 7};
//const unsigned jitGridSmapleKernelBase::permMask8bitA5[16]   = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
//
//const unsigned jitGridSmapleKernelBase::shufMask16bitUni[16] = {0x05040100, 0x0D0C0908, 0x80808080, 0x80808080, 0x05040100, 0x0D0C0908, 0x80808080, 0x80808080,
//                                                            0x05040100, 0x0D0C0908, 0x80808080, 0x80808080, 0x05040100, 0x0D0C0908, 0x80808080, 0x80808080};
//const unsigned jitGridSmapleKernelBase::permMask16bitA2[8]   = {0, 1, 4, 5, 2, 3, 6, 7};
//const unsigned jitGridSmapleKernelBase::permMask16bitA5[16]  = {0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15};
//
//const unsigned jitGridSmapleKernelBase::incVec[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

#define GET_OFF(field) offsetof(jGridSamplesExecArgs, field)

template <x64::cpu_isa_t isa>
jitUniGridSampleKernel<isa>::jitUniGridSampleKernel(const jGridSampleConfParams& jcp) :
        jitGridSampleKernelBase(jcp) {
    vlen = x64::cpu_isa_traits<isa>::vlen;
    dataTypeSize = jcp.inDataPrc.size();
    dataElPerVec = vlen / dataTypeSize;
    idxElPerVec = vlen / indicesTypeSize;
    if (dataTypeSize == 2)
        dataTypeShift = 1;
    else if (dataTypeSize == 4)
        dataTypeShift = 2;

    if (isa == x64::avx512_core) {
//        permMask8bitUni = permMask8bitA5;
        permGridMaskUni = permGridMask32bA5;
    } else if (isa == x64::avx2) {
//        permMask8bitUni = permMask8bitA2;
        permGridMaskUni = permGridMask32bA2;
    } else {
        // TODO:
    }
}

template <x64::cpu_isa_t isa>
void jitUniGridSampleKernel<isa>::create_ker() {
    auto code = x64::jit_generator::create_kernel();
    if (code != dnnl::impl::status::success)
        IE_THROW() << "Could not create Gather kernel. Error code: " << std::to_string(code);
    ker_ = (decltype(ker_))jit_ker();
}

template <x64::cpu_isa_t isa>
void jitUniGridSampleKernel<isa>::generate() {
    this->preamble();

    mov(regSrc, ptr[regParams + GET_OFF(src)]);
    mov(regDst, ptr[regParams + GET_OFF(dst)]);
    mov(regGrid, ptr[regParams + GET_OFF(grid)]);

//    mov(regWorkAmount, ptr[regParams + GET_OFF(workAmount)]);
    mov(regDstChStepB, ptr[regParams + GET_OFF(dstChStepB)]);

    mov(regChannelsNum, ptr[regParams + GET_OFF(channelsNum)]);
    mov(regAux1, ptr[regParams + GET_OFF(srcWidthFl)]);
    uni_vpbroadcastd(vSrcWidthFl, ptr[regAux1]);
    mov(regAux1, ptr[regParams + GET_OFF(srcHeightFl)]);
    uni_vpbroadcastd(vSrcHeightFl, ptr[regAux1]);

    mov(regAux1, reinterpret_cast<uintptr_t>(permGridMaskUni));
    uni_vmovups(vPermGridMask, ptr[regAux1]);

    uni_vpxor(vZeros, vZeros, vZeros);

//    auto& vAux0 = vAuxContainer[0];
//    auto& vAux1 = vAuxContainer[1];
//    auto& xAux0 = xmmAuxContainer[0];
//    auto& xAux1 = xmmAuxContainer[1];

    if (!jcp.dynamicShapes) {
//        mov(regWorkAmount, ptr[regParams + GET_OFF(workAmount)]);
//        mov(regAux1, ptr[regParams + GET_OFF(specIndicesSize)]);
//        uni_vpbroadcastd(vmmSpecIdxSizeB, ptr[regAux1]);
//        uni_vpslld(vmmSpecIdxSizeB, vmmSpecIdxSizeB, idxTypeShift); // multiply by indices type size.
//
//        mov(regAux1, ptr[regParams + GET_OFF(specIdxB)]);
//        uni_vmovups(vmmSpecIdxB, ptr[regAux1]);
//
//        if (jcp.beforeAxisSize != 1lu) {
//            mov(regAux1, ptr[regParams + GET_OFF(dataBeforeAxisSumB)]);
//            uni_vmovups(vmmSrcBeforeAxisSumB, ptr[regAux1]);
//        }
//
//        if (jcp.afterAxisSize == 1lu) { // Elementwise case.
//            uni_vmovd(reg32SpecIdxSizeB, xmmSpecIdxSizeB);
//            if (jcp.beforeAxisSize != 1lu) {
//                mov(regAux1, ptr[regParams + GET_OFF(axisAndAfterAxisSizeB)]);
//                uni_vpbroadcastd(vmmAxisAndAfterAxisSizeB, ptr[regAux1]);
//            }
//
//            mov(regAux1, ptr[regParams + GET_OFF(idxBatchSumB)]);
//            uni_vmovups(vmmIdxBatchSumB, ptr[regAux1]);
//
//            mov(regAux1, ptr[regParams + GET_OFF(betweenBatchAndAxisSize)]);
//            mov(regBetweenBatchAndAxisSize, ptr[regAux1]);
//            mov(regBetweenBatchAndAxisIter, ptr[regParams + GET_OFF(betweenBatchAndAxisIter)]);
//
//            if (jcp.specIdxSize < idxElPerVec) { // Short case.
//                if (jcp.specIdxSize != 1 && jcp.specIdxSize != 2 && jcp.specIdxSize != 4 && jcp.specIdxSize != 8 && jcp.specIdxSize != 16) {
//                    mov(regAux1, ptr[regParams + GET_OFF(permIdxMask)]);
//                    uni_vmovups(vmmPermIdxMask, ptr[regAux1]);
//                }
//                if (jcp.beforeAxisSize != 1lu) {
//                    mov(regAux1, ptr[regParams + GET_OFF(beforeAxisDiff)]);
//                    uni_vmovups(vmmBeforeAxDiffB, ptr[regAux1]);
//                    if (dataTypeSize != 1)
//                        uni_vpslld(vmmBeforeAxDiffB, vmmBeforeAxDiffB, dataTypeShift); // multiply by data type size
//                }
//                if (jcp.batchDims > 0lu) {
//                    mov(regAux1, ptr[regParams + GET_OFF(srcAfterBatchSizeB)]);
//                    uni_vpbroadcastd(vmmSrcAfterBatchSizeB, ptr[regAux1]);
//                }
//
//                process(true, false);
//            } else { // Long case.
//                uni_vmovd(reg32IdxIter, xmmSpecIdxB);
//                fillVlenVector();
//
//                process(false, false);
//            }
//        } else { // Blocked case.
//            if (jcp.afterAxisSize <= idxElPerVec) { // Short case.
//                mov(regAux1, ptr[regParams + GET_OFF(afterAxIdxB)]);
//                uni_vmovups(vmmAfterAxisIdxB, ptr[regAux1]);
//                mov(regAux1, ptr[regParams + GET_OFF(afterAxisPermMask)]);
//                uni_vmovups(vmmAfterAxisPermMask, ptr[regAux1]);
//                mov(regAux1, ptr[regParams + GET_OFF(specIdxDiff)]);
//                uni_vmovups(vmmSpecIdxDiff, ptr[regAux1]);
//                mov(regAux1, ptr[regParams + GET_OFF(srcAfterBatchSizeB)]);
//                uni_vpbroadcastd(vmmSrcAfterBatchSizeB, ptr[regAux1]);
//                mov(regAux1, ptr[regParams + GET_OFF(afterAxisSize)]);
//                uni_vpbroadcastd(vmmAfterAxisSize, ptr[regAux1]);
//
//                if (jcp.beforeAxisSize != 1lu) {
//                    mov(rSpecIdxAndAfterAxIterB, ptr[regParams + GET_OFF(specIdxAndAfterAxIterB)]);
//                    mov(rSpecIdxAndAfterAxSizeB, ptr[regParams + GET_OFF(specIdxAndAfterAxSizeB)]);
//                    if (jcp.specIdxSize * jcp.afterAxisSize < idxElPerVec) {
//                        mov(regAux1, ptr[regParams + GET_OFF(beforeAxisDiff)]);
//                        uni_vmovups(vmmBeforeAxDiffB, ptr[regAux1]);
//                    } else {
//                        mov(regAux1, ptr[regParams + GET_OFF(axisAndAfterAxisSizeB)]);
//                        uni_vpbroadcastd(vmmAxisAndAfterAxisSizeB, ptr[regAux1]);
//                    }
//                    const uint64_t specIdxAndAfterAxisSize = jcp.specIdxSize * jcp.afterAxisSize;
//                    if (specIdxAndAfterAxisSize != 1 && specIdxAndAfterAxisSize != 2 && specIdxAndAfterAxisSize != 4 &&
//                            specIdxAndAfterAxisSize != 8 && specIdxAndAfterAxisSize != 16) {
//                        mov(regAux1, ptr[regParams + GET_OFF(beforeAxisPermMask)]);
//                        uni_vmovups(vmmBeforeAxPermMask, ptr[regAux1]);
//                    }
//                }
//
//                process(true, true);
//            } else { // Long case.
//                IE_THROW() << "Gather kernel does not support static shape with after axis size greater than elements in vector.";
//            }
//        }
    } else { // Dynamic shapes.
//        mov(regWorkAmount, jcp.workAmount);
//        mov(regAux1, ptr[regParams + GET_OFF(start)]);
//        uni_vpbroadcastd(vmmSpecIdxB, ptr[regAux1]);
//        mov(regAux1, reinterpret_cast<uintptr_t>(incVec));
//        uni_vpaddd(vmmSpecIdxB, vmmSpecIdxB, ptr[regAux1]);
//        vcvtdq2ps(vmmSpecIdxB, vmmSpecIdxB);
//
//        // Formula: specIndices = (start % specIndicesSize) * idxTypeSize
//        mov(regAux1, ptr[regParams + GET_OFF(specIndicesSize)]);
//        uni_vpbroadcastd(vmmSpecIdxSizeB, ptr[regAux1]);
//        uni_vcvtdq2ps(vAux1, vmmSpecIdxSizeB);
//        uni_vdivps(vmmSrcBeforeAxisSumB, vmmSpecIdxB, vAux1);
//        uni_vroundps(vmmSrcBeforeAxisSumB, vmmSrcBeforeAxisSumB, 0x1);
//        uni_vfnmadd231ps(vmmSpecIdxB, vmmSrcBeforeAxisSumB, vAux1);
//        uni_vcvtps2dq(vmmSpecIdxB, vmmSpecIdxB);
//        uni_vpslld(vmmSpecIdxB, vmmSpecIdxB, idxTypeShift); // multiply by indices type size.
//        uni_vpslld(vmmSpecIdxSizeB, vmmSpecIdxSizeB, idxTypeShift); // multiply by indices type size.
//        uni_vmovd(reg32SpecIdxSizeB, xmmSpecIdxSizeB);
//
//        mov(regAux1, ptr[regParams + GET_OFF(betweenBatchAndAxisSize)]);
//        uni_vpbroadcastd(vAux1, ptr[regAux1]);
//        uni_vmovd(reg32BetweenBatchAndAxisSize, xAux1);
//        uni_vcvtdq2ps(vAux1, vAux1);
//        uni_vdivps(vmmIdxBatchSumB, vmmSrcBeforeAxisSumB, vAux1);
//        uni_vroundps(vmmIdxBatchSumB, vmmIdxBatchSumB, 0x1);
//        uni_vfnmadd231ps(vmmSrcBeforeAxisSumB, vmmIdxBatchSumB, vAux1);
//        uni_vcvtps2dq(vmmSrcBeforeAxisSumB, vmmSrcBeforeAxisSumB);
//        uni_vmovd(reg32BetweenBatchAndAxisIter, xmmSrcBeforeAxisSum);
//        uni_vcvtps2dq(vmmIdxBatchSumB, vmmIdxBatchSumB);
//
//        mov(regAux1, ptr[regParams + GET_OFF(axisAndAfterAxisSizeB)]);
//        uni_vpbroadcastd(vmmAxisAndAfterAxisSizeB, ptr[regAux1]);
//        // Formula: srcBeforeAxisSum = ((start / specIndicesSize) % betweenBatchAndAxis) * axisAndAfterAxisSize + srcAfterBatchSize * idxBatchSum
//        if (jcp.beforeAxisSize != 1lu) {
//            uni_vpmulld(vmmSrcBeforeAxisSumB, vmmSrcBeforeAxisSumB, vmmAxisAndAfterAxisSizeB);
//            mov(regAux1, ptr[regParams + GET_OFF(srcAfterBatchSizeB)]);
//            uni_vpbroadcastd(vAux0, ptr[regAux1]);
//            uni_vpmulld(vAux0, vAux0, vmmIdxBatchSumB);
//            uni_vpaddd(vmmSrcBeforeAxisSumB, vmmSrcBeforeAxisSumB, vAux0);
//        }
//
//        // Formula: idxBatchSum = specIdxSize * (start / afterBatchSize)
//        uni_vpmulld(vmmIdxBatchSumB, vmmIdxBatchSumB, vmmSpecIdxSizeB);
//
//        Xbyak::Label lBlock, lEnd;
//        mov(regAux2, ptr[regParams + GET_OFF(afterAxSize)]);
//        cmp(regAux2, 1);
//        jg(lBlock, T_NEAR);
//        {
//            Xbyak::Label lLessThanVector1, lTail1, lTail2, lE1;
//
//            cmp(regSpecIdxSizeB, vlen);
//            jl(lLessThanVector1, T_NEAR);
//                uni_vmovd(reg32IdxIter, xmmSpecIdxB);
//                fillVlenVector();
//
//                process(false, false);
//                jmp(lE1, T_NEAR);
//            L(lLessThanVector1);
//                mov(regAux1, ptr[regParams + GET_OFF(permIdxMask)]);
//                uni_vmovups(vmmPermIdxMask, ptr[regAux1]);
//                if (jcp.beforeAxisSize != 1lu) {
//                    mov(regAux1, ptr[regParams + GET_OFF(beforeAxisDiff)]);
//                    uni_vmovups(vmmBeforeAxDiffB, ptr[regAux1]);
//                    if (dataTypeSize != 1)
//                        uni_vpslld(vmmBeforeAxDiffB, vmmBeforeAxDiffB, dataTypeShift); // multiply by data type size
//                }
//                mov(regAux1, ptr[regParams + GET_OFF(srcAfterBatchSizeB)]);
//                uni_vpbroadcastd(vmmSrcAfterBatchSizeB, ptr[regAux1]);
//
//                process(true, false);
//            L(lE1);
//            jmp(lEnd, T_NEAR);
//        }
//        L(lBlock); {
//            mov(regAux1, ptr[regParams + GET_OFF(start)]);
//            uni_vpbroadcastd(vmmAfterAxisIdxB, ptr[regAux1]);
//            mov(regAux1, reinterpret_cast<uintptr_t>(incVec));
//            uni_vpaddd(vmmAfterAxisIdxB, vmmAfterAxisIdxB, ptr[regAux1]);
//            uni_vcvtdq2ps(vmmAfterAxisIdxB, vmmAfterAxisIdxB);
//
//            // afterAxIdxB = (start % afterAxSize) * idxTypeSize
//            movd(xAux0, reg32Aux1);
//            uni_vpbroadcastd(vAux1, xAux0);
//            uni_vcvtdq2ps(vAux1, vAux1);
//            uni_vdivps(vmmSrcBeforeAxisSumB, vmmAfterAxisIdxB, vAux1);
//            uni_vroundps(vmmSrcBeforeAxisSumB, vmmSrcBeforeAxisSumB, 0x1);
//            uni_vfnmadd231ps(vmmAfterAxisIdxB, vmmSrcBeforeAxisSumB, vAux1);
//            uni_vcvtps2dq(vmmAfterAxisIdxB, vmmAfterAxisIdxB);
//            uni_vpslld(vmmAfterAxisIdxB, vmmAfterAxisIdxB, idxTypeShift); // multiply by indices type size.
//
//            Xbyak::Label lLessThanVector2, lTail3, lTail4, lE2;
//
//            cmp(regAux2, dataElPerVec);
//            jl(lLessThanVector2, T_NEAR);
//                uni_vmovd(reg32IdxIter, xmmSpecIdxB);
//                fillVlenVector();
//
////                process(false, true);
//                jmp(lE2, T_NEAR);
//            L(lLessThanVector2);
//                auto& vAux2 = vAuxContainer[2];
//                // Calculate permute mask
//                uni_vmovd(xAux0, reg32Aux2);
//                uni_vpbroadcastd(vAux1, xAux0);
//                mov(regAux1, reinterpret_cast<uintptr_t>(&idxElPerVec));
//                uni_vpbroadcastd(vAux0, ptr[regAux1]);
//                uni_vpsubd(vmmAfterAxisPermMask, vAux0, vAux1);
//                mov(regAux1, reinterpret_cast<uintptr_t>(incVec));
//                uni_vpaddd(vmmAfterAxisPermMask, vmmAfterAxisPermMask, ptr[regAux1]);
//                for (int i = 0; i < 6; i++) {
//                    if (isa == x64::avx512_core) {
//                        Xbyak::Opmask kMask2 = Xbyak::Opmask(vAux2.getIdx());
//                        vpcmpgtd(kMask2, vAux0, vmmAfterAxisPermMask);
//                        uni_vpsubd(vmmAfterAxisPermMask | kMask2, vmmAfterAxisPermMask, vAux1);
//                    } else {
//                        vpcmpgtd(vAux2, vAux0, vmmAfterAxisPermMask);
//                        vpandn(vAux2, vAux2, vAux1);
//                        uni_vpsubd(vmmAfterAxisPermMask, vmmAfterAxisPermMask, vAux2);
//                    }
//                }
//
//                process(true, true);
//            L(lE2);
//        }
//        L(lEnd);
    }

    this->postamble();
}

template <x64::cpu_isa_t isa>
void jitUniGridSampleKernel<isa>::denormalizeRawCoordinates(const Vmm& vWCoord, const Vmm& vHCoord) {
    uni_vfmadd132ps(vWCoord, vWCoef, vWCoef);
    uni_vfmadd132ps(vHCoord, vHCoef, vHCoef);
    if (!jcp.alignCorners) {
        uni_vsubps(vWCoord, vWCoord, vHalf);
        uni_vsubps(vHCoord, vHCoord, vHalf);
    }
}

// TODO: Optimize for AVX5
template <x64::cpu_isa_t isa>
void jitUniGridSampleKernel<isa>::getZeroMask(const Vmm* vAuxPool, const Vmm& vWCoord, const Vmm& vHCoord) {
    const auto& vMask = vAuxPool[0]; // Output storage
    const auto& vAux = vAuxPool[1];

    uni_vcmpps(vAux, vSrcWidthFl, vWCoord, 0x1); // vWCoord < vSrcWidthFl
    uni_vcmpps(vMask, vWCoord, vZeros, 0x2); // vWCoord >= vZeros
    uni_vpand(vMask, vAux, vMask);

    uni_vcmpps(vAux, vSrcHeightFl, vHCoord, 0x1); // vHCoord < vSrcHeightFl
    uni_vpand(vMask, vAux, vMask);
    uni_vcmpps(vAux, vHCoord, vZeros, 0x2); // vHCoord >= vZeros
    uni_vpand(vMask, vAux, vMask);
}

// Returns source values obtained with padded coordinates in vAuxPool[0].
// Requires vAuxPool length 4.
template <x64::cpu_isa_t isa>
void jitUniGridSampleKernel<isa>::getPadded(const Vmm* vAuxPool, const Vmm& vWCoord, const Vmm& vHCoord) {
    const auto& vAux0 = vAuxPool[0]; // Output storage
    const auto& vAux1 = vAuxPool[1];
    const auto& vAux2 = vAuxPool[2];

    if (jcp.paddingMode == PaddingMode::ZEROS) {
        // TODO: Optimize for AVX5
        // SRC HEIGHT and WIDTH!
//        uni_vcmpps(vAux0, vSrcWidthFl, vWCoord, 0x1); // vWCoord < vSrcWidthFl
//        uni_vcmpps(vAux1, vWCoord, vZeros, 0x2); // vWCoord >= vZeros
//        uni_vpand(vAux1, vAux0, vAux1);
//
//        uni_vcmpps(vAux0, vSrcHeightFl, vHCoord, 0x1); // vHCoord < vSrcHeightFl
//        uni_vpand(vAux1, vAux0, vAux1);
//        uni_vcmpps(vAux0, vHCoord, vZeros, 0x2); // vHCoord >= vZeros
//        uni_vpand(vAux1, vAux0, vAux1);

        // ADD CHANNEL SHIFT
//        uni_vmovups(vAux2, vHCoord);
//        uni_vfmadd132ps(vAux2, vWCoord, vSrcWidthFl);
//        uni_vcvtps2dq(vAux2, vAux2);
//
//        uni_vpgatherdd(vAux0, ptr[regSrcIter + vAux2], vAux1); // TODO: 64b 16b 8b?
    } else if (jcp.paddingMode == PaddingMode::BORDER) {
        // TODO:
    } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
        // TODO:
    }
}

// Produces data interpolation.
// Requires vAuxPool length 6.
template <x64::cpu_isa_t isa>
void jitUniGridSampleKernel<isa>::interpolation(const Vmm* vAuxPool, const Vmm& vWCoord, const Vmm& vHCoord) {
    const auto& vAux0 = vAuxPool[0];
    const auto& vAux1 = vAuxPool[1];

    if (jcp.interpolationMode == InterpolationMode::BILINEAR) {
        const auto& vAux2 = vAuxPool[2];
        const auto& vAux3 = vAuxPool[3];
        const auto& vAux4 = vAuxPool[4];
        const auto& vAux5 = vAuxPool[5];

        uni_vmovups(vAux0, vWCoord);
        uni_vmovups(vAux1, vHCoord);
        uni_vroundps(vWCoord, vWCoord, 0x1); // Round floor
        uni_vroundps(vHCoord, vHCoord, 0x1); // Round floor
        uni_vsubps(vAux0, vAux0, vWCoord); // dx -> vAux0
        uni_vsubps(vAux1, vAux1, vHCoord); // dy -> vAux1

        if (jcp.paddingMode == PaddingMode::ZEROS) {
            getZeroMask(&vAuxPool[4], vWCoord, vHCoord);
            uni_vmovups(vAux5, vAux4);
        }

        // PER CHANNEL LOOP
        Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
        const Xbyak::Reg64& rChannel = regAux1;
        const Xbyak::Reg64& rDstTmp = regAux2;
        mov(rChannel, 0);
        mov(rDstTmp, regDst);
        L(lChannelLoopBegin);
        {
            cmp(rChannel, regChannelsNum);
            jge(lChannelLoopEnd, T_NEAR);

            if (jcp.paddingMode == PaddingMode::ZEROS) {
                // (x; y)
                uni_vmovups(vAux2, vHCoord);
                uni_vfmadd132ps(vAux2, vWCoord, vSrcWidthFl);
                vpbroadcastd(vAux3, rChannel);
                uni_vmulps(vAux3, vAux3, vSrcWidthFl);
                uni_vfmadd231ps(vAux2, vAux3, vSrcHeightFl);
                uni_vcvtps2dq(vAux2, vAux2);
                if (dataTypeSize > 1)
                    uni_vpslld(vAux2, vAux2, dataTypeShift); // multiply by source data type size.
                uni_vpxor(vWCoord, vWCoord, vWCoord);
                uni_vpgatherdd(vWCoord, ptr[regSrc + vAux2], vAux4); // v00 -> vWCoord TODO: 64b 16b 8b?
                uni_vmovups(vAux4, vAux5);
                if (jcp.inDataPrc == InferenceEngine::Precision::FP32) {
                    uni_vfmsub213ps(vWCoord, vAux0, vWCoord); // q0 = -(v00 - dx * v00) -> vWCoord
                } else if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                    // TODO:
                }

                // (x + 1; y)
                uni_vcvtdq2ps(vAux2, vAux2);
                uni_vaddps(vAux2, vAux2, vDataTypeSize);
                uni_vcvtps2dq(vAux2, vAux2);
                uni_vpxor(vAux3, vAux3, vAux3);
                uni_vpgatherdd(vAux3, ptr[regSrc + vAux2], vAux4); // TODO: 64b 16b 8b?
                uni_vmovups(vAux4, vAux5);
                if (jcp.inDataPrc == InferenceEngine::Precision::FP32) {
                    uni_vfmsub231ps(vWCoord, vAux3, vAux0); // q0 = -q0 + dx * v01
                } else if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                    // TODO:
                }

                // (x + 1; y + 1)
                uni_vcvtdq2ps(vAux2, vAux2);
                if (dataTypeSize != 1)
                    uni_vpslld(vSrcWidthFl, vSrcWidthFl, dataTypeShift); // multiply by source data type size.
                uni_vaddps(vAux2, vAux2, vSrcWidthFl);
                uni_vcvtps2dq(vAux2, vAux2);
                uni_vpxor(vAux3, vAux3, vAux3);
                uni_vpgatherdd(vAux3, ptr[regSrc + vAux2], vAux4); // TODO: 64b 16b 8b?
                uni_vmovups(vAux4, vAux5);

                // (x; y + 1)
                uni_vcvtdq2ps(vAux2, vAux2);
                uni_vsubps(vAux2, vAux2, vDataTypeSize);
                uni_vcvtps2dq(vAux2, vAux2);
                uni_vpxor(vHCoord, vHCoord, vHCoord);
                uni_vpgatherdd(vHCoord, ptr[regSrc + vAux2], vAux4); // TODO: 64b 16b 8b?
                if (jcp.inDataPrc == InferenceEngine::Precision::FP32) {
                    uni_vfmsub213ps(vHCoord, vAux0, vHCoord); // q1 = -(v10 - dx * v10) -> vHCoord
                    uni_vfmsub231ps(vHCoord, vAux3, vAux0); // q1 = -q1 + dx * v11
                    // Res = q0 + dy * (q1 - q0)
                    uni_vsubps(vAux0, vHCoord, vWCoord);
                    uni_vfmadd132ps(vAux0, vWCoord, vAux1);
                } else if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                    // TODO:
                }
            }

            uni_vmovups(ptr[rDstTmp], vAux0);
            add(rDstTmp, regDstChStepB);

//            getPadded(&vAuxPool[2], vWCoord, vHCoord);
//            uni_vaddps(vWCoord, vWCoord, vFOnes); // TODO: Store in separate on AVX5
//            getPadded(&vAuxPool[3], vWCoord, vHCoord);
//            uni_vaddps(vHCoord, vHCoord, vFOnes); // TODO: Store in separate on AVX5
//            getPadded(&vAuxPool[4], vWCoord, vHCoord);
//            uni_vsubps(vWCoord, vWCoord, vFOnes); // TODO: Store in separate on AVX5
//            getPadded(&vAuxPool[5], vWCoord, vHCoord);

//            if (jcp.inDataPrc == InferenceEngine::Precision::FP32) {
//                // q0 = v00 - dx * (v01 - v00)
//                uni_vsubps(vWCoord, vAux3, vAux2);
//                uni_vfmadd132ps(vWCoord, vAux2, vAux0);
//                // q1 = v10 - dx * (v11 - v10)
//                uni_vsubps(vHCoord, vAux4, vAux5);
//                uni_vfmadd132ps(vHCoord, vAux5, vAux0);
//                // q0 + dy * (q1 - q0)
//                uni_vsubps(vAux0, vHCoord, vWCoord);
//                uni_vfmadd132ps(vAux0, vWCoord, vAux1);
//            } else if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
//                // TODO:
//            }
            add(rChannel, 1);
            jmp(lChannelLoopBegin, T_NEAR);
            L(lChannelLoopEnd);
        }
    } else if (jcp.interpolationMode == InterpolationMode::BICUBIC) {
        // TODO:
    } else if (jcp.interpolationMode == InterpolationMode::NEAREST) {
        // TODO:
    }
}

// Requires vAuxPool length 4.
// Returns calculated shifts in vAuxPool[0] and mask in vAuxPool[1].
template <x64::cpu_isa_t isa>
void jitUniGridSampleKernel<isa>::calcCoordinates(const Vmm* vAuxPool, bool shiftFirst) {
    auto& vH = vAuxPool[0];
    auto& vW = vAuxPool[1];
    auto& vAux0 = vAuxPool[2];
    auto& vAux1 = vAuxPool[3];

//    if (shiftFirst)
//        add(regGrid, vlen);

    Xbyak::Label lSpacialLoop, lTail;
    L(lSpacialLoop);
    {
        sub(regWorkAmount, dataElPerVec);
        cmp(regWorkAmount, dataElPerVec);
        jl(lTail, T_NEAR);

        if (isa == x64::avx512_core) {
            // Permute to XXXX.XXXX.YYYY.YYYY
            vpermd(vH, vPermGridMask, ptr[regGrid]);

            Xbyak::Zmm zmmH = Xbyak::Zmm(vH.getIdx());
            Xbyak::Ymm ymmW = Xbyak::Ymm(vW.getIdx());
            // Extract Y component
            vextractf64x4(ymmW, zmmH, 1);
        } else if (isa == x64::avx2) {
            uni_vmovups(vAux0, ptr[regGrid]);
            // Permute to XXXX.YYYY
            vpermd(vH, vPermGridMask, vAux0);
            // Extract Y component
            Xbyak::Xmm xmmW = Xbyak::Xmm(vW.getIdx());
            vextracti128(xmmW, vH, 1);
        } else if (isa == x64::sse41) {
            // TODO: SSE
        }

        add(regGrid, vlen);
        if (isa == x64::avx512_core) {
            // Permute to XXXX.XXXX.YYYY.YYYY
            vpermd(vAux0, vPermGridMask, ptr[regGrid]);

            Xbyak::Zmm zmmH = Xbyak::Zmm(vH.getIdx());
            Xbyak::Zmm zmmW = Xbyak::Zmm(vW.getIdx());
            Xbyak::Zmm zmmAux0 = Xbyak::Zmm(vAux0.getIdx());
            Xbyak::Ymm ymmAux1 = Xbyak::Ymm(vAux1.getIdx());
            // Extract X component
            // TODO: Change to vpermd if will be available registers
            vextractf64x4(ymmAux1, zmmAux0, 0);
            vinsertf64x4(zmmH, zmmH, ymmAux1, 1);
            // Extract Y component
            vextractf64x4(ymmAux1, zmmAux0, 1);
            vinsertf64x4(zmmW, zmmW, ymmAux1, 1);
        } else if (isa == x64::avx2) {
            uni_vmovups(vAux0, ptr[regGrid]);
            // Permute to XXXX.YYYY
            vpermd(vAux0, vPermGridMask, vAux0);
            Xbyak::Xmm xmmAux2 = Xbyak::Xmm(vAux1.getIdx());
            // Extract X component
            vextractf128(xmmAux2, vAux0, 0);
            vinsertf128(vH, vH, xmmAux2, 1);
            // Extract Y component
            vextractf128(xmmAux2, vAux0, 1);
            vinsertf128(vW, vW, xmmAux2, 1);
        } else if (isa == x64::sse41) {
            // TODO: SSE
        }

        if (!shiftFirst)
            add(regGrid, vlen);

        denormalizeRawCoordinates(vW, vH);

        interpolation(&vAuxPool[2], vW, vH);

        add(regDst, vlen);

        jmp(lSpacialLoop, T_NEAR);
    }

    L(lTail);
    tail(true);
}

//template <x64::cpu_isa_t isa>
//void jitUniGridSampleKernel<isa>::calcCoordinatesBlock(Vmm* vAuxPool, bool shiftFirst) {
//    // Most likely there will no significant performance gain vs memcpy in reference implementation on big blocks after axis,
//    // therefore no time was invested to this case yet.
//    IE_THROW() << "Unsupported case.";
//}

template <x64::cpu_isa_t isa>
void jitUniGridSampleKernel<isa>::process(bool isShortIdx, bool blocked) {
//    Xbyak::Label lTailProc, lEndProc;
//    cmp(regWorkAmount, dataElPerVec);
//    jl(lTailProc, T_NEAR);
        if (dataTypeSize == 4)
            process32b(isShortIdx, blocked);
        else if (dataTypeSize == 2)
            process16b(isShortIdx, blocked);
        else if (dataTypeSize == 1)
            process8b(isShortIdx, blocked);
//    jmp(lEndProc, T_NEAR);
//    L(lTailProc);
//        tail(isShortIdx, false, blocked);
//    L(lEndProc);
}

template <x64::cpu_isa_t isa>
void jitUniGridSampleKernel<isa>::process32b(bool isShortIdx, bool blocked) {
    Xbyak::Label lBatchLoop, lEnd;

    // First iteration
//    shiftIdxAndGather(vAuxContainer, isShortIdx, false, blocked);
//    uni_vmovups(ptr[regDst], vAuxContainer[2]);

    // Main loop
    L(lBatchLoop);
    {
        sub(regBatch, 1);
        cmp(regBatch, 0);
        jl(lEnd, T_NEAR);

        mov(regWorkAmount, ptr[regParams + GET_OFF(workAmount)]);
        shiftIdxAndGather(vAuxContainer, isShortIdx, true, blocked);

        if (jcp.dynamicShapes) {
            add(regSrc, ptr[regParams + GET_OFF(srcBatchStepB)]);
            add(regGrid, ptr[regParams + GET_OFF(gridBatchStepB)]);
            add(regDst, ptr[regParams + GET_OFF(dstBatchStepB)]);
        } else {
            add(regSrc, jcp.srcBatchStepB);
            add(regGrid, jcp.gridBatchStepB);
            add(regDst, jcp.dstBatchStepB);
        }
        jmp(lBatchLoop, T_NEAR);
    }

    L(lEnd);
}

template <x64::cpu_isa_t isa>
void jitUniGridSampleKernel<isa>::process16b(bool isShortIdx, bool blocked) {
//    Xbyak::Label lDstIdxLoop1, lTail;
//
//    Vmm vShufMask, vPermMask, vBuff0;
//    if (isa == x64::avx512_core) {
//        vPermMask = vAuxContainer[7];
//        vShufMask = vAuxContainer[8];
//        vBuff0    = vAuxContainer[9];
//    } else {
//        vPermMask = vAuxContainer[1];
//        vShufMask = vAuxContainer[4];
//        vBuff0    = vAuxContainer[5];
//    }
//
//    mov(regAux1, reinterpret_cast<uintptr_t>(shufMask16bitUni));
//    uni_vmovups(vShufMask, ptr[regAux1]);
//    mov(regAux1, reinterpret_cast<uintptr_t>(permMask16bitUni));
//    uni_vmovups(vPermMask, ptr[regAux1]);
//
//    // First iteration
//    shiftIdxAndGather(vAuxContainer, isShortIdx, false, blocked);
//    vpshufb(vBuff0, vAuxContainer[2], vShufMask);
//
//    shiftIdxAndGather(vAuxContainer, isShortIdx, true, blocked);
//    vpshufb(vAuxContainer[0], vAuxContainer[2], vShufMask);
//
//    vshufps(vAuxContainer[0], vBuff0, vAuxContainer[0], 0x44);
//    vpermd(vAuxContainer[0], vPermMask, vAuxContainer[0]);
//
//    uni_vmovups(ptr[regDst], vAuxContainer[0]);
//
//    // Main loop.
//    L(lDstIdxLoop1);
//    {
//        add(regDst, vlen);
//        sub(regWorkAmount, dataElPerVec);
//        cmp(regWorkAmount, dataElPerVec);
//        jl(lTail, T_NEAR);
//
//        shiftIdxAndGather(vAuxContainer, isShortIdx, true, blocked);
//        vpshufb(vBuff0, vAuxContainer[2], vShufMask);
//
//        shiftIdxAndGather(vAuxContainer, isShortIdx, true, blocked);
//        vpshufb(vAuxContainer[0], vAuxContainer[2], vShufMask);
//
//        vshufps(vAuxContainer[0], vBuff0, vAuxContainer[0], 0x44);
//        vpermd(vAuxContainer[0], vPermMask, vAuxContainer[0]);
//
//        uni_vmovups(ptr[regDst], vAuxContainer[0]);
//
//        jmp(lDstIdxLoop1, T_NEAR);
//    }
//
//    L(lTail);
//    tail(isShortIdx, true, blocked);
}

template <x64::cpu_isa_t isa>
void jitUniGridSampleKernel<isa>::process8b(bool isShortIdx, bool blocked) {
    Xbyak::Label lDstIdxLoop1, lTail;

//    Vmm vShufMask, vPermMask, vBuff0, vBuff1;
//    if (isa == x64::avx512_core) {
//        vPermMask = vAuxContainer[7];
//        vShufMask = vAuxContainer[8];
//        vBuff0    = vAuxContainer[9];
//        vBuff1    = vAuxContainer[10];
//    } else {
//        vPermMask = vAuxContainer[1];
//        vShufMask = vAuxContainer[4];
//        vBuff0    = vAuxContainer[5];
//        vBuff1    = vAuxContainer[6];
//    }
//    mov(regAux1, reinterpret_cast<uintptr_t>(shufMask8bitUni));
//    uni_vmovups(vShufMask, ptr[regAux1]);
//
//    // First iteration
//    shiftIdxAndGather(vAuxContainer, isShortIdx, false, blocked);
//    vpshufb(vBuff0, vAuxContainer[2], vShufMask);
//
//    shiftIdxAndGather(vAuxContainer, isShortIdx, true, blocked);
//    vpshufb(vAuxContainer[0], vAuxContainer[2], vShufMask);
//
//    vshufps(vBuff0, vBuff0, vAuxContainer[0], 0x0);
//
//    shiftIdxAndGather(vAuxContainer, isShortIdx, true, blocked);
//    vpshufb(vBuff1, vAuxContainer[2], vShufMask);
//
//    shiftIdxAndGather(vAuxContainer, isShortIdx, true, blocked);
//    vpshufb(vAuxContainer[0], vAuxContainer[2], vShufMask);
//
//    vshufps(vBuff1, vBuff1, vAuxContainer[0], 0x0);
//    vshufps(vAuxContainer[0], vBuff0, vBuff1, 0x88);
//
//    mov(regAux1, reinterpret_cast<uintptr_t>(permMask8bitUni));
//    uni_vmovups(vPermMask, ptr[regAux1]);
//
//    vpermd(vAuxContainer[0], vPermMask, vAuxContainer[0]);
//
//    uni_vmovups(ptr[regDst], vAuxContainer[0]);
//
//    // Main loop.
//    L(lDstIdxLoop1);
//    {
//        add(regDst, vlen);
//        sub(regWorkAmount, dataElPerVec);
//        cmp(regWorkAmount, dataElPerVec);
//        jl(lTail, T_NEAR);
//
//        shiftIdxAndGather(vAuxContainer, isShortIdx, true, blocked);
//        vpshufb(vBuff0, vAuxContainer[2], vShufMask);
//
//        shiftIdxAndGather(vAuxContainer, isShortIdx, true, blocked);
//        vpshufb(vAuxContainer[0], vAuxContainer[2], vShufMask);
//
//        vshufps(vBuff0, vBuff0, vAuxContainer[0], 0x0);
//
//        shiftIdxAndGather(vAuxContainer, isShortIdx, true, blocked);
//        vpshufb(vBuff1, vAuxContainer[2], vShufMask);
//
//        shiftIdxAndGather(vAuxContainer, isShortIdx, true, blocked);
//        vpshufb(vAuxContainer[0], vAuxContainer[2], vShufMask);
//
//        vshufps(vAuxContainer[0], vBuff1, vAuxContainer[0], 0x0);
//        vshufps(vAuxContainer[0], vBuff0, vAuxContainer[0], 0x88);
//
//        if (isa == x64::avx2) {
//            // Register vPermMask is invalidated by shiftIdxAndGather and must be initialized again.
//            mov(regAux1, reinterpret_cast<uintptr_t>(permMask8bitUni));
//            uni_vmovups(vPermMask, ptr[regAux1]);
//        }
//        vpermd(vAuxContainer[0], vPermMask, vAuxContainer[0]);
//
//        uni_vmovups(ptr[regDst], vAuxContainer[0]);
//
//        jmp(lDstIdxLoop1, T_NEAR);
//    }
//
//    L(lTail);
//    tail(isShortIdx, true, blocked);
}

// Requires vAuxPool length 4.
// Returns gathered data in vAuxPool[2].
template <x64::cpu_isa_t isa>
void jitUniGridSampleKernel<isa>::shiftIdxAndGather(const Vmm* vAuxPool, bool isShortIdx, bool shiftFirst, bool blocked) {

//    if (blocked) {
//        if (isShortIdx) {
//            calcSrcShiftShortBlock(vAuxPool, shiftFirst);
//        } else {
//            calcCoordinatesBlock(vAuxPool, shiftFirst);
//        }
//    } else {
//        if (isShortIdx) {
//            calcSrcShiftShort(vAuxPool, shiftFirst);
//        } else {
            calcCoordinates(vAuxPool, shiftFirst);
//        }
//    }
//    auto& kGatherMask = masksContainer[vAuxPool[1].getIdx()];
//    uni_vmovups(vAuxPool[2], vmmZeros);
//    uniVpGatherDd(vAuxPool[2], ptr[regSrc + vAuxPool[0]], kGatherMask);
}

template <x64::cpu_isa_t isa>
void jitUniGridSampleKernel<isa>::tail(bool isShortIdx, bool shiftFirst, bool blocked) {
//    auto& vSrcShift = vAuxContainer[0];
//    auto& kGatherMask = masksContainer[vAuxContainer[1].getIdx()];
//    auto& vAux0 = vAuxContainer[2];
//    auto& vAux1 = vAuxContainer[3];
//    auto& kAuxMask1 = masksContainer[vAux1.getIdx()];
//    Xbyak::Label lEnd;
//
//    const int secondStepCycles = 4 / dataTypeSize;
//    for (int p = 0; p < secondStepCycles; p++) {
//        cmp(regWorkAmount, 0);
//        jle(lEnd, T_NEAR);
//
//        if (isShortIdx) {
//            if (blocked) {
//                calcSrcShiftShortBlock(vAuxContainer, p > 0 || shiftFirst);
//            } else {
//                calcSrcShiftShort(vAuxContainer, p > 0 || shiftFirst);
//            }
//        } else {
//            if (blocked) {
//                calcCoordinatesBlock(vAuxContainer, p > 0 || shiftFirst);
//            } else {
//                calcCoordinates(vAuxContainer, p > 0 || shiftFirst);
//            }
//        }
//
//        fillRestWorkMask(kAuxMask1, vAux0, regWorkAmount, regAux1, rdx);
//
//        // Combining masks.
//        if (isa == x64::avx512_core) {
//            auto kMask1 = Xbyak::Opmask(kAuxMask1.getIdx());
//            auto kMaskG = Xbyak::Opmask(kGatherMask.getIdx());
//            kandd(kMaskG, kMaskG, kMask1);
//        } else if (isa == x64::avx2) {
//            auto& vGatherMask = vAuxContainer[kGatherMask.getIdx()];
//            vpand(vGatherMask, vGatherMask, vAux1);
//        }
//
//        uni_vmovups(vAux0, vmmZeros);
//        uniVpGatherDd(vAux0, ptr[regSrc + vSrcShift], kGatherMask);
//        if (dataTypeSize == 4) {
//            uni_vmovups_tail(ptr[regDst], kAuxMask1, vAux0);
//            sub(regWorkAmount, dataElPerVec);
//        } else {
//            storeVectorPart(regDst, regWorkAmount, vAux0, vAux1);
//        }
//    }
//    L(lEnd);
}

//template <>
//void jitUniGridSampleKernel<x64::avx512_core>::fillRestWorkMask(Vmask& kDstMask, Vmm& vmmAux, const Xbyak::Reg64& rWorkRest,
//        const Xbyak::Reg64& rAux0, const Xbyak::Reg64& rAux1) {
//    Xbyak::Label lKmov;
//    Xbyak::Reg32 rOnes(rAux1.getIdx());
//    mov(rOnes, 0x0000FFFF);
//    cmp(rWorkRest, idxElPerVec);
//    jge(lKmov);
//        Xbyak::Reg8 rShift(Xbyak::Operand::CL);
//        mov(rShift, idxElPerVec);
//        sub(rShift, rWorkRest);
//        shr(rOnes, rShift);
//    L(lKmov);
//    kmovw(kDstMask, rOnes);
//}

//template <>
//void jitUniGridSampleKernel<x64::avx2>::fillRestWorkMask(Vmask& kDstMask, Vmm& vAux, const Xbyak::Reg64& rWorkRest,
//        const Xbyak::Reg64& rAux0, const Xbyak::Reg64& rAux1) {
//    Xbyak::Label lEnd;
//    mov(rAux0, rWorkRest);
//    Xbyak::Reg32 rOnes(rAux1.getIdx());
//    mov(rOnes, 0xFFFFFFFF);
//    Xbyak::Xmm xmmAux(vAux.getIdx());
//    uni_vmovups(kDstMask, vmmZeros);
//    for (uint8_t i = 0; i < idxElPerVec; i++) {
//        cmp(rAux0, 0);
//        je(lEnd, T_NEAR);
//
//        if (i % 4 == 0)
//            uni_vmovups(xmmAux, xmmZeros);
//
//        vpinsrd(xmmAux, xmmAux, rOnes, i % 4);
//        vinserti128(kDstMask, kDstMask, xmmAux, i / 4);
//        sub(rAux0, 1);
//    }
//    L(lEnd);
//}

//template <x64::cpu_isa_t isa>
//void jitUniGridSampleKernel<isa>::storeVectorPart(const Xbyak::Reg64& rDst, const Xbyak::Reg64& rToStoreCounter, Vmm& vmmSrc, Vmm& vAux) {
//    Xbyak::Label lEnd;
//    Xbyak::Xmm xAux(vAux.getIdx());
//    for (int j = 0; j < vlen / vlenXmm; j++) {
//        if (isa == x64::avx2)
//            vextracti128(xAux, vmmSrc, j);
//        else if (isa == x64::avx512_core)
//            vextracti64x2(xAux, vmmSrc, j);
//
//        for (int k = 0; k < 4; k++) {
//            cmp(rToStoreCounter, 0);
//            jle(lEnd, T_NEAR);
//
//            if (dataTypeSize == 4)
//                uni_vpextrd(ptr[rDst], xAux, k);
//            else if (dataTypeSize == 2)
//                uni_vpextrw(ptr[rDst], xAux, k * 2);
//            else if (dataTypeSize == 1)
//                uni_vpextrb(ptr[rDst], xAux, k * 4);
//
//            add(rDst, dataTypeSize);
//            sub(rToStoreCounter, 1);
//        }
//    }
//    L(lEnd);
//}

//template <>
//void jitUniGridSampleKernel<x64::avx512_core>::fillVlenVector() {
//    mov(reg32Aux1, vlen);
//    vpbroadcastd(vmmVecLenB, reg32Aux1);
//}
//template <>
//void jitUniGridSampleKernel<x64::avx2>::fillVlenVector() {
//    vpcmpeqd(vmmVecLenB, vmmVecLenB, vmmVecLenB);
//    vpsrld(vmmVecLenB, vmmVecLenB, 31); // Right shift to 1.
//    uni_vpslld(vmmVecLenB, vmmVecLenB, 5);  // Left shift to 32.
//}

//template <x64::cpu_isa_t isa>
//bool jitUniGridSampleKernel<isa>::isSupportedConfiguration(uint64_t afterAxisSize) {
//    if (!jcp.dynamicShapes && afterAxisSize <= idxElPerVec) {
//        if (afterAxisSize > 1 && isa == x64::avx2 && (dataTypeSize == 1 || dataTypeSize == 2))
//            // There are no enough registers for these cases.
//            return false;
//
//        return true;
//    }
//    if (jcp.dynamicShapes && afterAxisSize == 1) {
//        return true;
//    }
//    return false;
//}

template struct jitUniGridSampleKernel<x64::avx512_core>;
template struct jitUniGridSampleKernel<x64::avx2>;
template struct jitUniGridSampleKernel<x64::sse41>;

}   // namespace intel_cpu
}   // namespace ov
