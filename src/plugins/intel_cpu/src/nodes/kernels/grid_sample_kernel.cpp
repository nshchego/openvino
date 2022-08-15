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
void jitGridSampleKernel<isa>::create_ker() {
    auto code = x64::jit_generator::create_kernel();
    if (code != dnnl::impl::status::success)
        IE_THROW() << "Could not create GridSample kernel. Error code: " << std::to_string(code);
    ker_ = (decltype(ker_))jit_ker();
}

template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::generate() {
    this->preamble();

    mov(regSrc, ptr[regParams + GET_OFF(src)]);
    mov(regDst, ptr[regParams + GET_OFF(dst)]);
    mov(regGrid, ptr[regParams + GET_OFF(grid)]);

    mov(regAux1, ptr[regParams + GET_OFF(srcWidthFl)]);
    uni_vpbroadcastd(vSrcWidthF, ptr[regAux1]);
    mov(regAux1, ptr[regParams + GET_OFF(srcHeightFl)]);
    uni_vpbroadcastd(vSrcHeightF, ptr[regAux1]);

    mov(regSrcChannelStepB, ptr[regParams + GET_OFF(srcChannelStepB)]);
    mov(regDstChannelStepB, ptr[regParams + GET_OFF(dstChannelStepB)]);
    mov(regChannelsNum, ptr[regParams + GET_OFF(channelsNum)]);

    mov(regAux1, reinterpret_cast<uintptr_t>(permGridMaskUni));
    uni_vmovups(vPermGridMask, ptr[regAux1]);

    mov(regAux1, ptr[regParams + GET_OFF(one)]);
    uni_vpbroadcastd(vOnesF, ptr[regAux1]);

    uni_vpxor(vZeros, vZeros, vZeros);

    if (jcp.alignCorners) {
        mov(regAux1, ptr[regParams + GET_OFF(wDenormCoef)]);
        uni_vpbroadcastd(vWDenormCoef, ptr[regAux1]);
        mov(regAux1, ptr[regParams + GET_OFF(hDenormCoef)]);
        uni_vpbroadcastd(vHDenormCoef, ptr[regAux1]);
    } else {
        mov(regAux1, ptr[regParams + GET_OFF(halfVal)]);
        uni_vpbroadcastd(vHalf, ptr[regAux1]);
    }
    if (isa == x64::avx512_core) {
        if (jcp.paddingMode == PaddingMode::ZEROS) {
            mov(regAux1, dataTypeSize);
            vpbroadcastd(vDataTypeSize, reg32Aux1);
            mov(regAux1, ptr[regParams + GET_OFF(srcWidthB)]);
            uni_vpbroadcastd(vSrcWidthB, ptr[regAux1]);
        } else if (jcp.paddingMode == PaddingMode::BORDER) {
            mov(regAux1, ptr[regParams + GET_OFF(srcHeightSub1Fl)]);
            uni_vpbroadcastd(vSrcHeightSub1Fl, ptr[regAux1]);
            mov(regAux1, ptr[regParams + GET_OFF(srcWidthSub1Fl)]);
            uni_vpbroadcastd(vSrcWidthSub1Fl, ptr[regAux1]);
        } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
            mov(regAux1, ptr[regParams + GET_OFF(srcHeightMul2Fl)]);
            uni_vpbroadcastd(vSrcHeightMul2Fl, ptr[regAux1]);
            mov(regAux1, ptr[regParams + GET_OFF(srcWidthMul2Fl)]);
            uni_vpbroadcastd(vSrcWidthMul2Fl, ptr[regAux1]);
            mov(regAux1, ptr[regParams + GET_OFF(srcHeightMul2Sub1Fl)]);
            uni_vpbroadcastd(vSrcHeightMul2Sub1Fl, ptr[regAux1]);
            mov(regAux1, ptr[regParams + GET_OFF(srcWidthMul2Sub1Fl)]);
            uni_vpbroadcastd(vSrcWidthMul2Sub1Fl, ptr[regAux1]);
            if (jcp.alignCorners) {
                mov(reg32Aux1, 0x7fffffff);
                vpbroadcastd(vAbsMask, reg32Aux1);
            }
        }

        if (jcp.interpolationMode == InterpolationMode::BICUBIC) {
            mov(reg32Aux1, 0xbf400000); // -0.75f
            vpbroadcastd(vBicubConst, reg32Aux1);
            mov(reg32Aux1, 0x3fa00000); // 1.25f
            vpbroadcastd(vBicub2Const, reg32Aux1);
            mov(reg32Aux1, 0x40100000); // 2.25f
            vpbroadcastd(vBicub3Const, reg32Aux1);
            mov(reg32Aux1, 0x40000000); // 2.f
            vpbroadcastd(v2val, reg32Aux1); // TODO: rename
            mov(reg32Aux1, 0x3fc00000); // 1.5f
            vpbroadcastd(v2Bicub3Const, reg32Aux1);
        }
    }

    process();

    this->postamble();
}

template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::denormalizeRawCoordinates(const Vmm& vWCoord, const Vmm& vHCoord) {
    if (jcp.alignCorners) {
        uni_vfmadd132ps(vWCoord, vWDenormCoef, vWDenormCoef);
        uni_vfmadd132ps(vHCoord, vHDenormCoef, vHDenormCoef);
    } else {
        uni_vfmadd132ps(vWCoord, vSrcWidthF, vSrcWidthF);
        vfmsub132ps(vWCoord, vHalf, vHalf);
        uni_vfmadd132ps(vHCoord, vSrcHeightF, vSrcHeightF);
        vfmsub132ps(vHCoord, vHalf, vHalf);
    }
}

template <>
void jitGridSampleKernel<x64::avx512_core>::zerosPadding0(const Vmm& vCoord, const Vmm& vUpperBound, const Vmask& kDst, const Vmask& kAux) {
    vcmpps(kAux, vCoord, vUpperBound, 0x1);   // vCoord < vUpperBound
    vcmpps(kDst | kAux, vZeros, vCoord, 0x2); // vCoord >= vZeros
}

//template <>
//void jitGridSampleKernel<x64::avx2>::zerosPadding0(const Vmm& vCoord, const Vmm& vUpperBound, const Vmask& kDst, const Vmask& kAux) {
//
//}
//
//template <>
//void jitGridSampleKernel<x64::sse41>::zerosPadding0(const Vmm& vCoord, const Vmm& vUpperBound, const Vmask& kDst, const Vmask& kAux) {
//
//}

template <>
void jitGridSampleKernel<x64::avx512_core>::zerosPadding1(const Vmm& vCoord, const Vmm& vUpperBound, const Vmask& kDst, const Vmask& kAux) {
    vcmpps(kDst | kAux, vCoord, vUpperBound, 0x1); // vCoord < vUpperBound
    vcmpps(kDst | kDst, vZeros, vCoord, 0x2);      // vCoord >= vZeros
}

//template <>
//void jitGridSampleKernel<x64::avx2>::zerosPadding1(const Vmm& vCoord, const Vmm& vUpperBound, const Vmask& kDst, const Vmask& kAux) {
//
//}
//
//template <>
//void jitGridSampleKernel<x64::sse41>::zerosPadding1(const Vmm& vCoord, const Vmm& vUpperBound, const Vmask& kDst, const Vmask& kAux) {
//
//}

template <>
void jitGridSampleKernel<x64::avx512_core>::zerosPadding(const Vmm& vWCoord, const Vmm& vHCoord, const Vmask& kDst, const Vmask& kAux) {
    zerosPadding0(vWCoord, vSrcWidthF, kDst, kDst);
    zerosPadding1(vHCoord, vSrcHeightF, kDst, kDst);
}

template <>
void jitGridSampleKernel<x64::avx2>::zerosPadding(const Vmm& vWCoord, const Vmm& vHCoord, const Vmask& kDst, const Vmask& kAux) {
    uni_vcmpps(kAux, vSrcWidthF, vWCoord, 0x1); // vWCoord < vSrcWidthF
    uni_vcmpps(kDst, vWCoord, vZeros, 0x2);     // vWCoord >= vZeros
    uni_vpand(kDst, kAux, kDst);

    uni_vcmpps(kAux, vSrcHeightF, vHCoord, 0x1); // vHCoord < vSrcHeightF
    uni_vpand(kDst, kAux, kDst);
    uni_vcmpps(kAux, vHCoord, vZeros, 0x2);      // vHCoord >= vZeros
    uni_vpand(kDst, kAux, kDst);
}

template <>
void jitGridSampleKernel<x64::sse41>::zerosPadding(const Vmm& vWCoord, const Vmm& vHCoord, const Vmask& kDst, const Vmask& kAux) {
    uni_vcmpps(kAux, vSrcWidthF, vWCoord, 0x1); // vWCoord < vSrcWidthF
    uni_vcmpps(kDst, vWCoord, vZeros, 0x2);     // vWCoord >= vZeros
    uni_vpand(kDst, kAux, kDst);

    uni_vcmpps(kAux, vSrcHeightF, vHCoord, 0x1); // vHCoord < vSrcHeightF
    uni_vpand(kDst, kAux, kDst);
    uni_vcmpps(kAux, vHCoord, vZeros, 0x2);      // vHCoord >= vZeros
    uni_vpand(kDst, kAux, kDst);
}

template <>
void jitGridSampleKernel<x64::avx512_core>::borderPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const Vmm& vUpperBound, const Vmask& kAux) {
    vrangeps(vCoordDst, vCoordOrigin, vUpperBound, 0x0); // vWCoord >= vSrcWidthF
    vrangeps(vCoordDst, vCoordDst, vZeros, 0x1);         // vWCoord < vZeros
}

template <>
void jitGridSampleKernel<x64::avx2>::borderPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const Vmm& vUpperBound, const Vmask& kAux) {
//    vcmpps(kAux, vSrcWidthF, vWCoord, 0x2); // vWCoord >= vSrcWidthF
//    vmovups(vWCoord | kAux, vSrcWidthSub1Fl);
//    vcmpps(kAux, vWCoord, vZeros, 0x1); // vWCoord < vZeros
//    vmovups(vWCoord | kAux, vZeros);
//
//    vcmpps(kAux, vSrcHeightF, vHCoord, 0x2); // vHCoord >= vSrcHeightF
//    vmovups(vHCoord | kAux, vSrcHeightSub1Fl);
//    vcmpps(kAux, vHCoord, vZeros, 0x1); // vHCoord < vZeros
//    vmovups(vHCoord | kAux, vZeros);
}

template <>
void jitGridSampleKernel<x64::sse41>::borderPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const Vmm& vUpperBound, const Vmask& kAux) {
}

template <>
void jitGridSampleKernel<x64::avx512_core>::reflectionPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const Vmm& vAux, const Vmask& kAux, const uint8_t dim) {
    Vmm vSrcDimFl, vSrcDimMul2Fl, vSrcDimMul2Sub1Fl;
    if (dim == 0) {
        vSrcDimFl = vSrcWidthF;
        vSrcDimMul2Fl = vSrcWidthMul2Fl;
        vSrcDimMul2Sub1Fl = vSrcWidthMul2Sub1Fl;
    } else {
        vSrcDimFl = vSrcHeightF;
        vSrcDimMul2Fl = vSrcHeightMul2Fl;
        vSrcDimMul2Sub1Fl = vSrcHeightMul2Sub1Fl;
    }

    if (jcp.alignCorners) {
        // abs(x) % D21
        uni_vandps(vCoordDst, vCoordOrigin, vAbsMask); // abs(x)
        uni_vdivps(vAux, vCoordDst, vSrcDimMul2Sub1Fl);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, vSrcDimMul2Sub1Fl); // abs(x) % D21
    } else {
        // (x % D2 + D2) % D2
        if (vCoordDst.getIdx() != vCoordOrigin.getIdx())
            uni_vmovups(vCoordDst, vCoordOrigin);
        uni_vdivps(vAux, vCoordDst, vSrcDimMul2Fl);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, vSrcDimMul2Fl); // x % D2
        uni_vaddps(vCoordDst, vCoordDst, vSrcDimMul2Fl); // x % D2 + D2
        uni_vdivps(vAux, vCoordDst, vSrcDimMul2Fl);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfnmadd231ps(vCoordDst, vAux, vSrcDimMul2Fl); // (x % D2 + D2) % D2
    }

    uni_vsubps(vAux, vSrcDimMul2Sub1Fl, vCoordDst);
    vcmpps(kAux, vSrcDimFl, vCoordDst, 0x2); // vCoordDst >= vSrcDimFl
    vmovups(vCoordDst | kAux, vAux);
}

template <>
void jitGridSampleKernel<x64::avx2>::reflectionPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const Vmm& vAux, const Vmask& kAux, const uint8_t dim) {
    Vmm vSrcDimFl;
    if (dim == 0) {
        vSrcDimFl = vSrcWidthF;
    } else {
        vSrcDimFl = vSrcHeightF;
    }
    const auto& vSrcDimMul2Fl = kAux;
//    uni_vmulps(vSrcDimMul2Fl, vSrcDimFl, 2);
    // (x % W2 + W2) % W2
    uni_vdivps(vAux, vCoordDst, vSrcDimMul2Fl);
    uni_vroundps(vAux, vAux, 0x1); // Round floor
    uni_vfmsub231ps(vCoordDst, vAux, vSrcDimMul2Fl); // x % W2
    uni_vsubps(vCoordDst, vSrcDimMul2Fl, vCoordDst); // x % W2 + W2
    uni_vdivps(vAux, vCoordDst, vSrcDimMul2Fl);
    uni_vroundps(vAux, vAux, 0x1); // Round floor
    uni_vfmsub231ps(vCoordDst, vAux, vSrcDimMul2Fl); // (x % W2 + W2) % W2

    uni_vsubps(vAux, vSrcDimMul2Fl, vOnesF);
    uni_vsubps(vAux, vAux, vCoordDst);
    vcmpps(kAux, vSrcDimFl, vCoordDst, 0x2); // vCoordDst >= vSrcDimFl
//        vmovups(vCoordDst | kMask, vAux);
}

template <>
void jitGridSampleKernel<x64::sse41>::reflectionPadding(const Vmm& vCoordDst, const Vmm& vCoordOrigin, const Vmm& vAux, const Vmask& kAux, const uint8_t dim) {
// TODO:
}

template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::bicubicCoefficients(const Vmm& vCoef, const Vmm& vDDim, uint8_t idx) {
    if (idx == 0) {
        uni_vmovups(vCoef, vDDim);
        vfnmadd132ps(vCoef, vOnesF, v2val);
        uni_vfmadd231ps(vCoef, vDDim, vDDim);
        uni_vmulps(vCoef, vCoef, vDDim);
        uni_vmulps(vCoef, vCoef, vBicubConst);
    } else if (idx == 1) {
        uni_vmovups(vCoef, vDDim);
        vfmsub132ps(vCoef, vBicub3Const, vBicub2Const);
        uni_vmulps(vCoef, vCoef, vDDim);
        uni_vfmadd132ps(vCoef, vOnesF, vDDim);
    } else if (idx == 2) {
        uni_vmulps(vCoef, vDDim, vDDim);
        vfmadd132ps(vCoef, vBicubConst, vBicub2Const);
        vfmsub231ps(vCoef, vDDim, v2Bicub3Const);
        uni_vmulps(vCoef, vCoef, vDDim);
    } else if (idx == 3) {
        uni_vmulps(vCoef, vBicubConst, vDDim);
        uni_vmulps(vCoef, vCoef, vDDim);
        vfnmadd132ps(vCoef, vCoef, vDDim);
    }
}

// Returns source values obtained with padded coordinates in vAuxPool[0].
// Requires vAuxPool length 4.
template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::getPadded(const Vmm* vAuxPool, const Vmm& vWCoord, const Vmm& vHCoord) {
    const auto& vAux0 = vAuxPool[0]; // Output storage
    const auto& vAux1 = vAuxPool[1];
    const auto& vAux2 = vAuxPool[2];

    if (jcp.paddingMode == PaddingMode::ZEROS) {
        // TODO: Optimize for AVX5
        // SRC HEIGHT and WIDTH!
//        uni_vcmpps(vAux0, vSrcWidthF, vWCoord, 0x1); // vWCoord < vSrcWidthF
//        uni_vcmpps(vAux1, vWCoord, vZeros, 0x2); // vWCoord >= vZeros
//        uni_vpand(vAux1, vAux0, vAux1);
//
//        uni_vcmpps(vAux0, vSrcHeightF, vHCoord, 0x1); // vHCoord < vSrcHeightF
//        uni_vpand(vAux1, vAux0, vAux1);
//        uni_vcmpps(vAux0, vHCoord, vZeros, 0x2); // vHCoord >= vZeros
//        uni_vpand(vAux1, vAux0, vAux1);

        // ADD CHANNEL SHIFT
//        uni_vmovups(vAux2, vHCoord);
//        uni_vfmadd132ps(vAux2, vWCoord, vSrcWidthF);
//        uni_vcvtps2dq(vAux2, vAux2);
//
//        uni_vpgatherdd(vAux0, ptr[regSrcIter + vAux2], vAux1); // TODO: 64b 16b 8b?
    } else if (jcp.paddingMode == PaddingMode::BORDER) {
        // TODO:
    } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
        // TODO:
    }
}

template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::interpolation(const Vmm* vAuxPool, const Vmm& vWCoord, const Vmm& vHCoord) {
    if (jcp.interpolationMode == InterpolationMode::BILINEAR) {
        bilinearInterpolation(vAuxPool, vWCoord, vHCoord);
    } else if (jcp.interpolationMode == InterpolationMode::BICUBIC) {
        bicubicInterpolation(vAuxPool, vWCoord, vHCoord);
    } else if (jcp.interpolationMode == InterpolationMode::NEAREST) {
        nearestInterpolation(vAuxPool, vWCoord, vHCoord);
    }
}

template <>
void jitGridSampleKernel<x64::avx512_core>::nearestInterpolation(const Vmm* vAuxPool, const Vmm& vWCoord, const Vmm& vHCoord) {
    const auto& vSrcShift  = vWCoord;
    const auto& vAux       = vAuxPool[0];
    const auto& kMask0   = k1;
    const auto& kAuxMask = k2;

    uni_vroundps(vWCoord, vWCoord, 0x0); // Round near
    uni_vroundps(vHCoord, vHCoord, 0x0); // Round near

    if (jcp.paddingMode == PaddingMode::ZEROS) {
        zerosPadding(vWCoord, vHCoord, kMask0, kAuxMask);
    } else if (jcp.paddingMode == PaddingMode::BORDER) {
        borderPadding(vWCoord, vWCoord, vSrcWidthSub1Fl, kAuxMask);
        borderPadding(vHCoord, vHCoord, vSrcHeightSub1Fl, kAuxMask);
    } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
        reflectionPadding(vWCoord, vWCoord, vAux, kAuxMask, 0);
        reflectionPadding(vHCoord, vHCoord, vAux, kAuxMask, 1);
    }

    uni_vfmadd231ps(vWCoord, vHCoord, vSrcWidthF);
    uni_vcvtps2dq(vSrcShift, vWCoord);
    if (dataTypeSize > 1)
        uni_vpslld(vSrcShift, vSrcShift, dataTypeShift); // multiply by source data type size.

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
            kmovw(kAuxMask, kMask0);
            uni_vpxor(vAux, vAux, vAux);
        } else {
            kxnorw(kAuxMask, kAuxMask, kAuxMask);
        }
        uni_vpgatherdd(vAux, ptr[rSrcTmp + vSrcShift], kAuxMask);
        if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
            uni_vcvtdq2ps(vAux, vAux);
        }

        uni_vmovups(ptr[rDstTmp], vAux);
        add(rSrcTmp, regSrcChannelStepB);
        add(rDstTmp, regDstChannelStepB);
        add(rChannel, 1);

        jmp(lChannelLoopBegin, T_NEAR);
        L(lChannelLoopEnd);
    }
}

template <>
void jitGridSampleKernel<x64::avx2>::nearestInterpolation(const Vmm* vAuxPool, const Vmm& vWCoord, const Vmm& vHCoord) {

}

template <>
void jitGridSampleKernel<x64::sse41>::nearestInterpolation(const Vmm* vAuxPool, const Vmm& vWCoord, const Vmm& vHCoord) {

}

// Produces data interpolation.
template <>
void jitGridSampleKernel<x64::avx512_core>::bilinearInterpolation(const Vmm* vAuxPool, const Vmm& vWCoord, const Vmm& vHCoord) {
    const auto &vDX = vAuxPool[0];
    const auto &vDY = vAuxPool[1];
    const auto &vQ0 = vAuxPool[2];
    const auto &vQ1 = vAuxPool[3];
    const auto &shift00 = vWCoord;
    const auto &shift01 = vHCoord;
    const auto &shift10 = vAuxPool[4];
    const auto &shift11 = vAuxPool[5];
    const auto &vAux3 = vAuxPool[6];
    const auto &kMask00 = k1;
    const auto &kMask01 = k2;
    const auto &kMask10 = k3;
    const auto &kMask11 = k4;
    const auto &kAuxMask = k5;

    uni_vmovups(vDX, vWCoord);
    uni_vmovups(vDY, vHCoord);
    uni_vroundps(vWCoord, vWCoord, 0x1); // Round floor
    uni_vroundps(vHCoord, vHCoord, 0x1); // Round floor
    uni_vsubps(vDX, vDX, vWCoord);
    uni_vsubps(vDY, vDY, vHCoord);

    uni_vaddps(shift10, vWCoord, vOnesF);
    uni_vaddps(shift11, vHCoord, vOnesF);
    if (jcp.paddingMode == PaddingMode::ZEROS) {
        zerosPadding(vWCoord, vHCoord, kMask00, kAuxMask); // (y; x)
        zerosPadding(shift10, vHCoord, kMask01, kAuxMask); // (y; x + 1)
        zerosPadding(shift10, shift11, kMask11, kAuxMask); // (y + 1; x + 1)
        zerosPadding(vWCoord, shift11, kMask10, kAuxMask); // (y + 1; x)

        uni_vfmadd231ps(vWCoord, vHCoord, vSrcWidthF); // (y; x)
        uni_vcvtps2dq(shift00, vWCoord);
        if (dataTypeSize > 1)
            uni_vpslld(shift00, shift00, dataTypeShift); // multiply by source data type size.
        uni_vpaddd(shift01, shift00, vDataTypeSize);
        uni_vpaddd(shift10, shift00, vSrcWidthB); // shift11??
        uni_vpaddd(shift11, shift10, vDataTypeSize); // sub??
    } else if (jcp.paddingMode == PaddingMode::BORDER) {
        borderPadding(vWCoord, vWCoord, vSrcWidthSub1Fl, kAuxMask);
        borderPadding(vHCoord, vHCoord, vSrcHeightSub1Fl, kAuxMask);
        borderPadding(shift10, shift10, vSrcWidthSub1Fl, kAuxMask);
        borderPadding(shift11, shift11, vSrcHeightSub1Fl, kAuxMask);
    } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
        reflectionPadding(vWCoord, vWCoord, vQ0, kAuxMask, 0);
        reflectionPadding(vHCoord, vHCoord, vQ0, kAuxMask, 1);
        reflectionPadding(shift10, shift10, vQ0, kAuxMask, 0);
        reflectionPadding(shift11, shift11, vQ0, kAuxMask, 1);
    }
    if (jcp.paddingMode == PaddingMode::BORDER || jcp.paddingMode == PaddingMode::REFLECTION) {
        uni_vmovups(vAux3, shift11);
        // W * y + x
        uni_vfmadd132ps(vAux3, vWCoord, vSrcWidthF);   // (y + 1; x)
        uni_vfmadd231ps(vWCoord, vHCoord, vSrcWidthF); // (y; x)
        uni_vfmadd132ps(vHCoord, shift10, vSrcWidthF); // (y; x + 1)
        uni_vfmadd132ps(shift11, shift10, vSrcWidthF); // (y + 1; x + 1)
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
    const Xbyak::Reg64 &rSrcTmp = regAux2;
    const Xbyak::Reg64 &rDstTmp = regAux3;
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
        uni_vpgatherdd(vAux3, ptr[rSrcTmp + shift11], kAuxMask);
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
        uni_vpgatherdd(vQ1, ptr[rSrcTmp + shift10], kAuxMask);
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

        uni_vmovups(ptr[rDstTmp], vQ1);
        add(rSrcTmp, regSrcChannelStepB);
        add(rDstTmp, regDstChannelStepB);
        add(rChannel, 1);

        jmp(lChannelLoopBegin, T_NEAR);
        L(lChannelLoopEnd);
    }
}

template <>
void jitGridSampleKernel<x64::avx2>::bilinearInterpolation(const Vmm* vAuxPool, const Vmm& vWCoord, const Vmm& vHCoord) {
//    const auto& vDX = vAuxPool[0];
//    const auto& vDY = vAuxPool[1];
//    const auto& vGatherShift = vAuxPool[2];
//    const auto& vAux3 = vAuxPool[3];
//    const auto& vQ0 = vAuxPool[4];
//    const auto& vQ1 = vAuxPool[5];
//    const auto& kMask0 = k1;
//    const auto& kMask1 = k2;
//
////uni_vmovups(ptr[regDst], vWCoord);
//    uni_vmovups(vDX, vWCoord);
//    uni_vmovups(vDY, vHCoord);
//    uni_vroundps(vWCoord, vWCoord, 0x1); // Round floor
//    uni_vroundps(vHCoord, vHCoord, 0x1); // Round floor
//    uni_vsubps(vDX, vDX, vWCoord);
//    uni_vsubps(vDY, vDY, vHCoord);
//
////uni_vmovups(ptr[regDst], vWCoord);
////        if (jcp.paddingMode == PaddingMode::ZEROS) {
////            zerosPadding(vWCoord, vHCoord, kMask0, kMask1);
////            uni_kmovq(kMask1, kMask0);
////        }
////kmovq(regAux1, kMask0);
////uni_vmovups(ptr[regDst], vWCoord);
//
//    // PER CHANNEL LOOP
//    Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
//    const Xbyak::Reg64& rChannel = regAux1;
//    const Xbyak::Reg64& rSrcTmp = regAux2;
//    const Xbyak::Reg64& rDstTmp = regAux3;
//    mov(rChannel, 0);
//    mov(rSrcTmp, regSrc);
//    mov(rDstTmp, regDst);
////        uni_vfmadd132ps(vHCoord, vWCoord, vSrcWidthF);
////        uni_vcvtps2dq(vHCoord, vHCoord);
////        if (dataTypeSize > 1)
////            uni_vpslld(vHCoord, vHCoord, dataTypeShift); // multiply by source data type size.
//    L(lChannelLoopBegin);
//    {
//        cmp(rChannel, regChannelsNum);
//        jge(lChannelLoopEnd, T_NEAR);
//
//        if (jcp.paddingMode == PaddingMode::ZEROS) {
//            // (x; y)
//            zerosPadding(vWCoord, vHCoord, kMask0, kMask1);
//            uni_vmovups(vGatherShift, vHCoord);
//            uni_vfmadd132ps(vGatherShift, vWCoord, vSrcWidthF);
//            uni_vcvtps2dq(vGatherShift, vGatherShift);
//            if (dataTypeSize > 1)
//                uni_vpslld(vGatherShift, vGatherShift, dataTypeShift);
//            uni_vpxor(vQ0, vQ0, vQ0);
//            uni_vpgatherdd(vQ0, ptr[rSrcTmp + vGatherShift], kMask0); // v00 -> vQ0 TODO: 64b 16b 8b?
////uni_vmovups(ptr[rDstTmp], vOnesF);
////                uni_kmovq(kMask0, kMask1);
//            if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
//                uni_vcvtdq2ps(vQ0, vQ0);
//            }
//            uni_vfmsub213ps(vQ0, vDX, vQ0); // q0 = -(v00 - dx * v00)
//
//            // (x + 1; y)
////                uni_vpaddd(vGatherShift, vGatherShift, vDataTypeSize);
//            uni_vaddps(vWCoord, vWCoord, vOnesF);
//            zerosPadding(vWCoord, vHCoord, kMask0, kMask1);
//            uni_vmovups(vGatherShift, vHCoord);
//            uni_vfmadd132ps(vGatherShift, vWCoord, vSrcWidthF);
//            uni_vcvtps2dq(vGatherShift, vGatherShift);
//            if (dataTypeSize > 1)
//                uni_vpslld(vGatherShift, vGatherShift, dataTypeShift);
//            uni_vpxor(vAux3, vAux3, vAux3);
////uni_vmovups(ptr[rDstTmp], vGatherShift);
//            uni_vpgatherdd(vAux3, ptr[rSrcTmp + vGatherShift], kMask0); // TODO: 64b 16b 8b?
////uni_vmovups(ptr[rDstTmp], vAux3);
////                uni_kmovq(kMask0, kMask1);
//            // q0 = -q0 + dx * v01
//            if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
//                uni_vcvtdq2ps(vAux3, vAux3);
//            }
//            uni_vfmsub231ps(vQ0, vAux3, vDX);
//
//            // (x + 1; y + 1)
////                uni_vpaddd(vGatherShift, vGatherShift, vSrcWidthB);
//            uni_vaddps(vHCoord, vHCoord, vOnesF);
//            zerosPadding(vWCoord, vHCoord, kMask0, kMask1);
//            uni_vmovups(vGatherShift, vHCoord);
//            uni_vfmadd132ps(vGatherShift, vWCoord, vSrcWidthF);
//            uni_vcvtps2dq(vGatherShift, vGatherShift);
//            if (dataTypeSize > 1)
//                uni_vpslld(vGatherShift, vGatherShift, dataTypeShift);
//            uni_vpxor(vAux3, vAux3, vAux3);
//            uni_vpgatherdd(vAux3, ptr[rSrcTmp + vGatherShift], kMask0); // TODO: 64b 16b 8b?
////                uni_kmovq(kMask0, kMask1);
//
//            // (x; y + 1)
////                uni_vpsubd(vGatherShift, vGatherShift, vDataTypeSize);
//            uni_vsubps(vWCoord, vWCoord, vOnesF);
//            zerosPadding(vWCoord, vHCoord, kMask0, kMask1);
//            uni_vmovups(vGatherShift, vHCoord);
//            uni_vfmadd132ps(vGatherShift, vWCoord, vSrcWidthF);
//            uni_vcvtps2dq(vGatherShift, vGatherShift);
//            if (dataTypeSize > 1)
//                uni_vpslld(vGatherShift, vGatherShift, dataTypeShift);
//            uni_vpxor(vQ1, vQ1, vQ1);
//            uni_vpgatherdd(vQ1, ptr[rSrcTmp + vGatherShift], kMask0); // TODO: 64b 16b 8b?
////                uni_kmovq(kMask0, kMask1);
//            uni_vsubps(vHCoord, vHCoord, vOnesF);
//            if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
//                uni_vcvtdq2ps(vQ1, vQ1);
//            }
//
//            uni_vfmsub213ps(vQ1, vDX, vQ1); // q1 = -(v10 - dx * v10)
//            uni_vfmsub231ps(vQ1, vAux3, vDX); // q1 = -q1 + dx * v11
//            // Res = q0 + dy * (q1 - q0)
//            uni_vsubps(vQ1, vQ1, vQ0);
//            uni_vfmadd132ps(vQ1, vQ0, vDY);
//
//            if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
//                uni_vcvtps2dq(vQ1, vQ1);
//            }
//        }
//
//        uni_vmovups(ptr[rDstTmp], vQ1);
//        add(rSrcTmp, regSrcChannelStepB);
//        add(rDstTmp, regDstChannelStepB);
//        add(rChannel, 1);
//
//        jmp(lChannelLoopBegin, T_NEAR);
//        L(lChannelLoopEnd);
//    }
//} else if (jcp.interpolationMode == InterpolationMode::BICUBIC) {
//        // TODO:
//    } else if (jcp.interpolationMode == InterpolationMode::NEAREST) {
//    // TODO:
//}
        }

template <>
void jitGridSampleKernel<x64::sse41>::bilinearInterpolation(const Vmm* vAuxPool, const Vmm& vWCoord, const Vmm& vHCoord) {

}

template <>
void jitGridSampleKernel<x64::avx512_core>::bicubicInterpolation(const Vmm* vAuxPool, const Vmm& vWCoord, const Vmm& vHCoord) {
    const auto& vHTop      = vAuxPool[0];
    const auto& vWLeft     = vAuxPool[1];
    const auto& vDX        = vAuxPool[2];
    const auto& vDY        = vAuxPool[3];
    const auto& vXDotProd  = vAuxPool[4];
    const auto& vYDotProd  = vDX;
    const auto& vSrcShift0 = vAuxPool[5];
    const auto& vSrcShift  = vAuxPool[6];
    const auto& vCX0       = vAuxPool[7];
    const auto& vCX1       = vAuxPool[8];
    const auto& vCX2       = vAuxPool[9];
    const auto& vCX3       = vAuxPool[10];
    const auto& vAux       = vAuxPool[11]; // &vWLeft
    const auto& kMask0   = k1;
    const auto& kMask1   = k2;
    const auto& kMask2   = k3;
    const auto& kMask3   = k4;
    const auto& kAuxMask = k5;
    const auto& kMaskH   = k6;

    uni_vroundps(vWLeft, vWCoord, 0x1); // Round floor
    uni_vroundps(vHTop, vHCoord, 0x1);  // Round floor
    uni_vsubps(vDX, vWCoord, vWLeft);
    uni_vsubps(vDY, vHCoord, vHTop);
    uni_vsubps(vWLeft, vWLeft, vOnesF);
    uni_vsubps(vHTop, vHTop, vOnesF);
//uni_vmovups(ptr[regDst], vHTop);

    bicubicCoefficients(vCX0, vDX, 0); // TODO: for
    bicubicCoefficients(vCX1, vDX, 1);
    bicubicCoefficients(vCX2, vDX, 2);
    bicubicCoefficients(vCX3, vDX, 3);
//uni_vmovups(ptr[regDst], vCX3);

    if (jcp.paddingMode == PaddingMode::ZEROS) {
        uni_vmovups(vSrcShift0, vWLeft);
        uni_vfmadd231ps(vSrcShift0, vHTop, vSrcWidthF);
        uni_vcvtps2dq(vSrcShift0, vSrcShift0);
        if (dataTypeSize > 1)
            uni_vpslld(vSrcShift0, vSrcShift0, dataTypeShift); // multiply by source data type size.

        zerosPadding0(vWLeft, vSrcWidthF, kMask0, kAuxMask);
        uni_vaddps(vWLeft, vWLeft, vOnesF);
        zerosPadding0(vWLeft, vSrcWidthF, kMask1, kAuxMask);
        uni_vaddps(vWLeft, vWLeft, vOnesF);
        zerosPadding0(vWLeft, vSrcWidthF, kMask2, kAuxMask);
        uni_vaddps(vWLeft, vWLeft, vOnesF);
        zerosPadding0(vWLeft, vSrcWidthF, kMask3, kAuxMask);
    } else if (jcp.paddingMode == PaddingMode::BORDER) {
    } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
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
                zerosPadding0(vHCoord, vSrcHeightF, kMaskH, kMaskH);
                kandw(kAuxMask, kMaskH, kMask0);
                uni_vpxor(vAux, vAux, vAux);
            } else if (jcp.paddingMode == PaddingMode::BORDER) {
                borderPadding(vSrcShift0, vHCoord, vSrcHeightSub1Fl, kAuxMask);
                uni_vmulps(vSrcShift0, vSrcShift0, vSrcWidthF);
                uni_vmovups(vWCoord, vWLeft);
                borderPadding(vSrcShift, vWCoord, vSrcWidthSub1Fl, kAuxMask);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                uni_vcvtps2dq(vSrcShift, vSrcShift);
                if (dataTypeSize > 1)
                    uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
                kxnorw(kAuxMask, kAuxMask, kAuxMask);
            } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
                reflectionPadding(vSrcShift0, vHCoord, vAux, kAuxMask, 1);
//if (i == 0) {
//    uni_vmovups(ptr[rDstTmp], vHCoord);
//}
                uni_vmulps(vSrcShift0, vSrcShift0, vSrcWidthF);
                uni_vmovups(vWCoord, vWLeft);
                reflectionPadding(vSrcShift, vWCoord, vAux, kAuxMask, 0);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                uni_vcvtps2dq(vSrcShift, vSrcShift);
                if (dataTypeSize > 1)
                    uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
                kxnorw(kAuxMask, kAuxMask, kAuxMask);
            }
            uni_vpgatherdd(vAux, ptr[rSrcTmp + vSrcShift], kAuxMask);
//if (i == 0) {
//    uni_vmovups(ptr[rDstTmp], vAux);
//}
            if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                uni_vcvtdq2ps(vAux, vAux);
            }
            uni_vmulps(vXDotProd, vAux, vCX0);
// TODO: cycle 3?
            // (y - 1 + i; x)
            if (jcp.paddingMode == PaddingMode::ZEROS) {
                kandw(kAuxMask, kMaskH, kMask1);
                uni_vpxor(vAux, vAux, vAux);
                uni_vpaddd(vSrcShift, vSrcShift, vDataTypeSize);
            } else if (jcp.paddingMode == PaddingMode::BORDER) {
                uni_vaddps(vWCoord, vWCoord, vOnesF);
                borderPadding(vSrcShift, vWCoord, vSrcWidthSub1Fl, kAuxMask);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                uni_vcvtps2dq(vSrcShift, vSrcShift);
                if (dataTypeSize > 1)
                    uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
                kxnorw(kAuxMask, kAuxMask, kAuxMask);
            } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
                uni_vaddps(vWCoord, vWCoord, vOnesF);
                reflectionPadding(vSrcShift, vWCoord, vAux, kAuxMask, 0);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                uni_vcvtps2dq(vSrcShift, vSrcShift);
                if (dataTypeSize > 1)
                    uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
                kxnorw(kAuxMask, kAuxMask, kAuxMask);
            }
            uni_vpgatherdd(vAux, ptr[rSrcTmp + vSrcShift], kAuxMask);
            if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                uni_vcvtdq2ps(vAux, vAux);
            }
            uni_vfmadd231ps(vXDotProd, vAux, vCX1);

            // (y - 1 + i; x + 1)
            if (jcp.paddingMode == PaddingMode::ZEROS) {
                kandw(kAuxMask, kMaskH, kMask2);
                uni_vpxor(vAux, vAux, vAux);
                uni_vpaddd(vSrcShift, vSrcShift, vDataTypeSize);
            } else if (jcp.paddingMode == PaddingMode::BORDER) {
                uni_vaddps(vWCoord, vWCoord, vOnesF);
                borderPadding(vSrcShift, vWCoord, vSrcWidthSub1Fl, kAuxMask);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                uni_vcvtps2dq(vSrcShift, vSrcShift);
                if (dataTypeSize > 1)
                    uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
                kxnorw(kAuxMask, kAuxMask, kAuxMask);
            } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
                uni_vaddps(vWCoord, vWCoord, vOnesF);
                reflectionPadding(vSrcShift, vWCoord, vAux, kAuxMask, 0);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                uni_vcvtps2dq(vSrcShift, vSrcShift);
                if (dataTypeSize > 1)
                    uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
                kxnorw(kAuxMask, kAuxMask, kAuxMask);
            }
            uni_vpgatherdd(vAux, ptr[rSrcTmp + vSrcShift], kAuxMask);
            if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                uni_vcvtdq2ps(vAux, vAux);
            }
            uni_vfmadd231ps(vXDotProd, vAux, vCX2);

            // (y - 1 + i; x + 2)
            if (jcp.paddingMode == PaddingMode::ZEROS) {
                kandw(kAuxMask, kMaskH, kMask3);
                uni_vpxor(vAux, vAux, vAux);
                uni_vpaddd(vSrcShift, vSrcShift, vDataTypeSize);
            } else if (jcp.paddingMode == PaddingMode::BORDER) {
                uni_vaddps(vWCoord, vWCoord, vOnesF);
                borderPadding(vSrcShift, vWCoord, vSrcWidthSub1Fl, kAuxMask);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                uni_vcvtps2dq(vSrcShift, vSrcShift);
                if (dataTypeSize > 1)
                    uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
                kxnorw(kAuxMask, kAuxMask, kAuxMask);
            } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
                uni_vaddps(vWCoord, vWCoord, vOnesF);
                reflectionPadding(vSrcShift, vWCoord, vAux, kAuxMask, 0);
                uni_vaddps(vSrcShift, vSrcShift0, vSrcShift);
                uni_vcvtps2dq(vSrcShift, vSrcShift);
                if (dataTypeSize > 1)
                    uni_vpslld(vSrcShift, vSrcShift, dataTypeShift);
                kxnorw(kAuxMask, kAuxMask, kAuxMask);
            }
            uni_vpgatherdd(vAux, ptr[rSrcTmp + vSrcShift], kAuxMask);
            if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
                uni_vcvtdq2ps(vAux, vAux);
            }
            uni_vfmadd231ps(vXDotProd, vAux, vCX3);
//if (i == 0) {
//    uni_vmovups(ptr[rDstTmp], vXDotProd);
//}
//if (i == 0) {
//    uni_vmovups(ptr[rDstTmp], vSrcWidthB);
//}

            if (i != 3) {
                uni_vaddps(vHCoord, vHCoord, vOnesF);
                if (jcp.paddingMode == PaddingMode::ZEROS) {
                    uni_vpaddd(vSrcShift, vSrcShift, vSrcWidthB);
                }
            }
//if (i == 0) {
//    uni_vmovups(ptr[rDstTmp], vSrcShift);
//}

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

template <>
void jitGridSampleKernel<x64::avx2>::bicubicInterpolation(const Vmm* vAuxPool, const Vmm& vWCoord, const Vmm& vHCoord) {

}

template <>
void jitGridSampleKernel<x64::sse41>::bicubicInterpolation(const Vmm* vAuxPool, const Vmm& vWCoord, const Vmm& vHCoord) {

}

template <>
void jitGridSampleKernel<x64::avx512_core>::getCoordinates(const Vmm& vHCoord, const Vmm& vWCoord, const Vmm& vAux0) {
    uni_vpermd(vWCoord, vPermGridMask, ptr[regGrid]); // Permute to XXXX.XXXX.YYYY.YYYY
    Xbyak::Ymm ymmH = Xbyak::Ymm(vHCoord.getIdx());
    vextractf64x4(ymmH, vWCoord, 1); // Extract Y component

    add(regGrid, vlen);

    uni_vpermd(vAux0, vPermGridMask, ptr[regGrid]); // Permute to XXXX.XXXX.YYYY.YYYY
    Xbyak::Ymm ymmAux0 = Xbyak::Ymm(vAux0.getIdx());
    vinsertf64x4(vWCoord, vWCoord, ymmAux0, 1); // Extract X component
    vextractf64x4(ymmAux0, vAux0, 1); // Extract Y component
    vinsertf64x4(vHCoord, vHCoord, ymmAux0, 1);

    add(regGrid, vlen);
}

template <>
void jitGridSampleKernel<x64::avx2>::getCoordinates(const Vmm& vHCoord, const Vmm& vWCoord, const Vmm& vAux0) {
    uni_vpermd(vWCoord, vPermGridMask, ptr[regGrid]); // Permute to XXXX.YYYY
    Xbyak::Xmm xmmH = Xbyak::Xmm(vHCoord.getIdx());
    vextracti128(xmmH, vWCoord, 1); // Extract Y component

    add(regGrid, vlen);

    uni_vpermd(vAux0, vPermGridMask, ptr[regGrid]); // Permute to XXXX.YYYY
    Xbyak::Xmm xmmAux0 = Xbyak::Xmm(vAux0.getIdx());
    vinsertf128(vWCoord, vWCoord, xmmAux0, 1); // Extract X component
    vextractf128(xmmAux0, vAux0, 1); // Extract Y component
    vinsertf128(vHCoord, vHCoord, xmmAux0, 1);

    add(regGrid, vlen);
}

template <>
void jitGridSampleKernel<x64::sse41>::getCoordinates(const Vmm& vHCoord, const Vmm& vWCoord, const Vmm& vAux0) {
    pshufd(vWCoord, ptr[regGrid], 0xD8);
    shufpd(vHCoord, vWCoord, 0x2);

    add(regGrid, vlen);

    pshufd(vAux0, ptr[regGrid], 0xD8);
    shufpd(vWCoord, vAux0, 0x0);
    shufpd(vHCoord, vAux0, 0x3);

    add(regGrid, vlen);
}

template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::spatialLoop(const Vmm* vAuxPool) {
    auto& vHCoord = vAuxPool[0];
    auto& vWCoord = vAuxPool[1];
    auto& vAux0 = vAuxPool[2];

    Xbyak::Label lSpacialLoop, lTail;
    L(lSpacialLoop);
    {
        cmp(regWorkAmount, dataElPerVec);
        jl(lTail, T_NEAR);

        getCoordinates(vHCoord, vWCoord, vAux0);
        denormalizeRawCoordinates(vWCoord, vHCoord);
        interpolation(&vAuxPool[2], vWCoord, vHCoord);

        sub(regWorkAmount, dataElPerVec);
        add(regDst, vlen);

        jmp(lSpacialLoop, T_NEAR);
    }

    L(lTail);
    tail(true);
}

template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::process() {
    if (jcp.dynamicShapes) {
        Xbyak::Label lBatchLoop, lEnd;
        mov(regBatch, ptr[regParams + GET_OFF(batchNum)]);
        L(lBatchLoop);
        {
            cmp(regBatch, 0);
            jle(lEnd, T_NEAR);

            mov(regWorkAmount, ptr[regParams + GET_OFF(workAmount)]);
            spatialLoop(vAuxContainer);

            add(regSrc, ptr[regParams + GET_OFF(srcBatchStepB)]);
            add(regDst, ptr[regParams + GET_OFF(dstBatchStepB)]);

            sub(regBatch, 1);
            jmp(lBatchLoop, T_NEAR);
        }
        L(lEnd);
    } else {
        for (uint64_t i = 0lu; i < jcp.batchNum; i++) {
            mov(regWorkAmount, ptr[regParams + GET_OFF(workAmount)]);
            spatialLoop(vAuxContainer);

            add(regSrc, jcp.srcBatchStepB);
            add(regDst, jcp.dstBatchStepB);
        }
    }

}

template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::tail(bool isShortIdx, bool shiftFirst, bool blocked) {
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
//void jitGridSampleKernel<x64::avx512_core>::fillRestWorkMask(Vmask& kDstMask, Vmm& vmmAux, const Xbyak::Reg64& rWorkRest,
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
//void jitGridSampleKernel<x64::avx2>::fillRestWorkMask(Vmask& kDstMask, Vmm& vAux, const Xbyak::Reg64& rWorkRest,
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
//void jitGridSampleKernel<isa>::storeVectorPart(const Xbyak::Reg64& rDst, const Xbyak::Reg64& rToStoreCounter, Vmm& vmmSrc, Vmm& vAux) {
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
//void jitGridSampleKernel<x64::avx512_core>::fillVlenVector() {
//    mov(reg32Aux1, vlen);
//    vpbroadcastd(vmmVecLenB, reg32Aux1);
//}
//template <>
//void jitGridSampleKernel<x64::avx2>::fillVlenVector() {
//    vpcmpeqd(vmmVecLenB, vmmVecLenB, vmmVecLenB);
//    vpsrld(vmmVecLenB, vmmVecLenB, 31); // Right shift to 1.
//    uni_vpslld(vmmVecLenB, vmmVecLenB, 5);  // Left shift to 32.
//}

//template <x64::cpu_isa_t isa>
//bool jitGridSampleKernel<isa>::isSupportedConfiguration(uint64_t afterAxisSize) {
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

template struct jitGridSampleKernel<x64::avx512_core>;
template struct jitGridSampleKernel<x64::avx2>;
template struct jitGridSampleKernel<x64::sse41>;

}   // namespace intel_cpu
}   // namespace ov
