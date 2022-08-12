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
    uni_vpbroadcastd(vSrcWidthFl, ptr[regAux1]);
    mov(regAux1, ptr[regParams + GET_OFF(srcHeightFl)]);
    uni_vpbroadcastd(vSrcHeightFl, ptr[regAux1]);

    mov(regSrcChannelStepB, ptr[regParams + GET_OFF(srcChannelStepB)]);
    mov(regDstChannelStepB, ptr[regParams + GET_OFF(dstChannelStepB)]);
    mov(regChannelsNum, ptr[regParams + GET_OFF(channelsNum)]);

    mov(regAux1, reinterpret_cast<uintptr_t>(permGridMaskUni));
    uni_vmovups(vPermGridMask, ptr[regAux1]);

    mov(regAux1, ptr[regParams + GET_OFF(one)]);
    uni_vpbroadcastd(vOnes, ptr[regAux1]);

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
//            uni_vcvtps2dq(vSrcWidthB, vSrcWidthFl); // TODO: move to init?
//            if (dataTypeSize != 1)
//                uni_vpslld(vSrcWidthB, vSrcWidthB, dataTypeShift); // multiply by source data type size.
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
        }
    }

    process(true, true);

    this->postamble();
}

template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::denormalizeRawCoordinates(const Vmm& vWCoord, const Vmm& vHCoord) {
    if (jcp.alignCorners) {
        uni_vfmadd132ps(vWCoord, vWDenormCoef, vWDenormCoef);
        uni_vfmadd132ps(vHCoord, vHDenormCoef, vHDenormCoef);
    } else {
        uni_vfmadd132ps(vWCoord, vSrcWidthFl, vSrcWidthFl);
        vfmsub132ps(vWCoord, vHalf, vHalf);
        uni_vfmadd132ps(vHCoord, vSrcHeightFl, vSrcHeightFl);
        vfmsub132ps(vHCoord, vHalf, vHalf);
    }
}

template <>
void jitGridSampleKernel<x64::avx512_core>::getZeroMask(const Vmm& vWCoord, const Vmm& vHCoord, const Vmask& kDst, const Vmask& kAux) {
    vcmpps(kAux, vWCoord, vSrcWidthFl, 0x1); // vWCoord < vSrcWidthFl
    vcmpps(kDst | kAux, vZeros, vWCoord, 0x2); // vWCoord >= vZeros

    vcmpps(kAux | kDst, vHCoord, vSrcHeightFl, 0x1); // vHCoord < vSrcHeightFl
    vcmpps(kDst | kAux, vZeros, vHCoord, 0x2); // vHCoord >= vZeros
}

template <>
void jitGridSampleKernel<x64::avx2>::getZeroMask(const Vmm& vWCoord, const Vmm& vHCoord, const Vmask& kDst, const Vmask& kAux) {
    uni_vcmpps(kAux, vSrcWidthFl, vWCoord, 0x1); // vWCoord < vSrcWidthFl
    uni_vcmpps(kDst, vWCoord, vZeros, 0x2); // vWCoord >= vZeros
    uni_vpand(kDst, kAux, kDst);

    uni_vcmpps(kAux, vSrcHeightFl, vHCoord, 0x1); // vHCoord < vSrcHeightFl
    uni_vpand(kDst, kAux, kDst);
    uni_vcmpps(kAux, vHCoord, vZeros, 0x2); // vHCoord >= vZeros
    uni_vpand(kDst, kAux, kDst);
}

template <>
void jitGridSampleKernel<x64::sse41>::getZeroMask(const Vmm& vWCoord, const Vmm& vHCoord, const Vmask& kDst, const Vmask& kAux) {
    uni_vcmpps(kAux, vSrcWidthFl, vWCoord, 0x1); // vWCoord < vSrcWidthFl
    uni_vcmpps(kDst, vWCoord, vZeros, 0x2); // vWCoord >= vZeros
    uni_vpand(kDst, kAux, kDst);

    uni_vcmpps(kAux, vSrcHeightFl, vHCoord, 0x1); // vHCoord < vSrcHeightFl
    uni_vpand(kDst, kAux, kDst);
    uni_vcmpps(kAux, vHCoord, vZeros, 0x2); // vHCoord >= vZeros
    uni_vpand(kDst, kAux, kDst);
}

template <>
void jitGridSampleKernel<x64::avx512_core>::getBorderPadding(const Vmm& vWCoord, const Vmm& vHCoord, const Vmask& kAux) {
    vrangeps(vWCoord, vWCoord, vSrcWidthSub1Fl, 0x0); // vWCoord >= vSrcWidthFl
    vrangeps(vWCoord, vWCoord, vZeros, 0x1); // vWCoord < vZeros

    vrangeps(vHCoord, vHCoord, vSrcHeightSub1Fl, 0x0); // vHCoord >= vSrcHeightFl
    vrangeps(vHCoord, vHCoord, vZeros, 0x1); // vHCoord < vZeros
}

template <>
void jitGridSampleKernel<x64::avx2>::getBorderPadding(const Vmm& vWCoord, const Vmm& vHCoord, const Vmask& kAux) {
//    vcmpps(kAux, vSrcWidthFl, vWCoord, 0x2); // vWCoord >= vSrcWidthFl
//    vmovups(vWCoord | kAux, vSrcWidthSub1Fl);
//    vcmpps(kAux, vWCoord, vZeros, 0x1); // vWCoord < vZeros
//    vmovups(vWCoord | kAux, vZeros);
//
//    vcmpps(kAux, vSrcHeightFl, vHCoord, 0x2); // vHCoord >= vSrcHeightFl
//    vmovups(vHCoord | kAux, vSrcHeightSub1Fl);
//    vcmpps(kAux, vHCoord, vZeros, 0x1); // vHCoord < vZeros
//    vmovups(vHCoord | kAux, vZeros);
}

template <>
void jitGridSampleKernel<x64::sse41>::getBorderPadding(const Vmm& vWCoord, const Vmm& vHCoord, const Vmask& kAux) {
}

template <>
void jitGridSampleKernel<x64::avx512_core>::reflectionPadding(const Vmm& vCoord, const Vmm& vAux, const Vmask& kAux, const uint8_t dim) {
    Vmm vSrcDimFl, vSrcDimMul2Fl, vSrcDimMul2Sub1Fl;
    if (dim == 0) {
        vSrcDimFl = vSrcWidthFl;
        vSrcDimMul2Fl = vSrcWidthMul2Fl;
        vSrcDimMul2Sub1Fl = vSrcWidthMul2Sub1Fl;
    } else {
        vSrcDimFl = vSrcHeightFl;
        vSrcDimMul2Fl = vSrcHeightMul2Fl;
        vSrcDimMul2Sub1Fl = vSrcHeightMul2Sub1Fl;
    }

    if (jcp.alignCorners) {
        // abs(x) % D21
        vpabsd(vCoord, vCoord);
        uni_vdivps(vAux, vCoord, vSrcDimMul2Sub1Fl);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfnmadd231ps(vCoord, vAux, vSrcDimMul2Sub1Fl); // x % D21
    } else {
        // (x % D2 + D2) % D2
        uni_vdivps(vAux, vCoord, vSrcDimMul2Fl);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfmsub231ps(vCoord, vAux, vSrcDimMul2Fl); // -x % D2 TODO: vfnmadd231
        uni_vsubps(vCoord, vCoord, vSrcDimMul2Fl); // -(x % D2 + D2)
        uni_vdivps(vAux, vCoord, vSrcDimMul2Fl);
        uni_vroundps(vAux, vAux, 0x3); // Round floor
        uni_vfmsub231ps(vCoord, vAux, vSrcDimMul2Fl); // (x % D2 + D2) % D2
    }

    uni_vsubps(vAux, vSrcDimMul2Sub1Fl, vCoord);
    vcmpps(kAux, vSrcDimFl, vCoord, 0x2); // vCoord >= vSrcDimFl
    vmovups(vCoord | kAux, vAux);
}

template <>
void jitGridSampleKernel<x64::avx2>::reflectionPadding(const Vmm& vCoord, const Vmm& vAux, const Vmask& kAux, const uint8_t dim) {
    Vmm vSrcDimFl;
    if (dim == 0) {
        vSrcDimFl = vSrcWidthFl;
    } else {
        vSrcDimFl = vSrcHeightFl;
    }
    const auto& vSrcDimMul2Fl = kAux;
//    uni_vmulps(vSrcDimMul2Fl, vSrcDimFl, 2);
    // (x % W2 + W2) % W2
    uni_vdivps(vAux, vCoord, vSrcDimMul2Fl);
    uni_vroundps(vAux, vAux, 0x1); // Round floor
    uni_vfmsub231ps(vCoord, vAux, vSrcDimMul2Fl); // x % W2
    uni_vsubps(vCoord, vSrcDimMul2Fl, vCoord); // x % W2 + W2
    uni_vdivps(vAux, vCoord, vSrcDimMul2Fl);
    uni_vroundps(vAux, vAux, 0x1); // Round floor
    uni_vfmsub231ps(vCoord, vAux, vSrcDimMul2Fl); // (x % W2 + W2) % W2

    uni_vsubps(vAux, vSrcDimMul2Fl, vOnes);
    uni_vsubps(vAux, vAux, vCoord);
    vcmpps(kAux, vSrcDimFl, vCoord, 0x2); // vCoord >= vSrcDimFl
//        vmovups(vCoord | kMask, vAux);
}

template <>
void jitGridSampleKernel<x64::sse41>::reflectionPadding(const Vmm& vCoord, const Vmm& vAux, const Vmask& kAux, const uint8_t dim) {
// TODO:
}

//template <>
//void jitGridSampleKernel<x64::avx512_core>::reflectionWithAlign(const Vmm& vCoord, const Vmm& vAux, const Vmask& kAux, const uint8_t dim) {
//    Vmm vSrcDimFl, vSrcDimMul2Fl, vSrcDimMul2Sub1Fl;
//    if (dim == 0) {
//        vSrcDimFl = vSrcWidthFl;
//        vSrcDimMul2Fl = vSrcWidthMul2Fl;
//        vSrcDimMul2Sub1Fl = vSrcWidthMul2Sub1Fl;
//    } else {
//        vSrcDimFl = vSrcHeightFl;
//        vSrcDimMul2Fl = vSrcHeightMul2Fl;
//        vSrcDimMul2Sub1Fl = vSrcHeightMul2Sub1Fl;
//    }
//    // abs(x) % D21
//    vpabsb(vCoord, vCoord);
//    uni_vdivps(vAux, vCoord, vSrcDimMul2Sub1Fl);
//    uni_vroundps(vAux, vAux, 0x3); // Round floor
//    uni_vfnmadd231ps(vCoord, vAux, vSrcDimMul2Sub1Fl); // x % D21 vfnmadd231ps
////    uni_vsubps(vCoord, vCoord, vSrcDimMul2Fl); // -(x % W2 + W2)
////    uni_vdivps(vAux, vCoord, vSrcDimMul2Fl);
////    uni_vroundps(vAux, vAux, 0x3); // Round floor
////    uni_vfmsub231ps(vCoord, vAux, vSrcDimMul2Fl); // (x % W2 + W2) % W2
////
//    uni_vsubps(vAux, vSrcDimMul2Sub1Fl, vCoord);
//    vcmpps(kAux, vSrcDimFl, vCoord, 0x2); // vCoord >= vSrcDimFl
//    vmovups(vCoord | kAux, vAux);
//}
//
//template <>
//void jitGridSampleKernel<x64::avx2>::reflectionWithAlign(const Vmm& vCoord, const Vmm& vAux, const Vmask& kAux, const uint8_t dim) {
//
//}
//
//template <>
//void jitGridSampleKernel<x64::sse41>::reflectionWithAlign(const Vmm& vCoord, const Vmm& vAux, const Vmask& kAux, const uint8_t dim) {
//
//}

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
template <>
void jitGridSampleKernel<x64::avx512_core>::interpolation(const Vmm* vAuxPool, const Vmm& vWCoord, const Vmm& vHCoord) {
    if (jcp.interpolationMode == InterpolationMode::BILINEAR) {
        const auto& vDX     = vAuxPool[0];
        const auto& vDY     = vAuxPool[1];
        const auto& vQ0     = vAuxPool[2];
        const auto& vQ1     = vAuxPool[3];
        const auto& shift00 = vWCoord;
        const auto& shift01 = vHCoord;
        const auto& shift10 = vAuxPool[4];
        const auto& shift11 = vAuxPool[5];
        const auto& vAux3   = vAuxPool[6];
        const auto& kMask00  = k1;
        const auto& kMask01  = k2;
        const auto& kMask10  = k3;
        const auto& kMask11  = k4;
        const auto& kAuxMask = k5;

        uni_vmovups(vDX, vWCoord);
        uni_vmovups(vDY, vHCoord);
        uni_vroundps(vWCoord, vWCoord, 0x1); // Round floor
        uni_vroundps(vHCoord, vHCoord, 0x1); // Round floor
        uni_vsubps(vDX, vDX, vWCoord);
        uni_vsubps(vDY, vDY, vHCoord);

        uni_vaddps(shift10, vWCoord, vOnes);
        uni_vaddps(shift11, vHCoord, vOnes);
        if (jcp.paddingMode == PaddingMode::ZEROS) {
            getZeroMask(vWCoord, vHCoord, kMask00, kAuxMask); // (y; x)
            getZeroMask(shift10, vHCoord, kMask01, kAuxMask); // (y; x + 1)
            getZeroMask(shift10, shift11, kMask11, kAuxMask); // (y + 1; x + 1)
            getZeroMask(vWCoord, shift11, kMask10, kAuxMask); // (y + 1; x)

            uni_vfmadd231ps(vWCoord, vHCoord, vSrcWidthFl); // (y; x)
            uni_vcvtps2dq(shift00, vWCoord);
            if (dataTypeSize > 1)
                uni_vpslld(shift00, shift00, dataTypeShift); // multiply by source data type size.
            uni_vpaddd(shift01, shift00, vDataTypeSize);
            uni_vpaddd(shift10, shift00, vSrcWidthB);
            uni_vpaddd(shift11, shift10, vDataTypeSize);
        } else if (jcp.paddingMode == PaddingMode::BORDER) {
            getBorderPadding(vWCoord, vHCoord, kAuxMask);
            getBorderPadding(shift10, shift11, kAuxMask);
        } else if (jcp.paddingMode == PaddingMode::REFLECTION) {
            reflectionPadding(vWCoord, vQ0, kAuxMask, 0);
            reflectionPadding(vHCoord, vQ0, kAuxMask, 1);
            reflectionPadding(shift10, vQ0, kAuxMask, 0);
            reflectionPadding(shift11, vQ0, kAuxMask, 1);
        }
        if (jcp.paddingMode == PaddingMode::BORDER || jcp.paddingMode == PaddingMode::REFLECTION) {
            uni_vmovups(vAux3, shift11);
            // W * y + x
            uni_vfmadd132ps(vAux3, vWCoord, vSrcWidthFl);   // (y + 1; x)
            uni_vfmadd231ps(vWCoord, vHCoord, vSrcWidthFl); // (y; x)
            uni_vfmadd132ps(vHCoord, shift10, vSrcWidthFl); // (y; x + 1)
            uni_vfmadd132ps(shift11, shift10, vSrcWidthFl); // (y + 1; x + 1)
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
        const Xbyak::Reg64& rChannel = regAux1;
        const Xbyak::Reg32& rChannel32 = Xbyak::Reg32(rChannel.getIdx());
        const Xbyak::Reg64& rSrcTmp = regAux2;
        const Xbyak::Reg64& rDstTmp = regAux3;
        mov(rChannel, 0);
        mov(rSrcTmp, regSrc);
        mov(rDstTmp, regDst);
        L(lChannelLoopBegin);
        {
            cmp(rChannel, regChannelsNum);
            jge(lChannelLoopEnd, T_NEAR);

//            if (jcp.paddingMode == PaddingMode::ZEROS) {
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
//            } //else if (jcp.paddingMode == PaddingMode::BORDER || jcp.paddingMode == PaddingMode::REFLECTION) {
//                // (y; x)
//                kxnorw(kAuxMask, kAuxMask, kAuxMask);
//                uni_vpgatherdd(vQ0, ptr[rSrcTmp + shift00], kAuxMask); // v00 -> vQ0
//                if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
//                    uni_vcvtdq2ps(vQ0, vQ0);
//                }
//                uni_vfmsub213ps(vQ0, vDX, vQ0); // q0 = -(v00 - dx * v00)
//
//                // (y; x + 1)
//                kxnorw(kAuxMask, kAuxMask, kAuxMask);
//                uni_vpgatherdd(vAux3, ptr[rSrcTmp + shift01], kAuxMask);
//                if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
//                    uni_vcvtdq2ps(vAux3, vAux3);
//                }
//                uni_vfmsub231ps(vQ0, vAux3, vDX); // q0 = -q0 + dx * v01
//
//                // (y + 1; x + 1)
//                kxnorw(kAuxMask, kAuxMask, kAuxMask);
//                uni_vpgatherdd(vAux3, ptr[rSrcTmp + shift11], kAuxMask);
//                if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
//                    uni_vcvtdq2ps(vAux3, vAux3);
//                }
//
//                // (y + 1; x)
//                kxnorw(kAuxMask, kAuxMask, kAuxMask);
//                uni_vpgatherdd(vQ1, ptr[rSrcTmp + shift10], kAuxMask);
//                if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
//                    uni_vcvtdq2ps(vQ1, vQ1);
//                }
//
//                uni_vfmsub213ps(vQ1, vDX, vQ1); // q1 = -(v10 - dx * v10)
//                uni_vfmsub231ps(vQ1, vAux3, vDX); // q1 = -q1 + dx * v11
//                // Res = q0 + dy * (q1 - q0)
//                uni_vsubps(vQ1, vQ1, vQ0);
//                uni_vfmadd132ps(vQ1, vQ0, vDY);
//
//                if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
//                    uni_vcvtps2dq(vQ1, vQ1);
//                }
//            }

            uni_vmovups(ptr[rDstTmp], vQ1);
            add(rSrcTmp, regSrcChannelStepB);
            add(rDstTmp, regDstChannelStepB);
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

template <>
void jitGridSampleKernel<x64::avx2>::interpolation(const Vmm* vAuxPool, const Vmm& vWCoord, const Vmm& vHCoord) {
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
////            getZeroMask(vWCoord, vHCoord, kMask0, kMask1);
////            uni_kmovq(kMask1, kMask0);
////        }
////kmovq(regAux1, kMask0);
////uni_vmovups(ptr[regDst], vWCoord);
//
//    // PER CHANNEL LOOP
//    Xbyak::Label lChannelLoopBegin, lChannelLoopEnd;
//    const Xbyak::Reg64& rChannel = regAux1;
//    const Xbyak::Reg32& rChannel32 = Xbyak::Reg32(rChannel.getIdx());
//    const Xbyak::Reg64& rSrcTmp = regAux2;
//    const Xbyak::Reg64& rDstTmp = regAux3;
//    mov(rChannel, 0);
//    mov(rSrcTmp, regSrc);
//    mov(rDstTmp, regDst);
////        uni_vfmadd132ps(vHCoord, vWCoord, vSrcWidthFl);
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
//            getZeroMask(vWCoord, vHCoord, kMask0, kMask1);
//            uni_vmovups(vGatherShift, vHCoord);
//            uni_vfmadd132ps(vGatherShift, vWCoord, vSrcWidthFl);
//            uni_vcvtps2dq(vGatherShift, vGatherShift);
//            if (dataTypeSize > 1)
//                uni_vpslld(vGatherShift, vGatherShift, dataTypeShift);
//            uni_vpxor(vQ0, vQ0, vQ0);
//            uni_vpgatherdd(vQ0, ptr[rSrcTmp + vGatherShift], kMask0); // v00 -> vQ0 TODO: 64b 16b 8b?
////uni_vmovups(ptr[rDstTmp], vOnes);
////                uni_kmovq(kMask0, kMask1);
//            if (jcp.inDataPrc == InferenceEngine::Precision::I32) {
//                uni_vcvtdq2ps(vQ0, vQ0);
//            }
//            uni_vfmsub213ps(vQ0, vDX, vQ0); // q0 = -(v00 - dx * v00)
//
//            // (x + 1; y)
////                uni_vpaddd(vGatherShift, vGatherShift, vDataTypeSize);
//            uni_vaddps(vWCoord, vWCoord, vOnes);
//            getZeroMask(vWCoord, vHCoord, kMask0, kMask1);
//            uni_vmovups(vGatherShift, vHCoord);
//            uni_vfmadd132ps(vGatherShift, vWCoord, vSrcWidthFl);
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
//            uni_vaddps(vHCoord, vHCoord, vOnes);
//            getZeroMask(vWCoord, vHCoord, kMask0, kMask1);
//            uni_vmovups(vGatherShift, vHCoord);
//            uni_vfmadd132ps(vGatherShift, vWCoord, vSrcWidthFl);
//            uni_vcvtps2dq(vGatherShift, vGatherShift);
//            if (dataTypeSize > 1)
//                uni_vpslld(vGatherShift, vGatherShift, dataTypeShift);
//            uni_vpxor(vAux3, vAux3, vAux3);
//            uni_vpgatherdd(vAux3, ptr[rSrcTmp + vGatherShift], kMask0); // TODO: 64b 16b 8b?
////                uni_kmovq(kMask0, kMask1);
//
//            // (x; y + 1)
////                uni_vpsubd(vGatherShift, vGatherShift, vDataTypeSize);
//            uni_vsubps(vWCoord, vWCoord, vOnes);
//            getZeroMask(vWCoord, vHCoord, kMask0, kMask1);
//            uni_vmovups(vGatherShift, vHCoord);
//            uni_vfmadd132ps(vGatherShift, vWCoord, vSrcWidthFl);
//            uni_vcvtps2dq(vGatherShift, vGatherShift);
//            if (dataTypeSize > 1)
//                uni_vpslld(vGatherShift, vGatherShift, dataTypeShift);
//            uni_vpxor(vQ1, vQ1, vQ1);
//            uni_vpgatherdd(vQ1, ptr[rSrcTmp + vGatherShift], kMask0); // TODO: 64b 16b 8b?
////                uni_kmovq(kMask0, kMask1);
//            uni_vsubps(vHCoord, vHCoord, vOnes);
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
void jitGridSampleKernel<x64::sse41>::interpolation(const Vmm* vAuxPool, const Vmm& vWCoord, const Vmm& vHCoord) {

}

// Requires vAuxPool length 4.
// Returns calculated shifts in vAuxPool[0] and mask in vAuxPool[1].
template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::calcCoordinates(const Vmm* vAuxPool, bool shiftFirst) {
//template <>
//void jitGridSampleKernel<x64::avx512_core>::calcCoordinates(const Vmm* vAuxPool, bool shiftFirst) {
    auto& vHCoord = vAuxPool[0];
    auto& vWCoord = vAuxPool[1];
    auto& vAux0 = vAuxPool[2];
    auto& vAux1 = vAuxPool[3];

//    if (shiftFirst)
//        add(regGrid, vlen);

    Xbyak::Label lSpacialLoop, lTail;
    L(lSpacialLoop);
    {
        cmp(regWorkAmount, dataElPerVec);
        jl(lTail, T_NEAR);

//        if (isa == x64::avx512_core) {
                {
                    // Permute to XXXX.XXXX.YYYY.YYYY
                    uni_vpermd(vWCoord, vPermGridMask, ptr[regGrid]);

                    Xbyak::Zmm zmmW = Xbyak::Zmm(vWCoord.getIdx());
                    Xbyak::Ymm ymmH = Xbyak::Ymm(vHCoord.getIdx());
                    // Extract Y component
                    vextractf64x4(ymmH, zmmW, 1);
                }
//        } else if (isa == x64::avx2) {
//            uni_vmovups(vAux0, ptr[regGrid]);
//            // Permute to XXXX.YYYY
//            uni_vpermd(vHCoord, vPermGridMask, vAux0);
//            // Extract Y component
//            Xbyak::Xmm xmmW = Xbyak::Xmm(vWCoord.getIdx());
//            vextracti128(xmmW, vHCoord, 1);
//        } else if (isa == x64::sse41) {
//            // TODO: SSE
//        }

        add(regGrid, vlen);
//        if (isa == x64::avx512_core) {
                {
                    // Permute to XXXX.XXXX.YYYY.YYYY
                    uni_vpermd(vAux0, vPermGridMask, ptr[regGrid]);

                    Xbyak::Zmm zmmH = Xbyak::Zmm(vHCoord.getIdx());
                    Xbyak::Zmm zmmW = Xbyak::Zmm(vWCoord.getIdx());
                    Xbyak::Zmm zmmAux0 = Xbyak::Zmm(vAux0.getIdx());
                    Xbyak::Ymm ymmAux1 = Xbyak::Ymm(vAux1.getIdx());
                    // Extract X component
                    // TODO: Change to vpermd if will be available registers
                    vextractf64x4(ymmAux1, zmmAux0, 0);
                    vinsertf64x4(zmmW, zmmW, ymmAux1, 1);
                    // Extract Y component
                    vextractf64x4(ymmAux1, zmmAux0, 1);
                    vinsertf64x4(zmmH, zmmH, ymmAux1, 1);
                }
//        } else if (isa == x64::avx2) {
//            uni_vmovups(vAux0, ptr[regGrid]);
//            // Permute to XXXX.YYYY
//            uni_vpermd(vAux0, vPermGridMask, vAux0);
//            Xbyak::Xmm xmmAux2 = Xbyak::Xmm(vAux1.getIdx());
//            // Extract X component
//            vextractf128(xmmAux2, vAux0, 0);
//            vinsertf128(vHCoord, vHCoord, xmmAux2, 1);
//            // Extract Y component
//            vextractf128(xmmAux2, vAux0, 1);
//            vinsertf128(vWCoord, vWCoord, xmmAux2, 1);
//        } else if (isa == x64::sse41) {
//            // TODO: SSE
//        }

//        if (!shiftFirst)
            add(regGrid, vlen);

        denormalizeRawCoordinates(vWCoord, vHCoord);

        interpolation(&vAuxPool[2], vWCoord, vHCoord);

        sub(regWorkAmount, dataElPerVec);
        add(regDst, vlen);

        jmp(lSpacialLoop, T_NEAR);
    }

    L(lTail);
    tail(true);
}

//template <x64::cpu_isa_t isa>
//void jitGridSampleKernel<isa>::calcCoordinatesBlock(Vmm* vAuxPool, bool shiftFirst) {
//    // Most likely there will no significant performance gain vs memcpy in reference implementation on big blocks after axis,
//    // therefore no time was invested to this case yet.
//    IE_THROW() << "Unsupported case.";
//}

template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::process(bool isShortIdx, bool blocked) {
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
void jitGridSampleKernel<isa>::process32b(bool isShortIdx, bool blocked) {

    // First iteration
//    shiftIdxAndGather(vAuxContainer, isShortIdx, false, blocked);
//    uni_vmovups(ptr[regDst], vAuxContainer[2]);

    if (jcp.dynamicShapes) {
        Xbyak::Label lBatchLoop, lEnd;
        mov(regBatch, ptr[regParams + GET_OFF(batchNum)]);
        L(lBatchLoop);
        {
            cmp(regBatch, 0);
            jle(lEnd, T_NEAR);

            mov(regWorkAmount, ptr[regParams + GET_OFF(workAmount)]);
            shiftIdxAndGather(vAuxContainer, isShortIdx, true, blocked);

            add(regSrc, ptr[regParams + GET_OFF(srcBatchStepB)]);
            add(regDst, ptr[regParams + GET_OFF(dstBatchStepB)]);

            sub(regBatch, 1);
            jmp(lBatchLoop, T_NEAR);
        }
        L(lEnd);
    } else {
        for (uint64_t i = 0lu; i < jcp.batchNum; i++) {
            mov(regWorkAmount, ptr[regParams + GET_OFF(workAmount)]);
            shiftIdxAndGather(vAuxContainer, isShortIdx, true, blocked);

            add(regSrc, jcp.srcBatchStepB);
            add(regDst, jcp.dstBatchStepB);
        }
    }

}

template <x64::cpu_isa_t isa>
void jitGridSampleKernel<isa>::process16b(bool isShortIdx, bool blocked) {
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
void jitGridSampleKernel<isa>::process8b(bool isShortIdx, bool blocked) {
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
void jitGridSampleKernel<isa>::shiftIdxAndGather(const Vmm* vAuxPool, bool isShortIdx, bool shiftFirst, bool blocked) {

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
