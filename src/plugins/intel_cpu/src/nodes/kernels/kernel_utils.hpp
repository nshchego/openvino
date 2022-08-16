// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu/x64/jit_generator.hpp"
#include <dnnl_types.h>


namespace ov {
namespace intel_cpu {

class jitKernelBase: public dnnl::impl::cpu::x64::jit_generator {
protected:

void uni_vfmsub132ps(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
                     const Xbyak::Operand &op) {
    // Note: x1 gets overriden by x1*op
    // This is incorrect if x1 == x2
    if (dnnl::impl::cpu::x64::is_subset(dnnl::impl::cpu::x64::avx2, dnnl::impl::cpu::x64::isa_all) &&
            dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
        vfmsub132ps(x1, x2, op);
    } else if (dnnl::impl::cpu::x64::is_subset(dnnl::impl::cpu::x64::avx, dnnl::impl::cpu::x64::isa_all) &&
               dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx)) {
        assert(x1.getIdx() != x2.getIdx());
        vmulps(x1, x1, op);
        vsubps(x1, x1, x2);
    } else {
        assert(x1.getIdx() != x2.getIdx());
        mulps(x1, op);
        subps(x1, x2);
    }
}

void uni_vfmsub231ps(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2, const Xbyak::Operand &op) {
    // Note: x1 gets overriden by x1*x2
    // This is incorrect if x1 == op
//    if (is_valid_isa(avx2))
    vfmsub231ps(x1, x2, op);
//    else if (is_valid_isa(avx)) {
//        assert(!x1.isEqualIfNotInherited(op));
//        vmulps(x1, x1, x2);
//        vsubps(x1, x1, op);
//    } else {
//        assert(!x1.isEqualIfNotInherited(op));
//        mulps(x1, x2);
//        subps(x1, op);
//    }
}

void uni_vfmsub231ps(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2, const Xbyak::Operand &op) {
//    if (is_valid_isa(avx2))
        vfmsub231ps(x1, x2, op);
//    else {
//        // Note: x1 gets overriden by x1*x2
//        // This is incorrect if x1 == op
//        assert(!x1.isEqualIfNotInherited(op));
//        vmulps(x1, x1, x2);
//        vsubps(x1, x1, op);
//    }
}

void uni_kmovd(const Xbyak::Opmask& kDst, const Xbyak::Opmask& kSrc) {
    kmovq(kDst, kSrc);
}

void uni_kmovd(const Xbyak::Xmm& vDst, const Xbyak::Xmm& vSrc) {
    uni_vmovups(vDst, vSrc);
}

void uni_kxnorw(const Xbyak::Opmask& kDst, const Xbyak::Opmask& kSrc1, const Xbyak::Opmask& kSrc2) {
    kxnorw(kDst, kSrc1, kSrc2);
}

void uni_kxnorw(const Xbyak::Xmm& vDst, const Xbyak::Xmm& vSrc1, const Xbyak::Xmm& vSrc2) {
    uni_vpxor(vDst, vSrc1, vSrc2);
    if (dnnl::impl::cpu::x64::is_subset(dnnl::impl::cpu::x64::avx, dnnl::impl::cpu::x64::isa_all) &&
              dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx)) {
        vandnps(vDst, vSrc1, vSrc2);
    } else {
        andnps(vDst, vSrc1);
    }
}

//void uni_vpgatherdd(const Xbyak::Ymm& vDst, const Xbyak::Address& srcAddr, const Xbyak::Ymm& vMask) {
//    vpgatherdd(vDst, srcAddr, vMask);
//}

void uni_vpgatherdd(const Xbyak::Xmm& vDst, const Xbyak::Address& srcAddr, const Xbyak::Xmm& vMask) {
    vpgatherdd(vDst, srcAddr, vMask);
}

void uni_vpgatherdd(const Xbyak::Zmm& vDst, const Xbyak::Address& srcAddr, const Xbyak::Opmask& kMask) {
    vpgatherdd(vDst | kMask, srcAddr);
}

void uni_vpermd(const Xbyak::Xmm& vDst, const Xbyak::Xmm& vMask, const Xbyak::Operand& src) {
    //
}

void uni_vpermd(const Xbyak::Zmm& vDst, const Xbyak::Zmm& vMask, const Xbyak::Operand& src) {
    vpermd(vDst, vMask, src);
}
};

}
}