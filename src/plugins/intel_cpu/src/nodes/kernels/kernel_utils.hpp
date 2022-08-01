// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu/x64/jit_generator.hpp"
#include <dnnl_types.h>


namespace ov {
namespace intel_cpu {

void uni_vfmsub231ps(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
                     const Xbyak::Operand &op) {
    // Note: x1 gets overriden by x1*x2
    // This is incorrect if x1 == op
    if (is_valid_isa(avx2))
        vfmsub231ps(x1, x2, op);
    else if (is_valid_isa(avx)) {
        assert(!x1.isEqualIfNotInherited(op));
        vmulps(x1, x1, x2);
        vsubps(x1, x1, op);
    } else {
        assert(!x1.isEqualIfNotInherited(op));
        mulps(x1, x2);
        subps(x1, op);
    }
}

void uni_vfmsub231ps(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
                     const Xbyak::Operand &op) {
    if (is_valid_isa(avx2))
        vfmsub231ps(x1, x2, op);
    else {
        // Note: x1 gets overriden by x1*x2
        // This is incorrect if x1 == op
        assert(!x1.isEqualIfNotInherited(op));
        vmulps(x1, x1, x2);
        vsubps(x1, x1, op);
    }
}

void uni_vpgatherdd(const Xbyak::Ymm& vDst, const Xbyak::Address& srcAddr, const Xbyak::Ymm& vMask) {
    vpgatherdd(vDst, srcAddr, vMask);
}

void uni_vpgatherdd(const Xbyak::Zmm& vDst, const Xbyak::Address& srcAddr, const Xbyak::Opmask& kMask) {
    vpgatherdd(vDst | kMask, srcAddr);
}

}
}