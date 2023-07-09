// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_kernel_base.hpp"

using namespace ov::intel_cpu::kernel;
using namespace dnnl::impl::cpu;
using namespace Xbyak;


JitKernelBase::JitKernelBase(const char* name, x64::cpu_isa_t isa) :
        x64::jit_generator(name, nullptr, dnnl::impl::cpu::x64::MAX_CODE_SIZE, true, isa) {
    if (isa == x64::avx512_core) {
        vcvtneps2bf16.reset(new jit_uni_vcvtneps2bf16(this, x64::avx512_core));
    } else if (isa == x64::avx2) {
        vcvtneps2bf16.reset(new jit_uni_vcvtneps2bf16(this, x64::avx2));
    } else if (isa == x64::sse41) {
        vcvtneps2bf16.reset(new jit_uni_vcvtneps2bf16(this, x64::sse41));
    }
}

void JitKernelBase::uni_vfmsub132ps(const Xmm& vmm_dst,
                                    const Xmm& vSrc,
                                    const Operand& op) {
    if (isValidIsa(x64::avx2)) {
        vfmsub132ps(vmm_dst, vSrc, op);
    } else if (isValidIsa(x64::avx)) {
        assert(vmm_dst.getIdx() != vSrc.getIdx());
        vmulps(vmm_dst, vmm_dst, op);
        vsubps(vmm_dst, vmm_dst, vSrc);
    } else {
        assert(vmm_dst.getIdx() != vSrc.getIdx());
        mulps(vmm_dst, op);
        subps(vmm_dst, vSrc);
    }
}

void JitKernelBase::uni_vfnmadd132ps(const Xmm& vmm_dst,
                                     const Xmm& vSrc,
                                     const Operand& op) {
    if (isValidIsa(x64::avx2)) {
        vfnmadd132ps(vmm_dst, vSrc, op);
    } else if (isValidIsa(x64::avx)) {
        assert(vmm_dst.getIdx() != vSrc.getIdx());
        vmulps(vmm_dst, vmm_dst, op);
        vsubps(vmm_dst, vSrc, vmm_dst);
    } else {
        assert(vmm_dst.getIdx() != vSrc.getIdx());
        mulps(vmm_dst, op);
        subps(vSrc, vmm_dst);
        movups(vmm_dst, vSrc);
    }
}

void JitKernelBase::uni_vfmsub231ps(const Xmm& vmm_dst,
                                    const Xmm& vSrc,
                                    const Operand& op) {
    if (isValidIsa(x64::avx2)) {
        vfmsub231ps(vmm_dst, vSrc, op);
    } else if (isValidIsa(x64::avx)) {
        assert(!vmm_dst.isEqualIfNotInherited(op));
        vmulps(vSrc, vSrc, op);
        vsubps(vmm_dst, vSrc, vmm_dst);
    } else {
        assert(!vmm_dst.isEqualIfNotInherited(op));
        mulps(vSrc, op);
        subps(vSrc, vmm_dst);
        movups(vmm_dst, vSrc);
    }
}

void JitKernelBase::uni_vpaddd(const Ymm& vmm_dst,
                               const Ymm& vSrc,
                               const Operand& op) {
    if (isValidIsa(x64::avx2)) {
        vpaddd(vmm_dst, vSrc, op);
    } else if (isValidIsa(x64::avx)) {
        Xmm xmmDst(vmm_dst.getIdx());
        vmovups(vmm_dst, vSrc);
        if (op.isYMM()) {
            Ymm ymmOp(op.getIdx());
            Xmm xmmOp(op.getIdx());
            paddd(xmmDst, xmmOp);
            vperm2f128(vmm_dst, vmm_dst, vmm_dst, 0x1);
            vperm2f128(ymmOp, ymmOp, ymmOp, 0x1);
            paddd(xmmDst, xmmOp);
            vperm2f128(vmm_dst, vmm_dst, vmm_dst, 0x1);
            vperm2f128(ymmOp, ymmOp, ymmOp, 0x1);
        } else if (op.isMEM()) {
            const int vlen = x64::cpu_isa_traits<x64::sse41>::vlen;
            paddd(xmmDst, op.getAddress());
            vperm2f128(vmm_dst, vmm_dst, vmm_dst, 0x1);
            paddd(xmmDst, ptr[op.getAddress().getRegExp() + vlen]);
            vperm2f128(vmm_dst, vmm_dst, vmm_dst, 0x1);
        } else {
            IE_THROW() << "Not supported operand type.";
        }
    } else if (isValidIsa(x64::sse41)) {
        assert(vmm_dst.getIdx() != vSrc.getIdx());
        paddd(vmm_dst, op);
    } else {
        IE_THROW() << "Not defined behavior for instruction 'vpaddd' in current instructions set.";
    }
}

void JitKernelBase::uni_vaddpd(const Xmm& vmm_dst, const Operand &op1, const Operand &op2) {
    if (isValidIsa(x64::avx)) {
        vaddpd(vmm_dst, op1, op2);
    } else {
        if (vmm_dst.getIdx() != op1.getIdx()) {
            movupd(vmm_dst, op1);
        }
        addpd(vmm_dst, op2);
    }
}

void JitKernelBase::uni_vpsubd(const Ymm& vmm_dst,
                               const Ymm& vSrc,
                               const Operand& op) {
    if (isValidIsa(x64::avx2)) {
        vpsubd(vmm_dst, vSrc, op);
    } else if (isValidIsa(x64::avx)) {
        Xmm xmmDst(vmm_dst.getIdx());
        vmovups(vmm_dst, vSrc);
        if (op.isYMM()) {
            Ymm ymmOp(op.getIdx());
            Xmm xmmOp(op.getIdx());
            psubd(xmmDst, xmmOp);
            vperm2f128(vmm_dst, vmm_dst, vmm_dst, 0x1);
            vperm2f128(ymmOp, ymmOp, ymmOp, 0x1);
            psubd(xmmDst, xmmOp);
            vperm2f128(vmm_dst, vmm_dst, vmm_dst, 0x1);
            vperm2f128(ymmOp, ymmOp, ymmOp, 0x1);
        } else if (op.isMEM()) {
            const int vlen = x64::cpu_isa_traits<x64::sse41>::vlen;
            psubd(xmmDst, op.getAddress());
            vperm2f128(vmm_dst, vmm_dst, vmm_dst, 0x1);
            psubd(xmmDst, ptr[op.getAddress().getRegExp() + vlen]);
            vperm2f128(vmm_dst, vmm_dst, vmm_dst, 0x1);
        } else {
            IE_THROW() << "Not supported operand type.";
        }
    } else if (isValidIsa(x64::sse41)) {
        if (vmm_dst.getIdx() != vSrc.getIdx()) {
            movups(vmm_dst, vSrc);
        }
        psubd(vmm_dst, op);
    } else {
        IE_THROW() << "Not defined behavior for instruction 'vpsubd' in current instructions set.";
    }
}

void JitKernelBase::uni_vmulpd(const Xmm& vmm_dst,
                               const Operand& op1,
                               const Operand& op2) {
    if (isValidIsa(x64::avx)) {
        vmulpd(vmm_dst, op1, op2);
    } else {
        if (vmm_dst.getIdx() != op1.getIdx()) {
            movupd(vmm_dst, op1);
        }
        mulpd(vmm_dst, op2);
    }
}

void JitKernelBase::uni_vpmullq(const Xmm& vmm_dst,
                                const Xmm& vmm_src,
                                const Operand& op) {
    if (get_isa() == x64::avx512_core) {
        vpmullq(vmm_dst, vmm_src, op);
    } else if (get_isa() == x64::avx2) {
        auto ymm_aux_0 = RegistersPool::Reg<Ymm>(registersPool);
        auto ymm_aux_1 = RegistersPool::Reg<Ymm>(registersPool);
        // There is no multiply int64 instruction on AVX2 and SSE41, thus the WA is used.
        // vmm_src0 -> ab; vmm_src1 -> cd;
        uni_vpsrlq(ymm_aux_0, vmm_src, 32);
        uni_vpmuludq(ymm_aux_0, ymm_aux_0, op);      // a * d
        uni_vpsrlq(ymm_aux_1, op, 32);
        uni_vpmuludq(ymm_aux_1, ymm_aux_1, vmm_src); // b * c
        uni_vpaddq(ymm_aux_1, ymm_aux_1, ymm_aux_0); // a * d + b * c
        uni_vpsllq(ymm_aux_1, ymm_aux_1, 32);
        uni_vpmuludq(ymm_aux_0, vmm_src, op);        // b * d
        uni_vpaddq(vmm_dst, ymm_aux_0, ymm_aux_1);   // (a * d + b * c) << 32 + b * d
    } else {
        auto xmm_aux_0 = RegistersPool::Reg<Xmm>(registersPool);
        auto xmm_aux_1 = RegistersPool::Reg<Xmm>(registersPool);
        // There is no multiply int64 instruction on AVX2 and SSE41, thus the WA is used.
        // vmm_src0 -> ab; vmm_src1 -> cd;
        uni_vpsrlq(xmm_aux_0, vmm_src, 32);
        uni_vpmuludq(xmm_aux_0, xmm_aux_0, op);      // a * d
        uni_vpsrlq(xmm_aux_1, op, 32);
        uni_vpmuludq(xmm_aux_1, xmm_aux_1, vmm_src); // b * c
        uni_vpaddq(xmm_aux_1, xmm_aux_1, xmm_aux_0); // a * d + b * c
        uni_vpsllq(xmm_aux_1, xmm_aux_1, 32);
        uni_vpmuludq(xmm_aux_0, vmm_src, op);        // b * d
        uni_vpaddq(vmm_dst, xmm_aux_0, xmm_aux_1);   // (a * d + b * c) << 32 + b * d
    }
}

void JitKernelBase::uni_vdivps(const Xmm& vmm_dst,
                               const Operand& op1,
                               const Operand& op2) {
    if (isValidIsa(x64::avx)) {
        vdivps(vmm_dst, op1, op2);
    } else {
        if (!vmm_dst.isEqualIfNotInherited(op1)) {
            movups(vmm_dst, op1);
        }
        divps(vmm_dst, op2);
    }
}

void JitKernelBase::uni_vdivpd(const Xmm& vmm_dst,
                               const Operand& op1,
                               const Operand& op2) {
    if (isValidIsa(x64::avx)) {
        vdivpd(vmm_dst, op1, op2);
    } else {
        if (vmm_dst.getIdx() != op1.getIdx()) {
            movupd(vmm_dst, op1);
        }
        divpd(vmm_dst, op2);
    }
}

void JitKernelBase::uni_vsqrtpd(const Xmm& vmm_dst, const Operand &op) {
    if (isValidIsa(x64::avx)) {
        vsqrtpd(vmm_dst, op);
    } else {
        sqrtpd(vmm_dst, op);
    }
}

void JitKernelBase::uni_vandps(const Xmm& vmm_dst,
                               const Xmm& vSrs,
                               const Operand &op) {
    if (isValidIsa(x64::avx)) {
        vandps(vmm_dst, vSrs, op);
    } else {
        if (vmm_dst.getIdx() != vSrs.getIdx()) {
            movups(vmm_dst, vSrs);
        }
        andps(vmm_dst, op);
    }
}

void JitKernelBase::uni_vandpd(const Xmm& vmm_dst,
                               const Xmm& vSrs,
                               const Operand &op) {
    if (isValidIsa(x64::avx)) {
        vandpd(vmm_dst, vSrs, op);
    } else {
        if (vmm_dst.getIdx() != vSrs.getIdx()) {
            movupd(vmm_dst, vSrs);
        }
        andpd(vmm_dst, op);
    }
}

void JitKernelBase::uni_vandnps(const Xmm& vmm_dst,
                                const Xmm& vSrs,
                                const Operand &op) {
    if (isValidIsa(x64::avx)) {
        vandnps(vmm_dst, vSrs, op);
    } else {
        if (!vmm_dst.isEqualIfNotInherited(vSrs)) {
            movups(vmm_dst, vSrs);
        }
        andnps(vmm_dst, op);
    }
}

void JitKernelBase::uni_vorpd(const Xmm& vmm_dst,
                              const Xmm& vSrs,
                              const Operand &op) {
    if (isValidIsa(x64::avx)) {
        vorpd(vmm_dst, vSrs, op);
    } else {
        if (vmm_dst.getIdx() != vSrs.getIdx()) {
            movupd(vmm_dst, vSrs);
        }
        orpd(vmm_dst, op);
    }
}

void JitKernelBase::uni_vcmppd(const Xmm& vmm_dst,
                               const Xmm &vSrs,
                               const Operand &op,
                               const uint8_t imm) {
    if (isValidIsa(x64::avx)) {
        vcmppd(vmm_dst, vSrs, op, imm);
    } else {
        if (vmm_dst.getIdx() != vSrs.getIdx()) {
            movupd(vmm_dst, vSrs);
        }
        cmppd(vmm_dst, op, imm);
    }
}

void JitKernelBase::uni_vmaxpd(const Xmm& vmm_dst, const Operand &op1, const Operand &op2) {
    if (isValidIsa(x64::avx)) {
        vmaxpd(vmm_dst, op1, op2);
    } else {
        if (vmm_dst.getIdx() != op1.getIdx()) {
            movupd(vmm_dst, op1);
        }
        maxpd(vmm_dst, op2);
    }
}

void JitKernelBase::uni_vminpd(const Xmm& vmm_dst, const Operand &op1, const Operand &op2) {
    if (isValidIsa(x64::avx)) {
        vminpd(vmm_dst, op1, op2);
    } else {
        if (vmm_dst.getIdx() != op1.getIdx()) {
            movupd(vmm_dst, op1);
        }
        minpd(vmm_dst, op2);
    }
}

void JitKernelBase::uni_vpmaxsq(const Xmm& vmm_dst, const Xmm& vmm_src, const Operand& op) {
    if (get_isa() == x64::avx512_core) {
        vpmaxsq(vmm_dst, vmm_src, op);
    } else if (get_isa() == x64::avx2) {
        if (vmm_dst.isYMM()) {
            auto ymm_aux = RegistersPool::Reg<Ymm>(registersPool);

            vpcmpgtq(ymm_aux, vmm_src, op);
            vandpd(vmm_dst, vmm_src, ymm_aux);
            vandnpd(ymm_aux, ymm_aux, op);
            vorpd(vmm_dst, vmm_dst, ymm_aux);
        } else {
            auto xmm_aux = RegistersPool::Reg<Xmm>(registersPool);

            vpcmpgtq(xmm_aux, vmm_src, op);
            vandpd(vmm_dst, vmm_src, xmm_aux);
            vandnpd(xmm_aux, xmm_aux, op);
            vorpd(vmm_dst, vmm_dst, xmm_aux);
        }
    } else {
        auto xmm_aux = RegistersPool::Reg<Xmm>(registersPool);

        movups(xmm_aux, vmm_src);
        pcmpgtq(xmm_aux, op);
        andpd(xmm_aux, vmm_src);
        if (vmm_dst.getIdx() != vmm_src.getIdx()) {
            movups(vmm_dst, vmm_src);
        }
        pcmpgtq(vmm_dst, op);
        andnpd(vmm_dst, op);
        orpd(vmm_dst, xmm_aux);
    }
}

void JitKernelBase::uni_vpminsq(const Xmm& vmm_dst, const Xmm& vmm_src, const Operand& op) {
    if (get_isa() == x64::avx512_core) {
        vpminsq(vmm_dst, vmm_src, op);
    } else if (get_isa() == x64::avx2) {
        if (vmm_dst.isYMM()) {
            auto ymm_aux = RegistersPool::Reg<Ymm>(registersPool);

            vpcmpgtq(ymm_aux, vmm_src, op);
            vandpd(vmm_dst, vmm_src, ymm_aux);
            vandnpd(ymm_aux, ymm_aux, op);
            vorpd(vmm_dst, vmm_dst, ymm_aux);
        } else {
            auto xmm_aux = RegistersPool::Reg<Xmm>(registersPool);

            vpcmpgtq(xmm_aux, vmm_src, op);
            vandpd(vmm_dst, vmm_src, xmm_aux);
            vandnpd(xmm_aux, xmm_aux, op);
            vorpd(vmm_dst, vmm_dst, xmm_aux);
        }
    } else {
        auto xmm_aux = RegistersPool::Reg<Xmm>(registersPool);

        movups(xmm_aux, vmm_src);
        pcmpgtq(xmm_aux, op);
        andpd(xmm_aux, vmm_src);
        if (vmm_dst.getIdx() != vmm_src.getIdx()) {
            movups(vmm_dst, vmm_src);
        }
        pcmpgtq(vmm_dst, op);
        andnpd(vmm_dst, op);
        orpd(vmm_dst, xmm_aux);
    }
}

void JitKernelBase::uni_vcvtpd2dq(const Xbyak::Xmm &vmm_dst, const Xbyak::Operand &op) {
    if (isValidIsa(x64::avx)) {
        vcvtpd2dq(vmm_dst, op);
    } else {
        cvtpd2dq(vmm_dst, op);
    }
}

void JitKernelBase::uni_vcvtpd2ps(const Xbyak::Xmm &vmm_dst, const Xbyak::Operand &op) {
    if (isValidIsa(x64::avx)) {
        vcvtpd2ps(vmm_dst, op);
    } else {
        cvtpd2ps(vmm_dst, op);
    }
}

void JitKernelBase::gatherdd(const Xmm&    vmm_dst,
                             const Reg64&  rSrcPtr,
                             const Xmm&    vSrcShift,
                             const Opmask& kReadMask,
                             const bool useMask,
                             const bool zeroFill) {
    if (kReadMask.getIdx() == 0) {
        IE_THROW() << "The vpgatherdd instruction cannot use the register k0 as mask.";
    }
    if (!useMask)
        kxnord(kReadMask, kReadMask, kReadMask);
    if (zeroFill)
        uni_vpxor(vmm_dst, vmm_dst, vmm_dst);

    vpgatherdd(vmm_dst | kReadMask, ptr[rSrcPtr + vSrcShift]);
}

void JitKernelBase::gatherdd(const Xmm&   vmm_dst,
                             const Reg64& rSrcPtr,
                             const Xmm&   vSrcShift,
                             const Xmm&   vReadMask,
                             const bool useMask,
                             const bool zeroFill) {
    if (vmm_dst.getIdx() == vSrcShift.getIdx() || vmm_dst.getIdx() == vReadMask.getIdx() || vSrcShift.getIdx() == vReadMask.getIdx()) {
        IE_THROW() << "Any pair of the index, mask, or destination registers cannot be the same.";
    }
    if (zeroFill)
        pxor(vmm_dst, vmm_dst); // Don't use vpxor. It zeros the rest of the YMM register.

    if (isValidIsa(x64::avx2)) {
        if (!useMask)
            uni_vpcmpeqd(vReadMask, vReadMask, vReadMask);

        vpgatherdd(vmm_dst, ptr[rSrcPtr + vSrcShift], vReadMask);
    } else {
        auto rAux = getReg64();
        Reg32 r32Aux = Reg32(rAux.getIdx());
        const uint8_t elPerVec = x64::cpu_isa_traits<x64::sse41>::vlen / sizeof(int);

        for (uint8_t i = 0; i < elPerVec; i++) {
            Label lLoopNext;
            if (useMask) {
                uni_vpextrd(r32Aux, vReadMask, i);
                cmp(r32Aux, 0); // TODO: check significant bit
                je(lLoopNext, T_NEAR);
            }
            uni_vpextrd(r32Aux, vSrcShift, i);
            pinsrd(vmm_dst, ptr[rSrcPtr + rAux], i);

            if (useMask)
                L(lLoopNext);
        }
    }
}

void JitKernelBase::gatherdd(const Ymm&   vmm_dst,
                             const Reg64& rSrcPtr,
                             const Ymm&   vSrcShift,
                             const Ymm&   vReadMask,
                             const bool useMask,
                             const bool zeroFill) {
    if (vmm_dst.getIdx() == vSrcShift.getIdx() || vmm_dst.getIdx() == vReadMask.getIdx() || vSrcShift.getIdx() == vReadMask.getIdx()) {
        IE_THROW() << "Any pair of the index, mask, or destination registers cannot be the same.";
    }
    if (isValidIsa(x64::avx2)) {
        if (!useMask)
            uni_vpcmpeqd(vReadMask, vReadMask, vReadMask);
        if (zeroFill)
            uni_vpxor(vmm_dst, vmm_dst, vmm_dst);

        vpgatherdd(vmm_dst, ptr[rSrcPtr + vSrcShift], vReadMask);
    } else {
        Xmm xmmDst      = Xmm(vmm_dst.getIdx()),
                   xmmSrcShft  = Xmm(vSrcShift.getIdx()),
                   xmmReadMask = Xmm(vReadMask.getIdx());
        for (uint8_t i = 0; i < 2; i++) {
            gatherdd(xmmDst, rSrcPtr, xmmSrcShft, xmmReadMask, useMask, zeroFill);

            vperm2f128(vmm_dst, vmm_dst, vmm_dst, 0x1);
            vperm2f128(vSrcShift, vSrcShift, vSrcShift, 0x1);
            if (useMask)
                vperm2f128(vReadMask, vReadMask, vReadMask, 0x1);
        }
    }
}

// void JitKernelBase::gatherqq(const Xmm&   vmm_dst,
//                              const Reg64& rSrcPtr,
//                              const Xmm&   vSrcShift,
//                              const Xmm&   vReadMask,
//                              const bool useMask,
//                              const bool zeroFill) {
//     if (vmm_dst.getIdx() == vSrcShift.getIdx() || vmm_dst.getIdx() == vReadMask.getIdx() || vSrcShift.getIdx() == vReadMask.getIdx()) {
//         IE_THROW() << "Any pair of the index, mask, or destination registers cannot be the same.";
//     }
//     if (zeroFill)
//         pxor(vmm_dst, vmm_dst); // Don't use vpxor. It zeros the rest of the YMM register.

//     if (isValidIsa(x64::avx2)) {
//         if (!useMask)
//             uni_vpcmpeqq(vReadMask, vReadMask, vReadMask);

//         vpgatherqq(vmm_dst, ptr[rSrcPtr + vSrcShift], vReadMask);
//     } else {
//         auto rAux = getReg64();
//         // Reg32 r32Aux = Reg32(rAux.getIdx());
//         const uint8_t elPerVec = x64::cpu_isa_traits<x64::sse41>::vlen / sizeof(int);

//         for (uint8_t i = 0; i < elPerVec; i++) {
//             Label lLoopNext;
//             if (useMask) {
//                 uni_vpextrd(rAux, vReadMask, i);
//                 cmp(rAux, 0); // TODO: check significant bit
//                 je(lLoopNext, T_NEAR);
//             }
//             uni_vpextrd(rAux, vSrcShift, i);
//             pinsrd(vmm_dst, ptr[rSrcPtr + rAux], i);

//             if (useMask)
//                 L(lLoopNext);
//         }
//     }
// }

void JitKernelBase::uni_vpbroadcastd(const Xmm &x, const Operand &op) {
    if (isValidIsa(x64::avx2)) {
        vpbroadcastd(x, op);
    } else if (isValidIsa(x64::avx)) {
        if (op.isMEM()) {
            vbroadcastss(x, op.getAddress());
        } else {
            vmovss(x, x, op);
            vpshufd(x, x, 0x0);
        }
    } else {
        movss(x, op);
        pshufd(x, x, 0x0);
    }
}

void JitKernelBase::uni_vpbroadcastd(const Ymm &x, const Operand &op) {
    if (isValidIsa(x64::avx2)) {
        vpbroadcastd(x, op);
    } else {
        if (op.isMEM()) {
            vbroadcastss(x, op.getAddress());
        } else {
            const Xmm t(x.getIdx());
            if (!t.isEqualIfNotInherited(op)) {
                vmovss(t, t, op);
            }
            vinsertf128(x, x, t, 1);
            vshufps(x, x, x, 0);
        }
    }
}

void JitKernelBase::fillRestWorkMask(const Opmask& dstMask,
                                     const Reg64& rWorkRest) {
    auto rOnes = getReg64();

    mov(rOnes, 0xFFFFFFFFFFFFFFFF);
    shlx(rOnes, rOnes, rWorkRest);
    not_(rOnes);
    kmovq(dstMask, rOnes);
}

void JitKernelBase::fillRestWorkMask(const Xmm& xmmDstMask,
                                     const Reg64& rWorkRest,
                                     const uint64_t typeSize) {
    if (!one_of(typeSize, 1u, 2u, 4u, 8u)) {
        IE_THROW() << "Could not fill data with type size " << typeSize;
    }
    Label lEnd;
    auto r32Ones = getReg32();
    Reg64 r64Ones(r32Ones.getIdx());
    auto elPerVec = x64::cpu_isa_traits<x64::sse41>::vlen / typeSize;

    mov(r64Ones, 0xFFFFFFFFFFFFFFFF);
    for (uint8_t i = 0; i < elPerVec; i++) {
        cmp(rWorkRest, i);
        jle(lEnd, T_NEAR);

        if (typeSize == 1) {
            pinsrb(xmmDstMask, r32Ones, i);
        } else if (typeSize == 2) {
            pinsrw(xmmDstMask, r32Ones, i);
        } else if (typeSize == 4) {
            pinsrd(xmmDstMask, r32Ones, i);
        } else if (typeSize == 8) {
            pinsrq(xmmDstMask, r64Ones, i);
        }
    }
    L(lEnd);
}

void JitKernelBase::fillRestWorkMask(const Ymm& ymmDstMask,
                                     const Reg64& rWorkRest,
                                     const uint64_t typeSize) {
    if (!one_of(typeSize, 1u, 2u, 4u, 8u)) {
        IE_THROW() << "Could not fill data with type size " << typeSize;
    }
    Label lEnd;
    auto elPerVec = x64::cpu_isa_traits<x64::sse41>::vlen / typeSize;
    auto r32Ones = getReg32();
    Reg64 r64Ones(r32Ones.getIdx());
    Xmm xmmDstMask(ymmDstMask.getIdx());

    mov(r64Ones, 0xFFFFFFFFFFFFFFFF);
    uni_vpxor(ymmDstMask, ymmDstMask, ymmDstMask);
    for (uint8_t i = 0; i < 2; i++) {
        Label lPerm;
        for (uint8_t j = 0; j < elPerVec; j++) {
            cmp(rWorkRest, i * elPerVec + j);
            jle(i == 0 ? lEnd : lPerm, T_NEAR);

            if (typeSize == 1) {
                pinsrb(xmmDstMask, r32Ones, j);
            } else if (typeSize == 2) {
                pinsrw(xmmDstMask, r32Ones, j);
            } else if (typeSize == 4) {
                pinsrd(xmmDstMask, r32Ones, j);
            } else if (typeSize == 8) {
                pinsrq(xmmDstMask, r64Ones, j);
            }
        }
        cmp(rWorkRest, elPerVec);
        je(lEnd, T_NEAR);
        L(lPerm);
        vperm2f128(ymmDstMask, ymmDstMask, ymmDstMask, 0x1);
    }
    L(lEnd);
}

void JitKernelBase::load(const Xmm&     vmm_dst,
                         const Address& srcAddr,
                         const Reg64&   rLoadNum,
                         const size_t   typeSize,
                         const bool     zeroFilling) {
    if (!one_of(typeSize, 1u, 2u, 4u, 8u)) {
        IE_THROW() << "Could not load data with type size " << typeSize;
    }
    const uint8_t elPerVec = x64::cpu_isa_traits<x64::sse41>::vlen / typeSize;
    Label lEnd;
    if (zeroFilling)
        pxor(vmm_dst, vmm_dst);

    for (uint8_t i = 0; i < elPerVec; i++) {
        cmp(rLoadNum, i);
        jle(lEnd, T_NEAR);

        const size_t offset = i * typeSize;
        if (typeSize == 1)
            pinsrb(vmm_dst, ptr[srcAddr.getRegExp() + offset], i);
        else if (typeSize == 2)
            pinsrw(vmm_dst, ptr[srcAddr.getRegExp() + offset], i);
        else if (typeSize == 4)
            pinsrd(vmm_dst, ptr[srcAddr.getRegExp() + offset], i);
        else if (typeSize == 8)
            pinsrq(vmm_dst, ptr[srcAddr.getRegExp() + offset], i);
    }
    L(lEnd);
}

void JitKernelBase::load(const Ymm&     vmm_dst,
                         const Address& srcAddr,
                         const Reg64&   rLoadNum,
                         const size_t   typeSize,
                         const bool     zeroFilling) {
    if (!one_of(typeSize, 1u, 2u, 4u, 8u)) {
        IE_THROW() << "Could not load data with type size " << typeSize;
    }
    const size_t elPerXmm = x64::cpu_isa_traits<x64::sse41>::vlen / typeSize;
    Label lEnd;
    if (zeroFilling)
        uni_vpxor(vmm_dst, vmm_dst, vmm_dst);
    Xmm xmmDst(vmm_dst.getIdx());

    for (size_t i = 0lu; i < 2lu; i++) {
        Label lPerm;
        const size_t idx = i * elPerXmm;
        const size_t offset0 = idx * typeSize;

        for (size_t j = 0lu; j < elPerXmm; j++) {
            cmp(rLoadNum, j + idx);
            jle(i == 0 ? lEnd : lPerm, T_NEAR);

            const size_t offset = offset0 + j * typeSize;
            if (typeSize == 1)
                pinsrb(xmmDst, ptr[srcAddr.getRegExp() + offset], j);
            else if (typeSize == 2)
                pinsrw(xmmDst, ptr[srcAddr.getRegExp() + offset], j);
            else if (typeSize == 4)
                pinsrd(xmmDst, ptr[srcAddr.getRegExp() + offset], j);
            else if (typeSize == 8)
                pinsrq(xmmDst, ptr[srcAddr.getRegExp() + offset], j);
        }

        L(lPerm);
        vperm2f128(vmm_dst, vmm_dst, vmm_dst, 0x1);
    }
    L(lEnd);
}

void JitKernelBase::store(const Address& dstAddr,
                          const Xmm&     vSrc,
                          const Reg64&   rToStoreNum,
                          const size_t          typeSize) {
    if (!one_of(typeSize, 1u, 2u, 4u, 8u)) {
        IE_THROW() << "Could not store data with type size " << typeSize;
    }
    Label lEnd;
    const size_t elPerVec = x64::cpu_isa_traits<x64::sse41>::vlen / typeSize;

    for (size_t i = 0; i < elPerVec; i++) {
        cmp(rToStoreNum, i);
        jle(lEnd, T_NEAR);

        const size_t offset = i * typeSize;
        if (typeSize == 1) {
            uni_vpextrb(ptr[dstAddr.getRegExp() + offset], vSrc, i);
        } else if (typeSize == 2) {
            uni_vpextrw(ptr[dstAddr.getRegExp() + offset], vSrc, i);
        } else if (typeSize == 4) {
            uni_vpextrd(ptr[dstAddr.getRegExp() + offset], vSrc, i);
        } else if (typeSize == 8) {
            uni_vpextrq(ptr[dstAddr.getRegExp() + offset], vSrc, i);
        }
    }
    L(lEnd);
}

void JitKernelBase::store(const Address& dstAddr,
                          const Ymm&     vSrc,
                          const Reg64&   rToStoreNum,
                          const size_t          typeSize) {
    if (!one_of(typeSize, 1u, 2u, 4u, 8u)) {
        IE_THROW() << "Could not store data with type size " << typeSize;
    }
    Label lEnd;
    Xmm xmmSrc(vSrc.getIdx());
    const size_t elPerXmm = x64::cpu_isa_traits<x64::sse41>::vlen / typeSize;

    for (int i = 0; i < 2; i++) {
        Label lPerm;
        const size_t idx = i * elPerXmm;
        const size_t offset0 = idx * typeSize;

        for (size_t j = 0; j < elPerXmm; j++) {
            cmp(rToStoreNum, j + idx);
            jle(i == 0 ? lEnd : lPerm, T_NEAR);

            const size_t offset = offset0 + j * typeSize;
            if (typeSize == 8) {
                uni_vpextrq(ptr[dstAddr.getRegExp() + offset], xmmSrc, j);
            } else if (typeSize == 4) {
                uni_vpextrd(ptr[dstAddr.getRegExp() + offset], xmmSrc, j);
            } else if (typeSize == 2) {
                uni_vpextrw(ptr[dstAddr.getRegExp() + offset], xmmSrc, j);
            } else if (typeSize == 1) {
                uni_vpextrb(ptr[dstAddr.getRegExp() + offset], xmmSrc, j);
            }
        }

        L(lPerm);
        vperm2f128(vSrc, vSrc, vSrc, 0x1);
    }
    L(lEnd);
}

void JitKernelBase::memMovDD(const Reg64& rDst,
                             const Reg64& rSrc,
                             const Xmm&   vReadMask,
                             const Xmm&   vSrcShift,
                             const Reg64& rToStoreNum,
                             const bool          useMask,
                             const bool          zeroFill) {
    Label lEnd;
    auto rAux = getReg64();
    Reg32 r32Aux = Reg32(rAux.getIdx());
    const uint8_t typeSize = sizeof(int);
    const uint8_t elPerVec = x64::cpu_isa_traits<x64::sse41>::vlen / typeSize;

    for (uint8_t i = 0; i < elPerVec; i++) {
        cmp(rToStoreNum, i);
        jle(lEnd, T_NEAR);

        Label lLoopNext;
        if (useMask) {
            uni_vpextrd(r32Aux, vReadMask, i);
            cmp(r32Aux, 0);
            if (zeroFill) {
                Label lNotZero;
                jne(lNotZero, T_NEAR);
                mov(ptr[rDst.getReg() + i * typeSize], r32Aux);
                jmp(lLoopNext, T_NEAR);
                L(lNotZero);
            } else {
                je(lLoopNext, T_NEAR);
            }
        }
        uni_vpextrd(r32Aux, vSrcShift, i);
        mov(r32Aux, ptr[rSrc.getReg() + rAux]);
        mov(ptr[rDst.getReg() + i * typeSize], r32Aux);

        L(lLoopNext);
    }
    L(lEnd);
}

void JitKernelBase::memMovDD(const Reg64& rDst,
                             const Reg64& rSrc,
                             const Ymm&   vReadMask,
                             const Ymm&   vSrcShift,
                             const Reg64& rToStoreNum,
                             const bool          useMask,
                             const bool          zeroFill) {
    Label lEnd;
    if (isValidIsa(x64::avx2)) {
        auto vAux = RegistersPool::Reg<Ymm>(registersPool);
        gatherdd(vAux, rSrc, vSrcShift, vReadMask, useMask, zeroFill);
        store(ptr[rDst], vAux, rToStoreNum, sizeof(int));
    } else if (isValidIsa(x64::avx)) {
        const uint8_t typeSize = sizeof(int);
        const uint8_t elPerXmm = x64::cpu_isa_traits<x64::sse41>::vlen / typeSize;
        Xmm xmmReadMask  = Xmm(vReadMask.getIdx()),
                   xmmSrcShft   = Xmm(vSrcShift.getIdx());
        for (uint8_t i = 0; i < 2; i++) {
            memMovDD(rDst, rSrc, xmmReadMask, xmmSrcShft, rToStoreNum, useMask, zeroFill);

            if (i == 0) {
                cmp(rToStoreNum, elPerXmm);
                jle(lEnd, T_NEAR);
                sub(rToStoreNum, elPerXmm);
                add(rDst, typeSize * elPerXmm);
            } else {
                add(rToStoreNum, elPerXmm);
                sub(rDst, typeSize * elPerXmm);
            }

            vperm2f128(vSrcShift, vSrcShift, vSrcShift, 0x1);
            if (useMask)
                vperm2f128(vReadMask, vReadMask, vReadMask, 0x1);
        }
    }
    L(lEnd);
}

void JitKernelBase::load_vector(const Xmm& vmm_dst,
                               const Address &srcAdr,
                               const ov::element::Type& dstPrc,
                               const ov::element::Type& srcPrc) {
    Xmm xmmDst = Xmm(vmm_dst.getIdx());
    Ymm ymmDst = Ymm(vmm_dst.getIdx());

    switch (srcPrc) {
        case ov::element::f64:
            if (x64::mayiuse(x64::avx512_core) && one_of(dstPrc, ov::element::i64, ov::element::i32, ov::element::f32)) {
                if (dstPrc == ov::element::i64) {
                    vcvtpd2qq(vmm_dst, srcAdr);
                } else if (dstPrc == ov::element::i32) {
                    uni_vcvtpd2dq(vmm_dst.isZMM() ? ymmDst : vmm_dst, srcAdr);
                } else if (dstPrc == ov::element::f32) {
                    uni_vcvtpd2ps(vmm_dst.isZMM() ? ymmDst : vmm_dst, srcAdr);
                }
            } else if (!x64::mayiuse(x64::avx512_core) && one_of(dstPrc, ov::element::f32, ov::element::i32)) {
                if (dstPrc == ov::element::f32) {
                    uni_vcvtpd2ps(xmmDst, srcAdr);
                } else if (dstPrc == ov::element::i32) {
                    uni_vcvtpd2dq(xmmDst, srcAdr);
                }
            } else {
                uni_vmovups(vmm_dst, srcAdr);
            }
            break;
        case ov::element::i64:
            if (x64::mayiuse(x64::avx512_core) && one_of(dstPrc, ov::element::f64, ov::element::f32)) {
                if (dstPrc == ov::element::f64) {
                    vcvtqq2pd(vmm_dst, srcAdr);
                } else if (dstPrc == ov::element::f32) {
                    vcvtqq2ps(vmm_dst.isZMM() ? ymmDst : vmm_dst, srcAdr);
                }
            } else {
                uni_vmovups(vmm_dst, srcAdr);
            }
            break;
        case ov::element::f32:
            if (dstPrc == ov::element::i32) {
                uni_vcvtps2dq(vmm_dst, srcAdr);
            } else {
                uni_vmovups(vmm_dst, srcAdr);
            }
            break;
        case ov::element::i32:
            if (dstPrc == ov::element::f64) {
                uni_vcvtdq2pd(vmm_dst, srcAdr);
            } else if (dstPrc == ov::element::f32) {
                uni_vcvtdq2ps(vmm_dst, srcAdr);
            } else {
                uni_vmovups(vmm_dst, srcAdr);
            }
            break;
        case ov::element::bf16:
            uni_vpmovzxwd(vmm_dst, srcAdr);
            uni_vpslld(vmm_dst, vmm_dst, 16);
            break;
        case ov::element::u16:
            if (one_of(dstPrc, ov::element::f32, ov::element::i32)) {
                uni_vpmovzxwd(vmm_dst, srcAdr);
            } else {
                uni_vmovups(vmm_dst, srcAdr);
            }
            break;
        case ov::element::i16:
            if (one_of(dstPrc, ov::element::f32, ov::element::i32)) {
                uni_vpmovsxwd(vmm_dst, srcAdr);
            } else {
                uni_vmovups(vmm_dst, srcAdr);
            }
            break;
        case ov::element::i8:
            if (one_of(dstPrc, ov::element::f32, ov::element::i32)) {
                uni_vpmovsxbd(vmm_dst, srcAdr);
            } else {
                uni_vmovups(vmm_dst, srcAdr);
            }
            break;
        case ov::element::u8:
            if (one_of(dstPrc, ov::element::f32, ov::element::i32)) {
                uni_vpmovzxbd(vmm_dst, srcAdr);
            } else {
                uni_vmovups(vmm_dst, srcAdr);
            }
            break;
        default:
            IE_THROW() << "Unsupported source precision: " << srcPrc;
    }

    switch (dstPrc) {
        case ov::element::f32:
            if (!x64::mayiuse(x64::avx512_core) && (srcPrc == ov::element::i64)) {
                // Do conversion later.
            }
            if (one_of(srcPrc, ov::element::u8, ov::element::i8, ov::element::i16, ov::element::u16)) {
                uni_vcvtdq2ps(vmm_dst, vmm_dst);
            }
            break;
        case ov::element::i32:
            if (x64::mayiuse(x64::avx512_core)) {
                if (srcPrc == ov::element::i64) {
                    vpmovsqd(vmm_dst, vmm_dst);
                }
            } else {
                if (srcPrc == ov::element::i64) {
                    // Do conversion later.
                }
            }
            if (one_of(srcPrc, ov::element::bf16)) {
                uni_vcvtps2dq(vmm_dst, vmm_dst);
            }
            break;
        case ov::element::i64:
        case ov::element::f64:
            break;
        default:
            IE_THROW() << "Unsupported destination precision: " << dstPrc;
    }
}

void JitKernelBase::load_scalar(const Xmm& vmm_dst,
                               const Address &srcAdr,
                               const ov::element::Type& dstPrc,
                               const ov::element::Type& srcPrc) {
    Address src_adr_bcst(srcAdr.getBit(), true, srcAdr.getRegExp());

    switch (srcPrc) {
        case ov::element::f64:
            if (x64::mayiuse(x64::avx512_core) && one_of(dstPrc, ov::element::i64, ov::element::i32, ov::element::f32)) {
                if (dstPrc == ov::element::i64) {
                    vcvtpd2qq(vmm_dst, src_adr_bcst);
                } else if (dstPrc == ov::element::i32) {
                    vcvtpd2dq(vmm_dst, src_adr_bcst);
                } else if (dstPrc == ov::element::f32) {
                    vcvtpd2ps(vmm_dst, src_adr_bcst);
                }
            } else {
                uni_vmovsd(vmm_dst, srcAdr);
            }
            break;
        case ov::element::i64:
            if (x64::mayiuse(x64::avx512_core) && one_of(dstPrc, ov::element::f64, ov::element::f32)) {
                if (dstPrc == ov::element::f64) {
                    vcvtqq2pd(vmm_dst, src_adr_bcst);
                } else if (dstPrc == ov::element::f32) {
                    vcvtqq2ps(vmm_dst, src_adr_bcst);
                }
            } else {
                uni_vmovsd(vmm_dst, srcAdr);
            }
            break;
        case ov::element::f32:
            if (x64::mayiuse(x64::avx512_core) && one_of(dstPrc, ov::element::f64, ov::element::i32)) {
                if (dstPrc == ov::element::f64) {
                    vcvtps2pd(vmm_dst, src_adr_bcst);
                } else if (dstPrc == ov::element::i32) {
                    vcvtps2dq(vmm_dst, src_adr_bcst);
                }
            } else {
                uni_vmovss(vmm_dst, srcAdr);
            }
            break;
        case ov::element::i32:
            if (x64::mayiuse(x64::avx512_core) && one_of(dstPrc, ov::element::f32, ov::element::f64)) {
                if (dstPrc == ov::element::f32) {
                    vcvtdq2ps(vmm_dst, src_adr_bcst);
                } else if (dstPrc == ov::element::f64) {
                    vcvtdq2pd(vmm_dst, src_adr_bcst);
                }
            } else {
                uni_vmovss(vmm_dst, srcAdr);
            }
            break;
        case ov::element::bf16:
            uni_vpinsrw(vmm_dst, vmm_dst, srcAdr, 0);
            uni_vpslld(vmm_dst, vmm_dst, 16);
            break;
        case ov::element::i16:
            uni_vpinsrw(vmm_dst, vmm_dst, srcAdr, 0);
            uni_vpmovsxwd(vmm_dst, srcAdr);
            break;
        case ov::element::u16:
            uni_vpinsrw(vmm_dst, vmm_dst, srcAdr, 0);
            uni_vpmovzxwd(vmm_dst, srcAdr);
            break;
        case ov::element::i8:
            pinsrb(vmm_dst, srcAdr, 0);
            uni_vpmovsxbd(vmm_dst, vmm_dst);
            break;
        case ov::element::u8:
            pinsrb(vmm_dst, srcAdr, 0);
            uni_vpmovzxbd(vmm_dst, vmm_dst);
            break;
        default:
            IE_THROW() << "Unsupported source precision: " << srcPrc;
    }

    switch (dstPrc) {
        case ov::element::f32:
            if (x64::mayiuse(x64::avx512_core)) {
                if (one_of(srcPrc, ov::element::u8, ov::element::i8, ov::element::u16, ov::element::i16)) {
                    uni_vcvtdq2ps(vmm_dst, vmm_dst);
                }
            } else {
                if (srcPrc == ov::element::f64) {
                    uni_vcvtpd2ps(vmm_dst, vmm_dst);
                } else if (srcPrc == ov::element::i64) {
                    // Do conversion later.
                } else if (one_of(srcPrc, ov::element::u8, ov::element::i8, ov::element::u16, ov::element::i16, ov::element::i32)) {
                    uni_vcvtdq2ps(vmm_dst, vmm_dst);
                }
            }
            break;
        case ov::element::i32:
            if (!x64::mayiuse(x64::avx512_core)) {
                if (srcPrc == ov::element::i64) {
                    // Do conversion later.
                } else if (one_of(srcPrc, ov::element::f32, ov::element::bf16)) {
                    uni_vcvtps2dq(vmm_dst, vmm_dst);
                }
            } else if (srcPrc == ov::element::i64) {
                vpmovsqd(vmm_dst, vmm_dst);
            }
            break;
        case ov::element::i64:
        case ov::element::f64:
            break;
        default:
            IE_THROW() << "Unsupported destination precision: " << dstPrc;
    }
}

void JitKernelBase::load_with_bcst(const Xmm& vmm_dst,
                                   const Address &srcAdr,
                                   const ov::element::Type& dstPrc,
                                   const ov::element::Type& srcPrc) {
    Address src_adr_bcst(srcAdr.getBit(), true, srcAdr.getRegExp());

    switch (srcPrc) {
        case ov::element::f64:
            if (x64::mayiuse(x64::avx512_core) && one_of(dstPrc, ov::element::i64, ov::element::i32, ov::element::f32)) {
                if (dstPrc == ov::element::i64) {
                    vcvtpd2qq(vmm_dst, src_adr_bcst);
                } else if (dstPrc == ov::element::i32) {
                    vcvtpd2dq(vmm_dst, src_adr_bcst);
                } else if (dstPrc == ov::element::f32) {
                    vcvtpd2ps(vmm_dst, src_adr_bcst);
                }
            } else {
                uni_vbroadcastsd(vmm_dst, srcAdr); // does not work with XMM, use vpbroadcastq instead
            }
            break;
        case ov::element::i64:
            if (x64::mayiuse(x64::avx512_core) && one_of(dstPrc, ov::element::f64, ov::element::f32)) {
                if (dstPrc == ov::element::f64) {
                    vcvtqq2pd(vmm_dst, src_adr_bcst);
                } else if (dstPrc == ov::element::f32) {
                    vcvtqq2ps(vmm_dst, src_adr_bcst);
                }
            } else {
                uni_vbroadcastsd(vmm_dst, srcAdr);
            }
            break;
        case ov::element::f32:
            if (x64::mayiuse(x64::avx512_core) && one_of(dstPrc, ov::element::f64, ov::element::i32)) {
                if (dstPrc == ov::element::f64) {
                    vcvtps2pd(vmm_dst, src_adr_bcst);
                } else if (dstPrc == ov::element::i32) {
                    vcvtps2dq(vmm_dst, src_adr_bcst);
                }
            } else {
                uni_vbroadcastss(vmm_dst, srcAdr);
            }
            break;
        case ov::element::i32:
            if (x64::mayiuse(x64::avx512_core) && one_of(dstPrc, ov::element::f32, ov::element::f64)) {
                if (dstPrc == ov::element::f32) {
                    vcvtdq2ps(vmm_dst, src_adr_bcst);
                } else if (dstPrc == ov::element::f64) {
                    vcvtdq2pd(vmm_dst, src_adr_bcst);
                }
            } else {
                uni_vbroadcastss(vmm_dst, srcAdr);
            }
            break;
        case ov::element::bf16:
            uni_vpinsrw(vmm_dst, vmm_dst, srcAdr, 0);
            uni_vpslld(vmm_dst, vmm_dst, 16);
            break;
        case ov::element::i16:
            uni_vpinsrw(vmm_dst, vmm_dst, srcAdr, 0);
            uni_vpmovsxwd(vmm_dst, srcAdr);
            break;
        case ov::element::u16:
            uni_vpinsrw(vmm_dst, vmm_dst, srcAdr, 0);
            uni_vpmovzxwd(vmm_dst, srcAdr);
            break;
        case ov::element::i8:
            if (dstPrc == ov::element::i32) {
                pinsrb(vmm_dst, srcAdr, 0);
                uni_vpmovsxbd(vmm_dst, vmm_dst);
            } else {
                vpbroadcastb(vmm_dst, srcAdr);
            }
            break;
        case ov::element::u8:
            if (dstPrc == ov::element::i32) {
                pinsrb(vmm_dst, srcAdr, 0);
                uni_vpmovzxbd(vmm_dst, vmm_dst);
            } else {
                vpbroadcastb(vmm_dst, srcAdr);
            }
            break;
        default:
            IE_THROW() << "Unsupported source precision: " << srcPrc;
    }

    switch (dstPrc) {
        case ov::element::f32:
            if (x64::mayiuse(x64::avx512_core)) {
                if (one_of(srcPrc, ov::element::u8, ov::element::i8, ov::element::u16, ov::element::i16)) {
                    uni_vcvtdq2ps(vmm_dst, vmm_dst);
                }
            } else {
                if (srcPrc == ov::element::f64) {
                    uni_vcvtpd2ps(vmm_dst, vmm_dst);
                } else if (srcPrc == ov::element::i64) {
                    // Do conversion later.
                } else if (one_of(srcPrc, ov::element::u8, ov::element::i8, ov::element::u16, ov::element::i16, ov::element::i32)) {
                    uni_vcvtdq2ps(vmm_dst, vmm_dst);
                }
            }
            break;
        case ov::element::i32:
            if (!x64::mayiuse(x64::avx512_core)) {
                if (srcPrc == ov::element::i64) {
                    // Do conversion later.
                } else if (one_of(srcPrc, ov::element::f32, ov::element::bf16)) {
                    uni_vcvtps2dq(vmm_dst, vmm_dst);
                }
            } else if (srcPrc == ov::element::i64) {
                vpmovsqd(vmm_dst, vmm_dst);
            }
            break;
        case ov::element::i64:
        case ov::element::f64:
            break;
        default:
            IE_THROW() << "Unsupported destination precision: " << dstPrc;
    }
}

void JitKernelBase::store_vector(const Address &dstAdr,
                                 const Xmm &vSrc,
                                 const ov::element::Type& dstPrc,
                                 const ov::element::Type& srcPrc) {
    auto xmmSrc = Xmm(vSrc.getIdx());
    auto ymmSrc = Ymm(vSrc.getIdx());

    switch (srcPrc) {
        case ov::element::f64:
            if (dstPrc == ov::element::f32) {
                uni_vcvtpd2ps(x64::mayiuse(x64::avx512_core) ? ymmSrc : xmmSrc, vSrc);
            } else if (dstPrc == ov::element::i64) {
                if (x64::mayiuse(x64::avx512_core)) {
                    vcvtpd2qq(vSrc, vSrc);
                } else {
                    // TODO
                }
            } else if (dstPrc == ov::element::i32) {
                vcvtpd2dq(ymmSrc, vSrc);
            }
            break;
        case ov::element::i64:
            if (dstPrc == ov::element::f32 || dstPrc == ov::element::bf16) {
                if (x64::mayiuse(x64::avx512_core)) {
                    vcvtqq2ps(ymmSrc, vSrc);
                } else {
                    // TODO
                }
            } else if (dstPrc == ov::element::f64) {
                if (x64::mayiuse(x64::avx512_core)) {
                    vcvtqq2pd(vSrc, vSrc);
                } else {
                    // TODO
                }
            }
            break;
        case ov::element::f32:
            if (dstPrc == ov::element::i64) {
                if (x64::mayiuse(x64::avx512_core)) {
                    vcvtps2qq(vSrc, ymmSrc);
                } else {
                    // TODO
                }
            } else if ((dstPrc == ov::element::u8 || dstPrc == ov::element::u16) && x64::mayiuse(x64::avx512_core)) {
                vcvtps2udq(vSrc, vSrc);
            } else if (dstPrc != ov::element::f32 && dstPrc != ov::element::bf16) {
                uni_vcvtps2dq(vSrc, vSrc);
            }
            break;
        case ov::element::i32:
            if (dstPrc == ov::element::f32 || dstPrc == ov::element::bf16) {
                uni_vcvtdq2ps(vSrc, vSrc);
            }
            break;
        default:
            IE_THROW() << "Unsupported source precision: " << srcPrc;
    }

    switch (dstPrc) {
        case ov::element::f64:
            uni_vmovups(dstAdr, vSrc);
            break;
        case ov::element::f32:
            if (srcPrc.size() == 8) {
                uni_vmovups(dstAdr, ymmSrc);
            } else {
                uni_vmovups(dstAdr, vSrc);
            }
            break;
        case ov::element::i64:
            uni_vmovups(dstAdr, vSrc);
            break;
        case ov::element::i32:
            if (srcPrc == ov::element::i64) {
                if (x64::mayiuse(x64::avx512_core)) {
                    vpmovsqd(dstAdr, vSrc);
                } else {
                    // TODO
                }
            } else if (srcPrc == ov::element::f64) {
                uni_vmovups(dstAdr, ymmSrc);
            } else {
                uni_vmovups(dstAdr, vSrc);
            }
            break;
        case ov::element::bf16:
            vcvtneps2bf16->emit_code({static_cast<size_t>(ymmSrc.getIdx())}, {static_cast<size_t>(ymmSrc.getIdx())});
            vmovdqu16(dstAdr, ymmSrc);
            break;
        case ov::element::i16:
            if (x64::mayiuse(x64::avx512_core)) {
                vpmovsdw(dstAdr, vSrc);
            } else {
                uni_vpackssdw(vSrc, vSrc, vSrc);
                if (x64::mayiuse(x64::avx)) {
                    vpermq(ymmSrc, ymmSrc, 0x08);
                    uni_vmovdqu(dstAdr, xmmSrc);
                } else {
                    movq(dstAdr, xmmSrc);
                }
            }
            break;
        case ov::element::u16:
            if (x64::mayiuse(x64::avx512_core)) {
                vpmovusdw(dstAdr, xmmSrc);
            } else {
                uni_vpackusdw(vSrc, vSrc, vSrc);
                if (x64::mayiuse(x64::avx)) {
                    vpermq(ymmSrc, ymmSrc, 0x08);
                    uni_vmovdqu(dstAdr, xmmSrc);
                } else {
                    movq(dstAdr, xmmSrc);
                }
            }
            break;
        case ov::element::i8:
            if (x64::mayiuse(x64::avx512_core)) {
                if (srcPrc == ov::element::i64) {
                    vpmovsqb(dstAdr, vSrc);
                } else {
                    vpmovsdb(dstAdr, vSrc);
                }
            } else {
                uni_vpackssdw(vSrc, vSrc, vSrc);
                if (x64::mayiuse(x64::avx)) {
                    vpermq(ymmSrc, ymmSrc, 0x08);
                }
                uni_vpacksswb(vSrc, vSrc, vSrc);
                if (x64::mayiuse(x64::avx)) {
                    vmovq(dstAdr, xmmSrc);
                } else {
                    movd(dstAdr, xmmSrc);
                }
            }
            break;
        case ov::element::u8:
            if (x64::mayiuse(x64::avx512_core)) {
                if (srcPrc == ov::element::i64) {
                    vpmovusqb(dstAdr, vSrc);
                } else {
                    vpmovusdb(dstAdr, vSrc);
                }
            } else {
                uni_vpackusdw(vSrc, vSrc, vSrc);
                if (x64::mayiuse(x64::avx)) {
                    vpermq(ymmSrc, ymmSrc, 0x08);
                }
                uni_vpackuswb(vSrc, vSrc, vSrc);
                if (x64::mayiuse(x64::avx)) {
                    vmovq(dstAdr, xmmSrc);
                } else {
                    movd(dstAdr, xmmSrc);
                }
            }
            break;
        default:
            IE_THROW() << "Unsupported destination precision: " << dstPrc;
    }
}

void JitKernelBase::store_scalar(const Address &dstAdr,
                                const Xmm &vSrc,
                                const ov::element::Type& dstPrc,
                                const ov::element::Type& srcPrc) {
    switch (srcPrc) {
        case ov::element::f64:
            if (dstPrc == ov::element::f32) {
                uni_vcvtpd2ps(vSrc, vSrc);
            } else if (dstPrc == ov::element::i64) {
                if (x64::mayiuse(x64::avx512_core)) {
                    vcvtpd2qq(vSrc, vSrc);
                } else {
                    // TODO
                }
            } else if (dstPrc == ov::element::i32) {
                uni_vcvtpd2dq(vSrc, vSrc);
            }
            break;
        case ov::element::i64:
            if (dstPrc == ov::element::f32 || dstPrc == ov::element::bf16) {
                if (x64::mayiuse(x64::avx512_core)) {
                    vcvtqq2ps(vSrc, vSrc);
                } else {
                    // TODO
                }
            } else if (dstPrc == ov::element::i32) {
                if (x64::mayiuse(x64::avx512_core)) {
                    vpmovsqd(vSrc, vSrc);
                } else {
                    // TODO
                }
            }
            break;
        case ov::element::f32:
            if (dstPrc == ov::element::i64) {
                if (x64::mayiuse(x64::avx512_core)) {
                    vcvtps2qq(vSrc, vSrc);
                } else {
                    // TODO
                }
            } else if (dstPrc == ov::element::u8 && x64::mayiuse(x64::avx512_core)) {
                vcvtps2udq(vSrc, vSrc);
            } else if (dstPrc != ov::element::f32 && dstPrc != ov::element::bf16) {
                uni_vcvtps2dq(vSrc, vSrc);
            }
            break;
        case ov::element::i32:
            if (dstPrc == ov::element::f32 || dstPrc == ov::element::bf16) {
                uni_vcvtdq2ps(vSrc, vSrc);
            }
            break;
        default:
            IE_THROW() << "Unsupported source precision: " << srcPrc;
    }

    switch (dstPrc) {
        case ov::element::f64:
        case ov::element::i64:
            uni_vmovsd(dstAdr, vSrc);
            break;
        case ov::element::f32:
        case ov::element::i32:
            uni_vmovss(dstAdr, vSrc);
            break;
        case ov::element::bf16:
            uni_vpsrld(vSrc, vSrc, 16);
            uni_vpextrw(dstAdr, vSrc, 0x0);
            break;
        case ov::element::i16:
            uni_vpackssdw(vSrc, vSrc, vSrc);
            uni_vpextrw(dstAdr, vSrc, 0x0);
            break;
        case ov::element::u16:
            uni_vpackusdw(vSrc, vSrc, vSrc);
            uni_vpextrw(dstAdr, vSrc, 0x0);
            break;
        case ov::element::i8:
            if (x64::mayiuse(x64::avx512_core)) {
                vpmovsdb(vSrc, vSrc);
            } else {
                uni_vpackssdw(vSrc, vSrc, vSrc);
                uni_vpacksswb(vSrc, vSrc, vSrc);
            }
            uni_vpextrb(dstAdr, vSrc, 0x0);
            break;
        case ov::element::u8:
            if (x64::mayiuse(x64::avx512_core)) {
                vpmovusdb(vSrc, vSrc);
            } else {
                uni_vpackusdw(vSrc, vSrc, vSrc);
                uni_vpackuswb(vSrc, vSrc, vSrc);
            }
            uni_vpextrb(dstAdr, vSrc, 0);
            break;
        default:
            IE_THROW() << "Unsupported destination precision: " << dstPrc;
    }
}
