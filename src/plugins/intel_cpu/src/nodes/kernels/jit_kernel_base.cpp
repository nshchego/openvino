// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_kernel_base.hpp"

using namespace ov;
using namespace intel_cpu;
using namespace dnnl::impl::cpu;


void JitKernelBase::uni_vfmsub132ps(const Xbyak::Xmm&     x1,
                                    const Xbyak::Xmm&     x2,
                                    const Xbyak::Operand& op) {
    // Note: x1 gets overriden by x1*op
    // This is incorrect if x1 == x2
    if (isValidIsa(dnnl::impl::cpu::x64::avx2)) {
        vfmsub132ps(x1, x2, op);
    } else if (isValidIsa(dnnl::impl::cpu::x64::avx)) {
        assert(x1.getIdx() != x2.getIdx());
        vmulps(x1, x1, op);
        vsubps(x1, x1, x2);
    } else {
        assert(x1.getIdx() != x2.getIdx());
        mulps(x1, op);
        subps(x1, x2);
    }
}

void JitKernelBase::uni_vfnmadd132ps(const Xbyak::Xmm&     x1,
                                     const Xbyak::Xmm&     x2,
                                     const Xbyak::Operand& op) {
    // Note: x1 gets overriden by x1*op
    // This is incorrect if x1 == x2
    if (isValidIsa(dnnl::impl::cpu::x64::avx2)) {
        vfnmadd132ps(x1, x2, op);
    } else if (isValidIsa(dnnl::impl::cpu::x64::avx)) {
        assert(x1.getIdx() != x2.getIdx());
        vmulps(x1, x1, op);
        vsubps(x1, x2, x1);
    } else {
        assert(x1.getIdx() != x2.getIdx());
        mulps(x1, op);
        subps(x2, x1);
        vmovups(x1, x2);
    }
}

void JitKernelBase::uni_vfmsub231ps(const Xbyak::Xmm &x1,
                                    const Xbyak::Xmm &x2,
                                    const Xbyak::Operand &op) {
    // Note: x1 gets overriden by x1*x2
    // This is incorrect if x1 == op
    if (isValidIsa(dnnl::impl::cpu::x64::avx2)) {
        vfmsub231ps(x1, x2, op);
    } else if (isValidIsa(dnnl::impl::cpu::x64::avx)) {
        assert(!x1.isEqualIfNotInherited(op));
        vmulps(x2, x2, op);
        vsubps(x1, x2, x1);
    } else {
        assert(!x1.isEqualIfNotInherited(op));
        mulps(x2, op);
        subps(x2, x1);
        vmovups(x1, x2);
    }
}

void JitKernelBase::uni_vpaddd(const Xbyak::Ymm&     vDst,
                               const Xbyak::Ymm&     vSrc,
                               const Xbyak::Operand& op) {
    if (isValidIsa(dnnl::impl::cpu::x64::avx2)) {
        vpaddd(vDst, vSrc, op);
    } else if (isValidIsa(dnnl::impl::cpu::x64::avx)) {
        Xbyak::Xmm xmmDst(vDst.getIdx());
        vmovups(vDst, vSrc);
        if (op.isYMM()) {
            Xbyak::Ymm ymmOp(op.getIdx());
            Xbyak::Xmm xmmOp(op.getIdx());
            paddd(xmmDst, xmmOp);
            vperm2f128(vDst, vDst, vDst, 0x1);
            vperm2f128(ymmOp, ymmOp, ymmOp, 0x1);
            paddd(xmmDst, xmmOp);
            vperm2f128(vDst, vDst, vDst, 0x1);
            vperm2f128(ymmOp, ymmOp, ymmOp, 0x1);
        } else if (op.isMEM()) {
            const int vlen = dnnl::impl::cpu::x64::cpu_isa_traits<dnnl::impl::cpu::x64::sse41>::vlen;
            paddd(xmmDst, op.getAddress());
            vperm2f128(vDst, vDst, vDst, 0x1);
            paddd(xmmDst, ptr[op.getAddress().getRegExp() + vlen]);
            vperm2f128(vDst, vDst, vDst, 0x1);
        } else {
            IE_THROW() << "Not supported operand type.";
        }
    } else {
        IE_THROW() << "Not defined behavior for instruction 'vpaddd' in current instructions set.";
    }
}

void JitKernelBase::uni_vpsubd(const Xbyak::Ymm&    vDst,
                               const Xbyak::Ymm&    vSrc,
                               const Xbyak::Operand &op) {
    if (isValidIsa(dnnl::impl::cpu::x64::avx2)) {
        vpsubd(vDst, vSrc, op);
    } else if (isValidIsa(dnnl::impl::cpu::x64::avx)) {
        Xbyak::Xmm xmmDst(vDst.getIdx());
        vmovups(vDst, vSrc);
        if (op.isYMM()) {
            Xbyak::Ymm ymmOp(op.getIdx());
            Xbyak::Xmm xmmOp(op.getIdx());
            psubd(xmmDst, xmmOp);
            vperm2f128(vDst, vDst, vDst, 0x1);
            vperm2f128(ymmOp, ymmOp, ymmOp, 0x1);
            psubd(xmmDst, xmmOp);
            vperm2f128(vDst, vDst, vDst, 0x1);
            vperm2f128(ymmOp, ymmOp, ymmOp, 0x1);
        } else if (op.isMEM()) {
            const int vlen = dnnl::impl::cpu::x64::cpu_isa_traits<dnnl::impl::cpu::x64::sse41>::vlen;
            psubd(xmmDst, op.getAddress());
            vperm2f128(vDst, vDst, vDst, 0x1);
            psubd(xmmDst, ptr[op.getAddress().getRegExp() + vlen]);
            vperm2f128(vDst, vDst, vDst, 0x1);
        } else {
            IE_THROW() << "Not supported operand type.";
        }
    } else {
        IE_THROW() << "Not defined behavior for instruction 'vpsubd' in current instructions set.";
    }
}

void JitKernelBase::uni_kandd(const Xbyak::Xmm& vDst, const Xbyak::Xmm& vSrc1, const Xbyak::Xmm& vSrc2) {
    if (isValidIsa(dnnl::impl::cpu::x64::avx2)) {
        vpand(vDst, vSrc1, vSrc2);
    } else if (isValidIsa(dnnl::impl::cpu::x64::avx)) {
// TODO
    } else {
        if (vDst.getIdx() == vSrc2.getIdx()) {
            pand(vDst, vSrc1);
        } else {
            if (vDst.getIdx() != vSrc1.getIdx()) {
                mov(vDst, vSrc1);
            }
            pand(vDst, vSrc2);
        }
    }
}

void JitKernelBase::gatherdd(const Xbyak::Xmm&    vDst,
                             const Xbyak::Reg64&  rSrcPtr,
                             const Xbyak::Xmm&    vSrcShift,
                             const Xbyak::Opmask& kReadMask,
                             const bool useMask,
                             const bool zeroFill) {
    if (kReadMask.getIdx() == 0) {
        IE_THROW() << "The vpgatherdd instruction cannot use the register k0 as mask.";
    }
    if (!useMask)
        kxnorw(kReadMask, kReadMask, kReadMask);
    if (zeroFill)
        uni_vpxor(vDst, vDst, vDst);

    vpgatherdd(vDst | kReadMask, ptr[rSrcPtr + vSrcShift]); // TODO: Try do not use a mask.
}

void JitKernelBase::gatherdd(const Xbyak::Xmm&   vDst,
                             const Xbyak::Reg64& rSrcPtr,
                             const Xbyak::Xmm&   vSrcShift,
                             const Xbyak::Xmm&   vReadMask,
                             const bool useMask,
                             const bool zeroFill) {
    if (vDst.getIdx() == vSrcShift.getIdx() || vDst.getIdx() == vReadMask.getIdx() || vSrcShift.getIdx() == vReadMask.getIdx()) {
        IE_THROW() << "Any pair of the index, mask, or destination registers cannot be the same.";
    }
    if (zeroFill)
        pxor(vDst, vDst); // Don't use vpxor. It zeros the rest of the YMM register.

    if (isValidIsa(dnnl::impl::cpu::x64::avx2)) {
        if (!useMask)
            uni_vpcmpeqd(vReadMask, vReadMask, vReadMask);

        vpgatherdd(vDst, ptr[rSrcPtr + vSrcShift], vReadMask);
    } else {
        auto rAux = getReg64();
        Xbyak::Reg32 r32Aux = Xbyak::Reg32(rAux.getIdx());
        const uint8_t elPerVec = dnnl::impl::cpu::x64::cpu_isa_traits<dnnl::impl::cpu::x64::sse41>::vlen / sizeof(int);

        for (uint8_t i = 0; i < elPerVec; i++) {
            Xbyak::Label lLoopNext;
            if (useMask) {
                uni_vpextrd(r32Aux, vReadMask, i);
                cmp(r32Aux, 0); // TODO: check significant bit
                je(lLoopNext, T_NEAR);
            }
            uni_vpextrd(r32Aux, vSrcShift, i);
            pinsrd(vDst, ptr[rSrcPtr + (Xbyak::Reg64)rAux], i);

            if (useMask)
                L(lLoopNext);
        }
    }
}

void JitKernelBase::gatherdd(const Xbyak::Ymm&   vDst,
                             const Xbyak::Reg64& rSrcPtr,
                             const Xbyak::Ymm&   vSrcShift,
                             const Xbyak::Ymm&   vReadMask,
                             const bool useMask,
                             const bool zeroFill) {
    if (vDst.getIdx() == vSrcShift.getIdx() || vDst.getIdx() == vReadMask.getIdx() || vSrcShift.getIdx() == vReadMask.getIdx()) {
        IE_THROW() << "Any pair of the index, mask, or destination registers cannot be the same.";
    }
    if (isValidIsa(dnnl::impl::cpu::x64::avx2)) {
        if (!useMask)
            uni_vpcmpeqd(vReadMask, vReadMask, vReadMask);
        if (zeroFill)
            uni_vpxor(vDst, vDst, vDst);

        vpgatherdd(vDst, ptr[rSrcPtr + vSrcShift], vReadMask);
    } else {
        Xbyak::Xmm xmmDst      = Xbyak::Xmm(vDst.getIdx()),
                   xmmSrcShft  = Xbyak::Xmm(vSrcShift.getIdx()),
                   xmmReadMask = Xbyak::Xmm(vReadMask.getIdx());
        for (uint8_t i = 0; i < 2; i++) {
            gatherdd(xmmDst, rSrcPtr, xmmSrcShft, xmmReadMask, useMask, zeroFill);

            vperm2f128(vDst, vDst, vDst, 0x1);
            vperm2f128(vSrcShift, vSrcShift, vSrcShift, 0x1);
            if (useMask)
                vperm2f128(vReadMask, vReadMask, vReadMask, 0x1);
        }
    }
}

void JitKernelBase::uni_vpbroadcastd(const Xbyak::Xmm &x, const Xbyak::Operand &op) {
    if (isValidIsa(dnnl::impl::cpu::x64::avx2)) {
        vpbroadcastd(x, op);
    } else if (isValidIsa(dnnl::impl::cpu::x64::avx)) {
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

void JitKernelBase::uni_vpbroadcastd(const Xbyak::Ymm &x, const Xbyak::Operand &op) {
    if (isValidIsa(dnnl::impl::cpu::x64::avx2)) {
        vpbroadcastd(x, op);
    } else {
        if (op.isMEM()) {
            vbroadcastss(x, op.getAddress());
        } else {
            const Xbyak::Xmm t(x.getIdx());
            if (!t.isEqualIfNotInherited(op)) {
                vmovss(t, t, op);
            }
            vinsertf128(x, x, t, 1);
            vshufps(x, x, x, 0);
        }
    }
}

void JitKernelBase::fillRestWorkMask(const Xbyak::Opmask& dstMask,
                                     const Xbyak::Zmm&    zAux,
                                     const Xbyak::Reg64&  rWorkRest) {
    auto rAux0 = getReg64();
    auto rAux1 = getReg64();
    Xbyak::Label lKmov;
    Xbyak::Reg32 rOnes(rAux1.getIdx());
    const uint64_t typeSize = 4;
    const uint64_t elPerVec = dnnl::impl::cpu::x64::cpu_isa_traits<dnnl::impl::cpu::x64::avx512_core>::vlen / typeSize;

    mov(rOnes, 0x0000FFFF);
    cmp(rWorkRest, elPerVec);
    jge(lKmov);
    {
        Xbyak::Reg32 rShift(rAux0.getIdx());
        mov(rShift, elPerVec);
        sub(rShift, rWorkRest);
        shrx(rOnes, rOnes, rShift);
    }
    L(lKmov);
    kmovw(dstMask, rOnes);
}

void JitKernelBase::load(const Xbyak::Xmm&   vDst,
                         const Xbyak::Reg64& rSrc,
                         const Xbyak::Reg64& rLoadNum,
                         const uint8_t       typeSize,
                         const bool          zeroFilling) {
    if (!one_of(typeSize, 1, 2, 4, 8)) {
        IE_THROW() << "Could not load data with type size " << typeSize;
    }
    const uint8_t elPerVec = dnnl::impl::cpu::x64::cpu_isa_traits<dnnl::impl::cpu::x64::sse41>::vlen / typeSize;
    Xbyak::Label lLoopEnd;
    auto rRestCounter = getReg64();
    mov(rRestCounter, rLoadNum);
    if (zeroFilling)
        uni_vpxor(vDst, vDst, vDst);

    for (uint8_t i = 0; i < elPerVec; i++) {
        cmp(rRestCounter, 0);
        jle(lLoopEnd, T_NEAR);

        if (typeSize == 1)
            pinsrb(vDst, ptr[rSrc + i * typeSize], i);
        else if (typeSize == 2)
            pinsrw(vDst, ptr[rSrc + i * typeSize], i);
        else if (typeSize == 4)
            pinsrd(vDst, ptr[rSrc + i * typeSize], i);
        else if (typeSize == 8)
            pinsrq(vDst, ptr[rSrc + i * typeSize], i);

        dec(rRestCounter);
    }
    L(lLoopEnd);
}

void JitKernelBase::load(const Xbyak::Ymm&   vDst,
                         const Xbyak::Reg64& rSrc,
                         const Xbyak::Reg64& rLoadNum,
                         const uint8_t       typeSize,
                         const bool          zeroFilling) {
    if (!one_of(typeSize, 1, 2, 4, 8)) {
        IE_THROW() << "Could not load data with type size " << typeSize;
    }
    const uint8_t elPerVec = dnnl::impl::cpu::x64::cpu_isa_traits<dnnl::impl::cpu::x64::avx>::vlen / typeSize;
    Xbyak::Label lEnd, lLoop2End;
    auto rAux = getReg64();
    mov(rAux, rLoadNum);
    if (zeroFilling)
        uni_vpxor(vDst, vDst, vDst);
    Xbyak::Xmm xmmDst(vDst.getIdx());

    for (uint8_t i = 0; i < elPerVec / 2; i++) {
        cmp(rAux, 0);
        jle(lEnd, T_NEAR);

        if (typeSize == 1)
            pinsrb(xmmDst, ptr[rSrc + i * typeSize], i);
        else if (typeSize == 2)
            pinsrw(xmmDst, ptr[rSrc + i * typeSize], i);
        else if (typeSize == 4)
            pinsrd(xmmDst, ptr[rSrc + i * typeSize], i);
        else if (typeSize == 8)
            pinsrq(xmmDst, ptr[rSrc + i * typeSize], i);

        dec(rAux);
    }

    vperm2f128(vDst, vDst, vDst, 0x1);

    for (uint8_t i = 0; i < elPerVec / 2; i++) {
        cmp(rAux, 0);
        jle(lLoop2End, T_NEAR);

        const uint8_t offset = (i + elPerVec / 2) * typeSize;
        if (typeSize == 1)
            pinsrb(xmmDst, ptr[rSrc + offset], i);
        else if (typeSize == 2)
            pinsrw(xmmDst, ptr[rSrc + offset], i);
        else if (typeSize == 4)
            pinsrd(xmmDst, ptr[rSrc + offset], i);
        else if (typeSize == 8)
            pinsrq(xmmDst, ptr[rSrc + offset], i);

        dec(rAux);
    }
    L(lLoop2End);

    vperm2f128(vDst, vDst, vDst, 0x1);
    L(lEnd);
}

void JitKernelBase::store(const Xbyak::Reg64& rDst,
                          const Xbyak::Reg64& rToStoreNum,
                          const Xbyak::Xmm&   xmmSrc,
                          const size_t        typeSize,
                          const size_t        dstOffset) {
    Xbyak::Label lEnd;
    const uint64_t elPerVec = dnnl::impl::cpu::x64::cpu_isa_traits<dnnl::impl::cpu::x64::sse41>::vlen / typeSize;
    auto rRestCounter = getReg64();
    mov(rRestCounter, rToStoreNum);

    for (size_t k = 0; k < elPerVec; k++) {
        cmp(rRestCounter, 0);
        jle(lEnd, T_NEAR);

        const size_t offset = dstOffset + k * typeSize;
        if (typeSize == 8) {
            uni_vpextrq(ptr[rDst + offset], xmmSrc, k);
        } else if (typeSize == 4) {
            uni_vpextrd(ptr[rDst + offset], xmmSrc, k);
        } else if (typeSize == 2) {
            uni_vpextrw(ptr[rDst + offset], xmmSrc, k);
        } else if (typeSize == 1) {
            uni_vpextrb(ptr[rDst + offset], xmmSrc, k);
        }

        dec(rRestCounter);
    }
    L(lEnd);
}

void JitKernelBase::store(const Xbyak::Reg64& rDst,
                          const Xbyak::Reg64& rToStoreNum,
                          const Xbyak::Ymm&   vSrc,
                          const size_t        typeSize,
                          const size_t        dstOffset) {
    Xbyak::Label lEnd;
    Xbyak::Xmm xmmSrc(vSrc.getIdx());
    const size_t elPerVec = dnnl::impl::cpu::x64::cpu_isa_traits<dnnl::impl::cpu::x64::avx>::vlen / typeSize;

    for (int i = 0; i < 2; i++) {
        store(rDst, rToStoreNum, xmmSrc, typeSize, i == 0 ? dstOffset : (dstOffset + typeSize * elPerVec / 2));

        if (i == 0) {
            cmp(rToStoreNum, elPerVec / 2);
            jle(lEnd, T_NEAR);
            sub(rToStoreNum, elPerVec / 2);
        } else {
            add(rToStoreNum, elPerVec / 2);
        }

        vperm2f128(vSrc, vSrc, vSrc, 0x1);
    }
    L(lEnd);
}

// Makes gather from memory under the vReadMask and writes to the XMM/m128.
// It can fill in values not read from the source with zero.
void JitKernelBase::maskMov32(const Xbyak::Operand& opDst,
                              const Xbyak::Operand& opSrc,
                              const Xbyak::Xmm&     xmmReadMask,
                              const Xbyak::Xmm&     xmmSrcShift,
                              const Xbyak::Reg64&   rToStoreCounter,
                              const bool            useMask,
                              const bool            zeroMask) {
    Xbyak::Label lEnd;
    auto rAux = getReg64();
    Xbyak::Reg32 r32Aux = Xbyak::Reg32(rAux.getIdx());
    const uint8_t typeSize = 4;

    for (uint8_t i = 0; i < 4; i++) {
        cmp(rToStoreCounter, 0);
        jle(lEnd, T_NEAR);

        Xbyak::Label lLoopNext, lZeroMask;
        if (useMask) {
            uni_vpextrd(r32Aux, xmmReadMask, i);
            cmp(r32Aux, 0);
            je(lZeroMask, T_NEAR);
        }
        uni_vpextrd(r32Aux, xmmSrcShift, i);
        if (opDst.isXMM()) {
            Xbyak::Xmm xmmDst = Xbyak::Xmm(opDst.getIdx());
            pinsrd(xmmDst, ptr[opSrc.getReg() + (Xbyak::Reg64)rAux], i << 4);
        } else if (opDst.isREG()) {
            mov(r32Aux, ptr[opSrc.getReg() + (Xbyak::Reg64)rAux]);
            mov(ptr[opDst.getReg() + i * typeSize], r32Aux);
        }
        jmp(lLoopNext, T_NEAR);
        L(lZeroMask);
        if (zeroMask) {
            if (opDst.isXMM()) {
                Xbyak::Xmm xmmDst = Xbyak::Xmm(opDst.getIdx());
                pinsrd(xmmDst, r32Aux, i << 4);
            } else if (opDst.isREG()) {
                mov(ptr[opDst.getReg() + i * typeSize], r32Aux);
            }
        }

        L(lLoopNext);
        dec(rToStoreCounter);
    } // use VMASKMOVDQU?
    L(lEnd);
}

// Makes gather from memory under the vReadMask and writes to the YMM/m256.
// It can fill in values not read from the source with zero.
void JitKernelBase::maskMov32(const Xbyak::Operand& opDst,
                              const Xbyak::Operand& opSrc,
                              const Xbyak::Ymm&     vReadMask,
                              const Xbyak::Ymm&     vSrcShift,
                              const Xbyak::Reg64&   rToStoreCounter,
                              const bool            useMask,
                              const bool            zeroMask) {
    Xbyak::Label lEnd;
    if (isValidIsa(dnnl::impl::cpu::x64::avx2)) {
//            Xbyak::Ymm ymmDst(opDst.getIdx());
//            if (!useMask)
//                uni_vcmpps(vReadMask, vReadMask, vReadMask, 0x0);
//            if (zeroMask)
//                uni_vpxor(ymmDst, ymmDst, ymmDst);
//            vpgatherdd(ymmDst, ptr[Xbyak::Ymm(opSrc.getIdx()) + vSrcShift], vReadMask);
//                if (zeroMask)
//                    uni_vpxor(vAux, vAux, vAux);
//                gatherdd(vAux, ptr[vSrcShift + vSrcShift], vReadMask);
//                if (zeroMask)
//                    uni_vmovups(ptr[opDst.getReg()], vAux);
//                else
//                    uni_vmovups_tail(ptr[opDst.getReg()], vWriteMask, vAux);
    } else if (isValidIsa(dnnl::impl::cpu::x64::avx)) {
        Xbyak::Xmm xmmReadMask  = Xbyak::Xmm(vReadMask.getIdx()),
                   xmmSrcShft   = Xbyak::Xmm(vSrcShift.getIdx());
        for (uint8_t i = 0; i < 2; i++) {
            maskMov32(opDst, opSrc, xmmReadMask, xmmSrcShft, rToStoreCounter, useMask, zeroMask);

            if (i == 0) {
                cmp(rToStoreCounter, 0);
                jle(lEnd, T_NEAR);
            }

            if (opDst.isYMM()) {
                Xbyak::Ymm ymmDst = Xbyak::Ymm(opDst.getIdx());
                vperm2f128(ymmDst, ymmDst, ymmDst, 0x1);
            } else {
                if (i == 0)
                    add(opDst, sizeof(float) * 4);
                else
                    sub(opDst, sizeof(float) * 4);
            }
            vperm2f128(vSrcShift, vSrcShift, vSrcShift, 0x1);
            if (useMask)
                vperm2f128(vReadMask, vReadMask, vReadMask, 0x1);
        }
    }
    L(lEnd);
}

// Makes gather from memory under the vReadMask and writes to the XMM/m128 under the vWriteMask
// It can fill in values not read from the source with zero.
void JitKernelBase::maskMov32(const Xbyak::Operand& opDst,
                              const Xbyak::Operand& opSrc,
                              const Xbyak::Xmm&     vReadMask,
                              const Xbyak::Xmm&     vWriteMask,
                              const Xbyak::Xmm&     vSrcShift,
                              const Xbyak::Xmm&     vAux,
                              const Xbyak::Reg64&   rAux,
                              const bool            useMask,
                              const bool            zeroMask) {
    Xbyak::Reg32 r32Aux = Xbyak::Reg32(rAux.getIdx());
    const uint8_t typeSize = 4;

    for (uint8_t i = 0; i < 4; i++) {
        Xbyak::Label lLoopNext, lZeroMask;
        if (useMask) {
            uni_vpextrd(r32Aux, vReadMask, i);
            cmp(r32Aux, 0);
            je(lZeroMask, T_NEAR);
        }
        uni_vpextrd(r32Aux, vSrcShift, i);
        if (opDst.isXMM()) {
            Xbyak::Xmm vDst = Xbyak::Xmm(opDst.getIdx());
            uni_vpinsrd(vDst, vDst, ptr[opSrc.getReg() + rAux], i << 4);
        } else if (opDst.isREG()) {
            mov(rAux, ptr[opSrc.getReg() + rAux]);
            mov(ptr[opDst.getReg() + i * typeSize], r32Aux);
        }
        jmp(lLoopNext, T_NEAR);
        L(lZeroMask);
        if (zeroMask) {
            if (opDst.isXMM()) {
                Xbyak::Xmm vDst = Xbyak::Xmm(opDst.getIdx());
                uni_vpinsrd(vDst, vDst, r32Aux, i << 4);
            } else if (opDst.isREG()) {
                mov(ptr[opDst.getReg() + i * typeSize], r32Aux);
            }
        }
        L(lLoopNext);
    } // use VMASKMOVDQU?
}

// Gathers data from memory under the vReadMask and writes to the YMM/m256 under the vWriteMask
// It can fill in values not read from the source with zero.
void JitKernelBase::maskMov32(const Xbyak::Operand& opDst,
                              const Xbyak::Operand& opSrc,
                              const Xbyak::Ymm&     vReadMask,
                              const Xbyak::Ymm&     vWriteMask,
                              const Xbyak::Ymm&     vSrcShift,
                              const Xbyak::Ymm&     vAux,
                              const Xbyak::Reg64&   rAux,
                              const bool            useMask,
                              const bool            zeroMask) {
    if (isValidIsa(dnnl::impl::cpu::x64::avx2)) {
        Xbyak::Reg64 rSrc(opSrc.getIdx());
        if (opDst.isYMM()) {
            Xbyak::Ymm vDst = Xbyak::Ymm(opDst.getIdx());
            if (zeroMask)
                uni_vpxor(vDst, vDst, vDst);
            gatherdd(vDst, rSrc, vSrcShift, vReadMask);
        } else if (opDst.isREG()) {
            if (zeroMask)
                uni_vpxor(vAux, vAux, vAux);
            gatherdd(vAux, rSrc, vSrcShift, vReadMask);
            if (zeroMask)
                uni_vmovups(ptr[opDst.getReg()], vAux);
            else
                uni_vmovups_tail(ptr[opDst.getReg()], vWriteMask, vAux);
        }
    } else if (isValidIsa(dnnl::impl::cpu::x64::avx)) {
        Xbyak::Xmm xmmReadMask  = Xbyak::Xmm(vReadMask.getIdx()),
                   xmmWriteMask = Xbyak::Xmm(vWriteMask.getIdx()),
                   xmmSrcShft   = Xbyak::Xmm(vSrcShift.getIdx()),
                   xmmAux       = Xbyak::Xmm(vAux.getIdx());
        for (uint8_t i = 0; i < 2; i++) {
            maskMov32(opDst, opSrc, xmmReadMask, xmmWriteMask, xmmSrcShft, xmmAux, rAux, useMask);
            if (opDst.isYMM()) {
                Xbyak::Ymm ymmDst = Xbyak::Ymm(opDst.getIdx());
                vperm2f128(ymmDst, ymmDst, ymmDst, 0x1);
            } else {
                if (i == 0)
                    add(opDst, sizeof(float) * 4);
                else
                    sub(opDst, sizeof(float) * 4);
            }
            vperm2f128(vSrcShift, vSrcShift, vSrcShift, 0x1);
            if (useMask)
                vperm2f128(vReadMask, vReadMask, vReadMask, 0x1);
            if (zeroMask)
                vperm2f128(vWriteMask, vWriteMask, vWriteMask, 0x1);
        }

//        maskMov32(opDst, opSrc, xmmReadMask, xmmWriteMask, xmmSrcShft, xmmAux, rAux, useMask);
//        vperm2f128(vSrcShift, vSrcShift, vSrcShift, 0x1);
//        sub(opDst, sizeof(float) * 4);
    }
}






