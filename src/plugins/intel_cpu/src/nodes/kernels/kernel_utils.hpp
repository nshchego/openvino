// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu/x64/jit_generator.hpp"
#include <ie_common.h>
#include <dnnl_types.h>
#include <set>

namespace ov {
namespace intel_cpu {

template <typename Vmm>
class vRefWrap;

#define r64Ref() rRefWrap<Xbyak::Reg64>(this, r64Pool[getRegIdx()])

class jitKernelBase: public dnnl::impl::cpu::x64::jit_generator {
protected:

    jitKernelBase() {
        for (int i = 0; i < vecNum; i++) {
            vecSet.insert(i);
        }
        for (auto el : r64Pool) {
            regSet.insert(el.first);
        }
    }

    inline bool isValidIsa(dnnl::impl::cpu::x64::cpu_isa_t isa) {
        return is_subset(isa, dnnl::impl::cpu::x64::isa_all) && dnnl::impl::cpu::x64::mayiuse(isa);
    }

    void uni_vfmsub132ps(const Xbyak::Xmm &x1,
                         const Xbyak::Xmm &x2,
                         const Xbyak::Operand &op) {
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

    void uni_vfnmadd132ps(const Xbyak::Xmm &x1,
                          const Xbyak::Xmm &x2,
                          const Xbyak::Operand &op) {
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
            subps(x1, x2);
        }
    }

    void uni_vfmsub231ps(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2, const Xbyak::Operand &op) {
        // Note: x1 gets overriden by x1*x2
        // This is incorrect if x1 == op
        if (isValidIsa(dnnl::impl::cpu::x64::avx2)) {
            vfmsub231ps(x1, x2, op);
        } else if (isValidIsa(dnnl::impl::cpu::x64::avx)) {
            assert(!x1.isEqualIfNotInherited(op));
            vmulps(x1, x1, x2);
            vsubps(x1, x1, op);
        } else {
            assert(!x1.isEqualIfNotInherited(op));
            mulps(x1, x2);
            subps(x1, op);
        }
    }

    void uni_kmovd(const Xbyak::Opmask& kDst, const Xbyak::Opmask& kSrc) {
        kmovd(kDst, kSrc);
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

    void uni_vpgatherdd(const Xbyak::Xmm&    vDst,
                        const Xbyak::Reg64&  rSrcPtr,
                        const Xbyak::Xmm&    vSrcShift,
                        const Xbyak::Opmask& kReadMask,
                        const bool useMask   = true,
                        const bool zeroFill  = false) {
        if (kReadMask.getIdx() == 0) {
            IE_THROW() << "The vpgatherdd instruction cannot use the register k0 as mask.";
        }
        if (!useMask)
            kxnorw(kReadMask, kReadMask, kReadMask);
        if (zeroFill)
            uni_vpxor(vDst, vDst, vDst);

        vpgatherdd(vDst | kReadMask, ptr[rSrcPtr + vSrcShift]);
    }

    void uni_vpgatherdd(const Xbyak::Xmm&   vDst,
                        const Xbyak::Reg64& rSrcPtr,
                        const Xbyak::Xmm&   vSrcShift,
                        const Xbyak::Xmm&   vReadMask,
                        const bool useMask  = true,
                        const bool zeroFill = false) {
        if (vDst.getIdx()== vSrcShift.getIdx() || vDst.getIdx() == vReadMask.getIdx() || vSrcShift.getIdx() == vReadMask.getIdx()) {
            IE_THROW() << "Any pair of the index, mask, or destination registers cannot be the same.";
        }
        if (isValidIsa(dnnl::impl::cpu::x64::avx2)) {
            if (!useMask)
                uni_vcmpps(vReadMask, vReadMask, vReadMask, 0x0);
            if (zeroFill)
                uni_vpxor(vDst, vDst, vDst);

            vpgatherdd(vDst, ptr[rSrcPtr + vSrcShift], vReadMask);
        } else {
            auto rAux = r64Ref();
            Xbyak::Reg32 r32Aux = Xbyak::Reg32(rAux.getIdx());
            const uint8_t elPerVec = dnnl::impl::cpu::x64::cpu_isa_traits<dnnl::impl::cpu::x64::sse41>::vlen / sizeof(int);

            for (uint8_t i = 0; i < elPerVec; i++) {
                Xbyak::Label lLoopNext;
                if (useMask) {
                    uni_vpextrd(r32Aux, vReadMask, i);
                    cmp(r32Aux, 0);
                    if (zeroFill) {
                        Xbyak::Label lNonZero;
                        jne(lNonZero, T_NEAR);
                        pinsrd(vDst, r32Aux, i); // Don't use vpinsrd. It zeros the rest of the YMM, ZMM registers.
                        jmp(lLoopNext, T_NEAR);
                        L(lNonZero);
                    } else {
                        je(lLoopNext, T_NEAR);
                    }
                }
                uni_vpextrd(r32Aux, vSrcShift, i);
                pinsrd(vDst, ptr[rSrcPtr + (Xbyak::Reg64)rAux], i);

                if (useMask)
                    L(lLoopNext);
            }
        }
    }

    void uni_vpgatherdd(const Xbyak::Ymm&   vDst,
                        const Xbyak::Reg64& rSrcPtr,
                        const Xbyak::Ymm&   vSrcShift,
                        const Xbyak::Ymm&   vReadMask,
                        const bool useMask  = true,
                        const bool zeroFill = false) {
        if (vDst.getIdx()== vSrcShift.getIdx() || vDst.getIdx() == vReadMask.getIdx() || vSrcShift.getIdx() == vReadMask.getIdx()) {
            IE_THROW() << "Any pair of the index, mask, or destination registers cannot be the same.";
        }
        if (isValidIsa(dnnl::impl::cpu::x64::avx2)) {
            if (!useMask)
                uni_vcmpps(vReadMask, vReadMask, vReadMask, 0x0);
            if (zeroFill)
                uni_vpxor(vDst, vDst, vDst);
            vpgatherdd(vDst, ptr[rSrcPtr + vSrcShift], vReadMask);
        } else {
            Xbyak::Xmm xmmDst      = Xbyak::Xmm(vDst.getIdx()),
                       xmmSrcShft  = Xbyak::Xmm(vSrcShift.getIdx()),
                       xmmReadMask = Xbyak::Xmm(vReadMask.getIdx());
            for (uint8_t i = 0; i < 2; i++) {
                uni_vpgatherdd(xmmDst, rSrcPtr, xmmSrcShft, xmmReadMask, useMask, zeroFill);

                vperm2f128(vDst, vDst, vDst, 0x1);
                vperm2f128(vSrcShift, vSrcShift, vSrcShift, 0x1);
                if (useMask)
                    vperm2f128(vReadMask, vReadMask, vReadMask, 0x1);
            }
        }
    }

    void uni_vpermd(const Xbyak::Ymm& vDst, const Xbyak::Ymm& vMask, const Xbyak::Operand& src) {
        if (isValidIsa(dnnl::impl::cpu::x64::avx2)) {
            vpermd(vDst, vMask, src);
        } else if (isValidIsa(dnnl::impl::cpu::x64::avx)) {

        }
    }

    void uni_vpermd(const Xbyak::Zmm& vDst, const Xbyak::Zmm& vMask, const Xbyak::Operand& src) {
        vpermd(vDst, vMask, src);
    }

    void uni_vpbroadcastd(const Xbyak::Xmm &x, const Xbyak::Operand &op) {
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

    void uni_vpbroadcastd(const Xbyak::Ymm &x, const Xbyak::Operand &op) {
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

    void fillRestWorkMask(const Xbyak::Opmask& kDstMask,
                          const Xbyak::Zmm& zAux,
                          const Xbyak::Reg64& rWorkRest) {
        auto rAux0 = r64Ref();
        auto rAux1 = r64Ref();
        Xbyak::Label lKmov;
        Xbyak::Reg32 rOnes(rAux1.getIdx());
        const uint32_t typeSize = 4;
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
        kmovw(kDstMask, rOnes);
    }

    void loadEl2vec32(const Xbyak::Xmm&   vDst,
                      const Xbyak::Reg64& rSrc,
                      const Xbyak::Reg64& rLoadNum,
                      const bool zeroFilling = false) {
        auto rAux = r64Ref();
        const int typeSize = sizeof(int);
        const int elPerVec = dnnl::impl::cpu::x64::cpu_isa_traits<dnnl::impl::cpu::x64::sse41>::vlen / typeSize;
        Xbyak::Label lLoopEnd;
        if (rLoadNum.getIdx() != rAux.getIdx())
            mov(rAux, rLoadNum);
        if (zeroFilling)
            uni_vpxor(vDst, vDst, vDst);

        for (int i = 0; i < elPerVec; i++) {
            cmp(rAux, 0);
            jle(lLoopEnd, T_NEAR);

            pinsrd(vDst, ptr[rSrc + i * typeSize], i);

            dec(rAux);
        }
        L(lLoopEnd);
    }

    void loadEl2vec32(const Xbyak::Ymm&   vDst,
                      const Xbyak::Reg64& rSrc,
                      const Xbyak::Reg64& rLoadNum,
                      const bool zeroFilling = false) {
        auto rAux = r64Ref();
        const uint8_t typeSize = sizeof(int);
        const uint8_t elPerVec = dnnl::impl::cpu::x64::cpu_isa_traits<dnnl::impl::cpu::x64::avx>::vlen / typeSize;
        Xbyak::Label lLoopEnd0, lLoopEnd1;
        if (rLoadNum.getIdx() != rAux.getIdx())
            mov(rAux, rLoadNum);
        if (zeroFilling)
            uni_vpxor(vDst, vDst, vDst);
        Xbyak::Xmm xmmDst(vDst.getIdx());

        for (uint8_t i = 0; i < elPerVec / 2; i++) {
            cmp(rAux, 0);
            je(lLoopEnd0, T_NEAR);

            pinsrd(xmmDst, ptr[rSrc + i * typeSize], i);

            dec(rAux);
        }

        vperm2f128(vDst, vDst, vDst, 0x1);
        for (uint8_t i = 0; i < elPerVec / 2; i++) {
            cmp(rAux, 0);
            je(lLoopEnd1, T_NEAR);

            pinsrd(xmmDst, ptr[rSrc + i * typeSize], i);

            dec(rAux);
        }

        L(lLoopEnd1);
        vperm2f128(vDst, vDst, vDst, 0x1);
        L(lLoopEnd0);
    }

    void storeVectorPart(const Xbyak::Reg64& rDst,
                         const Xbyak::Reg64& rToStoreCounter,
                         const Xbyak::Xmm&   xmmSrc,
                         const uint64_t      typeSize) {
        Xbyak::Label lEnd;
        const uint64_t elPerVec = dnnl::impl::cpu::x64::cpu_isa_traits<dnnl::impl::cpu::x64::sse41>::vlen / typeSize;

        for (int k = 0; k < elPerVec; k++) {
            cmp(rToStoreCounter, 0);
            jle(lEnd, T_NEAR);

            if (typeSize == 8) {
                uni_vpextrq(ptr[rDst], xmmSrc, k);
            } else if (typeSize == 4) {
                uni_vpextrd(ptr[rDst], xmmSrc, k);
            } else if (typeSize == 2) {
                uni_vpextrw(ptr[rDst], xmmSrc, k);
            } else if (typeSize == 1) {
                uni_vpextrb(ptr[rDst], xmmSrc, k);
            }

            add(rDst, typeSize);
            dec(rToStoreCounter);
        }
        L(lEnd);
    }

    void storeVectorPart(const Xbyak::Reg64& rDst,
                         const Xbyak::Reg64& rToStoreCounter,
                         const Xbyak::Ymm&   ymmSrc,
                         const uint64_t      typeSize) {
        Xbyak::Label lEnd;
        Xbyak::Xmm xmmSrc(ymmSrc.getIdx());
        for (int i = 0; i < 2; i++) {
            storeVectorPart(rDst, rToStoreCounter, xmmSrc, typeSize);

            if (i == 0) {
                cmp(rToStoreCounter, 0);
                jle(lEnd, T_NEAR);
            }

            if (isValidIsa(dnnl::impl::cpu::x64::avx2)) {
                vperm2i128(ymmSrc, ymmSrc, ymmSrc, 0x1);
            } else if (isValidIsa(dnnl::impl::cpu::x64::avx)) {
                vperm2f128(ymmSrc, ymmSrc, ymmSrc, 0x1);
            }
        }
        L(lEnd);
    }

    // Makes gather from memory under the vReadMask and writes to the XMM/m128.
    // It can fill in values not read from the source with zero.
    void maskMov32(const Xbyak::Operand& opDst,
                   const Xbyak::Operand& opSrc,
                   const Xbyak::Xmm&     xmmReadMask,
                   const Xbyak::Xmm&     xmmSrcShift,
                   const Xbyak::Reg64&   rToStoreCounter,
                   const bool useMask  = false,
                   const bool zeroMask = false) {
        Xbyak::Label lEnd;
        auto rAux = r64Ref();
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
    void maskMov32(const Xbyak::Operand& opDst,
                   const Xbyak::Operand& opSrc,
                   const Xbyak::Ymm&     vReadMask,
                   const Xbyak::Ymm&     vSrcShift,
                   const Xbyak::Reg64&   rToStoreCounter,
                   const bool useMask  = false,
                   const bool zeroMask = false) {
        Xbyak::Label lEnd;
        if (isValidIsa(dnnl::impl::cpu::x64::avx2)) {
    //        if (opDst.isYMM()) {
    //            Xbyak::Ymm vDst = Xbyak::Ymm(opDst.getIdx());
    //            if (zeroMask)
    //                uni_vpxor(vDst, vDst, vDst);
    //            uni_vpgatherdd(vDst, ptr[vSrcShift + vSrcShift], vReadMask);
    //        } else if (opDst.isREG()) {
    //            if (zeroMask)
    //                uni_vpxor(vAux, vAux, vAux);
    //            uni_vpgatherdd(vAux, ptr[vSrcShift + vSrcShift], vReadMask);
    //            if (zeroMask)
    //                uni_vmovups(ptr[opDst.getReg()], vAux);
    //            else
    //                uni_vmovups_tail(ptr[opDst.getReg()], vWriteMask, vAux);
    //        }
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
    void maskMov32(const Xbyak::Operand& opDst,
                   const Xbyak::Operand& opSrc,
                   const Xbyak::Xmm&     vReadMask,
                   const Xbyak::Xmm&     vWriteMask,
                   const Xbyak::Xmm&     vSrcShift,
                   const Xbyak::Xmm&     vAux,
                   const Xbyak::Reg64&   rAux,
                   const bool useMask  = false,
                   const bool zeroMask = false) {
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
    void maskMov32(const Xbyak::Operand& opDst,
                   const Xbyak::Operand& opSrc,
                   const Xbyak::Ymm&     vReadMask,
                   const Xbyak::Ymm&     vWriteMask,
                   const Xbyak::Ymm&     vSrcShift,
                   const Xbyak::Ymm&     vAux,
                   const Xbyak::Reg64&   rAux,
                   const bool useMask  = false,
                   const bool zeroMask = false) {
        if (isValidIsa(dnnl::impl::cpu::x64::avx2)) {
            Xbyak::Reg64 rSrc(opSrc.getIdx());
            if (opDst.isYMM()) {
                Xbyak::Ymm vDst = Xbyak::Ymm(opDst.getIdx());
                if (zeroMask)
                    uni_vpxor(vDst, vDst, vDst);
                uni_vpgatherdd(vDst, rSrc, vSrcShift, vReadMask);
            } else if (opDst.isREG()) {
                if (zeroMask)
                    uni_vpxor(vAux, vAux, vAux);
                uni_vpgatherdd(vAux, rSrc, vSrcShift, vReadMask);
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

    int getVecIdx() {
        if (vecSet.empty()) {
            IE_THROW() << "There is no available vector register.";
        }
        int idx = *vecSet.begin();
        vecSet.erase(vecSet.begin());
        return idx;
    }

    int getVecIdx(int& idx) {
        idx = getVecIdx();
        return idx;
    }

    void releaseVecIdx(int idx) {
        if (idx < 0 && idx >= vecNum) {
            IE_THROW() << "Invalid vector register index: " << idx << ".";
        }
        if (vecSet.count(idx)) {
            IE_THROW() << "Vector with index " << idx << " was already released.";
        }
        vecSet.insert(idx);
    }

    int getRegIdx() {
        if (regSet.empty()) {
            IE_THROW() << "There is no available x64 register.";
        }
        int idx = *regSet.rbegin();
        regSet.erase(*regSet.rbegin());
        return idx;
    }

    int getRegIdx(int& idx) {
        idx = getRegIdx();
        return idx;
    }

    void releaseRegIdx(int idx) {
        if (r64Pool.find(idx) == r64Pool.end()) {
            IE_THROW() << "Invalid x64 register index: " << idx << ".";
        }
        if (regSet.count(idx)) {
            IE_THROW() << "Register with index " << idx << " was already released.";
        }
        regSet.insert(idx);
    }

    const size_t vecNum = isValidIsa(dnnl::impl::cpu::x64::avx512_core) ? 32 :
                          isValidIsa(dnnl::impl::cpu::x64::avx2) || isValidIsa(dnnl::impl::cpu::x64::avx) ? 16 : 8;
    std::set<int> vecSet;

    std::unordered_map<int, Xbyak::Reg64> r64Pool = { {rdx.getIdx(), rdx}, {rbx.getIdx(), rbx}, {rbp.getIdx(), rbp}, {rsi.getIdx(), rsi},
                                                      {r8.getIdx(), r8},   {r9.getIdx(), r9},   {r10.getIdx(), r10}, {r11.getIdx(), r11},
                                                      {r12.getIdx(), r12}, {r13.getIdx(), r13}, {r14.getIdx(), r14}, {r15.getIdx(), r15} };
    std::set<int> regSet;

    template <typename Vmm>
    class vRefWrap {
        jitKernelBase* ker;
        Vmm& ref;
    public:
        vRefWrap(jitKernelBase* ker, Vmm& ref) : ker(ker), ref(ref) {}
        ~vRefWrap() {
            ker->releaseVecIdx(ref.getIdx());
        }
        operator Vmm() {
            return ref;
        }
        int getIdx() {
            return ref.getIdx();
        }
    };

    template <typename RegType>
    class rRefWrap {
        jitKernelBase* ker;
        RegType& ref;
    public:
        rRefWrap(jitKernelBase* ker, RegType& ref) : ker(ker), ref(ref) {}
        ~rRefWrap() {
            ker->releaseRegIdx(ref.getIdx());
        }
        operator RegType() {
            return ref;
        }
        int getIdx() {
            return ref.getIdx();
        }
    };
};

}
}
