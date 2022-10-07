// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu/x64/jit_generator.hpp"
#include <ie_common.h>
#include <dnnl_types.h>
#include "utils/general_utils.h"

#include <set>

namespace ov {
namespace intel_cpu {

template <typename Vmm>
class vRefWrap;

#define r64Ref() rRefWrap<Xbyak::Reg64>(this, r64Pool[getRegIdx()])

class JitKernelBase: public dnnl::impl::cpu::x64::jit_generator {
protected:

    JitKernelBase();

    inline bool isValidIsa(dnnl::impl::cpu::x64::cpu_isa_t isa) {
        return is_subset(isa, dnnl::impl::cpu::x64::isa_all) && dnnl::impl::cpu::x64::mayiuse(isa);
    }

    void uni_vfmsub132ps(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2, const Xbyak::Operand &op);

    void uni_vfnmadd132ps(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2, const Xbyak::Operand &op);

    void uni_vfmsub231ps(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2, const Xbyak::Operand &op);

    void uni_vpaddd(const Xbyak::Ymm& vDst, const Xbyak::Ymm& vSrc, const Xbyak::Operand& op);

    void uni_vpsubd(const Xbyak::Ymm& vDst, const Xbyak::Ymm& vSrc, const Xbyak::Operand &op);

    void uni_kmovd(const Xbyak::Opmask& kDst, const Xbyak::Opmask& kSrc) {
        kmovd(kDst, kSrc);
    }

    void uni_kmovd(const Xbyak::Xmm& vDst, const Xbyak::Xmm& vSrc) {
        uni_vmovups(vDst, vSrc);
    }

    void uni_kxnorw(const Xbyak::Opmask& kDst, const Xbyak::Opmask& kSrc1, const Xbyak::Opmask& kSrc2) {
        kxnorw(kDst, kSrc1, kSrc2);
    }

    void uni_kandd(const Xbyak::Opmask& kDst, const Xbyak::Opmask& kSrc1, const Xbyak::Opmask& kSrc2) {
        kandd(kDst, kSrc1, kSrc2);
    }

    void uni_kandd(const Xbyak::Xmm& kDst, const Xbyak::Xmm& kSrc1, const Xbyak::Xmm& kSrc2);

    void uni_kxnorw(const Xbyak::Xmm& vDst, const Xbyak::Xmm& vSrc1, const Xbyak::Xmm& vSrc2) {
        uni_vpxor(vDst, vSrc1, vSrc2);
        if (dnnl::impl::cpu::x64::is_subset(dnnl::impl::cpu::x64::avx, dnnl::impl::cpu::x64::isa_all) &&
                  dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx)) {
            vandnps(vDst, vSrc1, vSrc2);
        } else {
            andnps(vDst, vSrc1);
        }
    }

    void gatherdd(const Xbyak::Xmm&    vDst,
                  const Xbyak::Reg64&  rSrcPtr,
                  const Xbyak::Xmm&    vSrcShift,
                  const Xbyak::Opmask& kReadMask,
                  const bool useMask   = true,
                  const bool zeroFill  = false);

    void gatherdd(const Xbyak::Xmm&   vDst,
                  const Xbyak::Reg64& rSrcPtr,
                  const Xbyak::Xmm&   vSrcShift,
                  const Xbyak::Xmm&   vReadMask,
                  const bool useMask  = true,
                  const bool zeroFill = false);

    void gatherdd(const Xbyak::Ymm&   vDst,
                  const Xbyak::Reg64& rSrcPtr,
                  const Xbyak::Ymm&   vSrcShift,
                  const Xbyak::Ymm&   vReadMask,
                  const bool useMask  = true,
                  const bool zeroFill = false);

    void uni_vpermd(const Xbyak::Ymm& vDst, const Xbyak::Ymm& vMask, const Xbyak::Operand& src) {
        if (isValidIsa(dnnl::impl::cpu::x64::avx2)) {
            vpermd(vDst, vMask, src);
        } else if (isValidIsa(dnnl::impl::cpu::x64::avx)) {

        }
    }

    void uni_vpermd(const Xbyak::Zmm& vDst, const Xbyak::Zmm& vMask, const Xbyak::Operand& src) {
        vpermd(vDst, vMask, src);
    }

    void uni_vpbroadcastd(const Xbyak::Xmm &x, const Xbyak::Operand &op);

    void uni_vpbroadcastd(const Xbyak::Ymm &x, const Xbyak::Operand &op);

    void fillRestWorkMask(const Xbyak::Opmask& kDstMask,
                          const Xbyak::Zmm& zAux,
                          const Xbyak::Reg64& rWorkRest);

    void load(const Xbyak::Xmm&   vDst,
              const Xbyak::Reg64& rSrc,
              const Xbyak::Reg64& rLoadNum,
              const uint8_t       typeSize,
              const bool zeroFilling = false);

    void load(const Xbyak::Ymm&   vDst,
              const Xbyak::Reg64& rSrc,
              const Xbyak::Reg64& rLoadNum,
              const uint8_t       typeSize,
              const bool zeroFilling = false);

    void store(const Xbyak::Reg64& rDst,
               const Xbyak::Reg64& rToStoreNum,
               const Xbyak::Xmm&   xmmSrc,
               const size_t        typeSize,
               const size_t        dstOffset = 0lu);

    void store(const Xbyak::Reg64& rDst,
               const Xbyak::Reg64& rToStoreNum,
               const Xbyak::Ymm&   vSrc,
               const size_t        typeSize,
               const size_t        dstOffset = 0lu);

    // Makes gather from memory under the vReadMask and writes to the XMM/m128.
    // It can fill in values not read from the source with zero.
    void maskMov32(const Xbyak::Operand& opDst,
                   const Xbyak::Operand& opSrc,
                   const Xbyak::Xmm&     xmmReadMask,
                   const Xbyak::Xmm&     xmmSrcShift,
                   const Xbyak::Reg64&   rToStoreCounter,
                   const bool useMask  = false,
                   const bool zeroMask = false);

    // Makes gather from memory under the vReadMask and writes to the YMM/m256.
    // It can fill in values not read from the source with zero.
    void maskMov32(const Xbyak::Operand& opDst,
                   const Xbyak::Operand& opSrc,
                   const Xbyak::Ymm&     vReadMask,
                   const Xbyak::Ymm&     vSrcShift,
                   const Xbyak::Reg64&   rToStoreCounter,
                   const bool useMask  = false,
                   const bool zeroMask = false);

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
                   const bool zeroMask = false);

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
                   const bool zeroMask = false);

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
        JitKernelBase* ker;
        Vmm& ref;
    public:
        vRefWrap(JitKernelBase* ker, Vmm& ref) : ker(ker), ref(ref) {}
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
         JitKernelBase* ker;
        RegType& ref;
    public:
        rRefWrap(JitKernelBase* ker, RegType& ref) : ker(ker), ref(ref) {}
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
