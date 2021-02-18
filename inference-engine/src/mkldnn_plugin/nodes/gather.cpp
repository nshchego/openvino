// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"

#include "ie_parallel.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "common/cpu_memcpy.h"
#include <mkldnn_types.h>
#include <string>
#include <vector>

#include <chrono> // remove

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

using namespace mkldnn::impl::cpu;

struct jGatherConfParams {
    int32_t beforeAxisSize;
    int64_t indicesSize;
    int32_t axisDim;
    int64_t dictTypeSize;
    int64_t workAmount;
    int64_t idxIterator;
    int64_t start;
    int64_t end;
//    bool blockedIndices512 = false;
//    bool blockedIndices256 = false;
//    bool blockedIndices128 = false;
};

struct jGatherArgs {
    const void* src;
    void* dst;
    const int* indices;
    const int* dictTypeSize;
    const int* axisDim;
    const int* axDimSum;
    const int* shufMask8bitUni;
    const int* permMask8bitA2;
    const int* permMask8bitA5;
    const int* shufMask16bitUni;
    const int* permMask16bitA2;
    const int* permMask16bitA5;
    const int* minusOne;
    size_t idxStartB;
    size_t workAmount;
    int* tmp; // remove
    int* retVal; // remove
};

struct jitGatherKernelBase {
    void (*ker_)(const jGatherArgs *);
    void operator()(const jGatherArgs *args) {
        assert(ker_);
        ker_(args);
    }
    explicit jitGatherKernelBase(jGatherConfParams jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jitGatherKernelBase() {}

    virtual void create_ker() = 0;

    jGatherConfParams jcp_;
};

#define GET_OFF(field) offsetof(jGatherArgs, field)

template <x64::cpu_isa_t isa>
struct jitUniGatherKernel : public jitGatherKernelBase, public x64::jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jitUniGatherKernel)

    explicit jitUniGatherKernel(jGatherConfParams jcp) : jitGatherKernelBase(jcp), x64::jit_generator() {}

    void create_ker() override {
        x64::jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        this->preamble();

        mov(regSrc, ptr[regParams + GET_OFF(src)]);
        mov(regDst, ptr[regParams + GET_OFF(dst)]);
        mov(regIndices, ptr[regParams + GET_OFF(indices)]);

        mov(regIdxIter, ptr[regParams + GET_OFF(idxStartB)]);

        mov(regAux1, ptr[regParams + GET_OFF(dictTypeSize)]);
        uni_vpbroadcastd(vmmDictTypeSize, ptr[regAux1]);

        mov(regAux1, ptr[regParams + GET_OFF(axisDim)]);
        uni_vpbroadcastd(vmmAxDim, ptr[regAux1]);

        mov(regAux1, ptr[regParams + GET_OFF(axDimSum)]);
        uni_vpbroadcastd(vmmAxDimSum, ptr[regAux1]);

        mov(regAux1, ptr[regParams + GET_OFF(minusOne)]);
        uni_vpbroadcastd(vmmMinusOne, ptr[regAux1]);

        mov(regWorkAmount, ptr[regParams + GET_OFF(workAmount)]);

//        mov(regAux1, ptr[regParams + GET_OFF(tmp)]);
//        mov(regAux2, ptr[regParams + GET_OFF(retVal)]);

        elPerVec = vlen / jcp_.dictTypeSize;
        if (isa == x64::avx512_common) {
            vpcmpub(kMaskOnes, vmmGatherMask, vmmGatherMask, 0);
        }

        if (jcp_.dictTypeSize == 4) {
            Xbyak::Label lDstIdxLoop, lTail;
            L(lDstIdxLoop);
            {
                cmp(regWorkAmount, elPerVec);
                jl(lTail, T_NEAR);

                vpGatherDD(vmmDst);
                uni_vmovups(ptr[regDst], vmmDst);

                add(regDst, vlen);
                sub(regWorkAmount, elPerVec);

                jmp(lDstIdxLoop, T_NEAR);
            }
            L(lTail);
            tail();
        } else if (jcp_.dictTypeSize == 2) {
            auto& vmmShufMask = vmmAux8;
            mov(regAux1, ptr[regParams + GET_OFF(shufMask16bitUni)]);
            uni_vmovups(vmmShufMask, ptr[regAux1]);

            auto& vmmPermMask = vmmAux9;
            if (isa == x64::avx512_common) {
                mov(regAux1, ptr[regParams + GET_OFF(permMask16bitA5)]);
            } else {
                mov(regAux1, ptr[regParams + GET_OFF(permMask16bitA2)]);
            }
            uni_vmovups(vmmPermMask, ptr[regAux1]);

            Xbyak::Label lDstIdxLoop, lTail;
            L(lDstIdxLoop);
            {
                cmp(regWorkAmount, elPerVec);
                jl(lTail, T_NEAR);

                // TODO: On AVX512_VBMI can be replaced on VPERMB(VPERMB(Gather()), Gather())
                gatherAndGroup(vmmDst, vmmShufMask);
                gatherAndGroup(vmmAux4, vmmShufMask);
                vshufps(vmmDst, vmmDst, vmmAux4, 0x44);
                vpermd(vmmDst, vmmPermMask, vmmDst);

                uni_vmovups(ptr[regDst], vmmDst);

                add(regDst, vlen);
                sub(regWorkAmount, elPerVec);

                jmp(lDstIdxLoop, T_NEAR);
            }
            L(lTail);
            tail();
        } else if (jcp_.dictTypeSize == 1) {
            auto& vmmShufMask = vmmAux8;
            mov(regAux1, ptr[regParams + GET_OFF(shufMask8bitUni)]);
            uni_vmovups(vmmShufMask, ptr[regAux1]);

            auto& vmmPermMask = vmmAux9;
            if (isa == x64::avx512_common) {
                mov(regAux1, ptr[regParams + GET_OFF(permMask8bitA5)]);
            } else {
                mov(regAux1, ptr[regParams + GET_OFF(permMask8bitA2)]);
            }
            uni_vmovups(vmmPermMask, ptr[regAux1]);

            Xbyak::Label lDstIdxLoop, lTail;
            L(lDstIdxLoop);
            {
                cmp(regWorkAmount, elPerVec);
                jl(lTail, T_NEAR);

                gatherAndGroup(vmmDst, vmmShufMask);
                gatherAndGroup(vmmAux4, vmmShufMask);
                vshufps(vmmDst, vmmDst, vmmAux4, 0);

                gatherAndGroup(vmmAux4, vmmShufMask);
                gatherAndGroup(vmmAux5, vmmShufMask);
                vshufps(vmmAux4, vmmAux4, vmmAux5, 0);

                vshufps(vmmDst, vmmDst, vmmAux4, 0x88);
                vpermd(vmmDst, vmmPermMask, vmmDst);

                uni_vmovups(ptr[regDst], vmmDst);

                add(regDst, vlen);
                sub(regWorkAmount, elPerVec);

                jmp(lDstIdxLoop, T_NEAR);
            }
            L(lTail);
            tail();
        }

        this->postamble();
    }

protected:
    using Vmm = typename mkldnn::impl::utils::conditional3<isa == x64::sse41, Xbyak::Xmm, isa == x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    const uint32_t vlen = x64::cpu_isa_traits<isa>::vlen;
    const uint32_t vlenXmm = x64::cpu_isa_traits<x64::sse41>::vlen;
    const uint32_t vlenYmm = x64::cpu_isa_traits<x64::avx2>::vlen;
    int elPerVec;

    void tail() {
        Xbyak::Label lTailLoop, lCalc, lCmpTop, lPositiveIdx, lFinish;
        Xbyak::Reg32 regDictTypeSize32(regAux1.getIdx());
        Xbyak::Reg32 regAxDimSum32(regAux2.getIdx());
        Xbyak::Reg32 regAux3_32(regAux3.getIdx());
        Xbyak::Reg16 regAux3_16(regAux3.getIdx());
        Xbyak::Reg8  regAux3_8(regAux3.getIdx());
        uni_vpextrd(regDictTypeSize32, xmmDictTypeSize, 0);
        uni_vpextrd(regAxDimSum32, xmmAxDimSum, 0);
        L(lTailLoop);
        {
            cmp(regWorkAmount, 0);
            je(lFinish, T_NEAR);

            cmp(regIdxIter, jcp_.indicesSize);
            jl(lCalc, T_NEAR);
            mov(regIdxIter, 0);
            uni_vpaddd(vmmAxDimSum, vmmAxDimSum, vmmAxDim);
            uni_vpextrd(regAxDimSum32, xmmAxDimSum, 0);

            L(lCalc);
            mov(eax, ptr[regIndices + regIdxIter]);
            cmp(rax, 0);
            jge(lCmpTop, T_NEAR);
            mov(rax, 0);
            jmp(lPositiveIdx);
            L(lCmpTop);
            cmp(rax, jcp_.axisDim);
            jl(lPositiveIdx, T_NEAR);
            mov(rax, 0);
            L(lPositiveIdx);
            mul(regDictTypeSize32);
            add(eax, regAxDimSum32);
            if (jcp_.dictTypeSize == 4) {
                mov(regAux3_32, ptr[regSrc + rax]);
                mov(ptr[regDst], regAux3_32);
            } else if (jcp_.dictTypeSize == 2) {
                mov(regAux3_16, ptr[regSrc + rax]);
                mov(ptr[regDst], regAux3_16);
            } else if (jcp_.dictTypeSize == 1) {
                mov(regAux3_8, ptr[regSrc + rax]);
                mov(ptr[regDst], regAux3_8);
            }

            add(regIdxIter, sizeof(int));
            add(regDst, jcp_.dictTypeSize);
            sub(regWorkAmount, 1);
            jmp(lTailLoop, T_NEAR);
        }
        L(lFinish);
    }

    void fillIndicies(Xbyak::Xmm& dst, Xbyak::Xmm& mask) {
        Xbyak::Label lPerElements, lExit;

        cmp(regIdxIter, jcp_.indicesSize - vlenXmm);
        jg(lPerElements, T_NEAR);
            uni_vmovups(dst, ptr[regIndices + regIdxIter]);
            uni_vpmulld(dst, dst, xmmDictTypeSize);
            // Check boundaries
        vpcmpgtd(mask, dst, xmmMinusOne);
        vpcmpgtd(xmmAux1, xmmAxDim, dst);
        vpand(mask, mask, xmmAux1);
            uni_vpaddd(dst, dst, xmmAxDimSum);
            add(regIdxIter, vlenXmm);
        cmp(regIdxIter, jcp_.indicesSize);
        jl(lExit, T_NEAR);
            uni_vpaddd(vmmAxDimSum, vmmAxDimSum, vmmAxDim);
            mov(regIdxIter, 0);
        jmp(lExit, T_NEAR);

        L(lPerElements);
        for (uint8_t i = 0; i < 4; i++) {
            Xbyak::Label insertLabel;

            cmp(regIdxIter, jcp_.indicesSize);
            jl(insertLabel, T_NEAR);
                mov(regIdxIter, 0);
                uni_vpaddd(vmmAxDimSum, vmmAxDimSum, vmmAxDim);

            L(insertLabel);
            uni_vpbroadcastd(xmmAux1, ptr[regIndices + regIdxIter]);
            uni_vpmulld(xmmAux1, xmmAux1, xmmDictTypeSize);
            vpcmpgtd(xmmAux3, xmmAux1, xmmMinusOne);
            vpcmpgtd(xmmAux7, xmmAxDim, xmmAux1);
            vpand(xmmAux3, xmmAux3, xmmAux7);
            uni_vpaddd(xmmAux1, xmmAux1, xmmAxDimSum);
            vinsertps(dst, dst, xmmAux1, i << 4);
            vinsertps(mask, mask, xmmAux3, i << 4);
            add(regIdxIter, sizeof(int));
        }
        L(lExit);
    }

    void fillIndicies(Xbyak::Ymm& dst, Xbyak::Ymm& mask) {
        Xbyak::Label lPerXmm, lExit;

        cmp(regIdxIter, jcp_.indicesSize - vlenYmm);
        jg(lPerXmm, T_NEAR);
            uni_vmovups(dst, ptr[regIndices + regIdxIter]);
            uni_vpmulld(dst, dst, vmmDictTypeSize);
            // Check boundaries
        vpcmpgtd(mask, dst, vmmMinusOne);
        vpcmpgtd(vmmAux1, vmmAxDim, dst);
        vpand(mask, mask, vmmAux1);
//mov(regAux1, ptr[regParams + GET_OFF(tmp)]);
//uni_vmovups(ptr[regAux1], mask);
//        vpand(vmmSrcShifts, vmmSrcShifts, mask);
//        uni_vpcmpeqd(vmmGatherMask, vmmGatherMask, vmmGatherMask);
            uni_vpaddd(dst, dst, vmmAxDimSum);
            add(regIdxIter, vlenYmm);
        cmp(regIdxIter, jcp_.indicesSize);
        jl(lExit, T_NEAR);
            uni_vpaddd(vmmAxDimSum, vmmAxDimSum, vmmAxDim);
            mov(regIdxIter, 0);
        jmp(lExit, T_NEAR);
        L(lPerXmm);
            for (int i = 0; i < 2; i++) {
                fillIndicies(xmmAux0, xmmAux2);
                vinsertf128(dst, dst, xmmAux0, i);
                vinsertf128(mask, mask, xmmAux2, i);
            }
        L(lExit);
    }

    void fillIndicies(Xbyak::Zmm& dst, Xbyak::Opmask& mask) {
        Xbyak::Label lPerYmm, lExit;

        cmp(regIdxIter, jcp_.indicesSize - vlen);
        jg(lPerYmm, T_NEAR);
            uni_vmovups(dst, ptr[regIndices + regIdxIter]);
            uni_vpmulld(dst, dst, vmmDictTypeSize);
        vpcmpgtd(mask, dst, vmmMinusOne);
        vpcmpgtd(kMaskAux2, vmmAxDim, dst);
        kandd(mask, mask, kMaskAux2);
            uni_vpaddd(dst, dst, vmmAxDimSum);
            add(regIdxIter, vlen);
        cmp(regIdxIter, jcp_.indicesSize);
        jl(lExit, T_NEAR);
            uni_vpaddd(vmmAxDimSum, vmmAxDimSum, vmmAxDim);
            mov(regIdxIter, 0);
        jmp(lExit, T_NEAR);
        L(lPerYmm);
            for (int i = 0; i < 2; i++) {
                fillIndicies(ymmAux2, ymmAux10);
                vinsertf32x8(dst, dst, ymmAux2, i);
                vinsertf32x8(vmmAux11, vmmAux11, ymmAux10, i);
            }
            vpmovd2m(mask, ymmAux10);
        L(lExit);
    }

    void vpGatherDD(const Xbyak::Ymm& dst) {
        fillIndicies(vmmSrcShifts, vmmGatherMask);
//mov(regAux1, ptr[regParams + GET_OFF(tmp)]);
//uni_vmovups(ptr[regAux1], vmmSrcShifts);
        vpgatherdd(dst, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
    }

    void vpGatherDD(const Xbyak::Zmm& dst) {
        fillIndicies(vmmSrcShifts, kMaskAux1);
//        vpcmpub(kMaskAux1, vmmGatherMask, vmmGatherMask, 0);
        vpgatherdd(dst | kMaskAux1, ptr[regSrc + vmmSrcShifts]);
    }

    void gatherAndGroup(const Xbyak::Ymm& dst, const Xbyak::Ymm& shufMask) {
        vpGatherDD(dst);
        vpshufb(dst, dst, shufMask);
    }

    void gatherAndGroup(const Xbyak::Zmm& dst, const Xbyak::Zmm& shufMask) {
        vpGatherDD(dst);
        vpshufb(dst | kMaskOnes, dst, shufMask);
    }

    Xbyak::Reg64 regSrc = r8;
    Xbyak::Reg64 regDst = r9;
    Xbyak::Reg64 regIndices = r10;
    Xbyak::Reg64 regIdxIter = r11;
    Xbyak::Reg64 regWorkAmount = r12;
    Xbyak::Reg64 regAux1 = r13;
    Xbyak::Reg64 regAux2 = r14;
    Xbyak::Reg64 regAux3 = r15;

    Xbyak::Reg64 regParams = x64::abi_param1;

    Xbyak::Opmask kMaskOnes = Xbyak::Opmask(1);
    Xbyak::Opmask kMaskAux1 = Xbyak::Opmask(2);
    Xbyak::Opmask kMaskAux2 = Xbyak::Opmask(3);

    Xbyak::Xmm xmmAux0 = Xbyak::Xmm(0);
    Xbyak::Xmm xmmAux1 = Xbyak::Xmm(1);
    Xbyak::Xmm xmmAxDimSum = Xbyak::Xmm(2);
    Xbyak::Xmm xmmAxDim = Xbyak::Xmm(3);
    Xbyak::Xmm xmmDictTypeSize = Xbyak::Xmm(4);
    Xbyak::Xmm xmmSrcShifts = Xbyak::Xmm(5);
    Xbyak::Xmm xmmMinusOne = Xbyak::Xmm(6);
    Xbyak::Xmm xmmAux2 = Xbyak::Xmm(7);
    Xbyak::Xmm xmmAux3 = Xbyak::Xmm(8);
    Xbyak::Xmm xmmAux7 = Xbyak::Xmm(12);

    Xbyak::Ymm ymmAux2 = Xbyak::Ymm(7);
    Xbyak::Ymm ymmAux10 = Xbyak::Ymm(16);

    Vmm vmmAux0 = Vmm(0);
    Vmm vmmAux1 = Vmm(1);
    Vmm vmmAxDimSum = Vmm(2);
    Vmm vmmAxDim = Vmm(3);
    Vmm vmmDictTypeSize = Vmm(4);
    Vmm vmmSrcShifts = Vmm(5);
    Vmm vmmMinusOne = Vmm(6);
//    Vmm vmmAux2 = Vmm(7);
    Vmm vmmAux3 = Vmm(8);
    Vmm vmmAux4 = Vmm(9);
    Vmm vmmAux5 = Vmm(10);
    Vmm vmmGatherMask = Vmm(11);
    Vmm vmmAux7 = Vmm(12);
    Vmm vmmAux8 = Vmm(13);
    Vmm vmmAux9 = Vmm(14);
    Vmm vmmDst = Vmm(15);
    Vmm vmmAux11 = Vmm(17);
};

template <x64::cpu_isa_t isa>
struct jitUniGatherKernelThr : public jitGatherKernelBase, public x64::jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jitUniGatherKernelThr)

    explicit jitUniGatherKernelThr(jGatherConfParams jcp) : jitGatherKernelBase(jcp), x64::jit_generator() {}

    void create_ker() override {
        x64::jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        this->preamble();

        mov(regSrc, ptr[regParams + GET_OFF(src)]);
        mov(regDst, ptr[regParams + GET_OFF(dst)]);
        mov(regIndices, ptr[regParams + GET_OFF(indices)]);

        mov(regIdxIter, ptr[regParams + GET_OFF(idxStartB)]);

        mov(regAux1, ptr[regParams + GET_OFF(dictTypeSize)]);
        uni_vpbroadcastd(vmmDictTypeSize, ptr[regAux1]);

        mov(regAux1, ptr[regParams + GET_OFF(axisDim)]);
        uni_vpbroadcastd(vmmAxDim, ptr[regAux1]);

        mov(regAux1, ptr[regParams + GET_OFF(axDimSum)]);
        uni_vpbroadcastd(vmmAxDimSum, ptr[regAux1]);

        mov(regAux1, ptr[regParams + GET_OFF(minusOne)]);
        uni_vpbroadcastd(vmmMinusOne, ptr[regAux1]);

//        mov(regWorkAmount, ptr[regParams + GET_OFF(workAmount)]);

//        mov(regAux1, ptr[regParams + GET_OFF(tmp)]);
//        mov(regAux2, ptr[regParams + GET_OFF(retVal)]);

        const int64_t elPerVec = vlen / jcp_.dictTypeSize;
        if (isa == x64::avx512_common) {
            vpcmpub(kMaskOnes, vmmGatherMask, vmmGatherMask, 0);
        }
//        idxIterator_ = jcp_.idxIterator;
        int64_t idxIterator = jcp_.idxIterator;
//        const int64_t indicesSize = jcp_.indicesSize;

//        if (jcp_.dictTypeSize == 4) {
            int64_t workRest = jcp_.workAmount;
            for (; workRest >= elPerVec; workRest -= elPerVec) {
//                if (idxIterator <= indicesSize - vlenYmm) {
                    mov(regIdxIter, idxIterator);
//        mov(regWorkAmount, jcp_.workAmount);
//                Xbyak::Label lDstIdxLoop, lTail;
//                L(lDstIdxLoop);
//                {
//                    cmp(regWorkAmount, elPerVec);
//                    jl(lTail, T_NEAR);

                    uni_vmovups(vmmSrcShifts, ptr[regIndices + regIdxIter]);
                    uni_vpmulld(vmmSrcShifts, vmmSrcShifts, vmmDictTypeSize);
                    // Check boundaries
                    vpcmpgtd(vmmGatherMask, vmmSrcShifts, vmmMinusOne);
                    vpcmpgtd(vmmAux1, vmmAxDim, vmmSrcShifts);
                    vpand(vmmGatherMask, vmmGatherMask, vmmAux1);
//
                    uni_vpaddd(vmmSrcShifts, vmmSrcShifts, vmmAxDimSum);
                    idxIterator += vlenYmm;
//                    add(regIdxIter, vlenYmm);
//                    if (idxIterator == jcp_.indicesSize) {
//                        idxIterator = 0;
//                        uni_vpaddd(vmmAxDimSum, vmmAxDimSum, vmmAxDim);
//                    }
//                } else {
//                    for (int i = 0; i < 2; i++) {
//                        fillIndicies(xmmAux0, xmmAux2);
//                        vinsertf128(vmmSrcShifts, vmmSrcShifts, xmmAux0, i);
//                        vinsertf128(vmmGatherMask, vmmGatherMask, xmmAux2, i);
//                    }
//                }

                vpgatherdd(vmmDst, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
                uni_vmovups(ptr[regDst], vmmDst);

                add(regDst, vlen);
//                sub(regWorkAmount, elPerVec);

//                jmp(lDstIdxLoop, T_NEAR);
            }
//            L(lTail);
//            }
//            tail(workRest);
//        } else if (jcp_.dictTypeSize == 2) {
//            auto& vmmShufMask = vmmAux8;
//            mov(regAux1, ptr[regParams + GET_OFF(shufMask16bitUni)]);
//            uni_vmovups(vmmShufMask, ptr[regAux1]);
//
//            auto& vmmPermMask = vmmAux9;
//            if (isa == x64::avx512_common) {
//                mov(regAux1, ptr[regParams + GET_OFF(permMask16bitA5)]);
//            } else {
//                mov(regAux1, ptr[regParams + GET_OFF(permMask16bitA2)]);
//            }
//            uni_vmovups(vmmPermMask, ptr[regAux1]);
//
//            Xbyak::Label lDstIdxLoop, lTail;
//            L(lDstIdxLoop);
//            {
//                cmp(regWorkAmount, elPerVec);
//                jl(lTail, T_NEAR);
//
//                // TODO: On AVX512_VBMI can be replaced on VPERMB(VPERMB(Gather()), Gather())
//                gatherAndGroup(vmmDst, vmmShufMask);
//                gatherAndGroup(vmmAux4, vmmShufMask);
//                vshufps(vmmDst, vmmDst, vmmAux4, 0x44);
//                vpermd(vmmDst, vmmPermMask, vmmDst);
//
//                uni_vmovups(ptr[regDst], vmmDst);
//
//                add(regDst, vlen);
//                sub(regWorkAmount, elPerVec);
//
//                jmp(lDstIdxLoop, T_NEAR);
//            }
//            L(lTail);
//            tail();
//        } else if (jcp_.dictTypeSize == 1) {
//            auto& vmmShufMask = vmmAux8;
//            mov(regAux1, ptr[regParams + GET_OFF(shufMask8bitUni)]);
//            uni_vmovups(vmmShufMask, ptr[regAux1]);
//
//            auto& vmmPermMask = vmmAux9;
//            if (isa == x64::avx512_common) {
//                mov(regAux1, ptr[regParams + GET_OFF(permMask8bitA5)]);
//            } else {
//                mov(regAux1, ptr[regParams + GET_OFF(permMask8bitA2)]);
//            }
//            uni_vmovups(vmmPermMask, ptr[regAux1]);
//
//            Xbyak::Label lDstIdxLoop, lTail;
//            L(lDstIdxLoop);
//            {
//                cmp(regWorkAmount, elPerVec);
//                jl(lTail, T_NEAR);
//
//                gatherAndGroup(vmmDst, vmmShufMask);
//                gatherAndGroup(vmmAux4, vmmShufMask);
//                vshufps(vmmDst, vmmDst, vmmAux4, 0);
//
//                gatherAndGroup(vmmAux4, vmmShufMask);
//                gatherAndGroup(vmmAux5, vmmShufMask);
//                vshufps(vmmAux4, vmmAux4, vmmAux5, 0);
//
//                vshufps(vmmDst, vmmDst, vmmAux4, 0x88);
//                vpermd(vmmDst, vmmPermMask, vmmDst);
//
//                uni_vmovups(ptr[regDst], vmmDst);
//
//                add(regDst, vlen);
//                sub(regWorkAmount, elPerVec);
//
//                jmp(lDstIdxLoop, T_NEAR);
//            }
//            L(lTail);
//            tail();
//        }

        this->postamble();
    }

protected:
    using Vmm = typename mkldnn::impl::utils::conditional3<isa == x64::sse41, Xbyak::Xmm, isa == x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    const int64_t vlen = x64::cpu_isa_traits<isa>::vlen;
    const int64_t vlenXmm = x64::cpu_isa_traits<x64::sse41>::vlen;
    const int64_t vlenYmm = x64::cpu_isa_traits<x64::avx2>::vlen;

//    void tail(int64_t workRest) {
//        if (workRest == 0)
//            return;
////        Xbyak::Label lTailLoop, lCalc, lCmpTop, lPositiveIdx, lFinish;
//        Xbyak::Reg32 regDictTypeSize32(regAux1.getIdx());
//        Xbyak::Reg32 regAxDimSum32(regAux2.getIdx());
//        Xbyak::Reg32 regAux3_32(regAux3.getIdx());
//        Xbyak::Reg16 regAux3_16(regAux3.getIdx());
//        Xbyak::Reg8  regAux3_8(regAux3.getIdx());
//        uni_vpextrd(regDictTypeSize32, xmmDictTypeSize, 0);
//        uni_vpextrd(regAxDimSum32, xmmAxDimSum, 0);
////        L(lTailLoop);
////        {
////            cmp(regWorkAmount, 0);
////            je(lFinish, T_NEAR);
//
//        for (; workRest > 0; workRest--) {
//            Xbyak::Label lCmpTop, lPositiveIdx;
//
//            if (idxIterator_ == jcp_.indicesSize) {
////            cmp(regIdxIter, jcp_.indicesSize);
////            jl(lCalc, T_NEAR);
////            mov(regIdxIter, 0);
//                idxIterator_ = 0;
//                uni_vpaddd(vmmAxDimSum, vmmAxDimSum, vmmAxDim);
//                uni_vpextrd(regAxDimSum32, xmmAxDimSum, 0);
//            }
//
////            L(lCalc);
//            mov(regIdxIter, idxIterator_);
//            mov(eax, ptr[regIndices + regIdxIter]);
//            cmp(rax, 0);
//            jge(lCmpTop, T_NEAR);
//            mov(rax, 0);
//            jmp(lPositiveIdx);
//            L(lCmpTop);
//            cmp(rax, jcp_.axisDim);
//            jl(lPositiveIdx, T_NEAR);
//            mov(rax, 0);
//            L(lPositiveIdx);
//            mul(regDictTypeSize32);
//            add(eax, regAxDimSum32);
//            if (jcp_.dictTypeSize == 4) {
//                mov(regAux3_32, ptr[regSrc + rax]);
//                mov(ptr[regDst], regAux3_32);
//            } else if (jcp_.dictTypeSize == 2) {
//                mov(regAux3_16, ptr[regSrc + rax]);
//                mov(ptr[regDst], regAux3_16);
//            } else if (jcp_.dictTypeSize == 1) {
//                mov(regAux3_8, ptr[regSrc + rax]);
//                mov(ptr[regDst], regAux3_8);
//            }
//
//            idxIterator_ += sizeof(int);
////            add(regIdxIter, sizeof(int));
//            add(regDst, jcp_.dictTypeSize);
////            sub(regWorkAmount, 1);
////            jmp(lTailLoop, T_NEAR);
//        }
////        L(lFinish);
//    }
//
//    void fillIndicies(Xbyak::Xmm& dst, Xbyak::Xmm& mask) {
//        if (idxIterator_ <= jcp_.indicesSize - vlenXmm) {
//            mov(regIdxIter, idxIterator_);
//            uni_vmovups(dst, ptr[regIndices + regIdxIter]);
//            uni_vpmulld(dst, dst, xmmDictTypeSize);
//            // Check boundaries
//            vpcmpgtd(mask, dst, xmmMinusOne);
//            vpcmpgtd(xmmAux1, xmmAxDim, dst);
//            vpand(mask, mask, xmmAux1);
//
//            uni_vpaddd(dst, dst, xmmAxDimSum);
//            idxIterator_ += vlenXmm;
//            if (idxIterator_ == jcp_.indicesSize) {
//                uni_vpaddd(vmmAxDimSum, vmmAxDimSum, vmmAxDim);
//                idxIterator_ = 0;
//            }
//        } else {
//            for (uint8_t i = 0; i < 4; i++) {
//                if (idxIterator_ == jcp_.indicesSize) {
//                    idxIterator_ = 0;
//                    uni_vpaddd(vmmAxDimSum, vmmAxDimSum, vmmAxDim);
//                }
//
//                mov(regIdxIter, idxIterator_);
//                uni_vpbroadcastd(xmmAux1, ptr[regIndices + regIdxIter]);
//                uni_vpmulld(xmmAux1, xmmAux1, xmmDictTypeSize);
//                // Check boundaries
//                vpcmpgtd(xmmAux3, xmmAux1, xmmMinusOne);
//                vpcmpgtd(xmmAux7, xmmAxDim, xmmAux1);
//                vpand(xmmAux3, xmmAux3, xmmAux7);
//
//                uni_vpaddd(xmmAux1, xmmAux1, xmmAxDimSum);
//                vinsertps(dst, dst, xmmAux1, i << 4);
//                vinsertps(mask, mask, xmmAux3, i << 4);
//                idxIterator_ += sizeof(int);
//            }
//        }
//    }
//
//    void fillIndicies(Xbyak::Ymm& dst, Xbyak::Ymm& mask) {
//        if (idxIterator_ <= jcp_.indicesSize - vlenYmm) {
//            mov(regIdxIter, idxIterator_);
//            uni_vmovups(dst, ptr[regIndices + regIdxIter]);
////mov(regAux1, ptr[regParams + GET_OFF(tmp)]);
////uni_vmovups(ptr[regAux1], dst);
//            uni_vpmulld(dst, dst, vmmDictTypeSize);
//            // Check boundaries
//            vpcmpgtd(mask, dst, vmmMinusOne);
//            vpcmpgtd(vmmAux1, vmmAxDim, dst);
//            vpand(mask, mask, vmmAux1);
//
//            uni_vpaddd(dst, dst, vmmAxDimSum);
//            idxIterator_ += vlenYmm;
//            if (idxIterator_ == jcp_.indicesSize) {
//                idxIterator_ = 0;
//                uni_vpaddd(vmmAxDimSum, vmmAxDimSum, vmmAxDim);
//            }
//        } else {
//            for (int i = 0; i < 2; i++) {
//                fillIndicies(xmmAux0, xmmAux2);
//                vinsertf128(dst, dst, xmmAux0, i);
//                vinsertf128(mask, mask, xmmAux2, i);
//            }
//        }
//    }
//
//    void fillIndicies(Xbyak::Zmm& dst, Xbyak::Opmask& mask) {
////        Xbyak::Label lPerYmm, lExit;
////
////        cmp(regIdxIter, jcp_.indicesSize - vlen);
////        jg(lPerYmm, T_NEAR);
////            uni_vmovups(dst, ptr[regIndices + regIdxIter]);
////            uni_vpmulld(dst, dst, vmmDictTypeSize);
////        vpcmpgtd(mask, dst, vmmMinusOne);
////        vpcmpgtd(kMaskAux2, vmmAxDim, dst);
////        kandd(mask, mask, kMaskAux2);
////            uni_vpaddd(dst, dst, vmmAxDimSum);
////            add(regIdxIter, vlen);
////        cmp(regIdxIter, jcp_.indicesSize);
////        jl(lExit, T_NEAR);
////            uni_vpaddd(vmmAxDimSum, vmmAxDimSum, vmmAxDim);
////            mov(regIdxIter, 0);
////        jmp(lExit, T_NEAR);
////        L(lPerYmm);
////            for (int i = 0; i < 2; i++) {
////                fillIndicies(ymmAux2, ymmAux10);
////                vinsertf32x8(dst, dst, ymmAux2, i);
////                vinsertf32x8(vmmAux11, vmmAux11, ymmAux10, i);
////            }
////            vpmovd2m(mask, ymmAux10);
////        L(lExit);
//    }
//
//    void vpGatherDD(const Xbyak::Ymm& dst) {
//        fillIndicies(vmmSrcShifts, vmmGatherMask);
////mov(regAux1, ptr[regParams + GET_OFF(tmp)]);
////uni_vmovups(ptr[regAux1], vmmSrcShifts);
//        vpgatherdd(dst, ptr[regSrc + vmmSrcShifts], vmmGatherMask);
//    }
//
//    void vpGatherDD(const Xbyak::Zmm& dst) {
//        fillIndicies(vmmSrcShifts, kMaskAux1);
////        vpcmpub(kMaskAux1, vmmGatherMask, vmmGatherMask, 0);
//        vpgatherdd(dst | kMaskAux1, ptr[regSrc + vmmSrcShifts]);
//    }
//
//    void gatherAndGroup(const Xbyak::Ymm& dst, const Xbyak::Ymm& shufMask) {
//        vpGatherDD(dst);
//        vpshufb(dst, dst, shufMask);
//    }
//
//    void gatherAndGroup(const Xbyak::Zmm& dst, const Xbyak::Zmm& shufMask) {
//        vpGatherDD(dst);
//        vpshufb(dst | kMaskOnes, dst, shufMask);
//    }

//    int64_t idxIterator_;

    Xbyak::Reg64 regSrc = r8;
    Xbyak::Reg64 regDst = r9;
    Xbyak::Reg64 regIndices = r10;
    Xbyak::Reg64 regIdxIter = r11;
    Xbyak::Reg64 regWorkAmount = r12;
    Xbyak::Reg64 regAux1 = r13;
    Xbyak::Reg64 regAux2 = r14;
    Xbyak::Reg64 regAux3 = r15;

    Xbyak::Reg64 regParams = x64::abi_param1;

    Xbyak::Opmask kMaskOnes = Xbyak::Opmask(1);
    Xbyak::Opmask kMaskAux1 = Xbyak::Opmask(2);
    Xbyak::Opmask kMaskAux2 = Xbyak::Opmask(3);

    Xbyak::Xmm xmmAux0 = Xbyak::Xmm(0);
    Xbyak::Xmm xmmAux1 = Xbyak::Xmm(1);
    Xbyak::Xmm xmmAxDimSum = Xbyak::Xmm(2);
    Xbyak::Xmm xmmAxDim = Xbyak::Xmm(3);
    Xbyak::Xmm xmmDictTypeSize = Xbyak::Xmm(4);
    Xbyak::Xmm xmmSrcShifts = Xbyak::Xmm(5);
    Xbyak::Xmm xmmMinusOne = Xbyak::Xmm(6);
    Xbyak::Xmm xmmAux2 = Xbyak::Xmm(7);
    Xbyak::Xmm xmmAux3 = Xbyak::Xmm(8);
    Xbyak::Xmm xmmAux7 = Xbyak::Xmm(12);

    Xbyak::Ymm ymmAux2 = Xbyak::Ymm(7);
    Xbyak::Ymm ymmAux10 = Xbyak::Ymm(16);

    Vmm vmmAux0 = Vmm(0);
    Vmm vmmAux1 = Vmm(1);
    Vmm vmmAxDimSum = Vmm(2);
    Vmm vmmAxDim = Vmm(3);
    Vmm vmmDictTypeSize = Vmm(4);
    Vmm vmmSrcShifts = Vmm(5);
    Vmm vmmMinusOne = Vmm(6);
//    Vmm vmmAux2 = Vmm(7);
    Vmm vmmAux3 = Vmm(8);
    Vmm vmmAux4 = Vmm(9);
    Vmm vmmAux5 = Vmm(10);
    Vmm vmmGatherMask = Vmm(11);
    Vmm vmmAux7 = Vmm(12);
    Vmm vmmAux8 = Vmm(13);
    Vmm vmmAux9 = Vmm(14);
    Vmm vmmDst = Vmm(15);
    Vmm vmmAux11 = Vmm(17);
};


class GatherImpl: public ExtLayerBase {
public:
    explicit GatherImpl(const CNNLayer* layer) {
        std::string errPrefix = std::string("Gather layer with name '") + layer->name + "' ";
        if (layer->insData.size() < 2 || layer->insData.size() > 3)
            THROW_IE_EXCEPTION << errPrefix << "has incorrect number of input edges: " << layer->insData.size();
        if (layer->outData.size() != 1)
            THROW_IE_EXCEPTION << errPrefix << "has incorrect number of output edges: " << layer->outData.size();

        auto dictData = layer->insData[GATHER_DICTIONARY].lock();
        auto idxData = layer->insData[GATHER_INDEXES].lock();
        if (!dictData || !idxData)
            THROW_IE_EXCEPTION << errPrefix << "has nullable input data.";

        if (layer->insData.size() > GATHER_AXIS) {
            // TODO: implemnt when will be available via ngraph node
//            layer->insData[GATHER_AXIS]->getCreatorLayer()->blobs.begin()->second->cbuffer().as<const index_t *>() +
//            indexes->getTensorDesc().getBlockingDesc().getOffsetPadding();
        } else {
            axis_ = layer->GetParamAsInt("axis");
        }

        const SizeVector& dictionaryDims = dictData->getTensorDesc().getDims();
        if (dictionaryDims.size() == 0)
            THROW_IE_EXCEPTION << errPrefix << "has incorrect input dictionary dimension.";
        // Dictionary must be at least rank axis + 1
        IE_ASSERT(-static_cast<int>(dictionaryDims.size()) <= axis_ && axis_ < static_cast<int>(dictionaryDims.size()))
            << errPrefix << "has incorrect axis value!";
        if (axis_ < 0)
            axis_ += dictionaryDims.size();

        //  Find number of dictionaries, index range and data length
        for (int i = 0; i < axis_; i++)
            beforeAxisSize_ *= dictionaryDims[i];
        axisDim_ = dictionaryDims[axis_];
        afterAxisSize_ = 1lu;
        for (size_t i = axis_ + 1; i < dictionaryDims.size(); i++)
            afterAxisSize_ *= dictionaryDims[i];

        if (afterAxisSize_ == 0)
            THROW_IE_EXCEPTION << errPrefix << "has incorrect input parameters dimension.";

        const SizeVector& indexesDims = idxData->getTensorDesc().getDims();
        indicesSize_ = std::accumulate(indexesDims.begin(), indexesDims.end(), 1, std::multiplies<size_t>());

        Precision dictPrecision = dictData->getTensorDesc().getPrecision();
        if (dictPrecision == Precision::BF16 && !x64::mayiuse(x64::avx512_common)) {
            dictPrecision = Precision::FP32;
        }
        const Precision idxPrecision(Precision::I32);

        LayerConfig config;
        DataConfig dataConfigIdx, dataConfigDct;
        dataConfigDct.desc = TensorDesc(dictPrecision, dictionaryDims,
                dictData->getTensorDesc().getLayoutByDims(dictionaryDims));
        config.inConfs.push_back(dataConfigDct);
        dataConfigIdx.desc = TensorDesc(idxPrecision, indexesDims,
                idxData->getTensorDesc().getLayout());
        config.inConfs.push_back(dataConfigIdx);

        DataConfig dataConfigOut;
        const SizeVector& outDims = layer->outData[0]->getTensorDesc().getDims();
        dataConfigOut.desc = TensorDesc(dictPrecision, outDims,
                layer->outData[0]->getTensorDesc().getLayoutByDims(outDims));
        config.outConfs.push_back(dataConfigOut);
        config.dynBatchSupport = false;
        confs.push_back(config);

        dictTypeSize_ = dictPrecision.size();

        // Gather instruction is applicable just for 32 and 64 bit data and is not supported by SSE.
        if ((x64::mayiuse(x64::avx512_common) || x64::mayiuse(x64::avx2)) &&
                afterAxisSize_ == 1) {
            jGatherConfParams jcp;
            jcp.beforeAxisSize = beforeAxisSize_;
            jcp.indicesSize = indicesSize_ * idxPrecision.size();
            jcp.dictTypeSize = dictTypeSize_;
            jcp.axisDim = axisDim_;
            threadsNum_ = parallel_get_max_threads();
//            const auto vlen512 = x64::cpu_isa_traits<x64::avx512_common>::vlen;
            const auto vlen256 = x64::cpu_isa_traits<x64::avx2>::vlen;
//            const auto vlen128 = x64::cpu_isa_traits<x64::sse41>::vlen;
            if (x64::mayiuse(x64::avx512_common)) {
//                if (threadsNum > 2 && indicesSize_ >= 4) {
//                    if (indicesSize_ >= vlen512)  {
//                        if (indicesSize_ % vlen512 == 0) {
//                            jcp.blockedIndices512 = true;
//                        }
//                    } else if (indicesSize_ >= 32 && indicesSize_ % 32 == 0) {
//                        jcp.blockedIndices256 = true;
//                    } else if (indicesSize_ >= 16 && indicesSize_ % 16 == 0) {
//                        jcp.blockedIndices128 = true;
//                    }
//                    idxPrecision.size();
//                }
                jKernel_.reset(new jitUniGatherKernel<x64::avx512_common>(jcp));
            } else if (x64::mayiuse(x64::avx2)) {
                size_t workAmount = beforeAxisSize_ * indicesSize_;
                const size_t elPerVec = vlen256 / dictTypeSize_;
                if (workAmount >= elPerVec) {
                    jKernels_.resize(threadsNum_);
//                    const size_t wRest = (workAmount / vlen256) % (threadsNum - 1);
                    const size_t wpt  = ((workAmount / elPerVec) / threadsNum_ + 1) * elPerVec;

//printf("LAYER: %s\n", layer->name.c_str());
                    auto threadBody = [&](const int ithr, const int nthr) {
                        jGatherConfParams jcpThr = jcp;
                        jcpThr.start = std::min(wpt * ithr, workAmount);
                        jcpThr.end = std::min(wpt * (ithr + 1), workAmount);
                        jcpThr.workAmount = jcpThr.end - jcpThr.start;
//                        jcpThr.workAmount = std::min(wpt * (ithr + 1), workAmount) -
//                                std::min(wpt * ithr, workAmount);
                        jcpThr.idxIterator = (jcpThr.start % indicesSize_) * idxPrecision.size();
//printf("[%d] WA: %lu start: %ld; end: %ld; wa: %ld; ii: %ld\n", ithr, workAmount, jcpThr.start, jcpThr.end, jcpThr.workAmount, jcpThr.idxIterator);
                        jKernels_[ithr].reset(new jitUniGatherKernelThr<x64::avx2>(jcpThr));
                        jKernels_[ithr]->create_ker();
                    };
                    parallel_nt(threadsNum_, threadBody);
                } else {
                    jKernel_.reset(new jitUniGatherKernel<x64::avx2>(jcp));
                }
            }
            if (jKernel_)
                jKernel_->create_ker();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        switch (dictTypeSize_) {
            case sizeof(PrecisionTrait<Precision::I32>::value_type):
                return gather<PrecisionTrait<Precision::I32>::value_type>(inputs, outputs, resp);
            case sizeof(PrecisionTrait<Precision::I16>::value_type):
                return gather<PrecisionTrait<Precision::I16>::value_type>(inputs, outputs, resp);
            case sizeof(PrecisionTrait<Precision::I8>::value_type):
                return gather<PrecisionTrait<Precision::I8>::value_type>(inputs, outputs, resp);
            default:
                std::string errMsg = std::string("Gather layer has inputData with unsupported precision: ") +
                    inputs[GATHER_DICTIONARY]->getTensorDesc().getPrecision().name();
                errMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                return GENERAL_ERROR;
        }
    }

private:
    template <typename dataType>
    StatusCode gather(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) {
        auto& dictionary = inputs[GATHER_DICTIONARY];
        const int* srcIndices = inputs[GATHER_INDEXES]->cbuffer().as<const int*>() +
            inputs[GATHER_INDEXES]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        auto& output = outputs[0];

static unsigned c1 = 0;
static double t1 = 0.0;
c1++;
auto start1 = std::chrono::steady_clock::now();

        if (afterAxisSize_ == 1) {
            const dataType* srcDictData = dictionary->cbuffer().as<const dataType *>() + dictionary->getTensorDesc().getBlockingDesc().getOffsetPadding();
            dataType *dstData = output->buffer().as<dataType*>() + output->getTensorDesc().getBlockingDesc().getOffsetPadding();

            size_t workAmount = beforeAxisSize_ * indicesSize_;
            if (!jKernels_.empty()) {
                auto threadBody = [&](const int ithr, const int nthr) {
                    auto& ker = jKernels_[ithr];
                    size_t basStart = (ker->jcp_.start / indicesSize_) % beforeAxisSize_;
                    size_t idxStart = ker->jcp_.start % indicesSize_;
                    const int dictTypeSize = dictTypeSize_;
                    const int axisDimB = axisDim_ * dictTypeSize_;
                    const int axDimSumB = axisDimB * basStart;
                    const int minusOne = -1;
//printf("[%d] start: %ld; basStart: %lu; idxStart: %lu; axisDimB: %d; axDimSumB: %d\n", ithr, ker->jcp_.start, basStart, idxStart, axisDimB, axDimSumB);
//int tmp[16] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
//int retVal = 0;

                    auto arg = jGatherArgs();
                    arg.src = srcDictData;
                    arg.dst = dstData + basStart * indicesSize_ + idxStart;
                    arg.indices = srcIndices;
                    arg.dictTypeSize = &dictTypeSize;
                    arg.axisDim = &axisDimB;
                    arg.axDimSum = &axDimSumB;
                    arg.idxStartB = idxStart * sizeof(int);
                    arg.shufMask8bitUni  = shufMask8bitUni_;
                    arg.permMask8bitA2   = permMask8bitA2_;
                    arg.permMask8bitA5   = permMask8bitA5_;
                    arg.shufMask16bitUni = shufMask16bitUni_;
                    arg.permMask16bitA2  = permMask16bitA2_;
                    arg.permMask16bitA5  = permMask16bitA5_;
                    arg.minusOne = &minusOne;
//                    arg.workAmount = end - start;
//                    arg.tmp = tmp;
//                    arg.retVal = &retVal;
                    (*ker)(&arg);
//    std::string tmpStr = "tmp: ";
//for (int s = 0; s < 16; s++) {
//    tmpStr += std::to_string(tmp[s]) + "; ";
//}
//printf("%s\n", tmpStr.c_str());
//printf("retVal: %d\n", retVal);
                };
                parallel_nt(threadsNum_, threadBody);
            } else if (jKernel_) {
                auto threadBody = [&](const int ithr, const int nthr) {
                    size_t start(0lu), end(0lu);
                    splitter(workAmount, nthr, ithr, start, end);
                    if (start >= end)
                        return;
                    size_t basStart = 0lu, idxStart = 0lu;
                    parallel_it_init(start, basStart, beforeAxisSize_, idxStart, indicesSize_);
//                    if (ithr > 0)
//                        return;
//printf("[%d] start: %lu; end: %lu; basStart: %lu; idxStart: %lu\n", ithr, start, end, basStart, idxStart);
//int tmp[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
//int retVal = 0;

                    const int dictTypeSize = dictTypeSize_;
                    const int axisDimB = axisDim_ * dictTypeSize_;
                    const int axDimSumB = axisDimB * basStart;
                    const int minusOne = -1;

                    auto arg = jGatherArgs();
                    arg.src = srcDictData;
                    arg.dst = dstData + basStart * indicesSize_ + idxStart;
                    arg.indices = srcIndices;
                    arg.dictTypeSize = &dictTypeSize;
                    arg.axisDim = &axisDimB;
                    arg.axDimSum = &axDimSumB;
                    arg.idxStartB = idxStart * sizeof(int);
                    arg.shufMask8bitUni  = shufMask8bitUni_;
                    arg.permMask8bitA2   = permMask8bitA2_;
                    arg.permMask8bitA5   = permMask8bitA5_;
                    arg.shufMask16bitUni = shufMask16bitUni_;
                    arg.permMask16bitA2  = permMask16bitA2_;
                    arg.permMask16bitA5  = permMask16bitA5_;
                    arg.minusOne = &minusOne;
                    arg.workAmount = end - start;
//                    arg.tmp = tmp;
//                    arg.retVal = &retVal;
                    (*jKernel_)(&arg);
//    std::string tmpStr = "tmp: ";
//for (int s = 0; s < 16; s++) {
//    tmpStr += std::to_string(tmp[s]) + "; ";
//}
//printf("%s\n", tmpStr.c_str());
//printf("retVal: %d\n", retVal);
                };

                parallel_nt(0, threadBody);
            } else {
                auto threadBody = [&](const int ithr, const int nthr) {
                    size_t start(0lu), end(0lu);
                    splitter(workAmount, nthr, ithr, start, end);
                    if (start >= end)
                        return;
                    size_t basStart = 0lu, idxStart = 0lu;
                    parallel_it_init(start, basStart, beforeAxisSize_, idxStart, indicesSize_);

                    for (size_t i = basStart; i < beforeAxisSize_ && start < end; i++) {
                        const dataType* srcDictDataShifted = srcDictData + i * axisDim_;
                        dataType* dstDataShifted = dstData + i * indicesSize_;
                        for (size_t j = idxStart; j < indicesSize_ && start < end; j++, start++) {
                            dstDataShifted[j] = srcDictDataShifted[srcIndices[j]];
                        }
                        idxStart = 0lu;
                    }
                };

                parallel_nt(0, threadBody);
            }
        } else {
            const uint8_t *srcDictData = dictionary->cbuffer().as<const uint8_t *>() + dictionary->getTensorDesc().getBlockingDesc().getOffsetPadding();
            uint8_t* dstData = output->cbuffer().as<uint8_t*>() + output->getTensorDesc().getBlockingDesc().getOffsetPadding();
            const size_t len = afterAxisSize_ * dictionary->getTensorDesc().getPrecision().size();
            const size_t idxMlt = len * axisDim_;
            const size_t lenSrcIndexSize = len * indicesSize_;
            parallel_for(indicesSize_, [&](size_t i) {
                //  Index clipping
                size_t len_i = len * i;
                size_t dstSize = output->byteSize() - len_i;
                if (srcIndices[i] < axisDim_) {
                    size_t idxShift = len * srcIndices[i];
                    uint8_t* dstDataShifted = dstData + len_i;
                    const uint8_t* srcDictDataShifted = srcDictData + idxShift;
                    for (size_t j = 0; j < beforeAxisSize_; j++) {
                        size_t jlenSrcIndexSize = j * lenSrcIndexSize;
                        cpu_memcpy_s(dstDataShifted + jlenSrcIndexSize,
                                    dstSize - jlenSrcIndexSize,
                                    srcDictDataShifted + j * idxMlt,
                                    len);
                    }
                } else {
                    for (size_t j = 0; j < beforeAxisSize_; j++) {
                        memset(&dstData[len * (i + j * indicesSize_)], 0, len);
                    }
                }
            });
        }

auto end1 = std::chrono::steady_clock::now();
t1 += std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();
if (c1 % 1000 == 0) {
    std::cout << "GE PARALLEL SECTION: " << t1 / c1 << std::endl;
}

        return OK;
    }

    const size_t GATHER_DICTIONARY = 0;
    const size_t GATHER_INDEXES = 1;
    const size_t GATHER_AXIS = 2;

    int axis_ = 0;
    size_t beforeAxisSize_ = 1lu;
    size_t axisDim_ = 0lu;
    size_t afterAxisSize_ = 1lu;
    size_t indicesSize_ = 1lu;
    size_t dictTypeSize_ = 1lu;
    int threadsNum_;

    const int shufMask8bitUni_[16]  = {0x0C080400, 0x80808080, 0x80808080, 0x80808080, 0x0C080400, 0x80808080, 0x80808080, 0x80808080,
                                       0x0C080400, 0x80808080, 0x80808080, 0x80808080, 0x0C080400, 0x80808080, 0x80808080, 0x80808080};
    const int permMask8bitA2_[8]    = {0, 4, 1, 5, 2, 6, 3, 7};
    const int permMask8bitA5_[16]   = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
    const int shufMask16bitUni_[16] = {0x05040100, 0x0D0C0908, 0x80808080, 0x80808080, 0x05040100, 0x0D0C0908, 0x80808080, 0x80808080,
                                       0x05040100, 0x0D0C0908, 0x80808080, 0x80808080, 0x05040100, 0x0D0C0908, 0x80808080, 0x80808080};
    const int permMask16bitA2_[8]   = {0, 1, 4, 5, 2, 3, 6, 7};
    const int permMask16bitA5_[16]  = {0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15};

    std::shared_ptr<jitGatherKernelBase> jKernel_;
    std::vector<std::shared_ptr<jitGatherKernelBase>> jKernels_;
};


REG_FACTORY_FOR(GatherImpl, Gather);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
