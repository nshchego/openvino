
#include "unique.hpp"

using namespace dnnl::impl::cpu;
using InferenceEngine::Precision;

namespace ov {
namespace intel_cpu {

template <x64::cpu_isa_t isa>
void UniqueKernel<isa>::generate() {
    this->preamble();
    registersPool = RegistersPool::create(isa, {rax, rcx, rsp, rdi, k0});

    initVectors();
    process();

    registersPool.reset();
    this->postamble();
}

template <>
void UniqueKernel<x64::avx512_core>::initVectors() {
    auto rAux = getReg64();
    Xbyak::Reg32 rMask(rAux.getIdx()); // TODO use r64

    kMask0 = getMask();
    mov(rMask, 0B0101010101010101);
    kmovw(kMask0, rMask);

    kMask1 = getMask();
    mov(rMask, 0B1010101010101010);
    kmovw(kMask1, rMask);

    kMaskMinLast = getMask();
    mov(rMask, 0B0010101010101010);
    kmovw(kMaskMinLast, rMask);

    kMaskMaxFirst = getMask();
    mov(rMask, 0B0101010101010100);
    kmovw(kMaskMaxFirst, rMask);

    kFirstElMask = getMask();
    mov(rMask, 0B0000000000000001);
    kmovw(kFirstElMask, rMask);

    kLastElMask = getMask();
    mov(rMask, 0B1000000000000000);
    kmovw(kLastElMask, rMask);

    kTailMask = getMask();
    mov(rMask, 0xFFFFFFFF);
    kmovw(kTailMask, rMask);

    static const unsigned permElem[16]  = { 15, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 0 };
    mov(rAux, reinterpret_cast<uintptr_t>(permElem));
    vPermElem = getVmm();
    uni_vmovups(vPermElem, ptr[rAux]);

    const auto vecNum = registersPool->countFree<Vmm>() - 3;//2;
    for (int i = 0; i < vecNum; i++) {
        contiguousVec.push_back(getVmm());
    }
}

template <x64::cpu_isa_t isa>
void UniqueKernel<isa>::process() {
    sortInBlocks();
}

template <>
void UniqueKernel<x64::avx512_core>::alignTailMask(const Vmask& kDst, const Vmask& kSrc, bool even) {
    Xbyak::Label lEnd, lCopy;
    auto rAux = getReg64();

    kmovq(rAux, kSrc);
    popcnt(rAux, rAux);
    and_(rAux, 0x1);
    cmp(rAux, 0);
    if (even) {
        jne(kDst.getIdx() != kSrc.getIdx() ? lCopy : lEnd, T_NEAR);
    } else {
        je(kDst.getIdx() != kSrc.getIdx() ? lCopy : lEnd, T_NEAR);
    }

    kshiftrq(kDst, kSrc, 0x1);
    jmp(lEnd, T_NEAR);
    L(lCopy);
    kmovq(kDst, kSrc);
    L(lEnd);
}

template <>
void UniqueKernel<x64::avx512_core>::cmpPerm(const Vmm& vDst, const Vmm& vSrc1, const Vmm& vSrc2, const Vmask& kMinMask, const Vmask& kMaxMask, bool tail) {
    if (jcp.dataPrc == Precision::FP32) {
        vminps(vDst | kMinMask, vSrc1, vSrc2);
        vmaxps(vDst | kMaxMask, vSrc1, vSrc2);
    } else if (jcp.dataPrc == Precision::I32) {
        vpminsd(vDst | kMinMask, vSrc1, vSrc2);
        vpmaxsd(vDst | kMaxMask, vSrc1, vSrc2);
    }  else if (jcp.dataPrc == Precision::U8) {
        vpminub(vDst | kMinMask, vSrc1, vSrc2);
        vpmaxub(vDst | kMaxMask, vSrc1, vSrc2);
    } else if (jcp.dataPrc == Precision::I8) {
        vpminsb(vDst | kMinMask, vSrc1, vSrc2);
        vpmaxsb(vDst | kMaxMask, vSrc1, vSrc2);
    }
}

template <>
void UniqueKernel<x64::avx512_core>::permOnEdge(const Vmm& vSrc1, const Vmm& vSrc2, const Vmm& vOrigin1) {
    if (jcp.dataPrc.size() == 4) {
        uni_vmovups(vSrc1 | kLastElMask, vSrc2);
        vpermd(vSrc2 | kFirstElMask, vPermElem, vOrigin1);
    } else if (jcp.dataPrc.size() == 1) {
        vmovdqu8(vSrc1 | kLastElMask, vSrc2);
        vmovdqu8(vSrc2 | kFirstElMask, vSrc1);
    }
}

template <>
void UniqueKernel<x64::avx512_core>::sortContiguousVec(const Xbyak::Reg64& rBlockLen) {
    Xbyak::Label lFew, lEnd;
    auto vAux1 = getVmm();

    Xbyak::Label lLastCmp;
    const int iterNum = (contiguousVec.size() * dataElPerVec) / 2 - 1;
    auto vAux2 = getVmm();

    for (int i = 0; i < iterNum; i++) {
        cmp(rBlockLen, 2 * (i + 1));
        jle(lLastCmp, T_NEAR);
        Xbyak::Label lSecond, lNext;

        // Compare and permute pairs {0;1}{2;3}...{14;15}
        for (int v = 0; v < contiguousVec.size(); v++) {
            Xbyak::Label lNext1, lTail1;
            const auto &vToSort = contiguousVec[v];
            cmp(rBlockLen, dataElPerVec * (v + 1));
            jl(lTail1, T_NEAR);

            vpshufd(vAux1, vToSort, 0B10110001);
            cmpPerm(vToSort, vToSort, vAux1, kMask0, kMask1);
            jmp(lNext1, T_NEAR);

            L(lTail1);
            {
                cmp(rBlockLen, dataElPerVec * v);
                je(lSecond, T_NEAR);

                vpshufd(vAux1, vToSort, 0B10110001);
                kmovw(k0, kTailMask);
                alignTailMask(kTailMask, k0, true);
                kandw(kTailMask, kTailMask, kMask0);
                cmpPerm(vToSort, vToSort, vAux1, kTailMask, kMask1, false);
                kmovw(kTailMask, k0);
                jmp(lSecond, T_NEAR);
            }

            L(lNext1);
        }

        L(lSecond);
        // Compare and permute pairs {15';0}{1;2}...{13;14}{15;0'}, where n' are values form neighbor vectors.
        const auto& vFirst = contiguousVec[0];
        vpermd(vAux1, vPermElem, vFirst);
        for (int v = 0; v < contiguousVec.size() - 1; v++) {
            Xbyak::Label lLast, lNext2;
            cmp(rBlockLen, dataElPerVec * (v + 1));
            jle(lLast, T_NEAR); // TODO: jl?

            const auto& vCurr = contiguousVec[v];
            const auto& vNext = contiguousVec[v + 1];
            const auto& vCurrAux = v % 2 == 0 ? vAux1 : vAux2;
            const auto& vNextAux = v % 2 == 0 ? vAux2 : vAux1;
            const auto& kMaskMax = v == 0 ? kMaskMaxFirst : kMask0;

            vpermd(vNextAux, vPermElem, vNext);
            permOnEdge(vCurrAux, vNextAux, vCurr);
            cmpPerm(vCurr, vCurr, vCurrAux, kMask1, kMaskMax);
            jmp(lNext2, T_NEAR);

            L(lLast);
            {
                kmovw(k0, kTailMask);
                alignTailMask(kTailMask, k0, true);
                kandw(kTailMask, kTailMask, kMaskMinLast);
                cmpPerm(vCurr, vCurr, vCurrAux, kTailMask, kMaskMax);
                kmovw(kTailMask, k0);
                jmp(lNext, T_NEAR);
            }

            L(lNext2);
        }

        L(lNext);
    }

    L(lLastCmp);
    for (int v = 0; v < contiguousVec.size(); v++) {
        Xbyak::Label lNext3, lTail3;
        const auto &vToSort = contiguousVec[v];
        cmp(rBlockLen, dataElPerVec * (v + 1));
        jl(lTail3, T_NEAR);

        vpshufd(vAux1, vToSort, 0B10110001);
        cmpPerm(vToSort, vToSort, vAux1, kMask0, kMask1);
        jmp(lNext3, T_NEAR);

        L(lTail3);
        {
            cmp(rBlockLen, dataElPerVec * v);
            je(lEnd, T_NEAR);

            vpshufd(vAux1, vToSort, 0B10110001);
            cmpPerm(vToSort, vToSort, vAux1, kMask0, kMask1, true);
            jmp(lEnd, T_NEAR);
        }

        L(lNext3);
    }

    L(lEnd);
}

template <>
void UniqueKernel<x64::avx512_core>::sortInBlocks() {
    Xbyak::Label lBlocksLoop, lFinishLoad, lFinishStore, lEnd;
    auto rSrcPtr = getReg64();
    auto rDstPtr = getReg64();
    mov(rSrcPtr,  ptr[regParams + GET_OFF(srcPtr)]);
    mov(rDstPtr,  ptr[regParams + GET_OFF(dstPtr[FIRST_UNIQUE_IDX])]);

    auto rBlocksNum   = getReg64();
    auto rBlockLenPtr = getReg64();
    auto rBlockLen    = getReg64();

    mov(rBlocksNum,   ptr[regParams + GET_OFF(blocksNum)]);
    mov(rBlockLenPtr, ptr[regParams + GET_OFF(blockLen)]);

    // SORT IN BLOCKS
    // Loop over contiguous vectors.
    L(lBlocksLoop);
    {
        cmp(rBlocksNum, 0);
        jle(lEnd, T_NEAR);

        // Load to contiguous vector.
        mov(rBlockLen, ptr[rBlockLenPtr]);
        for (int v = 0; v < contiguousVec.size(); v++) {
            Xbyak::Label lLoadNext, lLoadTail;
            cmp(rBlockLen, dataElPerVec * (v + 1));
            jl(lLoadTail, T_NEAR);

            const auto& vec = contiguousVec[v];
            uni_vmovups(vec, ptr[rSrcPtr]);

            add(rSrcPtr, vlen);
            jmp(lLoadNext, T_NEAR);

            L(lLoadTail);
            {
                cmp(rBlockLen, dataElPerVec * v);
                je(lFinishLoad, T_NEAR);

                auto rRest = getReg64();
                mov(rRest, rBlockLen);
                sub(rRest, dataElPerVec * v);
                fillRestWorkMask(kTailMask, rRest, dataTypeSize);
                uni_vmovups((Vmm) vec | kTailMask, ptr[rSrcPtr]);
                imul(rRest, rRest, dataTypeSize);
                add(rSrcPtr, rRest);
                jmp(lFinishLoad, T_NEAR);
            }

            L(lLoadNext);
        }
        L(lFinishLoad);

        sortContiguousVec(rBlockLen);

        // Store from contiguous vector.
        for (int v = 0; v < contiguousVec.size(); v++) {
            Xbyak::Label lStoreNext, lStoreTail;
            cmp(rBlockLen, dataElPerVec * (v + 1));
            jl(lStoreTail, T_NEAR);

            const auto& vec = contiguousVec[v];
            uni_vmovups(ptr[rDstPtr], vec);

            add(rDstPtr, vlen);
            jmp(lStoreNext, T_NEAR);

            L(lStoreTail);
            {
                cmp(rBlockLen, dataElPerVec * v);
                je(lFinishStore, T_NEAR);

                uni_vmovups(ptr[rDstPtr] | kTailMask, vec);
                sub(rBlockLen, dataElPerVec * v);
                imul(rBlockLen, rBlockLen, dataTypeSize);
                add(rDstPtr, rBlockLen);
                jmp(lFinishStore, T_NEAR);
            }

            L(lStoreNext);
        }
        L(lFinishStore);

        dec(rBlocksNum);
        add(rBlockLenPtr, sizeof(int64_t));
        jmp(lBlocksLoop, T_NEAR);
    }
    // ALL BLOCKS SORTED

    L(lEnd);
}

// Gather samples { w/Nb + j*W, 2*w/Nb + j*w, ... , (Np - 1)*w/Np + j*w },
// where W - total kernel's work, Nb - blocks num, w = W/Nb, 0 <= j < Nb.
template <>
void UniqueKernel<x64::avx512_core>::gatherSamples() {
    Xbyak::Label lInBlock, lInMemory, lFinishLoad, lEnd;
    auto rSrcPtr     = getReg64();
    auto rSrcPtrStep = getReg64();
    mov(rSrcPtr,      ptr[regParams + GET_OFF(dstPtr[UNIQUE_DATA])]);
    mov(rSrcPtrStep,  ptr[regParams + GET_OFF(samplesIdxStep)]);

    auto vSrcIdx   = getVmm();
    auto rSamplesLen = getReg64();

    mov(rSamplesLen, ptr[regParams + GET_OFF(samplesIdxPtr)]);
    uni_vmovups(vSrcIdx, ptr[rSamplesLen]);
    mov(rSamplesLen, ptr[regParams + GET_OFF(samplesLen)]);

    // If samples num <= block size, sort in block.
    // If samples num > block size, sort in memory.
    // Load to contiguous vector.
    cmp(rSamplesLen, contiguousVec.size() * dataElPerVec);
    jg(lInMemory, T_NEAR);

    for (int v = 0; v < contiguousVec.size(); v++) {
        Xbyak::Label lLoadNext, lLoadTail;
        cmp(rSamplesLen, dataElPerVec * (v + 1));
        jl(lLoadTail, T_NEAR);

        const auto& vec = contiguousVec[v];
        gatherdd(vec, rSrcPtr, vSrcIdx, kTailMask, false);

        add(rSrcPtr, rSrcPtrStep);
        jmp(lLoadNext, T_NEAR);

        L(lLoadTail);
        {
            cmp(rSamplesLen, dataElPerVec * v);
            je(lFinishLoad, T_NEAR);

            auto rRest = getReg64();
            mov(rRest, rSamplesLen);
            sub(rRest, dataElPerVec * v);
            fillRestWorkMask(kTailMask, rRest, dataTypeSize);
            kmovq(k0, kTailMask);
            gatherdd(vec, rSrcPtr, vSrcIdx, kTailMask, true);
            jmp(lFinishLoad, T_NEAR);
        }

        L(lLoadNext);
    }
    L(lFinishLoad);

    kmovq(kTailMask, k0);
    sortContiguousVec(rSamplesLen);

    // Store from contiguous vector.
    Xbyak::Label lFinishStore;
    auto rDstPtr = getReg64();
    mov(rDstPtr, ptr[regParams + GET_OFF(samplesPtr)]);

    for (int v = 0; v < contiguousVec.size(); v++) {
        Xbyak::Label lStoreNext, lStoreTail;
        cmp(rSamplesLen, dataElPerVec * (v + 1));
        jl(lStoreTail, T_NEAR);

        const auto& vec = contiguousVec[v];
        uni_vmovups(ptr[rDstPtr], vec);

        add(rDstPtr, vlen);
        jmp(lStoreNext, T_NEAR);

        L(lStoreTail);
        {
            cmp(rSamplesLen, dataElPerVec * v);
            je(lFinishStore, T_NEAR);

            uni_vmovups(ptr[rDstPtr] | kTailMask, vec);
            sub(rSamplesLen, dataElPerVec * v);
            imul(rSamplesLen, rSamplesLen, dataTypeSize);
            add(rDstPtr, rSamplesLen);
            jmp(lFinishStore, T_NEAR);
        }

        L(lStoreNext);
    }
    L(lFinishStore);

    L(lEnd);
}

template <>
void UniqueKernel<x64::avx512_core>::exchangePartitions() {
    Xbyak::Label lBlocksLoop, lFinishLoad, lFinishStore, lEnd;
    auto rSrcPtr = getReg64();
    auto rDstPtr = getReg64();
    mov(rSrcPtr, ptr[regParams + GET_OFF(dstPtr[FIRST_UNIQUE_IDX])]);
    mov(rDstPtr, ptr[regParams + GET_OFF(dstPtr[UNIQUE_DATA])]);

    auto rPivotsNum   = getReg64();
    auto rBlocksNum   = getReg64();
    auto rBlockLenPtr = getReg64();
    auto rBlockLen    = getReg64();

    mov(rPivotsNum,   ptr[regParams + GET_OFF(blocksNum)]);
    mov(rBlocksNum,   ptr[regParams + GET_OFF(blocksNum)]);
    mov(rBlockLenPtr, ptr[regParams + GET_OFF(blockLen)]);

    L(lBlocksLoop);
    {
        cmp(rBlocksNum, 0);
        jle(lEnd, T_NEAR);

        // Load to contiguous vector.
        mov(rBlockLen, ptr[rBlockLenPtr]);
        for (int v = 0; v < contiguousVec.size(); v++) {
            Xbyak::Label lLoadNext, lLoadTail;
            cmp(rBlockLen, dataElPerVec * (v + 1));
            jl(lLoadTail, T_NEAR);

            const auto& vec = contiguousVec[v];
            uni_vmovups(vec, ptr[rSrcPtr]);

            add(rSrcPtr, vlen);
            jmp(lLoadNext, T_NEAR);

            L(lLoadTail);
            {
                cmp(rBlockLen, dataElPerVec * v);
                je(lFinishLoad, T_NEAR);

                auto rRest = getReg64();
                mov(rRest, rBlockLen);
                sub(rRest, dataElPerVec * v);
                fillRestWorkMask(kTailMask, rRest, dataTypeSize);
                uni_vmovups((Vmm) vec | kTailMask, ptr[rSrcPtr]);
                imul(rRest, rRest, dataTypeSize);
                add(rSrcPtr, rRest);
                jmp(lFinishLoad, T_NEAR);
            }

            L(lLoadNext);
        }
        L(lFinishLoad);

        sortContiguousVec(rBlockLen);

        // Store from contiguous vector.
        for (int v = 0; v < contiguousVec.size(); v++) {
            Xbyak::Label lStoreNext, lStoreTail;
            cmp(rBlockLen, dataElPerVec * (v + 1));
            jl(lStoreTail, T_NEAR);

            const auto& vec = contiguousVec[v];
            uni_vmovups(ptr[rDstPtr], vec);

            add(rDstPtr, vlen);
            jmp(lStoreNext, T_NEAR);

            L(lStoreTail);
            {
                cmp(rBlockLen, dataElPerVec * v);
                je(lFinishStore, T_NEAR);

                uni_vmovups(ptr[rDstPtr] | kTailMask, vec);
                sub(rBlockLen, dataElPerVec * v);
                imul(rBlockLen, rBlockLen, dataTypeSize);
                add(rDstPtr, rBlockLen);
                jmp(lFinishStore, T_NEAR);
            }

            L(lStoreNext);
        }
        L(lFinishStore);

        dec(rBlocksNum);
        add(rBlockLenPtr, sizeof(int64_t));
        jmp(lBlocksLoop, T_NEAR);
    }
    // ALL BLOCKS SORTED

    L(lEnd);
}

template class UniqueKernel<x64::avx512_core>;
template class UniqueKernel<x64::avx2>;
template class UniqueKernel<x64::sse41>;

}   // namespace intel_cpu
}   // namespace ov
