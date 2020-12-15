// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"
#include "ie_parallel.hpp"
#include "jit_generator.hpp"
#include "common/cpu_memcpy.h"
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <ie_common.h>
#include <mkldnn_node.h>

#include <cmath>
#include <vector>
#include <string>
#include <chrono>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace mkldnn::impl;
using namespace mkldnn::impl::cpu;
using namespace mkldnn::impl::utils;

struct jitGreedyDecoderConfT {
    int classesNum;
};

struct jitArgsGreedyDecoder {
    const float* probs;
    float* tmpDst;
    int* maxClassIdx;
    float* tmpVal;
};

struct jitUniGreedyDecoderBase {
    void (*ker_)(const jitArgsGreedyDecoder *);
    void operator()(const jitArgsGreedyDecoder *args) { assert(ker_); ker_(args); }
    jitGreedyDecoderConfT jpp;
    explicit jitUniGreedyDecoderBase(jitGreedyDecoderConfT jpp) : ker_(nullptr), jpp(jpp) {}
    virtual ~jitUniGreedyDecoderBase() {}
};

#define GET_OFF(field) offsetof(jitArgsGreedyDecoder, field)

template <cpu::cpu_isa_t isa>
struct jitUniGreedyDecoderKernel : public jitUniGreedyDecoderBase, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jitUniGreedyDecoderKernel)

    explicit jitUniGreedyDecoderKernel(jitGreedyDecoderConfT jpp) : jitUniGreedyDecoderBase(jpp), jit_generator() {
        this->preamble();

        mov(regProbs, ptr[regParams + GET_OFF(probs)]);
//        mov(regTmpDst, ptr[regParams + GET_OFF(tmpDst)]);
        mov(regMaxClassIdx, ptr[regParams + GET_OFF(maxClassIdx)]);

        const uint32_t dataTypeSize = sizeof(float);
        const uint32_t elPerVector = vlen / dataTypeSize;
        const int tailLen = jpp.classesNum % elPerVector;
        uint8_t xmmNum = 1;
        if (isa == cpu::avx512_common) {
            xmmNum = 4;
        } else if (isa == cpu::avx2) {
            xmmNum = 2;
        }

        mov(regMaxIdx, 0);

        if (jpp.classesNum >= elPerVector) {
            uni_vmovups(vmmFirstV, ptr[regProbs]);
            const int iterations = jpp.classesNum / elPerVector;
            for (int i = 1; i < iterations; i++) {
                add(regProbs, vlen);
                uni_vmovups(vmmSecondV, ptr[regProbs]);
                vmaxps(vmmFirstV, vmmFirstV, vmmSecondV);
            }
            broadcastMax(vmmFirstV);
        }

        Xbyak::Label foundLabel, lookWithStep;
        // Tail
        if (tailLen > 0) {
            if (jpp.classesNum > elPerVector) {
                add(regProbs, vlen);
            }
            uni_vmovups(vmmFirstV, ptr[regProbs]);
            mov(regTmp, 0);
            mov(regTmpVal, -1);
            int c = 0;
            for (uint8_t x = 0; x < xmmNum; x++) {
                if (isa == cpu::avx512_common) {
                    vextractf32x4(xmm8, vmmFirstV, x);
                } else if (isa == cpu::avx2) {
                    vextractf128(xmm8, vmmFirstV, x);
                }
                if (jpp.classesNum < elPerVector && x == 0) {
                    uni_extractps(regMaxProb32, xmm8, x);
                }
                for (uint8_t i = 0; i < 4 && c < tailLen; i++, c++) {
                    Xbyak::Label nextLabel;
                    uni_extractps(regCurProb32, xmm8, i);
                    cmp(regCurProb32, regMaxProb32);
                    jle(nextLabel, T_NEAR);
                    mov(regMaxProb32, regCurProb32);
                    mov(regTmpVal, regTmp);
                    L(nextLabel);
                    inc(regTmp);
                }
            }
            Xbyak::Label foundInTail;
            cmp(regTmpVal, -1);
            jg(foundInTail, T_NEAR);
            jmp(lookWithStep);
            L(foundInTail);
            add(regTmpVal, (jpp.classesNum - tailLen));
            mov(regMaxIdx, regTmpVal);
            jmp(foundLabel, T_NEAR);
        }

        L(lookWithStep);
        Xbyak::Label findInVecLabel;
        const int iterations = jpp.classesNum / elPerVector;
        vpcmpeqd(vmmOnes, vmmOnes, vmmOnes);
        mov(regProbs, ptr[regParams + GET_OFF(probs)]);
        int idx = 0;
        for (; idx < iterations; idx++) {
            uni_vmovups(vmmFirstV, ptr[regProbs]);
            vcmpps(vmmGrtMask, vmmFirstV, vmmMaxVec, jit_generator::_cmp_eq_oq);
            vptest(vmmGrtMask, vmmOnes);
            jnz(findInVecLabel, T_NEAR);
            add(regProbs, vlen);
            inc(regMaxIdx);
        }
        L(findInVecLabel);
        mov(rax, elPerVector);
        mul(regMaxIdx);
        mov(regMaxIdx, eax);

        mov(regTmp, 0);
        for (int x = 0; x < xmmNum; x++) {
            if (isa == cpu::avx512_common) {
                vextractf32x4(xmm8, vmmFirstV, x);
            } else if (isa == cpu::avx2) {
                vextractf128(xmm8, vmmFirstV, x);
            }
            for (int i = 0; i < 4; i++) {
                Xbyak::Label nextLabel;
                uni_extractps(regCurProb32, xmm8, i);
                cmp(regCurProb32, regMaxProb);
                jne(nextLabel, T_NEAR);
                add(regMaxIdx, regTmp);
                jmp(foundLabel, T_NEAR);
                L(nextLabel);
                inc(regTmp);
            }
        }

        L(foundLabel);

        mov(ptr[regMaxClassIdx], regMaxIdx);

        this->postamble();

        ker_ = (decltype(ker_))this->getCode();
    }

    inline void uni_extractps(const Xbyak::Operand& op, const Xbyak::Xmm& xmm, uint8_t imm) {
        if (isa == cpu::avx512_common || isa == cpu::avx2) {
            vextractps(op, xmm, imm);
        } else {
            extractps(op, xmm, imm);
        }
    }

    inline void uni_insertps(const Xbyak::Xmm& x1, const Xbyak::Xmm& x2, const Xbyak::Operand& op, uint8_t imm) {
        if (isa == cpu::avx512_common || isa == cpu::avx2) {
            vinsertps(x1, x2, op, imm);
        } else {
            insertps(x1, op, imm);
        }
    }

    inline void findMaxInXmm(const Xbyak::Xmm& src, const Xbyak::Xmm& dst, const Xbyak::Reg32& maxVal) {
        uni_extractps(maxVal, src, 0);
        uni_insertps(dst, src, src, 0x0);
        for (int i = 1; i < 4; i++) {
            Xbyak::Label nextLabel;
            uni_extractps(regCurProb32, src, i);
            cmp(regCurProb, maxVal);
            jle(nextLabel, T_NEAR);
            mov(maxVal, regCurProb);
            uni_insertps(dst, src, src, i << 6);
            L(nextLabel);
        }
    }

    void broadcastMax(const Xbyak::Xmm& src) {
//        uni_extractps(regMaxProb, src, 0);
//        uni_insertps(xmmAux9, src, src, 0x0);
//        for (int i = 1; i < 4; i++) {
//            Xbyak::Label nextLabel;
//            uni_extractps(regCurProb, src, i);
//            cmp(regCurProb, regMaxProb);
//            jle(nextLabel, T_NEAR);
//            mov(regMaxProb, regCurProb);
//            uni_insertps(xmmAux9, src, src, i << 6);
//            L(nextLabel);
//        }
        findMaxInXmm(src, xmmAux9, regMaxProb32);
        uni_vbroadcastss(vmmMaxVec, xmmAux9);
    }

    inline void broadcastMax(const Xbyak::Ymm& src) {
        vextractf128(xmm8, src, 0);
        vextractf128(xmmAux9, src, 1);
        vmaxps(xmm8, xmmAux9);
        findMaxInXmm(xmm8, xmmAux9, regMaxProb32);
//        uni_extractps(regMaxProb32, xmm8, 0);
//        uni_insertps(xmmAux9, xmm8, xmm8, 0x0);
//        for (int i = 1; i < 4; i++) {
//            Xbyak::Label nextLabel;
//            uni_extractps(regCurProb32, xmm8, i);
//            cmp(regCurProb32, regMaxProb32);
//            jle(nextLabel, T_NEAR);
//            mov(regMaxProb, regCurProb32);
//            uni_insertps(xmmAux9, xmm8, xmm8, i << 6);
//            L(nextLabel);
//        }
        vbroadcastss(vmmMaxVec, xmmAux9);
    }

    void broadcastMax(const Xbyak::Zmm& src) {
        vextractf32x8(vmmAux5, src, 0);
        vextractf32x8(vmmAux6, src, 1);
        vmaxps(vmmAux5, vmmAux6);
        broadcastMax(vmmAux5);
//        vextractf32x4(xmm8, src, 0);
//        for (int i = 1; i < 4; i++) {
//            vextractf32x4(xmmAux9, src, i);
//            vmaxps(xmm8, xmmAux9);
//        }
//        findMaxInXmm(xmm8, xmmAux9, regMaxProb32);
//        uni_extractps(regMaxProb32, xmm8, 0);
//        uni_insertps(xmmAux9, xmm8, xmm8, 0x0);
//        for (int i = 1; i < 4; i++) {
//            Xbyak::Label nextLabel;
//            uni_extractps(regCurProb32, xmm8, i);
//            cmp(regCurProb32, regMaxProb32);
//            jle(nextLabel, T_NEAR);
//            mov(regMaxProb, regCurProb32);
//            uni_insertps(xmmAux9, xmm8, xmm8, i << 6);
//            L(nextLabel);
//        }
//        vbroadcastss(vmmMaxVec, xmmAux9);
    }

private:
    using Vmm = typename conditional3<isa == cpu::sse42, Xbyak::Xmm, isa == cpu::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    uint32_t vlen = cpu_isa_traits<isa>::vlen;

    Xbyak::Reg64 regProbs = r8;
    Xbyak::Reg64 regTmpDst = r9;
    Xbyak::Reg64 regMaxProb = r10;
    Xbyak::Reg64 regCurProb = r11;
    Xbyak::Reg64 regMaxIdx = r12;
    Xbyak::Reg64 regMaxClassIdx = r13;
    Xbyak::Reg64 regTmpVal = r14;
    Xbyak::Reg64 regTmp = r15;
    Xbyak::Reg32 regMaxProb32 = r10d;
    Xbyak::Reg32 regCurProb32 = r11d;

    Xbyak::Fpu regSt0 = st0;
    Xbyak::Fpu regSt1 = st1;

    Xbyak::Reg64 regParams = abi_param1;

    Xbyak::Xmm xmm8 = Xbyak::Xmm(8);
    Xbyak::Xmm xmmAux9 = Xbyak::Xmm(9);
    Vmm vmmFirstV = Vmm(0);
    Vmm vmmSecondV = Vmm(1);
    Vmm vmmGrtMask = Vmm(2);
    Vmm vmmOnes = Vmm(3);
    Vmm vmmMaxVec = Vmm(4);
    Vmm vmmAux5 = Vmm(5);
    Vmm vmmAux6 = Vmm(6);
};

class CTCGreedyDecoderImpl: public ExtLayerBase {
public:
    explicit CTCGreedyDecoderImpl(const CNNLayer* layer) : mergeRepeated_(true) {
        try {
            std::string errPrefix = "CTCGreedyDecoder layer with name '" + layer->name + "' ";
            if (layer->insData.size() != 2)
                THROW_IE_EXCEPTION << errPrefix << "has invalid number of input edges: " << layer->insData.size();
            if (layer->outData.size() != 1)
                THROW_IE_EXCEPTION << errPrefix << "has invalid number of outputs edges: " << layer->outData.size();

            auto inData = layer->insData[DATA_INDEX].lock();
            auto sequenceLenData = layer->insData[SEQUENCE_LENGTH_INDEX].lock();
            if (!inData || !sequenceLenData)
                THROW_IE_EXCEPTION << errPrefix << "has nullable inputs.";
            if (inData->getTensorDesc().getPrecision() != Precision::FP32)
                THROW_IE_EXCEPTION << errPrefix << "has unsupported 'data' input precision: " << inData->getTensorDesc().getPrecision();
            if (sequenceLenData->getTensorDesc().getPrecision() != Precision::FP32)
                THROW_IE_EXCEPTION << errPrefix << "has unsupported 'sequence_length' input precision: " << sequenceLenData->getTensorDesc().getPrecision();

            std::vector<DataConfigurator> inputConfigs{{ConfLayout::PLN, Precision::FP32}, {ConfLayout::PLN, Precision::FP32}};
            std::vector<DataConfigurator> outputConfigs{{ConfLayout::PLN, Precision::FP32}};
            addConfig(layer, inputConfigs, outputConfigs);

            mergeRepeated_ = layer->GetParamAsBool("ctc_merge_repeated", true);

            jitGreedyDecoderConfT jpp;
            jpp.classesNum = inData->getTensorDesc().getDims()[2];
            if (mayiuse(cpu::avx2)) {
                kernel_.reset(new jitUniGreedyDecoderKernel<cpu::avx2>(jpp));
            }
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

//    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
//                       ResponseDesc *resp) noexcept override {
//        const float* probabilities = inputs[DATA_INDEX]->cbuffer().as<const float*>() +
//            inputs[DATA_INDEX]->getTensorDesc().getBlockingDesc().getOffsetPadding();
//        const float* sequenceLengths = inputs[SEQUENCE_LENGTH_INDEX]->cbuffer().as<const float*>() +
//            inputs[SEQUENCE_LENGTH_INDEX]->getTensorDesc().getBlockingDesc().getOffsetPadding();
//        float* outputSequences = outputs[0]->buffer().as<float*>();
//
//        const size_t T = inputs[DATA_INDEX]->getTensorDesc().getDims()[0];
//        const size_t B = inputs[DATA_INDEX]->getTensorDesc().getDims()[1];
//        const int C = inputs[DATA_INDEX]->getTensorDesc().getDims()[2];
//        const size_t BC = C * B;
//        const size_t TB = T * B;
//        const size_t CB1 = C * (B - 1);
//
//        const int blankIndex = C - 1;
//
////printf("T: %lu; B: %lu; C: %d\n", T, B, C);
//
//static double du1 = 0.0;
//static int c1 = 0;
//auto start = std::chrono::steady_clock::now();
//
////        size_t workAmount = 0;
////        std::vector<size_t> sequenceLengthB(B, 0);
////        for (size_t b = 0; b < B; b++) {
////            for (size_t t = 0; t < T; t++) {
////                if (sequenceLengths[B * t + b] == 0.f)
////                    break;
////                sequenceLengthB[b]++;
////            }
////            workAmount += sequenceLengthB[b];
////        }
//
//        // Parallelization could not be made by T due to output index depends on merged classes and
//        // blank index thus could not be shared between threads. Better to parallelize by classes.
////        auto threadBody = [&](const int ithr, const int nthr) {
////            size_t start(0lu), end(0lu);
////            splitter(workAmount, nthr, ithr, start, end);
////            if (start >= end)
////                return;
////            size_t tStart = 0lu, bStart = 0lu;
////            int64_t cw = 0, st = start;
////            for (; bStart < B; bStart++) {
////                cw += sequenceLengthB[bStart];
////                if (cw >= st) {
////                    tStart = sequenceLengthB[bStart] + st - cw;
////                    break;
////                }
////            }
//
////            size_t workCounter = start;
////printf("[] tStart: %lu; bStart: %lu\n", tStart, bStart);
//
//            for (size_t b = 0; b < B; ++b) {
//                int prev_class_idx = -1;
//                size_t outputIndex = b * T;
//                const float* probs = probabilities + b * C;
////                size_t sequenceLength = sequenceLengthB[b];
////printf("[] sequenceLengths: %lu; b: %lu\n", sequenceLength, b);
//
//                for (size_t t = 0; t < T; ++t) {
//                    if (sequenceLengths[B * t + b] == 0.f) {
////printf("[] BREAK: t: %lu; b: %lu\n", t, b);
//                        break;
//                    }
////                    int maxClassIdx = 0;
//
////    std::string probsStr = "probs: ";
////for (int i = 0; i < C; i++) {
////    probsStr += std::to_string(probs[i]) + "; ";
////}
////printf(probsStr);
////                    if (kernel_ != nullptr) {
////                        float tmpDst[8];
////                        auto arg = jitArgsGreedyDecoder();
////                        arg.probs = probs;
////                        arg.tmpDst = tmpDst;
////                        arg.maxClassIdx = &maxClassIdx;
////                        (*kernel_)(&arg);
////                        probs += C;
////probsStr += "\ntmpDst: ";
////for (int i = 0; i < 8; i++) {
////    probsStr += std::to_string(tmpDst[i]) + "; ";
////}
////probsStr += "\tt: " + std::to_string(t) + "; b: " + std::to_string(b) + "; maxClassIdx: " + std::to_string(maxClassIdx);
////                    } else {
////                        const int threadsNum = parallel_get_max_threads();
////                        std::vector<int> maxClassPerThread(threadsNum, -1);
////                        std::vector<float> probsPerThread(threadsNum, std::numeric_limits<float>::lowest());
//
////static double du2 = 0.0;
////static int c2 = 0;
////auto start2 = std::chrono::steady_clock::now();
//
//                        const int threadsNum = parallel_get_max_threads();
//                        std::vector<int> maxClassPerThread(threadsNum, -1);
//                        std::vector<float> probsPerThread(threadsNum, std::numeric_limits<float>::lowest());
//
//                        auto threadBody = [&](const int ithr, const int nthr) {
//                            int start(0), end(0);
//                            splitter(C, nthr, ithr, start, end);
//                            if (start >= end)
//                                return;
//
//                            maxClassPerThread[ithr] = start;
//                            const float* probsThr = probs + start;
//                            probsPerThread[ithr] = probsThr[0];
//                            probsThr++;
//                            for (int c = start + 1; c < end; c++, probsThr++) {
//                                if (probsThr[0] > probsPerThread[ithr]) {
//                                    maxClassPerThread[ithr] = c;
//                                    probsPerThread[ithr] = probsThr[0];
//                                }
//                            }
//                        };
//                        parallel_nt(0, threadBody);
////                        probs += C;
//                        int maxClassIdx = maxClassPerThread[0];
//                        float maxProb = probsPerThread[0];
//                        for (int i = 1; i < threadsNum; i++) {
//                            if (probsPerThread[i] > maxProb) {
//                                maxClassIdx = maxClassPerThread[i];
//                                maxProb = probsPerThread[i];
//                            }
//                        }
//
////auto end2 = std::chrono::steady_clock::now();
////c2++;
////du2 += std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();
////if (c2 % 1000 == 0) {
////    printf("DU2: %f\n", du2 / c2);
////}
////                    }
////probsStr += "    t: " + std::to_string(t) + "; b: " + std::to_string(b) + "; maxClassIdx: " + std::to_string(maxClassIdx) +
////        "; outputIndex: " + std::to_string(outputIndex);
////printf("%s\n", probsStr.c_str());
////printf("t: %lu; b: %lu; maxClassIdx: %d\n", t, b, maxClassIdx);
//                    if (maxClassIdx < blankIndex &&
//                            !(mergeRepeated_ && maxClassIdx == prev_class_idx)) {
//                        outputSequences[outputIndex++] = static_cast<float>(maxClassIdx);
//                    }
//
//                    prev_class_idx = maxClassIdx;
//                    probs += BC;
//
////                    if (++workCounter >= end) {
////                        return;
////                    }
//                }
//                std::fill(outputSequences + outputIndex, outputSequences + (b + 1) * T, -1.f);
////                tStart = 0lu;
//            }
////        }; // thread body
//
////        parallel_nt(0, threadBody);
//
//auto end = std::chrono::steady_clock::now();
//c1++;
//du1 += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
//if (c1 % 100 == 0) {
//    printf("DU1: %f\n", du1 / c1);
//}
//
//        return OK;
//    }
//
//    const size_t DATA_INDEX = 0lu;
//    const size_t SEQUENCE_LENGTH_INDEX = 1lu;
//    bool mergeRepeated_;
//
//    std::shared_ptr<jitUniGreedyDecoderBase> kernel_;
//};


    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
static double du1 = 0.0;
static int c1 = 0;
auto start = std::chrono::steady_clock::now();

        const float* probabilities = inputs[DATA_INDEX]->cbuffer().as<const float*>() +
            inputs[DATA_INDEX]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const float* sequenceLengths = inputs[SEQUENCE_LENGTH_INDEX]->cbuffer().as<const float*>() +
            inputs[SEQUENCE_LENGTH_INDEX]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        float* outputSequences = outputs[0]->buffer().as<float*>();

        const size_t T = inputs[DATA_INDEX]->getTensorDesc().getDims()[0];
        const size_t B = inputs[DATA_INDEX]->getTensorDesc().getDims()[1];
        const int C = inputs[DATA_INDEX]->getTensorDesc().getDims()[2];
        const size_t BC = B * C;
        const size_t TB = T * B;
        const size_t CB1 = C * (B - 1);

        const int blankIndex = C - 1;

//printf("T: %lu; B: %lu; C: %d\n", T, B, C);
//auto start = std::chrono::steady_clock::now();

        std::vector<size_t> sequenceLengthB(B, 0);
        parallel_for(B, [&](size_t b) {
            for (size_t t = 0; t < T; t++) {
                if (sequenceLengths[B * t + b] == 0.f)
                    break;
                sequenceLengthB[b]++;
            }
        });

        size_t workAmount = 0;
        for (size_t b = 0; b < B; b++) {
            workAmount += sequenceLengthB[b];
        }

//auto start = std::chrono::steady_clock::now();

        auto threadBody = [&](const int ithr, const int nthr) {
            size_t start(0lu), end(0lu);
            splitter(workAmount, nthr, ithr, start, end);
            if (start >= end)
                return;
            size_t tStart = 0lu, bStart = 0lu;
            int64_t cw = 0, st = start;
            for (; bStart < B; bStart++) {
                cw += sequenceLengthB[bStart];
                if (cw >= st) {
                    tStart = sequenceLengthB[bStart] + st - cw;
                    break;
                }
            }

            size_t workCounter = start;
//printf("[] tStart: %lu; bStart: %lu\n", tStart, bStart);

            for (size_t b = bStart; b < B; ++b) {
                int prev_class_idx = -1;
                size_t outputIndex = b * T + tStart;
                const float* probs = probabilities + b * C + BC * tStart;
                size_t sequenceLength = sequenceLengthB[b];
//printf("[] sequenceLengths: %lu; b: %lu\n", sequenceLength, b);

                for (size_t t = tStart; t < sequenceLength; ++t) {
//                    if (sequenceLengths[B * t + b] == 0.f) {
////printf("[] BREAK: t: %lu; b: %lu\n", t, b);
//                        break;
//                    }
                    int maxClassIdx = 0;

//    std::string probsStr = "probs: ";
//for (int i = 0; i < C; i++) {
//    probsStr += std::to_string(probs[i]) + "; ";
//}
//printf("%s\n", probsStr.c_str());
                    if (kernel_ != nullptr) {
//                        float tmpDst[8];
//                        float tmpVal = -1.f;
                        auto arg = jitArgsGreedyDecoder();
                        arg.probs = probs;
//                        arg.tmpDst = tmpDst;
                        arg.maxClassIdx = &maxClassIdx;
//                        arg.tmpVal = static_cast<float*>(&tmpVal);
                        (*kernel_)(&arg);
                        probs += C;
//probsStr += "\ntmpDst: ";
//for (int i = 0; i < 8; i++) {
//    probsStr += std::to_string(tmpDst[i]) + "; ";
//}
//probsStr += "  t: " + std::to_string(t) + "; b: " + std::to_string(b) + "; maxClassIdx: " + std::to_string(maxClassIdx);
//probsStr += "  tmpVal: " + std::to_string(tmpVal);
                    } else {
//static double du2 = 0.0;
//static int c2 = 0;
//auto start2 = std::chrono::steady_clock::now();

                        float maxProb = probs[0];
                        ++probs;

                        for (int c = 1; c < C; ++c, ++probs) {
                            if (*probs > maxProb) {
                                maxClassIdx = c;
                                maxProb = *probs;
                            }
                        }
                    }
//auto end2 = std::chrono::steady_clock::now();
//c2++;
//du2 += std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();
//if (c2 % 1000 == 0) {
//    printf("DU2: %f\n", du2 / c2);
//}
//                    }
//probsStr += "  t: " + std::to_string(t) + "; b: " + std::to_string(b) + "; maxClassIdx: " + std::to_string(maxClassIdx) +
//        "; outputIndex: " + std::to_string(outputIndex);
//printf("%s\n", probsStr.c_str());
//                    if (maxClassIdx < blankIndex &&
//                            !(mergeRepeated_ && maxClassIdx == prev_class_idx)) {
                        outputSequences[outputIndex++] = static_cast<float>(maxClassIdx);
//                    }

//                    prev_class_idx = maxClassIdx;
                    probs += CB1;

                    if (++workCounter >= end) {
                        return;
                    }
                }
//                std::fill(outputSequences + outputIndex, outputSequences + (b + 1) * T, -1.f);
                tStart = 0lu;
            }
        }; // thread body

        parallel_nt(0, threadBody);

//auto end = std::chrono::steady_clock::now();


        parallel_for(B, [&](size_t b) {
            int prev_class_idx = -1;
            size_t outputIndex = b * T;
            const size_t sequenceLength = sequenceLengthB[b];
//printf("[] sequenceLengths: %lu; b: %lu\n", sequenceLength, b);

            float* shiftedOut = outputSequences + b * T;

//    std::string probsStr = "INDX: ";
//for (int i = 0; i < sequenceLength; i++) {
//    probsStr += std::to_string(outputSequences[i]) + "; ";
//}
//probsStr += "\nRES:\n";
            for (size_t t = 0; t < sequenceLength; ++t) {
                if (*shiftedOut < blankIndex &&
                        !(mergeRepeated_ && *shiftedOut == prev_class_idx)) {
                    outputSequences[outputIndex++] = *shiftedOut;
//                        probsStr += std::to_string(prev_class_idx) + "; ";
//                        probsStr += std::to_string(outputIndex) + ": " + std::to_string(outputSequences[outputIndex]) + "\n";
                }
                prev_class_idx = *shiftedOut;
                shiftedOut++;
            }
            std::fill(outputSequences + outputIndex, outputSequences + (b + 1) * T, -1.f);
        });
//printf("%s\n", probsStr.c_str());

auto end = std::chrono::steady_clock::now();
c1++;
du1 += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
if (c1 % 100 == 0) {
    printf("DU1: %f\n", du1 / c1);
}

        return OK;
    }

    const size_t DATA_INDEX = 0lu;
    const size_t SEQUENCE_LENGTH_INDEX = 1lu;
    bool mergeRepeated_;

    std::shared_ptr<jitUniGreedyDecoderBase> kernel_;
};

REG_FACTORY_FOR(CTCGreedyDecoderImpl, CTCGreedyDecoder);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
