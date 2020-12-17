// Copyright (C) 2020 Intel Corporation
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
        const uint32_t iterations = jpp.classesNum / elPerVector;
        const int tailLen = jpp.classesNum % elPerVector;
        const uint8_t xmmNum = vlen / cpu_isa_traits<cpu::sse42>::vlen;

        mov(regMaxIdx, 0);

        if (jpp.classesNum >= elPerVector) {
            uni_vmovups(vmmFirstV, ptr[regProbs]);
            for (uint32_t i = 1; i < iterations; i++) {
                add(regProbs, vlen);
                uni_vmovups(vmmSecondV, ptr[regProbs]);
                uni_vmaxps(vmmFirstV, vmmFirstV, vmmSecondV);
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
                    vextractf32x4(xmmAux8, Xbyak::Zmm(vmmFirstV.getIdx()), x);
                } else if (isa == cpu::avx2) {
                    vextractf128(xmmAux8, Xbyak::Ymm(vmmFirstV.getIdx()), x);
                }
                if (jpp.classesNum < elPerVector && x == 0) {
                    uni_extractps(regMaxProb32, xmmAux8, x);
                }
                for (uint8_t i = 0; i < 4 && c < tailLen; i++, c++) {
                    Xbyak::Label nextLabel;
                    uni_extractps(regCurProb32, xmmAux8, i);
                    cmp(regCurProb32, regMaxProb32);
                    jle(nextLabel, T_NEAR);
                    mov(regMaxProb32, regCurProb32);
                    mov(regTmpVal, regTmp);
                    L(nextLabel);
                    inc(regTmp);
                }
            }
            cmp(regTmpVal, -1);
            je(lookWithStep, T_NEAR);
            add(regTmpVal, (jpp.classesNum - tailLen));
            mov(regMaxIdx, regTmpVal);
            jmp(foundLabel, T_NEAR);
        }

        L(lookWithStep);
        Xbyak::Label findInVecLabel;
        if (isa == cpu::avx512_common) {
            uni_vpcmpeqd(xmmAux9, xmmAux9, xmmAux9);
            vbroadcastss(vmmOnes, xmmAux9);
        } else {
            uni_vpcmpeqd(vmmOnes, vmmOnes, vmmOnes);
        }
        mov(regProbs, ptr[regParams + GET_OFF(probs)]);
        for (uint32_t idx = 0; idx < iterations; idx++) {
            uni_vmovups(vmmFirstV, ptr[regProbs]);
            if (isa == cpu::avx512_common) {
                vcmpps(k_mask, vmmFirstV, vmmMaxVec, jit_generator::_cmp_eq_oq);
                kortestw(k_mask, k_mask);
            } else {
                vcmpps(vmmGrtMask, vmmFirstV, vmmMaxVec, jit_generator::_cmp_eq_oq);
                vptest(vmmGrtMask, vmmOnes);
            }
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
                vextractf32x4(xmmAux8, Xbyak::Zmm(vmmFirstV.getIdx()), x);
            } else if (isa == cpu::avx2) {
                vextractf128(xmmAux8, Xbyak::Ymm(vmmFirstV.getIdx()), x);
            }
            for (int i = 0; i < 4; i++) {
                Xbyak::Label nextLabel;
                uni_extractps(regCurProb32, xmmAux8, i);
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
        findMaxInXmm(src, xmmAux9, regMaxProb32);
        uni_vbroadcastss(vmmMaxVec, xmmAux9);
    }

    inline void broadcastMax(const Xbyak::Ymm& src) {
        vextractf128(xmmAux8, src, 0);
        vextractf128(xmmAux9, src, 1);
        vmaxps(xmmAux8, xmmAux9);
        findMaxInXmm(xmmAux8, xmmAux9, regMaxProb32);
        vbroadcastss(vmmMaxVec, xmmAux9);
    }

    void broadcastMax(const Xbyak::Zmm& src) {
        vextractf32x8(ymmAux5, src, 0);
        vextractf32x8(ymmAux6, src, 1);
        vmaxps(ymmAux5, ymmAux6);
        broadcastMax(ymmAux5);
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

    Xbyak::Reg64 regParams = abi_param1;

    Xbyak::Xmm xmmAux8 = Xbyak::Xmm(8);
    Xbyak::Xmm xmmAux9 = Xbyak::Xmm(9);
    Vmm vmmFirstV = Vmm(0);
    Vmm vmmSecondV = Vmm(1);
    Vmm vmmGrtMask = Vmm(2);
    Vmm vmmOnes = Vmm(3);
    Vmm vmmMaxVec = Vmm(4);
    Xbyak::Ymm ymmAux5 = Xbyak::Ymm(5);
    Xbyak::Ymm ymmAux6 = Xbyak::Ymm(6);

    Xbyak::Opmask k_mask;
};

class CTCGreedyDecoderSeqLenImpl: public ExtLayerBase {
public:
    explicit CTCGreedyDecoderSeqLenImpl(const CNNLayer* layer) : mergeRepeated_(true) {
        try {
            std::string errPrefix = "CTCGreedyDecoderSeqLen layer with name '" + layer->name + "' ";
            if (layer->insData.size() < 2 || layer->insData.size() > 3)
                THROW_IE_EXCEPTION << errPrefix << "has invalid number of input edges: " << layer->insData.size();
            if (layer->outData.size() != 2)
                THROW_IE_EXCEPTION << errPrefix << "has invalid number of outputs edges: " << layer->outData.size();

            auto inData = layer->insData[DATA_INDEX].lock();
            auto sequenceLenData = layer->insData[SEQUENCE_LENGTH_INDEX].lock();
            if (!inData || !sequenceLenData)
                THROW_IE_EXCEPTION << errPrefix << "has nullable inputs.";
            if (inData->getTensorDesc().getPrecision() != Precision::FP32)
                THROW_IE_EXCEPTION << errPrefix << "has unsupported 'data' input precision: " << inData->getTensorDesc().getPrecision();
            if (sequenceLenData->getTensorDesc().getPrecision() != Precision::I32)
                THROW_IE_EXCEPTION << errPrefix << "has unsupported 'sequence_length' input precision: " << sequenceLenData->getTensorDesc().getPrecision();

            std::vector<DataConfigurator> inputConfigs{{ConfLayout::PLN, Precision::FP32}, {ConfLayout::PLN, Precision::I32}};

            if (layer->insData.size() > BLANK_INDEX) {
                auto blankIndexData = layer->insData[BLANK_INDEX].lock();
                if (!blankIndexData)
                    THROW_IE_EXCEPTION << errPrefix << "has nullable inputs.";
                if (blankIndexData->getTensorDesc().getPrecision() != Precision::I32)
                    THROW_IE_EXCEPTION << errPrefix << "has unsupported 'blank_index' input precision: " << blankIndexData->getTensorDesc().getPrecision();
                inputConfigs.push_back({ConfLayout::PLN, Precision::I32});
            }
            std::vector<DataConfigurator> outputConfigs{{ConfLayout::PLN, Precision::I32}, {ConfLayout::PLN, Precision::I32}};
            addConfig(layer, inputConfigs, outputConfigs);

            mergeRepeated_ = layer->GetParamAsBool("merge_repeated", true);

            jitGreedyDecoderConfT jpp;
            jpp.classesNum = inData->getTensorDesc().getDims()[2];
            if (mayiuse(cpu::avx2)) {
                kernel_.reset(new jitUniGreedyDecoderKernel<cpu::avx2>(jpp));
            }
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        const float* probabilities = inputs[DATA_INDEX]->cbuffer().as<const float*>() +
            inputs[DATA_INDEX]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const int* sequenceLengths = inputs[SEQUENCE_LENGTH_INDEX]->cbuffer().as<const int*>() +
            inputs[SEQUENCE_LENGTH_INDEX]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        int* outputSequences = outputs[0]->buffer().as<int*>();

        const size_t T = inputs[DATA_INDEX]->getTensorDesc().getDims()[0];
        const size_t B = inputs[DATA_INDEX]->getTensorDesc().getDims()[1];
        const int C = inputs[DATA_INDEX]->getTensorDesc().getDims()[2];
        const size_t CN = C * B;
        const size_t TN = T * B;
        const size_t CN1 = C * (B - 1);

        int blankIndex = C - 1;
        if (inputs.size() > BLANK_INDEX)
            blankIndex = (inputs[BLANK_INDEX]->cbuffer().as<const int*>() +
                inputs[BLANK_INDEX]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0];

static double du1 = 0.0;
static int c1 = 0;
auto start = std::chrono::steady_clock::now();

        auto threadBody = [&](const int ithr, const int nthr) {
            size_t start(0lu), end(0lu);
            splitter(TN, nthr, ithr, start, end);
            if (start >= end)
                return;
            size_t bStart = start / T;
            size_t tStart = start % B;

            size_t workCounter = start;

            for (size_t b = bStart; b < B; ++b) {
                int prev_class_idx = -1;
                size_t outputIndex = b * T;
                const float* probs = probabilities + b * C;
                const size_t actualSeqLen = sequenceLengths[b];
                if (actualSeqLen > T)
                    return; // err

                for (size_t t = tStart; t < actualSeqLen; ++t) {
                    int maxClassIdx = 0;
                    float maxProb = probs[0];
                    ++probs;

//                        auto arg = jitArgsGreedyDecoder();
//                        arg.prob = probs;
//                        (*kernel_)(&arg);
                    for (int c = 1; c < C; ++c, ++probs) {
                        if (*probs > maxProb) {
                            maxClassIdx = c;
                            maxProb = *probs;
                        }
                    }
                    if (maxClassIdx != blankIndex && maxClassIdx < C &&
                            !(mergeRepeated_ && maxClassIdx == prev_class_idx)) {
                        outputSequences[outputIndex++] = static_cast<float>(maxClassIdx);
                    }

                    prev_class_idx = maxClassIdx;
                    probs += CN1;

                    if (++workCounter >= end) {
                        return;
                    }
                }
                tStart = 0lu;
            }
        }; // thread body

        parallel_nt(0, threadBody);

auto end = std::chrono::steady_clock::now();
c1++;
du1 += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
if (c1 % 100 == 0) {
    printf("DU1: %f\b", du1 / c1);
}

        return OK;
    }

    const size_t DATA_INDEX = 0lu;
    const size_t SEQUENCE_LENGTH_INDEX = 1lu;
    const size_t BLANK_INDEX = 2lu;
    bool mergeRepeated_;

    std::shared_ptr<jitUniGreedyDecoderBase> kernel_;
};

REG_FACTORY_FOR(CTCGreedyDecoderSeqLenImpl, CTCGreedyDecoderSeqLen);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
