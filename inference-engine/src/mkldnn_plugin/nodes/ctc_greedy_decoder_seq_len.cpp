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
        mov(regTmpDst, ptr[regParams + GET_OFF(tmpDst)]);
        mov(regMaxClassIdx, ptr[regParams + GET_OFF(maxClassIdx)]);

        const int dataStep = vlen / sizeof(float);

        uni_vmovups(vmmFirstV, ptr[regProbs]);
        int c = jpp.classesNum;
        for (; c > dataStep; c -= dataStep) {
            add(regProbs, vlen);
            uni_vmovups(vmmSecondV, ptr[regProbs]);
            vmaxps(vmmFirstV, vmmFirstV, vmmSecondV);
        }

        Xbyak::Label setLabel;
        Xbyak::Label nextLabel;
        Xbyak::Label loopLabel;
        uni_vmovups(ptr[regTmpDst], vmmFirstV);
        mov(regMaxProb, ptr[regTmpDst]);
        mov(regMaxIdx, 0);
        mov(regTmp, 0);
        L(loopLabel);
        {
            add(regTmpDst, sizeof(float));
            mov(regCurProb, ptr[regTmpDst]);
            cmp(regCurProb, regMaxProb);
            jg(setLabel, T_NEAR);
            jmp(nextLabel);
            L(setLabel);
            mov(regMaxProb, regCurProb);
            mov(regMaxIdx, regTmp);
            L(nextLabel);
            add(regTmp, 1);
            cmp(regTmp, dataStep);
            jl(loopLabel);
        }

//        float floatSize = sizeof(float);
        Xbyak::Label loopLabel2, foundLabel;
        mov(regProbs, ptr[regParams + GET_OFF(probs)]);
//        mov(st0, regMaxIdx);
//        fld(regSt0);
//        mov(st1, sizeof(float));
        mov(eax, regMaxIdx);
        mov(regTmp, sizeof(int));
        mul(regTmp);
//        mov();
//        fstp(ptr[regTmp]);
        add(regProbs, eax);
        mov(regTmp, 0);
        L(loopLabel2);
        {
            mov(regCurProb, ptr[regProbs]);
            cmp(regCurProb, regMaxProb);
            je(foundLabel, T_NEAR);
            add(regProbs, vlen);
            add(regTmp, dataStep);
            cmp(regTmp, jpp.classesNum);
            jl(loopLabel2);
        }
        L(foundLabel);
        mov(regMaxIdx, regTmp);

        mov(ptr[regMaxClassIdx], regMaxIdx);
//        for (; c >= 0; c--) {
//            ;
//        }

        this->postamble();

        ker_ = (decltype(ker_))this->getCode();
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
//    Xbyak::Reg64 regWorkAmount = r14;
    Xbyak::Reg64 regTmp = r15;

    Xbyak::Fpu regSt0 = st0;
    Xbyak::Fpu regSt1 = st1;

    Xbyak::Reg64 regParams = abi_param1;

    Vmm vmmFirstV = Vmm(0);
    Vmm vmmSecondV = Vmm(1);
//    Vmm vmmSrcIdx = Vmm(2);
//    Vmm vmmMult = Vmm(3);
//    Vmm vmmAcc = Vmm(4);
//    Vmm vmmOnes = Vmm(5);
//    Vmm vmmIndicesSteps = Vmm(7);
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
