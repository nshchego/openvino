// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"
#include "ie_parallel.hpp"

#include <cmath>
#include <vector>
#include <string>
#include <chrono>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class CTCGreedyDecoderImpl: public ExtLayerBase {
public:
    explicit CTCGreedyDecoderImpl(const CNNLayer* layer) : mergeRepeated_(true), isVersion1_(false) {
        try {
            std::string errPrefix = "CTCGreedyDecoder layer with name '" + layer->name + "' ";
            if (layer->insData.size() < 2 || layer->insData.size() > 3)
                THROW_IE_EXCEPTION << errPrefix << "has invalid number of input edges: " << layer->insData.size();
            if (layer->outData.empty() || layer->outData.size() > 2)
                THROW_IE_EXCEPTION << errPrefix << "has invalid number of outputs edges: " << layer->outData.size();
            if (layer->outData.size() == 1)
                isVersion1_ = true;

            auto inData = layer->insData[DATA_INDEX].lock();
            auto sequenceLenData = layer->insData[SEQUENCE_LENGTH_INDEX].lock();
            if (!inData || !sequenceLenData)
                THROW_IE_EXCEPTION << errPrefix << "has nullable inputs.";
            if (inData->getTensorDesc().getPrecision() != Precision::FP32)
                THROW_IE_EXCEPTION << errPrefix << "has unsupported 'data' input precision: " << inData->getTensorDesc().getPrecision();

            std::vector<DataConfigurator> inputConfigs;
            inputConfigs.push_back({ConfLayout::PLN, Precision::FP32});
            std::vector<DataConfigurator> outputConfigs;

            if (isVersion1_) {
                if (sequenceLenData->getTensorDesc().getPrecision() != Precision::FP32)
                    THROW_IE_EXCEPTION << errPrefix << "has unsupported 'sequence_length' input precision: " << sequenceLenData->getTensorDesc().getPrecision();
                inputConfigs.push_back({ConfLayout::PLN, Precision::FP32});
                outputConfigs.push_back({ConfLayout::PLN, Precision::FP32});
            } else {
                if (sequenceLenData->getTensorDesc().getPrecision() != Precision::I32)
                    THROW_IE_EXCEPTION << errPrefix << "has unsupported 'sequence_length' input precision: " << sequenceLenData->getTensorDesc().getPrecision();
                inputConfigs.push_back({ConfLayout::PLN, Precision::I32});
                if (layer->insData.size() > BLANK_INDEX) {
                    auto blankIndexData = layer->insData[BLANK_INDEX].lock();
                    if (!blankIndexData)
                        THROW_IE_EXCEPTION << errPrefix << "has nullable inputs.";
                    if (blankIndexData->getTensorDesc().getPrecision() != Precision::I32)
                        THROW_IE_EXCEPTION << errPrefix << "has unsupported 'blank_index' input precision: " << blankIndexData->getTensorDesc().getPrecision();
                    inputConfigs.push_back({ConfLayout::PLN, Precision::I32});
                }
                outputConfigs.push_back({ConfLayout::PLN, Precision::I32});
                outputConfigs.push_back({ConfLayout::PLN, Precision::I32});
            }
            mergeRepeated_ = layer->GetParamAsBool("merge_repeated", true);

            addConfig(layer, inputConfigs, outputConfigs);
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        const float* probabilities = inputs[DATA_INDEX]->cbuffer().as<const float*>() +
            inputs[DATA_INDEX]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        const size_t T = inputs[DATA_INDEX]->getTensorDesc().getDims()[0];
        const size_t B = inputs[DATA_INDEX]->getTensorDesc().getDims()[1];
        const int C = inputs[DATA_INDEX]->getTensorDesc().getDims()[2];
        const size_t CN = C * B;
        const size_t TN = T * B;
        const size_t CN1 = C * (B - 1);

        int blankIndex = C - 1;

static double du1 = 0.0;
static int c1 = 0;
auto start = std::chrono::steady_clock::now();

        if (isVersion1_) {
            const float* sequenceLengths = inputs[SEQUENCE_LENGTH_INDEX]->cbuffer().as<const float*>() +
                inputs[SEQUENCE_LENGTH_INDEX]->getTensorDesc().getBlockingDesc().getOffsetPadding();
            float* outputSequences = outputs[0]->buffer().as<float*>();

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

                    for (size_t t = tStart; t < T; ++t) {
                        if (sequenceLengths[B * t + b] == 0.f) {
                            break;
                        }
                        int maxClassIdx = 0;
                        float maxProb = probs[0];
                        ++probs;

                        for (int c = 1; c < C; ++c, ++probs) {
                            if (*probs > maxProb) {
                                maxClassIdx = c;
                                maxProb = *probs;
                            }
                        }
                        if (maxClassIdx < blankIndex &&
                                !(mergeRepeated_ && maxClassIdx == prev_class_idx)) {
                            outputSequences[outputIndex++] = static_cast<float>(maxClassIdx);
                        }

                        prev_class_idx = maxClassIdx;
                        probs += CN1;

                        if (++workCounter >= end) {
                            return;
                        }
                    }
                    std::fill(outputSequences + outputIndex, outputSequences + (b + 1) * T, -1.f);
                    tStart = 0lu;
                }
            }; // thread body

            parallel_nt(0, threadBody);
        } else {
            const int* sequenceLengths = inputs[SEQUENCE_LENGTH_INDEX]->cbuffer().as<const int*>() +
                inputs[SEQUENCE_LENGTH_INDEX]->getTensorDesc().getBlockingDesc().getOffsetPadding();
            if (inputs.size() > BLANK_INDEX)
                blankIndex = (inputs[BLANK_INDEX]->cbuffer().as<const int*>() +
                    inputs[BLANK_INDEX]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0];
            int* outputSequences = outputs[0]->buffer().as<int*>();

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
        }

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
    bool isVersion1_;
};

REG_FACTORY_FOR(CTCGreedyDecoderImpl, CTCGreedyDecoder);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
