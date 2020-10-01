// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"
#include "ie_parallel.hpp"
#include <chrono>

#include <cmath>


namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class CTCLossImpl : public ExtLayerBase {
public:
    explicit CTCLossImpl(const CNNLayer* layer) {
        _logPrefix = std::string("CTCLoss layer with name '") + layer->name + "'";

        if (layer->insData.size() != 4 && layer->insData.size() != 5)
            THROW_IE_EXCEPTION << _logPrefix << " has invalid inputs number.";

        _ctcMergeRepeated = layer->GetParamAsBool("ctc_merge_repeated", true);
        _preprocessCollapseRepeated = layer->GetParamAsBool("preprocess_collapse_repeated", false);
        _unique = layer->GetParamAsBool("unique", false);

        auto logitsData = layer->insData[0].lock();
        if (logitsData == nullptr)
            THROW_IE_EXCEPTION << _logPrefix << " has nullable logits data";
        auto logitsPrecision = logitsData->getTensorDesc().getPrecision();
        if (logitsPrecision == Precision::BF16)
            logitsPrecision = Precision::FP32;

        LayerConfig config;
        config.inConfs.resize(layer->insData.size());
        config.inConfs[0].desc = TensorDesc(logitsPrecision,
            logitsData->getTensorDesc().getDims(),
            TensorDesc::getLayoutByDims(logitsData->getTensorDesc().getDims()));
        auto intPrecision = Precision::I32;
        for (int i = 1; i < layer->insData.size(); i++) {
            auto data = layer->insData[i].lock();
            if (data == nullptr)
                THROW_IE_EXCEPTION << _logPrefix << " has nullable input data at " << i;
            config.inConfs[i].desc = TensorDesc(intPrecision,
                data->getTensorDesc().getDims(),
                TensorDesc::getLayoutByDims(data->getTensorDesc().getDims()));
        }

        DataConfig outConfig;
        auto& outDims = layer->outData[0]->getTensorDesc().getDims();
        outConfig.desc = TensorDesc(logitsPrecision,
            outDims,
            TensorDesc::getLayoutByDims(outDims));
        config.outConfs.push_back(outConfig);
        config.dynBatchSupport = false;

        confs.push_back(config);
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs,
                       std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        StatusCode returnCode = OK;

        const float* logits = inputs[0]->cbuffer().as<const float*>() +
            inputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const int* logitsLength = inputs[1]->cbuffer().as<const int*>() +
            inputs[1]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const int* labels = inputs[2]->cbuffer().as<const int*>() +
            inputs[2]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const int* labelsLength = inputs[3]->cbuffer().as<const int*>() +
            inputs[3]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        float* dstData = outputs[0]->buffer().as<float*>() +
            outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        const auto& logitsShape = inputs[0]->getTensorDesc().getDims();
        const auto batchNum = logitsShape[0];
        const auto maxTime = logitsShape[1];
        const auto classesNum = logitsShape[2];

        int blankIndex = classesNum - 1;
        if (inputs.size() > 4) {
            blankIndex = inputs[4]->cbuffer().as<const int*>()[0];
        }

        std::vector<int> targetD(2 * maxTime + 1);

        const size_t TC = maxTime * classesNum;

        auto thread_body = [&](const int ithr, const int nthr) {
            size_t start(0lu), end(0lu);
            splitter(batchNum, nthr, ithr, start, end);
            if (start >= end)
                return;

        for (size_t b = start; b < end; b++) {
            const int actualLogitLen = logitsLength[b];
            const int actualTargetLen = labelsLength[b];
            if (actualLogitLen < 0 || actualTargetLen < 0 || actualLogitLen > maxTime || actualTargetLen > maxTime
                    || actualTargetLen > actualLogitLen) {
                std::string errorMsg = _logPrefix + ". Logit or label length cannot be greater than max sequence length. "
                    + "Also a label length cannot be greater than a logit length"
                    + " and both cannot be negative.\nMaxSeqLen: "
                    + std::to_string(maxTime) + "; Logit len: " + std::to_string(actualLogitLen)
                    + "; Label len: " + std::to_string(actualTargetLen);
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                returnCode = GENERAL_ERROR;
                return;
            }

            const int* target = &labels[b * maxTime];
            // Decoding target: merge repeated characters if preprocess_collapse_repeated == True,
            // find unique elemnts if unique == True
            // Target indices with blanks before each index and a blank at the end.
            int decodedTargetLen = 0lu;
            if (_unique) {
                std::unordered_set<int> uniqVals;
                for (size_t t = 0lu; t < actualTargetLen; t++) {
                    if (uniqVals.find(target[t]) != uniqVals.end()) {
                        continue;
                    }
                    uniqVals.insert(target[t]);
                    targetD[decodedTargetLen++] = blankIndex;
                    targetD[decodedTargetLen++] = target[t];
                }
                targetD[decodedTargetLen++] = blankIndex;
            } else if (_preprocessCollapseRepeated) {
                int prevValue = target[0];
                targetD[decodedTargetLen++] = blankIndex;
                targetD[decodedTargetLen++] = target[0];
                for (size_t t = 1lu; t < actualTargetLen; t++) {
                    if (target[t] == prevValue) {
                        continue;
                    }
                    targetD[decodedTargetLen++] = blankIndex;
                    targetD[decodedTargetLen++] = target[t];
                    prevValue = target[t];
                }
                targetD[decodedTargetLen++] = blankIndex;
            } else {
                for (size_t t = 0lu; t < actualTargetLen; t++) {
                    targetD[decodedTargetLen++] = blankIndex;
                    targetD[decodedTargetLen++] = target[t];
                }
                targetD[decodedTargetLen++] = blankIndex;
            }

            const size_t BTC = b * TC;

            // logProbabilities = ln_softmax[b][t][c] = logits[b][t][c] - ln(sum_c(exp(logits[b][t])))
            // their:
            // logProbabilities = ln_softmax[b][t][c] = logits[b][t][c] - maxLogit - ln(sum_c(exp(logits[b][t] - maxLogit)))
static unsigned c1 = 0;
static double t1 = 0.;
static double t2 = 0.;
c1++;
auto start1 = std::chrono::steady_clock::now();

            std::vector<std::vector<float>> logProbabilities(actualLogitLen, std::vector<float>(decodedTargetLen));
            float maxLogit, expSum, addendum;
            size_t btcT = BTC;
            for (size_t t = 0lu; t < actualLogitLen; t++) {
                maxLogit = -std::numeric_limits<float>::max();
                for (size_t c = 0lu; c < classesNum; c++) {
                    if (logits[btcT + c] > maxLogit)
                        maxLogit = logits[btcT + c];
                }
                expSum = 0.f;
                for (size_t c = 0lu; c < classesNum; c++) {
                    expSum += std::exp(logits[btcT + c] - maxLogit);
                }
                addendum = -(maxLogit + std::log(expSum));
                for (size_t s = 0; s < decodedTargetLen; s++) {
                    logProbabilities[t][s] = logits[btcT + targetD[s]] + addendum;
                }
                btcT += classesNum;
            }

auto end1 = std::chrono::steady_clock::now();
t1 += std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();

            const auto float_inf = std::numeric_limits<float>::infinity();

            auto sumLogs = [&float_inf](float log1, float log2) {
                if (log1 == -float_inf) {
                    return log2;
                } else if (log2 == -float_inf) {
                    return log1;
                } else {
                    if (log1 > log2)
                        return log1 + std::log1pf(std::exp(log2 - log1));
                    else
                        return log2 + std::log1pf(std::exp(log1 - log2));
                }
            };

auto start2 = std::chrono::steady_clock::now();

            std::vector<std::vector<float>> logBeta(decodedTargetLen, std::vector<float>(actualLogitLen, -float_inf));

            for (int u = decodedTargetLen - 2; u < decodedTargetLen; u++)
                logBeta[u][actualLogitLen - 1] = 0.f;

            for (int t = actualLogitLen - 2; t >= 0; t--) {
                for (int u = std::max(0, decodedTargetLen - (2 * (actualLogitLen - t)));
                        u < std::min(decodedTargetLen, 2 * (t + 1)); u++) {
                    if (_ctcMergeRepeated || targetD[u] == blankIndex) {
                        logBeta[u][t] = sumLogs(logBeta[u][t],
                            logBeta[u][t + 1] + logProbabilities[t + 1][u]);
                    }

                    if (u + 1 < decodedTargetLen) {
                        logBeta[u][t] = sumLogs(logBeta[u][t],
                            logBeta[u + 1][t + 1] + logProbabilities[t + 1][u + 1]);
                    }

                    if (u + 2 < decodedTargetLen) {
                        const bool matching_labels_merge =
                            _ctcMergeRepeated && (targetD[u] == targetD[u + 2]);
                        if (targetD[u] != blankIndex && !matching_labels_merge) {
                            logBeta[u][t] = sumLogs(logBeta[u][t],
                                logBeta[u + 2][t + 1] + logProbabilities[t + 1][u + 2]);
                        }
                    }
                }
            }

            std::vector<float> logAlpha(decodedTargetLen, -float_inf);

            logAlpha[0] = logProbabilities[0][0];
            size_t label_0 = (decodedTargetLen > 1) ? 1 : 0;
            logAlpha[1] = logProbabilities[0][label_0];

auto end2 = std::chrono::steady_clock::now();
t2 += std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();

if (c1 % 100 == 0)
    std::cout << "T1: " << t1 / c1 << "; T2: " << t2 / c1 << std::endl;

            float log_p_z_x = -float_inf;
            for (int u = 0; u < decodedTargetLen; ++u) {
                log_p_z_x = sumLogs(log_p_z_x, logAlpha[u] + logBeta[u][0]);
            }
            dstData[b] = -log_p_z_x;
            }
        };
        parallel_nt(0, thread_body);

        return returnCode;
    } // execute

protected:
    bool _ctcMergeRepeated;
    bool _preprocessCollapseRepeated;
    bool _unique;

    std::string _logPrefix;
};

REG_FACTORY_FOR(CTCLossImpl, CTCLoss);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine

