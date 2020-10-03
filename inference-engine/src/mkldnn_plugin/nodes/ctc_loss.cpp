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
        const size_t batchNum = logitsShape[0];
        const auto maxTime = logitsShape[1];
        const size_t classesNum = logitsShape[2];

        int blankIndex = classesNum - 1;
        if (inputs.size() > 4) {
            blankIndex = inputs[4]->cbuffer().as<const int*>()[0];
        }

        const size_t TC = maxTime * classesNum;

static unsigned c1 = 0, c2 = 0;
static double t1 = 0.;
static double t2 = 0.;

        std::vector<std::vector<std::vector<float>>> logProbabilities(batchNum);
        for (size_t b = 0lu; b < batchNum; b++) {
            logProbabilities[b].resize(logitsLength[b]);
            for (size_t ll = 0; ll < logitsLength[b]; ll++) {
                logProbabilities[b][ll].resize(labelsLength[b] * 2 + 1);
            }
        }
        std::vector<int> decodedTargetLenB(batchNum, 0);
        std::vector<std::vector<int>> targetDB(batchNum);

        auto threadBody_1 = [&](const int ithr, const int nthr) {
            size_t start(0lu), end(0lu);
            splitter(batchNum, nthr, ithr, start, end);
            if (start >= end)
                return;

            for (size_t b = start; b < end; b++) {
                const int actualLogitLen = logitsLength[b];
                const int actualTargetLen = labelsLength[b];
                size_t decodedTargetLen = 0lu;
                if (actualLogitLen < 0 || actualTargetLen < 0 || actualLogitLen > maxTime || actualTargetLen > actualLogitLen) {
                    std::string errorMsg = _logPrefix + ". Logit length cannot be greater than max sequence length. "
                        + "Label length cannot be greater than a logit length"
                        + " and both cannot be negative.\nMaxSeqLen: "
                        + std::to_string(maxTime) + "; Logit len: " + std::to_string(actualLogitLen)
                        + "; Label len: " + std::to_string(actualTargetLen);
                    errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                    returnCode = GENERAL_ERROR;
                    return;
                }

                // Decoding target: merge repeated characters if preprocess_collapse_repeated == True,
                // find unique elemnts if unique == True
                // Target indices with blanks before each index and a blank at the end.
                const int* target = &labels[b * maxTime];
                targetDB[b].resize(actualTargetLen * 2 + 1);
                auto& targetD = targetDB[b];
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
                decodedTargetLenB[b] = decodedTargetLen;
            } // for batch
        }; // threadBody_1

        parallel_nt(0, threadBody_1);
//printf("threadBody_1 end\n");
/*
for (int b = 0; b < batchNum; b++) {
    printf("B %d; decodedTargetLen: %d\ntargetD: ", b, decodedTargetLenB[b]);
    for (int s = 0; s < decodedTargetLenB[b]; s++)
        printf("%d; ", targetDB[b][s]);
    printf("\n");
}
*/
        auto threadBody_2 = [&](const int ithr, const int nthr) {
auto start1 = std::chrono::steady_clock::now();

            size_t start(0lu), end(0lu);
            size_t sB(0lu), sT(0lu);
//            if (batchNum >= nthr) {
/*                splitter(batchNum, nthr, ithr, start, end);
                if (start >= end)
                    return;
                sB = start;*/
//            } else {
                size_t workAmount = 0;
                for (size_t b = 0; b < batchNum; b++)
                    workAmount += logitsLength[b];
                splitter(workAmount, nthr, ithr, start, end);
                if (start >= end)
                    return;
                int64_t cw = 0, st = start;
                for (; sB < batchNum; sB++) {
                    cw += logitsLength[sB];
                    if (cw >= st) {
                        sT = logitsLength[sB] + st - cw;
                        break;
                    }
                }
//            }
//printf("ithr: %d; start: %lu; end: %lu; sB: %lu; sT: %lu\n", ithr, start, end, sB, sT);
            size_t workCounter = start;

            for (size_t b = sB; b < batchNum; b++) {
                const size_t actualLogitLen = logitsLength[b];
                const size_t BTC = b * TC;
                auto& targetD = targetDB[b];

//c1++;
//auto start1 = std::chrono::steady_clock::now();

                double expSum = 0.0;
                size_t btcT = BTC + sT * classesNum;
                size_t decodedTargetLen = decodedTargetLenB[b];
                // logProbabilities = ln_softmax[b][t][c] = logits[b][t][c] - ln(sum_c(exp(logits[b][t])))
                for (size_t t = sT; t < actualLogitLen; t++) {
                    expSum = 0.0;
                    for (size_t c = 0lu; c < classesNum; c++) {
                        expSum += std::exp(logits[btcT + c]);
                    }
                    for (size_t s = 0; s < decodedTargetLen; s++) {
                        logProbabilities[b][t][s] = logits[btcT + targetD[s]] - std::log(expSum);
                    }
                    btcT += classesNum;
                    workCounter++;
                    if (workCounter >= end) {
//printf("end ithr %d: b: %lu; t: %lu\n", ithr, b, t);
//c1++;
auto end1 = std::chrono::steady_clock::now();
/*t1 += std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();
if (c1 % 100 == 0) {
    printf("T1: %f\n", t1 / c1);
}*/
//printf("T1: %f; ", static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count()));
                        return;
                    }
                }
                sT = 0lu;
            }  // batch
            }; // thread
printf("\n");
//c1++;
//auto start1 = std::chrono::steady_clock::now();

        parallel_nt(0, threadBody_2);

/*auto end1 = std::chrono::steady_clock::now();
t1 += std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();
if (c1 % 100 == 0) {
    printf("T1: %f\n", t1 / c1);
}*/

/*for (int b = 0; b < batchNum; b++) {
    std::cout << "B_" << b << std::endl;
    for (int t = 0; t < logitsLength[b]; t++) {
        for (int s = 0; s < decodedTargetLenB[b]; s++) {
            std::cout << logProbabilities[b][t][s] << ";";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}*/

//printf("threadBody_2 end\n");

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

        auto threadBody_3 = [&](const int ithr, const int nthr) {
            size_t start(0lu), end(0lu);
            splitter(batchNum, nthr, ithr, start, end);
            if (start >= end)
                return;

        for (size_t b = start; b < end; b++) {
//c2++;
//auto start2 = std::chrono::steady_clock::now();

            auto& targetD = targetDB[b];
            int actualLogitLen = logitsLength[b];
            int decodedTargetLen = decodedTargetLenB[b];
            std::vector<std::vector<float>> logBeta(decodedTargetLen, std::vector<float>(actualLogitLen, -float_inf));

            for (int u = decodedTargetLen - 2; u < decodedTargetLen; u++)
                logBeta[u][actualLogitLen - 1] = 0.f;

            for (int t = actualLogitLen - 2; t >= 0; t--) {
                for (int u = std::max(0, decodedTargetLen - (2 * (actualLogitLen - t)));
                        u < std::min(decodedTargetLen, 2 * (t + 1)); u++) {
                    if (_ctcMergeRepeated || targetD[u] == blankIndex) {
                        logBeta[u][t] = sumLogs(logBeta[u][t],
                            logBeta[u][t + 1] + logProbabilities[b][t + 1][u]);
                    }

                    if (u + 1 < decodedTargetLen) {
                        logBeta[u][t] = sumLogs(logBeta[u][t],
                            logBeta[u + 1][t + 1] + logProbabilities[b][t + 1][u + 1]);
                    }

                    if (u + 2 < decodedTargetLen) {
                        const bool matching_labels_merge =
                            _ctcMergeRepeated && (targetD[u] == targetD[u + 2]);
                        if (targetD[u] != blankIndex && !matching_labels_merge) {
                            logBeta[u][t] = sumLogs(logBeta[u][t],
                                logBeta[u + 2][t + 1] + logProbabilities[b][t + 1][u + 2]);
                        }
                    }
                }
            }
//printf("Final loop end\n");

            std::vector<float> logAlpha(decodedTargetLen, -float_inf);

            logAlpha[0] = logProbabilities[b][0][0];
            size_t label_0 = (decodedTargetLen > 1) ? 1 : 0;
            logAlpha[1] = logProbabilities[b][0][label_0];

/*auto end2 = std::chrono::steady_clock::now();
t2 += std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();

if (c2 % 100 == 0)
    std::cout << "T2: " << t2 / c1 << std::endl;*/

            float log_p_z_x = -float_inf;
            for (int u = 0; u < decodedTargetLen; ++u) {
                log_p_z_x = sumLogs(log_p_z_x, logAlpha[u] + logBeta[u][0]);
            }
            dstData[b] = -log_p_z_x;
            } // batch
        }; // threadBody_3

c2++;
auto start2 = std::chrono::steady_clock::now();

        parallel_nt(0, threadBody_3);

auto end2 = std::chrono::steady_clock::now();
t2 += std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();

if (c2 % 100 == 0)
    std::cout << "T2: " << t2 / c2 << std::endl;

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

