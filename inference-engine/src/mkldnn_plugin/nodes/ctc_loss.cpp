// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"
#include "ie_parallel.hpp"

#include <cmath>
#include <forward_list>

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

        LayerConfig config;
        config.inConfs.resize(layer->insData.size());
        for (int i = 0; i < layer->insData.size(); i++) {
            auto data = layer->insData[i].lock();
            if (data == nullptr)
                THROW_IE_EXCEPTION << _logPrefix << " has nullable input data";
            auto prc = data->getTensorDesc().getPrecision();
            if (prc == Precision::BF16)
                prc = Precision::FP32;
            config.inConfs[i].desc = TensorDesc(prc,
                data->getTensorDesc().getDims(),
                TensorDesc::getLayoutByDims(data->getTensorDesc().getDims()));
        }

        auto logitsData = layer->insData[0].lock();
        if (logitsData == nullptr)
            THROW_IE_EXCEPTION << _logPrefix << " has nullable logits data.";
        DataConfig outConfig;
        auto& outDims = layer->outData[0]->getTensorDesc().getDims();
        outConfig.desc = TensorDesc(logitsData->getTensorDesc().getPrecision(),
            outDims,
            TensorDesc::getLayoutByDims(outDims));
        config.outConfs.push_back(outConfig);
        config.dynBatchSupport = false;

        confs.push_back(config);
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs,
                       std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        switch (inputs[1]->getTensorDesc().getPrecision()) {
            case Precision::I32: {
                return processData<PrecisionTrait<Precision::I32>::value_type>(inputs, outputs, resp);
            }
            case Precision::I64: {
                return processData<PrecisionTrait<Precision::I64>::value_type>(inputs, outputs, resp);
            }
            default: {
                if (resp) {
                    std::string errorMsg = _logPrefix + " does not support precision '"
                            + std::string(inputs[1]->getTensorDesc().getPrecision().name()) + "'";
                    errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                }
                return GENERAL_ERROR;
            }
        }
    }

protected:
    struct PathNode {
        PathNode() {
            _children.reserve(3);
        }

        PathNode* initialize(int64_t depth, int64_t targetId, float logProbability, bool isBlank = false) {
            _children.clear();
            _depth = depth;
            _targetId = targetId;
            _logProbability = logProbability;
            _isBlank = isBlank;

            return this;
        }

        std::vector<PathNode*> _children;
        int64_t _depth;
        int64_t _targetId;
        float _logProbability;
        bool _isBlank;
    };

    struct NodesPull {
        NodesPull(size_t size) : pull(std::deque<PathNode>(size)), pullPointers(std::deque<PathNode*>(size)) {
            for (size_t i = 0lu; i < size; i++)
                pullPointers[i] = &pull[i];
        }
        PathNode* getNode() {
            if (pullPointers.empty()) {
std::cout << "getNode() pullPointers.empty()" << std::endl;
                // Something went wrong. Throw exception?
                pull.push_back(PathNode());
                return &pull[pull.size() -  1];
            }
            PathNode* res = pullPointers.back();
            pullPointers.pop_back();
            return res;
        }
        void returnNode(PathNode* nodePtr) {
            pullPointers.push_back(nodePtr);
        }
    protected:
        std::deque<PathNode> pull;
        std::deque<PathNode*> pullPointers;
    };

    template<typename I1>
    StatusCode processData(
                std::vector<Blob::Ptr>& inputs,
                std::vector<Blob::Ptr>& outputs,
                ResponseDesc* resp) noexcept {
        switch (inputs[2]->getTensorDesc().getPrecision()) {
            case Precision::I32: {
                return processData<I1, PrecisionTrait<Precision::I32>::value_type>(inputs, outputs, resp);
            }
            case Precision::I64: {
                return processData<I1, PrecisionTrait<Precision::I64>::value_type>(inputs, outputs, resp);
            }
            default: {
                if (resp) {
                    std::string errorMsg = _logPrefix + " does not support labels precision '"
                            + std::string(inputs[2]->getTensorDesc().getPrecision().name()) + "'";
                    errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                }
                return GENERAL_ERROR;
            }
        }
    }

    template<typename I1, typename I2>
    StatusCode processData(
                std::vector<Blob::Ptr>& inputs,
                std::vector<Blob::Ptr>& outputs,
                ResponseDesc* resp) noexcept {
        const float* logits = inputs[0]->cbuffer().as<const float*>() +
            inputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const I1* logitsLength = inputs[1]->cbuffer().as<const I1*>() +
            inputs[1]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const I2* labels = inputs[2]->cbuffer().as<const I2*>() +
            inputs[2]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        const I1* labelsLength = inputs[3]->cbuffer().as<const I1*>() +
            inputs[3]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        float* dstData = outputs[0]->buffer().as<float*>() +
            outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        const auto& logitsShape = inputs[0]->getTensorDesc().getDims();
        const auto batchNum = logitsShape[0];
        const auto maxTime = logitsShape[1];
        const auto classesNum = logitsShape[2];

        I2 blankIndex = classesNum - 1;
        if (inputs.size() > 4) {
            blankIndex = inputs[4]->cbuffer().as<const I2*>()[0];
        }

        std::vector<I2> targetD(maxTime);

        const size_t TC = maxTime * classesNum;

        for (size_t b = 0; b < batchNum; b++) {
//std::cout << "BATCH: " << b << std::endl;
            const I1 actualLogitLen = logitsLength[b];
            const I1 actualTargetLen = labelsLength[b];
            if (actualLogitLen > maxTime || actualTargetLen > maxTime || actualTargetLen > actualLogitLen) {
                std::string errorMsg = _logPrefix + ". Logit or label length cannot greater than max sequence length. "
                    + "Also a label length cannot be greater than a logit length.\nMaxSeqLen: "
                    + std::to_string(maxTime) + "; Logit len: " + std::to_string(actualLogitLen)
                    + "; Label len: " + std::to_string(actualTargetLen);
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                return GENERAL_ERROR;
            }

            const I2* target = &labels[b * maxTime];
            // Decoding target: merge repeated characters if preprocess_collapse_repeated == True,
            // find unique elemnts if unique == True
            size_t decodedTargetLen = 0lu;
            if (_unique) {
                std::unordered_set<I2> uniqVals;
                for (size_t t = 0lu; t < actualTargetLen; t++) {
                    if (uniqVals.find(target[t]) != uniqVals.end()) {
                        continue;
                    }
                    uniqVals.insert(target[t]);
                    targetD[decodedTargetLen++] = target[t];
                }
            } else if (_preprocessCollapseRepeated) {
                I2 prevValue = target[0];
                targetD[decodedTargetLen++] = target[0];
                for (size_t t = 1lu; t < actualTargetLen; t++) {
                    if (target[t] == prevValue) {
                        continue;
                    }
                    targetD[decodedTargetLen++] = target[t];
                    prevValue = target[t];
                }
            } else {
                std::copy(target, target + actualTargetLen, targetD.data());
                decodedTargetLen = actualTargetLen;
            }

            const size_t BTC = b * TC;

            std::vector<std::unordered_map<size_t, float>> logProbabilities(actualLogitLen);
            float logProb = 0.f, kExp = 0.f;
            for (size_t t = 0; t < actualLogitLen; t++) {
                kExp = 0.f;
                const size_t btcT = BTC + classesNum * t;
                for (size_t c = 0; c < classesNum; c++) {
                    kExp += std::exp(logits[btcT + c]);
                }
                // Not all are required. Add filter and check performance
                for (size_t s = 0; s < decodedTargetLen; s++) {
                    logProb = logits[btcT + targetD[s]] - std::log(kExp);
                    logProbabilities[t].insert({targetD[s], logProb});
                }
                logProb = logits[btcT + blankIndex] - std::log(kExp);
                logProbabilities[t].insert({blankIndex, logProb});
            }


            const auto float_inf = std::numeric_limits<float>::infinity();
            float lnSum = -float_inf;
            auto sumLogs = [&float_inf](float log1, float log2) {
                if (log1 == -float_inf) {
                    return log2;
                } else if (log2 != -float_inf) {
                    if (log1 > log2)
                        return log1 + std::log1pf(std::exp(log2 - log1));
                    else
                        return log2 + std::log1pf(std::exp(log1 - log2));
                }
                return -float_inf;
            };

            const int64_t singleTargetDepth = actualLogitLen - decodedTargetLen;
            const int64_t lastTargetID = decodedTargetLen - 1;
            const int64_t lastLogitID = actualLogitLen - 1;

            NodesPull nodesPull(actualLogitLen * 100);
            PathNode* zeroNode = nodesPull.getNode()->initialize(0, -1, -float_inf);
            std::vector<PathNode*> pathStack;
            pathStack.reserve(actualLogitLen);

            if (singleTargetDepth > 0)
                zeroNode->_children.push_back(nodesPull.getNode()->initialize(0, -1,
                    logProbabilities[0].find(blankIndex)->second, true));
            pathStack.push_back(zeroNode);
            pathStack.push_back(nodesPull.getNode()->initialize(0, 0,
                logProbabilities[0].find(targetD[0])->second));
//std::cout << "s: " << (pathStack.back()->_isBlank ? blankIndex : targetD[pathStack.back()->_targetId]) << std::endl;

            while (!pathStack.empty()) {
                int64_t nextDepth = pathStack.back()->_depth + 1;
                if (nextDepth == lastLogitID) {
                    if (pathStack.back()->_targetId == lastTargetID) {
                        if (!pathStack.back()->_isBlank) {
                            lnSum = sumLogs(lnSum, pathStack.back()->_logProbability + logProbabilities[nextDepth].find(targetD[pathStack.back()->_targetId])->second);
//std::cout << "s: " <<  targetD[pathStack.back()->_targetId] << std::endl;
//std::cout << "lnSum: " << lnSum << std::endl;
                            lnSum = sumLogs(lnSum, pathStack.back()->_logProbability + logProbabilities[nextDepth].find(blankIndex)->second);
//std::cout << "s: " <<  blankIndex << std::endl;
//std::cout << "lnSum: " << lnSum << std::endl;
                        } else {
                            lnSum = sumLogs(lnSum, pathStack.back()->_logProbability + logProbabilities[nextDepth].find(blankIndex)->second);
//std::cout << "s: " << blankIndex << std::endl;
//std::cout << "lnSum: " << lnSum << std::endl;
}
                    } else {
                        lnSum = sumLogs(lnSum, pathStack.back()->_logProbability + logProbabilities[nextDepth].find(targetD[pathStack.back()->_targetId + 1])->second);
//std::cout << "s: " <<  targetD[pathStack.back()->_targetId + 1] << std::endl;
//std::cout << "lnSum: " << lnSum << std::endl;
                    }

                    while (!pathStack.empty()) {
                        nodesPull.returnNode(pathStack.back());
                        pathStack.pop_back();
                        if (pathStack.empty() || !pathStack.back()->_children.empty())
                            break;
                    }
                    if (pathStack.empty())
                        break;
                    auto next = pathStack.back()->_children.back();
                    pathStack.back()->_children.pop_back();
                    pathStack.push_back(next);
//std::cout << "s: " <<  (pathStack.back()->_isBlank ? blankIndex : targetD[pathStack.back()->_targetId]) << std::endl;
                } else {
                    auto& children = pathStack.back()->_children;
                    if (pathStack.back()->_targetId + singleTargetDepth > pathStack.back()->_depth) {
                        if (!pathStack.back()->_isBlank)
                            children.push_back(nodesPull.getNode()->initialize(nextDepth, pathStack.back()->_targetId,
                                pathStack.back()->_logProbability + logProbabilities[nextDepth].find(targetD[pathStack.back()->_targetId])->second));
                        children.push_back(nodesPull.getNode()->initialize(nextDepth, pathStack.back()->_targetId,
                            pathStack.back()->_logProbability + logProbabilities[nextDepth].find(blankIndex)->second, true));
                        if (pathStack.back()->_targetId < lastTargetID)
                            children.push_back(nodesPull.getNode()->initialize(nextDepth, pathStack.back()->_targetId + 1,
                                pathStack.back()->_logProbability + logProbabilities[nextDepth].find(targetD[pathStack.back()->_targetId + 1])->second));
                    } else if (pathStack.back()->_targetId + singleTargetDepth == pathStack.back()->_depth) {
                        children.push_back(nodesPull.getNode()->initialize(nextDepth, pathStack.back()->_targetId + 1,
                            pathStack.back()->_logProbability + logProbabilities[nextDepth].find(targetD[pathStack.back()->_targetId + 1])->second));
                    }

                    pathStack.push_back(pathStack.back()->_children.back());
                    children.pop_back();
//std::cout << "s: " << (pathStack.back()->_isBlank ? blankIndex : targetD[pathStack.back()->_targetId]) << std::endl;
                }
            }


/*            size_t work_amount = actualLogitLen - decodedTargetLen + 1lu;
            std::vector<float> sumPerThread(parallel_get_max_threads(), -float_inf);

            // Looking for aligned paths
            auto thread_body = [&](const int ithr, const int nthr) {
                size_t start0(0lu), end0(0lu);
                splitter(work_amount, nthr, ithr, start0, end0);
                if (start0 >= end0)
                    return;
                if (ithr >= sumPerThread.size())
                    sumPerThread.push_back(-float_inf);

                std::function<void(size_t, size_t, size_t, float)> findPaths =
                        [&](size_t targetIdx, size_t start, size_t end, float prevLogProb) {
                    if (end > actualLogitLen) {
                        if (sumPerThread[ithr] == -float_inf) {
                            sumPerThread[ithr] = prevLogProb;
                        } else if (prevLogProb != -float_inf) {
                            if (sumPerThread[ithr] > prevLogProb)
                                sumPerThread[ithr] = sumPerThread[ithr] + std::log1pf(std::exp(prevLogProb - sumPerThread[ithr]));
                            else
                                sumPerThread[ithr] = prevLogProb + std::log1pf(std::exp(sumPerThread[ithr] - prevLogProb));
                        }
                        return;
                    }

                    size_t nextIdx = targetIdx + 1;
                    int64_t st64 = start;
                    float newLogProb = prevLogProb;
                    if (!_ctcMergeRepeated) {
                        for (size_t pos = start; pos < end; pos++) {
                            newLogProb = prevLogProb;
                            for (size_t bl = start; bl < pos; bl++) {
                                newLogProb += logProbabilities[bl].find(blankIndex)->second;
                            }
                            newLogProb += logProbabilities[pos].find(targetD[targetIdx])->second;
                            if (end == actualLogitLen) {
                                for (int64_t ble = pos + 1; ble < actualLogitLen; ble++) {
                                    newLogProb += logProbabilities[ble].find(blankIndex)->second;
                                }
                            }
                            findPaths(nextIdx, pos + 1, end + 1, newLogProb);
                        }
                    } else {
                        for (size_t pos = start; pos < end; pos++) {
                            newLogProb = prevLogProb;
                        size_t next_start = pos + 1;
                        for (size_t bl = start; bl < pos; bl++) {
                            newLogProb += logProbabilities[bl].find(blankIndex)->second;
                        }
                        if (end == actualLogitLen) {
                            for (int64_t ble = pos + 1; ble < actualLogitLen; ble++) {
                                newLogProb += logProbabilities[ble].find(blankIndex)->second;
                            }
                        }
                        if (targetIdx < decodedTargetLen - 1
                               && targetD[targetIdx] == targetD[targetIdx + 1]) {
                            newLogProb += logProbabilities[next_start++].find(blankIndex)->second;
                        }
                        for (int64_t bl = pos; bl >= st64; bl--) {
                            newLogProb += logProbabilities[bl].find(targetD[targetIdx])->second;
                            findPaths(nextIdx, next_start, end + 1, newLogProb);
                            if (bl > 0)
                                newLogProb -= logProbabilities[bl - 1].find(blankIndex)->second;
                            }
                        }
                    }
                }; // findPaths

                // First tartget symbol
                int64_t st64 = start0;
                float newLogProb = 0.f;
                if (!_ctcMergeRepeated) {
                    for (size_t pos = start0; pos < end0; pos++) {
                        newLogProb = 0.f;
                        for (size_t bl = 0; bl < pos; bl++) {
                            newLogProb += logProbabilities[bl].find(blankIndex)->second;
                        }
                        newLogProb += logProbabilities[pos].find(targetD[0])->second;
                        if (work_amount == actualLogitLen) {
                            for (int64_t ble = pos + 1; ble < actualLogitLen; ble++) {
                                newLogProb += logProbabilities[ble].find(blankIndex)->second;
                            }
                        }
                        if (decodedTargetLen > 1) {
                            findPaths(1, pos + 1, work_amount + 1, newLogProb);;
                        } else {
                            if (sumPerThread[ithr] == -float_inf)
                                sumPerThread[ithr] = newLogProb;
                            else if (newLogProb != -float_inf)
                                sumPerThread[ithr] = sumPerThread[ithr] + std::log1pf(std::exp(newLogProb - sumPerThread[ithr]));
                        }
                    }
                } else {
                    for (size_t pos = start0; pos < end0; pos++) {
                        newLogProb = 0.f;
                        size_t next_start = pos + 1;
                        for (size_t bl = 0; bl < pos; bl++) {
                            newLogProb += logProbabilities[bl].find(blankIndex)->second;
                        }
                        if (work_amount == actualLogitLen) {
                            for (int64_t ble = pos + 1; ble < actualLogitLen; ble++) {
                                newLogProb += logProbabilities[ble].find(blankIndex)->second;
                            }
                        }
                        if (decodedTargetLen > 1
                               && targetD[0] == targetD[1]) {
                            newLogProb += logProbabilities[next_start++].find(blankIndex)->second;
                        }
                        for (int64_t bl = pos; bl >= 0; bl--) {
                            newLogProb += logProbabilities[bl].find(targetD[0])->second;
                            if (decodedTargetLen > 1) {
                                findPaths(1, next_start, work_amount + 1, newLogProb);
                            } else {
                                if (sumPerThread[ithr] == -float_inf)
                                    sumPerThread[ithr] = newLogProb;
                                else if (newLogProb != -float_inf)
                                    sumPerThread[ithr] = sumPerThread[ithr] + std::log1pf(std::exp(newLogProb - sumPerThread[ithr]));
                            }
                            if (bl > 0)
                                newLogProb -= logProbabilities[bl - 1].find(blankIndex)->second;
                        }
                    }
                }
            }; // thread_body
*/
/*            parallel_nt(0, thread_body);

            float res = -float_inf;

            for (auto sum : sumPerThread) {
                if (res == -float_inf) {
                    res = sum;
                } else if (sum != -float_inf) {
                    if (res > sum)
                        res = res + std::log1pf(std::exp(sum - res));
                    else
                        res = sum + std::log1pf(std::exp(res - sum));
                }
            }*/

            dstData[b] = -lnSum;
        } // for (size_t b = 0; b < batchNum; b++)

        return OK;
    } // processData

    bool _ctcMergeRepeated;
    bool _preprocessCollapseRepeated;
    bool _unique;

    std::string _logPrefix;
};

REG_FACTORY_FOR(CTCLossImpl, CTCLoss);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine

