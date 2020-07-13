// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/shape_util.hpp"

namespace ngraph
{
namespace runtime
{
namespace reference
{

template <typename T, typename U>
void CTCLoss(const T* logits,
             const Shape& logitsShape,
             const U* logitsLength,
             const U* labels,
             const U* labelsLength,
             const U* blankIndexP,
             const bool preprocessCollapseRepeated,
             const bool ctcMergeRepeated,
             const bool unique,
             T* output) {
    const size_t batchNum = logitsShape[0];
    const size_t maxTime = logitsShape[1];
    const size_t classesNum = logitsShape[2];
std::cout<<"preprocessCollapseRepeated: "<<preprocessCollapseRepeated<<std::endl;
std::cout<<"ctcMergeRepeated: "<<ctcMergeRepeated<<std::endl;
std::cout<<"unique: "<<unique<<std::endl;
    U blankIndex = classesNum - 1;
    if (blankIndexP != nullptr) {
        blankIndex = blankIndexP[0];
    }

    U* targetD = new U[maxTime];
    U* pathS = new U[maxTime];

    //const size_t BC = batchNum * classesNum;
    const size_t TC = maxTime * classesNum;

    for (size_t b = 0; b < batchNum; b++) {
std::cout<<"Batch: "<<b<<std::endl;
        U actualLogitLen = logitsLength[b];
        U actualTargetLen = labelsLength[b];
std::cout<<"actualLogitLen: "<<actualLogitLen<<std::endl;
std::cout<<"actualTargetLen: "<<actualTargetLen<<std::endl;
        if (actualLogitLen >= maxTime || actualTargetLen >= maxTime || actualTargetLen > actualLogitLen) {
            delete[] targetD;
            delete[] pathS;
            throw ngraph_error(std::string(
                "Logit or label length cannot be more than max sequence length. Also a label length cannot be greater than a logit length.\nMaxSeqLen: ")
                + std::to_string(maxTime) + "; Logit len: " + std::to_string(actualLogitLen)
                + "; Label len: " + std::to_string(actualTargetLen));
        }

        const U* target = &labels[b * maxTime];
        // Decoding target: merge repeated characters if preprocess_collapse_repeated == True,
        // find unique elemnts if unique == True
        size_t decodedTargetLen = 0lu;
        if (unique) {
            std::unordered_set<U> uniqVals;
            for (size_t t = 0lu; t < actualTargetLen; t++) {
                if (uniqVals.find(target[t]) != uniqVals.end()) {
                    continue;
                }
                uniqVals.insert(target[t]);
                targetD[decodedTargetLen++] = target[t];
            }
        } else if (preprocessCollapseRepeated) {
            U prevValue = target[0];
            targetD[decodedTargetLen++] = target[0];
            for (size_t t = 1lu; t < actualTargetLen; t++) {
                if (target[t] == prevValue) {
                    continue;
                }
                targetD[decodedTargetLen++] = target[t];
                prevValue = target[t];
            }
        } else {
            for (size_t t = 0lu; t < actualTargetLen; t++) {
                targetD[decodedTargetLen++] = target[t];
            }
        }
std::cout<<"decodedTargetLen: "<<decodedTargetLen<<std::endl;

        //const size_t bC = b * classesNum;
        const size_t bC = b * classesNum;
        const size_t BTC = b * TC;

        std::vector<T> kExp(actualLogitLen, 0);
        for (size_t t = 0; t < actualLogitLen; t++) {
            for (size_t c = 0; c < classesNum; c++) {
                kExp[t] += std::exp(logits[BTC + classesNum * t + c]);
            }

        }

        T res = 0;

        // Looking for aligned paths
        std::function<void(size_t targetIdx, size_t start, size_t end)> findPaths = 
                [&](size_t targetIdx, size_t start, size_t end) {
            if (end > actualLogitLen) {
                T ex = 0;
                T denom = 1;
                for (size_t t = 0; t < actualLogitLen; t++) {
std::cout<<"t: "<<t<<"; S: "<<pathS[t]<<"; denom: "<<kExp[t]<<"; logit: "<<logits[BTC + classesNum * t + pathS[t]]<<std::endl;
                    ex += logits[BTC + classesNum * t + pathS[t]];
                    denom *= kExp[t];
                }
                T prod = std::exp(ex) / denom;
                res += std::log(prod);

                return;
            }

            size_t nextIdx = targetIdx + 1;
            int64_t st64 = start;
            if (!ctcMergeRepeated) {
                for (size_t pos = start; pos < end; pos++) {
                    for (size_t bl = start; bl < pos; bl++) {
                        pathS[bl] = blankIndex;
                    }
                    pathS[pos] = targetD[targetIdx];
                    findPaths(nextIdx, pos + 1, end + 1);
                }
            } else {
                for (size_t pos = start; pos < end; pos++) {
                    for (size_t bl = start; bl < pos; bl++) {
                        pathS[bl] = blankIndex;
                    }
                    for (int64_t bl = pos; bl >= st64; bl--) {
                        pathS[bl] = targetD[targetIdx];
                        findPaths(nextIdx, pos + 1, end + 1);
                    }
                }
            }
        };

        findPaths(0lu, 0lu, actualLogitLen - decodedTargetLen + 1lu);

        output[b] = -res;
    }

    delete[] targetD;
    delete[] pathS;

} // CTCLoss

} // reference
} // runtime
} // ngraph
