// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ngraph::Node> makeCTCGreedyDecoderSeqLen(
        const ngraph::Output<Node>& inputData,
        int blankIndex,
        bool mergeRepeated,
        const element::Type& idxPrec) {
    auto inputDataShape = inputData.get_shape();
    size_t T = inputDataShape[0];
    size_t B = inputDataShape[1];

    std::mt19937 gen(1);
    std::uniform_int_distribution<unsigned long> dist(0, T);

    std::vector<int> sequenceLenData(B);
    for (int b = 0; b < B; b++) {
        int len = dist(gen);
        sequenceLenData[b] = len;
    }

    auto sequenceLenNode = makeConstant(idxPrec, {B}, sequenceLenData);

    std::vector<int> blankIdxData = {blankIndex};
    auto blankIndexNode = makeConstant(idxPrec, {1}, blankIdxData);

    std::shared_ptr<opset1::CTCGreedyDecoder> CTCGreedyDecoderSeqLenNode;
    // = std::make_shared<opset1::CTCGreedyDecoder>(inputData, sequenceLenNode, mergeRepeated);

    return CTCGreedyDecoderSeqLenNode;
}
}  // namespace builder
}  // namespace ngraph
