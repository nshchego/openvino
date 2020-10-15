// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<Node> makeGatherND(
        const ngraph::Output<Node>& dataNode,
        std::vector<size_t>& indicesShape,
        std::vector<int>& indices,
        int blankIndex,
        const element::Type& iType) {
    auto indicesNode = makeConstant(iType, indicesShape, indices);

    auto gatherNDNode = std::make_shared<opset5::GatherND>(dataNode, indicesNode, blankIndex);

    return gatherNDNode;
}

}  // namespace builder
}  // namespace ngraph
