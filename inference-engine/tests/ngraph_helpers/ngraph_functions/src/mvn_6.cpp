// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<Node> makeMVN6(const Output<Node>& in,
                                std::vector<int>& axes,
                                bool normalizeVariance,
                                float eps,
                                std::string& epsMode) {
    auto axesNode = builder::makeConstant(element::i32, Shape{axes.size()}, axes);
    op::MVNEpsMode nEpsMode = op::MVNEpsMode::INSIDE_SQRT;
    if (epsMode == "outside_sqrt")
        nEpsMode = op::MVNEpsMode::OUTSIDE_SQRT;
    auto mvnNode = std::make_shared<op::v6::MVN>(in, axesNode, normalizeVariance, eps, nEpsMode);

    return mvnNode;
}

}  // namespace builder
}  // namespace ngraph
