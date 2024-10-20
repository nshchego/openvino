// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertConvertLike;

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertConvertLike : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertConvertLike", "0");
    ConvertConvertLike();
};

namespace ngraph {
namespace pass {
using ov::pass::ConvertConvertLike;
}  // namespace pass
}  // namespace ngraph
