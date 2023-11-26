// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>

// This transformation inserts Conversion node string->u8 and u8->string before and after the Extension node.

namespace ov {
namespace intel_cpu {

class ConvertStringU8ForExtension: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertStringU8ForExtension", "0");

    ConvertStringU8ForExtension();

    // bool isNativelySupported(const ov::Node::type_info_t& type) const;

    // std::shared_ptr<ov::Node> changeConstantPrecision(std::shared_ptr<op::v0::Constant>& constant) const;

    // bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace intel_cpu
}  // namespace ov