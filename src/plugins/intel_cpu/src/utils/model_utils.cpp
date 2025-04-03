// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "model_utils.hpp"

#include "openvino/op/convolution.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"

namespace ov::intel_cpu {

Config::ModelType getModelType(const std::shared_ptr<const Model>& model) {
    if (op::util::has_op_with_type<op::v1::Convolution>(model) ||
        op::util::has_op_with_type<op::v1::ConvolutionBackpropData>(model)) {
        return Config::ModelType::CNN;
    }

    if ((op::util::has_op_with_type<op::v13::ScaledDotProductAttention>(model) && model->get_variables().size() > 0) ||
        op::util::has_op_with_type<ov::op::PagedAttentionExtension>(model)) {
        return Config::ModelType::LLM;
    }

    return Config::ModelType::Unknown;
}

}  // namespace ov::intel_cpu
