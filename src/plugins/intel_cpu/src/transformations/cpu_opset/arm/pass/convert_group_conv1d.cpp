// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "convert_group_conv1d.hpp"

#include "openvino/core/rt_info.hpp"
#include <openvino/op/convolution.hpp>
#include <openvino/op/group_conv.hpp>
#include <openvino/op/reshape.hpp>
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov::intel_cpu;

template <class Conv>
ov::matcher_pass_callback ConvertConv1DBase::convert_conv1d_to_conv2d() {
    return [&](pass::pattern::Matcher& m) {
        auto conv = std::dynamic_pointer_cast<Conv>(m.get_match_root());
        if (!conv) {
            return false;
        }

        auto input_shape = conv->get_input_shape(0);
        // is Conv1D
        if (input_shape.size() != 3) {
            return false;
        }

        auto input   = conv->input_value(0);
        auto weights = conv->input_value(1);
        auto input2d_shape = input_shape;
        input2d_shape.push_back(1);
        auto in2d_shape = std::make_shared<op::v0::Constant>(ov::element::i64, Shape{4}, input2d_shape);

        auto weights2d_shape = weights.get_shape();
        weights2d_shape.push_back(1);
        auto w_shape = std::make_shared<op::v0::Constant>(ov::element::i64, Shape{weights2d_shape.size()}, weights2d_shape);

        auto input2d   = std::make_shared<op::v1::Reshape>(input, in2d_shape, true);
        auto weights2d = std::make_shared<op::v1::Reshape>(weights, w_shape, true);

        auto conv2d = std::make_shared<Conv>(input2d,
                                             weights2d,
                                             Strides{conv->get_strides()[0], 1},
                                             CoordinateDiff{conv->get_pads_begin()[0], 0},
                                             CoordinateDiff{conv->get_pads_end()[0], 0},
                                             Strides{conv->get_dilations()[0], 1},
                                             conv->get_auto_pad());

        auto in_shape = std::make_shared<op::v0::Constant>(ov::element::i64, Shape{3}, conv->get_output_shape(0));
        auto reshape = std::make_shared<op::v1::Reshape>(conv2d, in_shape, true);

        reshape->set_friendly_name(conv->get_friendly_name());
        copy_runtime_info(conv, {input2d, weights2d, conv2d, reshape});
        replace_node(conv, reshape);
        return true;
    };
}

ConvertConv1D::ConvertConv1D() {
    auto m = std::make_shared<pass::pattern::Matcher>(
        pass::pattern::wrap_type<op::v1::Convolution>({pass::pattern::any_input(pass::pattern::has_static_shape()),
                                                             pass::pattern::any_input(pass::pattern::has_static_shape())},
                                                             pass::pattern::has_static_shape()), "ConvertConvolutionToArm");
    register_matcher(m, convert_conv1d_to_conv2d<op::v1::Convolution>());
}

ConvertGroupConv1D::ConvertGroupConv1D() {
    auto m = std::make_shared<pass::pattern::Matcher>(
            pass::pattern::wrap_type<op::v1::GroupConvolution>({pass::pattern::any_input(pass::pattern::has_static_shape()),
                                                                      pass::pattern::any_input(pass::pattern::has_static_shape())},
                                                                      pass::pattern::has_static_shape()), "ConvertGroupConvolutionToArm");
    register_matcher(m, convert_conv1d_to_conv2d<op::v1::GroupConvolution>());
}