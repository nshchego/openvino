// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mha_fusion.hpp"

#include <openvino/core/rt_info.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/fake_quantize.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/softmax.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include "simplify_fakequantize.hpp"
#include "transformations/cpu_opset/x64/op/mha.hpp"

#include "itt.hpp"

using namespace ov::intel_cpu;

bool MHAFusionBase::valid_transpose_order(const std::shared_ptr<Node>& node, const std::vector<int64_t>& expected_order) {
    if (auto transpose_pattern = as_type_ptr<op::v0::Constant>(node)) {
        if (transpose_pattern->cast_vector<int64_t>() != expected_order) {
            return false;
        }
    } else {
        return false;
    }

    return true;
}

// TODO: draw pattern
MHAFloatFusion::MHAFloatFusion() {
    MATCHER_SCOPE(MHAFloatFusion);

    auto in0 = pass::pattern::any_input(pass::pattern::has_static_shape());
    auto in1 = pass::pattern::any_input(pass::pattern::has_static_shape());
    auto in2 = pass::pattern::wrap_type<op::v0::Constant>();
    auto in3 = pass::pattern::any_input(pass::pattern::has_static_shape());
    auto in4 = pass::pattern::wrap_type<op::v0::Constant>();
    auto in5 = pass::pattern::wrap_type<op::v0::Constant>();
    auto in6 = pass::pattern::wrap_type<op::v0::Constant>();
    auto in7 = pass::pattern::wrap_type<op::v0::Constant>();
    auto in8 = pass::pattern::any_input(pass::pattern::has_static_shape());
    auto in9 = pass::pattern::wrap_type<op::v0::Constant>();
    auto in10 = pass::pattern::wrap_type<op::v0::Constant>();
    auto transpose0 = std::make_shared<op::v1::Transpose>(in0, in4);
    auto transpose1 = std::make_shared<op::v1::Transpose>(in1, in5);
    auto mul = std::make_shared<op::v1::Multiply>(transpose1, in2);
    auto matmul0 = std::make_shared<op::v0::MatMul>(transpose0, mul);
    auto add = std::make_shared<op::v1::Add>(matmul0, in3);
    auto reshape0 = std::make_shared<op::v1::Reshape>(add, in6, true);
    auto softmax = std::make_shared<op::v1::Softmax>(reshape0);
    auto reshape1 = std::make_shared<op::v1::Reshape>(softmax, in7, true);
    auto transpose2 = std::make_shared<op::v1::Transpose>(in8, in9);
    auto matmul1 = std::make_shared<op::v0::MatMul>(reshape1, transpose2);
    auto transpose3 = std::make_shared<op::v1::Transpose>(matmul1, in10);

    matcher_pass_callback callback = [=](pass::pattern::Matcher& m) {
//std::cout << "MHAFloatFusion::callback" << std::endl;
        auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose0_in = pattern_to_output.at(in0);
        auto transpose1_in = pattern_to_output.at(in1);
        auto mul_in1 = pattern_to_output.at(in2);
        auto add_in1 = pattern_to_output.at(in3);
        auto transpose2_in = pattern_to_output.at(in8);

        if (transpose0_in.get_shape() != transpose1_in.get_shape() || transpose0_in.get_shape() != transpose2_in.get_shape()) {
            return false;
        }

        if (transpose0_in.get_shape().size() != 4) {
            return false;
        }

        auto expected_add_shape = Shape({transpose0_in.get_shape()[0], 1, 1, transpose0_in.get_shape()[1]});
        if (add_in1.get_shape() != expected_add_shape) {
            return false;
        }

        if (!valid_transpose_order(pattern_to_output.at(in4).get_node_shared_ptr(), {0, 2, 1, 3})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in5).get_node_shared_ptr(), {0, 2, 3, 1})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in9).get_node_shared_ptr(), {0, 2, 1, 3})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in10).get_node_shared_ptr(), {0, 2, 1, 3})) return false;

        std::vector<float> mul_scales;
        if (auto mul_node = as_type_ptr<op::v1::Multiply>(pattern_to_output.at(mul).get_node_shared_ptr())) {
            mul_scales = as_type_ptr<op::v0::Constant>(mul_node->get_input_node_shared_ptr(1))->cast_vector<float>();

            auto expected_shape = Shape({1, transpose0_in.get_shape()[2], 1, 1});
            if (mul_scales.size() != 1 && mul_node->get_input_shape(1) != expected_shape) {
                return false;
            }
        } else {
            return false;
        }

        auto matmul0_node = as_type_ptr<op::v0::MatMul>(pattern_to_output.at(matmul0).get_node_shared_ptr());
        if (!matmul0_node)
            return false;
        if (matmul0_node->get_transpose_a() || matmul0_node->get_transpose_b())
            return false;

        auto reshape0_node = as_type_ptr<op::v1::Reshape>(pattern_to_output.at(reshape0).get_node_shared_ptr());
        if (!reshape0_node)
            return false;

        if (auto reshape_pattern = as_type_ptr<op::v0::Constant>(pattern_to_output.at(in6).get_node_shared_ptr())) {
            if (reshape0_node->get_input_shape(0).size() != 4) {
                return false;
            }

            std::vector<int64_t> reshapeConstData = {static_cast<int64_t>(reshape0_node->get_input_shape(0)[0] *
                                                                          reshape0_node->get_input_shape(0)[1] *
                                                                          reshape0_node->get_input_shape(0)[2]),
                                                     -1};

            if (reshape_pattern->cast_vector<int64_t>() != reshapeConstData) {
                return false;
            }
        } else {
            return false;
        }

        if (auto reshape1_node = as_type_ptr<op::v1::Reshape>(pattern_to_output.at(reshape1).get_node_shared_ptr())) {
            if (reshape0_node->get_input_shape(0) != reshape1_node->get_output_shape(0)) {
                return false;
            }
        } else {
            return false;
        }

        auto softmax_node = as_type_ptr<op::v1::Softmax>(pattern_to_output.at(softmax).get_node_shared_ptr());
        if (!softmax_node)
            return false;
        if (softmax_node->get_axis() != 1)
            return false;

        auto matmul1_node = as_type_ptr<op::v0::MatMul>(pattern_to_output.at(matmul1).get_node_shared_ptr());
        if (!matmul1_node)
            return false;
        if (matmul1_node->get_transpose_a() || matmul1_node->get_transpose_b())
            return false;

        bool is_mul_first = true;
        auto transpose3_node = pattern_to_output.at(transpose3).get_node_shared_ptr();
        auto mha = std::make_shared<MHANode>(transpose0_in, transpose1_in, add_in1, transpose2_in, mul_scales, is_mul_first,
                                                            transpose3_node->get_output_element_type(0));
        mha->set_friendly_name(m.get_match_root()->get_friendly_name());
        copy_runtime_info({pattern_to_output.at(transpose0).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose1).get_node_shared_ptr(),
                                   pattern_to_output.at(mul).get_node_shared_ptr(),
                                   pattern_to_output.at(matmul0).get_node_shared_ptr(),
                                   pattern_to_output.at(add).get_node_shared_ptr(),
                                   pattern_to_output.at(reshape0).get_node_shared_ptr(),
                                   pattern_to_output.at(softmax).get_node_shared_ptr(),
                                   pattern_to_output.at(reshape1).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose2).get_node_shared_ptr(),
                                   pattern_to_output.at(matmul1).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose3).get_node_shared_ptr(),
                                  },
                                  mha);

        if (transformation_callback(mha)) {
            return false;
        }

        replace_node(m.get_match_root(), mha);

        return true;
    };

    auto m = std::make_shared<pass::pattern::Matcher>(transpose3, matcher_name);
    this->register_matcher(m, callback);
}

MHAFloatFusion2::MHAFloatFusion2() {
    MATCHER_SCOPE(MHAFloatFusion2);

    auto in0 = pass::pattern::any_input(pass::pattern::has_static_shape());
    auto in1 = pass::pattern::any_input(pass::pattern::has_static_shape());
    auto in3 = pass::pattern::any_input(pass::pattern::has_static_shape());
    auto in4 = pass::pattern::wrap_type<op::v0::Constant>();
    auto in5 = pass::pattern::wrap_type<op::v0::Constant>();
    auto in6 = pass::pattern::wrap_type<op::v0::Constant>();
    auto in7 = pass::pattern::wrap_type<op::v0::Constant>();
    auto in8 = pass::pattern::any_input(pass::pattern::has_static_shape());
    auto in9 = pass::pattern::wrap_type<op::v0::Constant>();
    auto in10 = pass::pattern::wrap_type<op::v0::Constant>();
    auto transpose0 = std::make_shared<op::v1::Transpose>(in0, in4);
    auto transpose1 = std::make_shared<op::v1::Transpose>(in1, in5);
    auto matmul0 = std::make_shared<op::v0::MatMul>(transpose0, transpose1);
    auto add = std::make_shared<op::v1::Add>(matmul0, in3);
    auto softmax = std::make_shared<op::v1::Softmax>(add);
    auto transpose2 = std::make_shared<op::v1::Transpose>(in8, in9);
    auto matmul1 = std::make_shared<op::v0::MatMul>(softmax, transpose2);
    auto transpose3 = std::make_shared<op::v1::Transpose>(matmul1, in10);

    matcher_pass_callback callback = [=](pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose0_in = pattern_to_output.at(in0);
        auto transpose1_in = pattern_to_output.at(in1);
        auto add_in1 = pattern_to_output.at(in3);
        auto transpose2_in = pattern_to_output.at(in8);

        if (transpose0_in.get_shape() != transpose1_in.get_shape() || transpose0_in.get_shape() != transpose2_in.get_shape()) {
            return false;
        }

        if (transpose0_in.get_shape().size() != 4) {
            return false;
        }

        auto expected_add_shape = Shape({transpose0_in.get_shape()[0], 1, 1, transpose0_in.get_shape()[1]});
        if (add_in1.get_shape() != expected_add_shape) {
            return false;
        }

        if (!valid_transpose_order(pattern_to_output.at(in4).get_node_shared_ptr(), {0, 2, 1, 3})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in5).get_node_shared_ptr(), {0, 2, 3, 1})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in9).get_node_shared_ptr(), {0, 2, 1, 3})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in10).get_node_shared_ptr(), {0, 2, 1, 3})) return false;

        auto matmul0_node = as_type_ptr<op::v0::MatMul>(pattern_to_output.at(matmul0).get_node_shared_ptr());
        if (!matmul0_node)
            return false;
        if (matmul0_node->get_transpose_a() || matmul0_node->get_transpose_b())
            return false;

        auto softmax_node = as_type_ptr<op::v1::Softmax>(pattern_to_output.at(softmax).get_node_shared_ptr());
        if (!softmax_node)
            return false;
        if (softmax_node->get_axis() != 3)
            return false;

        auto matmul1_node = as_type_ptr<op::v0::MatMul>(pattern_to_output.at(matmul1).get_node_shared_ptr());
        if (!matmul1_node)
            return false;
        if (matmul1_node->get_transpose_a() || matmul1_node->get_transpose_b())
            return false;

        auto transpose3_node = pattern_to_output.at(transpose3).get_node_shared_ptr();
        auto mha = std::make_shared<MHANode>(transpose0_in, transpose1_in, add_in1, transpose2_in, std::vector<float>(), false,
                                                            transpose3_node->get_output_element_type(0));
        mha->set_friendly_name(m.get_match_root()->get_friendly_name());
        copy_runtime_info({pattern_to_output.at(transpose0).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose1).get_node_shared_ptr(),
                                   pattern_to_output.at(matmul0).get_node_shared_ptr(),
                                   pattern_to_output.at(add).get_node_shared_ptr(),
                                   pattern_to_output.at(softmax).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose2).get_node_shared_ptr(),
                                   pattern_to_output.at(matmul1).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose3).get_node_shared_ptr(),
                                  },
                                  mha);

        if (transformation_callback(mha)) {
            return false;
        }

        replace_node(m.get_match_root(), mha);

        return true;
    };

    auto m = std::make_shared<pass::pattern::Matcher>(transpose3, matcher_name);
    this->register_matcher(m, callback);
}

// TODO: draw pattern
MHAQuantFusion::MHAQuantFusion() {
    MATCHER_SCOPE(MHAQuantFusion);

    auto in0 = pass::pattern::any_input(pass::pattern::has_static_shape());
    auto in1 = pass::pattern::any_input(pass::pattern::has_static_shape());
    auto in2 = pass::pattern::wrap_type<op::v0::Constant>();
    auto in3 = pass::pattern::any_input(pass::pattern::has_static_shape());
    auto in4 = pass::pattern::wrap_type<op::v0::Constant>();
    auto in5 = pass::pattern::wrap_type<op::v0::Constant>();
    auto in6 = pass::pattern::wrap_type<op::v0::Constant>();
    auto in7 = pass::pattern::wrap_type<op::v0::Constant>();
    auto in8 = pass::pattern::any_input(pass::pattern::has_static_shape());
    auto in9 = pass::pattern::wrap_type<op::v0::Constant>();
    auto in10 = pass::pattern::wrap_type<op::v0::Constant>();
    auto transpose0 = std::make_shared<op::v1::Transpose>(in0, in4);
    auto transpose1 = std::make_shared<op::v1::Transpose>(in1, in5);
    auto matmul0 = std::make_shared<op::v0::MatMul>(transpose0, transpose1);
    auto fakeQuantize0 = pass::pattern::wrap_type<op::v0::FakeQuantize>({matmul0,
                                                                                   pass::pattern::wrap_type<op::v0::Constant>(),
                                                                                   pass::pattern::wrap_type<op::v0::Constant>(),
                                                                                   pass::pattern::wrap_type<op::v0::Constant>(),
                                                                                   pass::pattern::wrap_type<op::v0::Constant>()});
    auto add = std::make_shared<op::v1::Add>(fakeQuantize0, in3);
    auto mul = std::make_shared<op::v1::Multiply>(add, in2);
    auto reshape0 = std::make_shared<op::v1::Reshape>(mul, in6, true);
    auto softmax = std::make_shared<op::v1::Softmax>(reshape0);
    auto reshape1 = std::make_shared<op::v1::Reshape>(softmax, in7, true);
    auto fakeQuantize1 = pass::pattern::wrap_type<op::v0::FakeQuantize>({reshape1,
                                                                                   pass::pattern::wrap_type<op::v0::Constant>(),
                                                                                   pass::pattern::wrap_type<op::v0::Constant>(),
                                                                                   pass::pattern::wrap_type<op::v0::Constant>(),
                                                                                   pass::pattern::wrap_type<op::v0::Constant>()});
    auto transpose2 = std::make_shared<op::v1::Transpose>(in8, in9);
    auto matmul1 = std::make_shared<op::v0::MatMul>(fakeQuantize1, transpose2);
    auto fakeQuantize2 = pass::pattern::wrap_type<op::v0::FakeQuantize>({matmul1,
                                                                                   pass::pattern::wrap_type<op::v0::Constant>(),
                                                                                   pass::pattern::wrap_type<op::v0::Constant>(),
                                                                                   pass::pattern::wrap_type<op::v0::Constant>(),
                                                                                   pass::pattern::wrap_type<op::v0::Constant>()});
    auto transpose3 = std::make_shared<op::v1::Transpose>(fakeQuantize2, in10);

    matcher_pass_callback callback = [=](pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose0_in = pattern_to_output.at(in0);
        auto transpose1_in = pattern_to_output.at(in1);
        auto add_in1 = pattern_to_output.at(in3);
        auto transpose2_in = pattern_to_output.at(in8);

        if (transpose0_in.get_shape() != transpose1_in.get_shape() || transpose0_in.get_shape() != transpose2_in.get_shape()) {
            return false;
        }

        if (transpose0_in.get_shape().size() != 4) {
            return false;
        }

        auto expected_add_shape = Shape({transpose0_in.get_shape()[0], 1, 1, transpose0_in.get_shape()[1]});
        if (add_in1.get_shape() != expected_add_shape) {
            return false;
        }

        std::vector<float> mul_scales;
        if (auto mul_node = as_type_ptr<op::v1::Multiply>(pattern_to_output.at(mul).get_node_shared_ptr())) {
            mul_scales = as_type_ptr<op::v0::Constant>(mul_node->get_input_node_shared_ptr(1))->cast_vector<float>();

            auto expected_shape = Shape({1, transpose0_in.get_shape()[2], 1, 1});
            if (mul_scales.size() != 1 && mul_node->get_input_shape(1) != expected_shape) {
                return false;
            }
        } else {
            return false;
        }

        if (!valid_transpose_order(pattern_to_output.at(in4).get_node_shared_ptr(), {0, 2, 1, 3})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in5).get_node_shared_ptr(), {0, 2, 3, 1})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in9).get_node_shared_ptr(), {0, 2, 1, 3})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in10).get_node_shared_ptr(), {0, 2, 1, 3})) return false;

        auto matmul0_node = as_type_ptr<op::v0::MatMul>(pattern_to_output.at(matmul0).get_node_shared_ptr());
        if (!matmul0_node)
            return false;
        if (matmul0_node->get_transpose_a() || matmul0_node->get_transpose_b())
            return false;

        std::vector<float> fq0_scale;
        auto fq0_node = as_type_ptr<op::v0::FakeQuantize>(pattern_to_output.at(fakeQuantize0).get_node_shared_ptr());
        if (fq0_node) {
            fq0_scale = simplifyToScale(fq0_node);
            if (!fq0_scale.size())
                return false;
        }

        auto reshape0_node = as_type_ptr<op::v1::Reshape>(pattern_to_output.at(reshape0).get_node_shared_ptr());
        if (!reshape0_node)
            return false;

        if (auto reshape_pattern = as_type_ptr<op::v0::Constant>(pattern_to_output.at(in6).get_node_shared_ptr())) {
            if (reshape0_node->get_input_shape(0).size() != 4) {
                return false;
            }

            std::vector<int64_t> reshapeConstData = {static_cast<int64_t>(reshape0_node->get_input_shape(0)[0] *
                                                                          reshape0_node->get_input_shape(0)[1] *
                                                                          reshape0_node->get_input_shape(0)[2]),
                                                     -1};

            if (reshape_pattern->cast_vector<int64_t>() != reshapeConstData) {
                return false;
            }
        } else {
            return false;
        }

        if (auto reshape1_node = as_type_ptr<op::v1::Reshape>(pattern_to_output.at(reshape1).get_node_shared_ptr())) {
            if (reshape0_node->get_input_shape(0) != reshape1_node->get_output_shape(0)) {
                return false;
            }
        } else {
            return false;
        }

        auto softmax_node = as_type_ptr<op::v1::Softmax>(pattern_to_output.at(softmax).get_node_shared_ptr());
        if (!softmax_node)
            return false;
        if (softmax_node->get_axis() != 1)
            return false;

        std::vector<float> fq1_scale;
        auto fq1_node = as_type_ptr<op::v0::FakeQuantize>(pattern_to_output.at(fakeQuantize1).get_node_shared_ptr());
        if (fq1_node) {
            fq1_scale = simplifyToScale(fq1_node);
            if (!fq1_scale.size())
                return false;
        } else {
            return false;
        }

        auto matmul1_node = as_type_ptr<op::v0::MatMul>(pattern_to_output.at(matmul1).get_node_shared_ptr());
        if (!matmul1_node)
            return false;
        if (matmul1_node->get_transpose_a() || matmul1_node->get_transpose_b())
            return false;

        std::vector<float> fq2_scale;
        if (auto fq_node = as_type_ptr<op::v0::FakeQuantize>(pattern_to_output.at(fakeQuantize2).get_node_shared_ptr())) {
            fq2_scale = simplifyToScale(fq_node);
            if (!fq2_scale.size())
                return false;
        }

        bool is_mul_first = false;
        auto transpose3_node = pattern_to_output.at(transpose3).get_node_shared_ptr();
        auto mha = std::make_shared<MHANode>(transpose0_in, transpose1_in, add_in1, transpose2_in, mul_scales, is_mul_first,
                                                            std::vector<float>(), fq0_scale, fq1_scale, fq2_scale,
                                                            element::undefined,
                                                            fq0_node ? fq0_node->get_output_element_type(0) : element::undefined,
                                                            fq1_node->get_output_element_type(0), transpose3_node->get_output_element_type(0));
        mha->set_friendly_name(m.get_match_root()->get_friendly_name());
        copy_runtime_info({pattern_to_output.at(transpose0).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose1).get_node_shared_ptr(),
                                   pattern_to_output.at(matmul0).get_node_shared_ptr(),
                                   pattern_to_output.at(fakeQuantize0).get_node_shared_ptr(),
                                   pattern_to_output.at(add).get_node_shared_ptr(),
                                   pattern_to_output.at(mul).get_node_shared_ptr(),
                                   pattern_to_output.at(reshape0).get_node_shared_ptr(),
                                   pattern_to_output.at(softmax).get_node_shared_ptr(),
                                   pattern_to_output.at(reshape1).get_node_shared_ptr(),
                                   pattern_to_output.at(fakeQuantize1).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose2).get_node_shared_ptr(),
                                   pattern_to_output.at(matmul1).get_node_shared_ptr(),
                                   pattern_to_output.at(fakeQuantize2).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose3).get_node_shared_ptr(),
                                  },
                                  mha);

        if (transformation_callback(mha)) {
            return false;
        }

        replace_node(m.get_match_root(), mha);

        return true;
    };

    auto m = std::make_shared<pass::pattern::Matcher>(transpose3, matcher_name);
    this->register_matcher(m, callback);
}

// TODO: draw pattern
MHAQuantFusion2::MHAQuantFusion2() {
    MATCHER_SCOPE(MHAQuantFusion2);

    auto in0 = pass::pattern::any_input(pass::pattern::has_static_shape());
    auto in1 = pass::pattern::any_input(pass::pattern::has_static_shape());
    auto in2 = pass::pattern::wrap_type<op::v0::Constant>();
    auto in3 = pass::pattern::any_input(pass::pattern::has_static_shape());
    auto in4 = pass::pattern::wrap_type<op::v0::Constant>();
    auto in5 = pass::pattern::wrap_type<op::v0::Constant>();
    auto in8 = pass::pattern::any_input(pass::pattern::has_static_shape());
    auto in9 = pass::pattern::wrap_type<op::v0::Constant>();
    auto in10 = pass::pattern::wrap_type<op::v0::Constant>();
    auto transpose0 = std::make_shared<op::v1::Transpose>(in0, in4);
    auto transpose1 = std::make_shared<op::v1::Transpose>(in1, in5);
    auto fakeQuantize0 = pass::pattern::wrap_type<op::v0::FakeQuantize>({transpose1,
                                                                                pass::pattern::wrap_type<op::v0::Constant>(),
                                                                                pass::pattern::wrap_type<op::v0::Constant>(),
                                                                                pass::pattern::wrap_type<op::v0::Constant>(),
                                                                                pass::pattern::wrap_type<op::v0::Constant>()});
    auto matmul0 = std::make_shared<op::v0::MatMul>(transpose0, fakeQuantize0);
    auto mul = std::make_shared<op::v1::Multiply>(matmul0, in2);
    auto add = std::make_shared<op::v1::Add>(mul, in3);
    auto softmax = std::make_shared<op::v1::Softmax>(add);
    auto transpose2 = std::make_shared<op::v1::Transpose>(in8, in9);
    auto matmul1 = std::make_shared<op::v0::MatMul>(softmax, transpose2);
    auto fakeQuantize1 = pass::pattern::wrap_type<op::v0::FakeQuantize>({matmul1,
                                                                                   pass::pattern::wrap_type<op::v0::Constant>(),
                                                                                   pass::pattern::wrap_type<op::v0::Constant>(),
                                                                                   pass::pattern::wrap_type<op::v0::Constant>(),
                                                                                   pass::pattern::wrap_type<op::v0::Constant>()});
    auto transpose3 = std::make_shared<op::v1::Transpose>(fakeQuantize1, in10);

    matcher_pass_callback callback = [=](pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose0_in = pattern_to_output.at(in0);
        auto transpose1_in = pattern_to_output.at(in1);
        auto add_in1 = pattern_to_output.at(in3);
        auto transpose2_in = pattern_to_output.at(in8);

        if (transpose0_in.get_shape() != transpose1_in.get_shape() || transpose0_in.get_shape() != transpose2_in.get_shape()) {
            return false;
        }

        if (transpose0_in.get_shape().size() != 4) {
            return false;
        }

        auto expected_add_shape = Shape({transpose0_in.get_shape()[0], 1, 1, transpose0_in.get_shape()[1]});
        if (add_in1.get_shape() != expected_add_shape) {
            return false;
        }

        std::vector<float> mul_scales;
        if (auto mul_node = as_type_ptr<op::v1::Multiply>(pattern_to_output.at(mul).get_node_shared_ptr())) {
            mul_scales = as_type_ptr<op::v0::Constant>(mul_node->get_input_node_shared_ptr(1))->cast_vector<float>();

            auto expected_shape = Shape({1, transpose0_in.get_shape()[2], 1, 1});
            if (mul_scales.size() != 1 && mul_node->get_input_shape(1) != expected_shape) {
                return false;
            }
        } else {
            return false;
        }

        if (!valid_transpose_order(pattern_to_output.at(in4).get_node_shared_ptr(), {0, 2, 1, 3})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in5).get_node_shared_ptr(), {0, 2, 3, 1})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in9).get_node_shared_ptr(), {0, 2, 1, 3})) return false;
        if (!valid_transpose_order(pattern_to_output.at(in10).get_node_shared_ptr(), {0, 2, 1, 3})) return false;

        auto matmul0_node = as_type_ptr<op::v0::MatMul>(pattern_to_output.at(matmul0).get_node_shared_ptr());
        if (!matmul0_node)
            return false;
        if (matmul0_node->get_transpose_a() || matmul0_node->get_transpose_b())
            return false;

        std::vector<float> fq0_scale;
        auto fq0_node = as_type_ptr<op::v0::FakeQuantize>(pattern_to_output.at(fakeQuantize0).get_node_shared_ptr());
        if (fq0_node) {
            fq0_scale = simplifyToScale(fq0_node);
            if (!fq0_scale.size())
                return false;
        } else {
            return false;
        }

        auto softmax_node = as_type_ptr<op::v1::Softmax>(pattern_to_output.at(softmax).get_node_shared_ptr());
        if (!softmax_node)
            return false;
        if (softmax_node->get_axis() != 3)
            return false;

        std::vector<float> fq1_scale;
        if (auto fq_node = as_type_ptr<op::v0::FakeQuantize>(pattern_to_output.at(fakeQuantize1).get_node_shared_ptr())) {
            fq1_scale = simplifyToScale(fq_node);
            if (!fq1_scale.size())
                return false;
        }

        auto matmul1_node = as_type_ptr<op::v0::MatMul>(pattern_to_output.at(matmul1).get_node_shared_ptr());
        if (!matmul1_node)
            return false;
        if (matmul1_node->get_transpose_a() || matmul1_node->get_transpose_b())
            return false;

        bool is_mul_first = true;
        auto transpose3_node = pattern_to_output.at(transpose3).get_node_shared_ptr();
        auto mha = std::make_shared<MHANode>(transpose0_in, transpose1_in, add_in1, transpose2_in, mul_scales, is_mul_first,
                                                            fq0_scale, std::vector<float>(), std::vector<float>(), fq1_scale,
                                                            fq0_node->get_output_element_type(0), element::undefined, element::undefined,
                                                            transpose3_node->get_output_element_type(0));
        mha->set_friendly_name(m.get_match_root()->get_friendly_name());
        copy_runtime_info({pattern_to_output.at(transpose0).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose1).get_node_shared_ptr(),
                                   pattern_to_output.at(fakeQuantize0).get_node_shared_ptr(),
                                   pattern_to_output.at(matmul0).get_node_shared_ptr(),
                                   pattern_to_output.at(mul).get_node_shared_ptr(),
                                   pattern_to_output.at(add).get_node_shared_ptr(),
                                   pattern_to_output.at(softmax).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose2).get_node_shared_ptr(),
                                   pattern_to_output.at(matmul1).get_node_shared_ptr(),
                                   pattern_to_output.at(fakeQuantize1).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose3).get_node_shared_ptr(),
                                  },
                                  mha);

        if (transformation_callback(mha)) {
            return false;
        }

        replace_node(m.get_match_root(), mha);

        return true;
    };

    auto m = std::make_shared<pass::pattern::Matcher>(transpose3, matcher_name);
    this->register_matcher(m, callback);
}
