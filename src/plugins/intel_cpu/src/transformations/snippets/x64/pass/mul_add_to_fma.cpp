// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mul_add_to_fma.hpp"

#include <ngraph/op/add.hpp>
#include <ngraph/op/multiply.hpp>
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include <snippets/itt.hpp>
#include "transformations/snippets/x64/op/fused_mul_add.hpp"

using namespace ov::intel_cpu;

pass::MulAddToFMA::MulAddToFMA() {
    MATCHER_SCOPE(MulAddToFMA);
    auto mul_input_1 = ov::pass::pattern::any_input();
    auto mul_input_2 = ov::pass::pattern::any_input();
    auto mul_m = ov::pass::pattern::wrap_type<op::v1::Multiply>({ mul_input_1, mul_input_2 }, ov::pass::pattern::consumers_count(1));
    auto add_input_2 = ov::pass::pattern::any_input();
    auto add_m = ov::pass::pattern::wrap_type<op::v1::Add>({ mul_m, add_input_2 });

    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(pass::itt::domains::SnippetsTransform, "Snippets::op::MulAddToFMA_callback")
        const auto& pattern_map = m.get_pattern_value_map();
        const auto multiply = pattern_map.at(mul_m).get_node_shared_ptr();
        const auto add = pattern_map.at(add_m).get_node_shared_ptr();

        if (transformation_callback(add)) {
            return false;
        }

        const auto& a = multiply->input_value(0);
        const auto& b = multiply->input_value(1);
        const auto& c = pattern_map.at(add_input_2);

        const auto fma = std::make_shared<FusedMulAdd>(a, b, c);
        copy_runtime_info({ a.get_node_shared_ptr(), b.get_node_shared_ptr(), c.get_node_shared_ptr() }, fma);
        fma->set_friendly_name(add->get_friendly_name());
        replace_node(add, fma);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(add_m, "MulAddToFMA");
    register_matcher(m, callback);
}
