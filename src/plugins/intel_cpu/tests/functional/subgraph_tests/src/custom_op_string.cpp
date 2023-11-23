// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/ov_tensor_utils.hpp>
#include <openvino/op/op.hpp>
#include <shared_test_classes/base/ov_subgraph.hpp>
#include "test_utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

using CustomOpStringCPUTestParams = std::tuple<ElementType, InputShape>;

class CustomOpString : public ov::op::Op {
public:
    OPENVINO_OP("CustomOpString");

    CustomOpString() = default;
    CustomOpString(const ov::OutputVector& args) : Op(args) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        const auto& inputs_count = input_values().size();
        OPENVINO_ASSERT(inputs_count == 1,
                        "Input count must be 1, Got: ",
                        inputs_count);
        OPENVINO_ASSERT(get_input_element_type(0) == ov::element::Type_t::string,
                        "The input must be string.");
        set_output_size(1);

        auto inShape = get_input_partial_shape(0);

        set_output_type(0, ov::element::Type_t::string, inShape);
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        OPENVINO_ASSERT(new_args.size() == 1, "Incorrect number of new arguments: ", new_args.size(), ". 1 is expected.");

        return std::make_shared<CustomOpString>(new_args);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override {
        for (size_t i = 0lu; i < inputs.size(); i++) {
            OPENVINO_ASSERT(inputs[i].get_shape().size() == static_cast<size_t>(get_input_partial_shape(i).rank().get_length()),
                "Invalid input shape rank: ", inputs[i].get_shape().size());
        }
        for (size_t i = 0llu; i < outputs.size(); i++) {
            OPENVINO_ASSERT(outputs[i].get_shape().size() == static_cast<size_t>(get_output_partial_shape(i).rank().get_length()),
                "Invalid outputs shape rank: ", outputs[i].get_shape().size());
        }

        const auto& in_0 = inputs[0];
        auto& out = outputs[0];

        memcpy(out.data(), in_0.data(), out.get_byte_size());

        return true;
    }

    bool evaluate(ov::TensorVector& output_values,
                  const ov::TensorVector& input_values,
                  const ov::EvaluationContext& evaluationContext) const override {
        return evaluate(output_values, input_values);
    }

    bool has_evaluate() const override {
        return true;
    }
};

class CustomOpStringCPUTest : public testing::WithParamInterface<CustomOpStringCPUTestParams>,
                                  virtual public SubgraphBaseTest,
                                  public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<CustomOpStringCPUTestParams>& obj) {
        ElementType inType;
        InputShape inputShape;
        std::tie(inType, inputShape) = obj.param;

        std::ostringstream result;
        result << "IS=" << inputShape << "_";
        result << "Prc=" << inType;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = utils::DEVICE_CPU;

        ElementType inType;
        InputShape inputShape;
        std::tie(inType, inputShape) = this->GetParam();

        init_input_shapes({inputShape});

        auto in_0 = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]);
        ov::OutputVector param_outs({in_0});
        auto custom_op = std::make_shared<CustomOpString>(param_outs);

        ov::ParameterVector input_params{in_0};
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(custom_op)};
        function = std::make_shared<ov::Model>(results, input_params, "CustomOpString");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            auto tensor = utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override {
        ASSERT_EQ(expected.size(), actual.size());
        ASSERT_EQ(expected.size(), function->get_results().size());
        const auto& results = function->get_results();
        for (size_t j = 0; j < results.size(); j++) {
            const auto result = results[j];
            for (size_t i = 0; i < result->get_input_size(); ++i) {
                utils::compare(expected[j], actual[j], abs_threshold, rel_threshold);
            }
        }
    }
};

TEST_P(CustomOpStringCPUTest, CompareWithRefs) {
    run();
}

const std::vector<InputShape> inputShapes = {
    {{}, {{2, 3, 16}}},
    {{2, 3, -1}, {{2, 3, 0}}},
    {{}, {{}}}
};

INSTANTIATE_TEST_SUITE_P(smoke_CustomOp,
                         CustomOpStringCPUTest,
                         ::testing::Combine(::testing::Values(ElementType::string), ::testing::ValuesIn(inputShapes)),
                         CustomOpStringCPUTest::getTestCaseName);

} // namespace CPULayerTestsDefinitions
