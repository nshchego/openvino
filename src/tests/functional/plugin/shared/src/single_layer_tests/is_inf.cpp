// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"
#include "ie_test_utils/common_test_utils/ov_tensor_utils.hpp"
#include "single_layer_tests/is_inf.hpp"

//#include "functional_test_utils/plugin_cache.hpp"

using namespace ov::test::subgraph;

std::string IsInfLayerTest::getTestCaseName(const testing::TestParamInfo<IsInfParams>& obj) {
    std::vector<InputShape> shapes;
    ElementType dataPrc;
    bool detectNegative, detectPositive;
    std::string targetName;
    ov::AnyMap additionalConfig;
    std::tie(shapes, detectNegative, detectPositive, dataPrc, targetName, additionalConfig) = obj.param;
    std::ostringstream results;

    results << "IS=(";
    for (const auto& shape : shapes) {
        results << CommonTestUtils::partialShape2str({shape.first}) << "_";
    }
    results << ")_TS=(";
    for (const auto& shape : shapes) {
        for (const auto& item : shape.second) {
            results << CommonTestUtils::vec2str(item) << "_";
        }
    }
    results << ")_detectNegative=" << (detectNegative ? "True" : "False") << "_";
    results << "detectPositive=" << (detectPositive ? "True" : "False") << "_";
    results << "dataPrc=" << dataPrc << "_";
    results << "trgDev=" << targetName;

    for (auto const& configItem : additionalConfig) {
        results << "_configItem=" << configItem.first << "_";
        configItem.second.print(results);
    }

    return results.str();
}

void IsInfLayerTest::SetUp() {
    std::vector<InputShape> shapes;
    ElementType dataPrc;
    bool detectNegative, detectPositive;
    std::string targetName;
    ov::AnyMap additionalConfig;
    std::tie(shapes, detectNegative, detectPositive, dataPrc, targetDevice, additionalConfig) = this->GetParam();

    init_input_shapes(shapes);
    configuration.insert(additionalConfig.begin(), additionalConfig.end());

    auto parameters = ngraph::builder::makeDynamicParams(dataPrc, { inputDynamicShapes.front() });
    parameters[0]->set_friendly_name("Data");

    ov::op::v10::IsInf::Attributes attributes {detectNegative, detectPositive};
    auto isInf = std::make_shared<ov::op::v10::IsInf>(parameters[0], attributes);
    function = std::make_shared<ngraph::Function>(isInf, parameters, "IsInf");
}

void IsInfLayerTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& input = function->inputs()[0];

    int32_t range = std::accumulate(targetInputStaticShapes[0].begin(), targetInputStaticShapes[0].end(), 1u, std::multiplies<uint32_t>());
    auto tensor = utils::create_and_fill_tensor(
            input.get_element_type(), targetInputStaticShapes[0], range, -range / 2, 1);

    auto pointer = tensor.data<element_type_traits<ov::element::Type_t::f32>::value_type>();
    testing::internal::Random random(1);
    random.Generate(range);
    for (size_t i = 0; i < range / 2; i++) {
        pointer[random.Generate(range)] = i % 2 == 0 ? std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity();
    }

    inputs.insert({input.get_node_shared_ptr(), tensor});
}
