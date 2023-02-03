// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include <cpp_interfaces/interface/ie_internal_plugin_config.hpp>

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

using convertLayerTestParamsSet = std::tuple<
        InputShape,        // input shapes
        ElementType,       // input precision
        ElementType,       // output precision
        ov::AnyMap,        // Additional network configuration
        CPUSpecificParams
>;

class ConvertCPULayerTest : public testing::WithParamInterface<convertLayerTestParamsSet>,
                            virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<convertLayerTestParamsSet> obj) {
        InputShape inputShape;
        ElementType inPrc, outPrc;
        CPUSpecificParams cpuParams;
        ov::AnyMap config;
        std::tie(inputShape, inPrc, outPrc, config, cpuParams) = obj.param;

        std::ostringstream result;

        result << "IS=" << CommonTestUtils::partialShape2str({inputShape.first}) << "_";
        result << "TS=";
        for (const auto& shape : inputShape.second) {
            result << CommonTestUtils::vec2str(shape) << "_";
        }
        result << "inputPRC=" << inPrc << "_";
        result << "targetPRC=" << outPrc << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams);

        for (auto const& configItem : config) {
            result << "_configItem=" << configItem.first << "_";
            configItem.second.print(result);
        }

        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;

        InputShape shapes;
        CPUSpecificParams cpuParams;
        std::tie(shapes, inPrc, outPrc, configuration, cpuParams) = GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        auto primitive = selectedType;
        if (primitive.empty())
            primitive = getPrimitiveType();
        // WA: I32 precision support disabled in snippets => primitive has to be changed
        // TODO: remove the WA after I32 is supported in snippets (ticket: 99803)
        if (inPrc == ElementType::i32 || inPrc == ElementType::i64 || outPrc == ElementType::i32 || outPrc == ElementType::i64)
            primitive = "unknown";

        if (inPrc == ElementType::i64 || inPrc == ElementType::u64) {
            auto i64Flag = configuration.find(PluginConfigInternalParams::KEY_CPU_NATIVE_I64);
            if (i64Flag == configuration.end() || i64Flag->second == PluginConfigParams::NO) {
                selectedType = makeSelectedTypeStr(primitive, ElementType::i32);
            } else {
                selectedType = makeSelectedTypeStr(primitive, ElementType::i64);
            }
        } else if (inPrc == ElementType::u8) {
            selectedType = makeSelectedTypeStr(primitive, ElementType::i8);
        } else {
            selectedType = makeSelectedTypeStr(primitive, inPrc);
        }

        for (size_t i = 0; i < shapes.second.size(); i++) {
            targetStaticShapes.push_back(std::vector<ov::Shape>{shapes.second[i]});
        }

        inputDynamicShapes.push_back(shapes.first);

        ov::ParameterVector params = ngraph::builder::makeDynamicParams(inPrc, inputDynamicShapes);
        auto conversion = ngraph::builder::makeConversion(params.front(), outPrc, ngraph::helpers::ConversionTypes::CONVERT);

        function = makeNgraphFunction(inPrc, params, conversion, "ConversionCPU");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        if (outPrc != ElementType::boolean) {
            SubgraphBaseTest::generate_inputs(targetInputStaticShapes);
            return;
        }

        // In the scenario where input precision is floating point and output precision is boolean,
        // for CPU plugin, the output precision boolean will be converted to u8 during common transformation,
        // the elements in the output tensor will retain the format of u8 with the range [0, 255].
        // But the output precision in ngraph reference is literal boolean, the elements are either 0 or 1.
        // Here input floating points values are set to be in the range of [-1, 1], so no extra precision
        // converting between actual output and expected output will be needed from the side of single layer tests.
        inputs.clear();
        const auto& funcInputs = function->inputs();

        auto shape = targetInputStaticShapes.front();
        size_t size = shape_size(shape);
        ov::Tensor tensor = utils::create_and_fill_tensor(funcInputs[0].get_element_type(), shape, 2 * size);

        if (inPrc == ElementType::f32) {
            auto *rawBlobDataPtr = static_cast<float *>(tensor.data());
            for (size_t i = 0; i < size; ++i) {
                rawBlobDataPtr[i] = rawBlobDataPtr[i] / size - 1;
            }
        } else if (inPrc == ElementType::bf16) {
            auto *rawBlobDataPtr = static_cast<ov::bfloat16 *>(tensor.data());
            for (size_t i = 0; i < size; ++i) {
                rawBlobDataPtr[i] = rawBlobDataPtr[i] / size - 1;
            }
        } else {
            FAIL() << "Generating inputs with precision" << inPrc << " isn't supported, if output precision is boolean.";
        }

        inputs.insert({funcInputs[0].get_node_shared_ptr(), tensor});
    }

private:
    ElementType inPrc, outPrc;
};

TEST_P(ConvertCPULayerTest, CompareWithRefs) {
    run();

    CheckPluginRelatedResults(compiledModel, std::set<std::string>{"Convert", "Subgraph"});
}

std::vector<InputShape> inShapes_4D_dynamic = {
        {
            // dynamic
            {{-1, -1, -1, -1}},
            // target
            {
                {2, 4, 4, 1},
                {2, 17, 5, 4},
                {1, 2, 3, 4}
            }
        },
        {
            // dynamic
            {{{1, 5}, {2, 22}, {2, 9}, {1, 4}}},
            // target
            {
                {2, 17, 5, 4},
                {5, 2, 3, 2},
                {1, 10, 4, 1},
            }
        }
};

// List of precisions natively supported by onednn.
const std::vector<ElementType> precisions = {
        ElementType::u8,
        ElementType::i8,
        ElementType::i32,
        ElementType::f32,
        ElementType::bf16
};

const std::vector<ElementType> precisions_floating_point = {
        ElementType::f32,
        ElementType::bf16
};

std::vector<CPUSpecificParams> memForm4D_dynamic = {
    CPUSpecificParams({nchw}, {nchw}, {}, "unknown"),
    CPUSpecificParams({nhwc}, {nhwc}, {}, "unknown"),
    CPUSpecificParams({nChw8c}, {nChw8c}, {}, "unknown"),
    CPUSpecificParams({nChw16c}, {nChw16c}, {}, "unknown")
};

const ov::AnyMap emptyConfig = {};
const ov::AnyMap i64Config = {{PluginConfigInternalParams::KEY_CPU_NATIVE_I64, PluginConfigParams::YES}};

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_Dynamic, ConvertCPULayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes_4D_dynamic),
                                ::testing::ValuesIn(precisions),
                                ::testing::ValuesIn(precisions),
                                ::testing::Values(emptyConfig),
                                ::testing::ValuesIn(memForm4D_dynamic)),
                        ConvertCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_FromI64_Dynamic, ConvertCPULayerTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inShapes_4D_dynamic),
                                 ::testing::Values(ElementType::i64),
                                 ::testing::ValuesIn(precisions),
                                 ::testing::Values(i64Config),
                                 ::testing::ValuesIn(memForm4D_dynamic)),
                         ConvertCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_ToI64_Dynamic, ConvertCPULayerTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inShapes_4D_dynamic),
                                 ::testing::ValuesIn(precisions),
                                 ::testing::Values(ElementType::i64),
                                 ::testing::Values(i64Config),
                                 ::testing::ValuesIn(memForm4D_dynamic)),
                         ConvertCPULayerTest::getTestCaseName);

std::vector<InputShape> inShapes_4D_static = {
    {{1, 2, 3, 4}, {{1, 2, 3, 4}}},
    {{1, 1, 1080, 1920}, {{1, 1, 1080, 1920}}},
};

std::vector<CPUSpecificParams> memForm4D_static_common = {
    CPUSpecificParams({nchw}, {nchw}, {}, {}),
    CPUSpecificParams({nhwc}, {nhwc}, {}, {}),
};

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest, ConvertCPULayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes_4D_static),
                                ::testing::ValuesIn(precisions),
                                ::testing::ValuesIn(precisions),
                                ::testing::Values(emptyConfig),
                                ::testing::ValuesIn(memForm4D_static_common)),
                        ConvertCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_FromI64, ConvertCPULayerTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inShapes_4D_static),
                                 ::testing::Values(ElementType::i64),
                                 ::testing::ValuesIn(precisions),
                                 ::testing::Values(i64Config),
                                 ::testing::ValuesIn(memForm4D_static_common)),
                         ConvertCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_ToI64, ConvertCPULayerTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inShapes_4D_static),
                                 ::testing::ValuesIn(precisions),
                                 ::testing::Values(ElementType::i64),
                                 ::testing::Values(i64Config),
                                 ::testing::ValuesIn(memForm4D_static_common)),
                         ConvertCPULayerTest::getTestCaseName);

std::vector<InputShape> inShapes_4D_blocked = {
    {{1, 16, 5, 5}, {{1, 16, 5, 5}}},
};

std::vector<CPUSpecificParams> memForm4D_static_blocked = {
    CPUSpecificParams({nChw16c}, {nChw16c}, {}, {})
};

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_Blocked, ConvertCPULayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes_4D_blocked),
                                ::testing::ValuesIn(precisions),
                                ::testing::ValuesIn(precisions),
                                ::testing::Values(emptyConfig),
                                ::testing::ValuesIn(filterCPUSpecificParams(memForm4D_static_blocked))),
                        ConvertCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_BOOL_Static, ConvertCPULayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes_4D_static),
                                ::testing::ValuesIn(precisions_floating_point),
                                ::testing::Values(ElementType::boolean),
                                ::testing::Values(emptyConfig),
                                ::testing::Values(CPUSpecificParams({nchw}, {nchw}, {}, {}))),
                        ConvertCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_BOOL_Dynamic, ConvertCPULayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes_4D_dynamic),
                                ::testing::ValuesIn(precisions_floating_point),
                                ::testing::Values(ElementType::boolean),
                                ::testing::Values(emptyConfig),
                                ::testing::Values(CPUSpecificParams({nchw}, {nchw}, {}, "unknown"))),
                        ConvertCPULayerTest::getTestCaseName);

} // namespace CPULayerTestsDefinitions
