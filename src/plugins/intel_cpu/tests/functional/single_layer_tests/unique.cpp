// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>

using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        std::vector<InputShape>,             // Input shapes
        std::tuple<bool, int>,               // Is flattened and axis
        bool,                                // Sorted
        ElementType,                         // Data precision
        CPUSpecificParams,                   // CPU specific params
        std::map<std::string, std::string>   // Additional config
> UniqueLayerTestCPUParams;

class UniqueLayerTestCPU : public testing::WithParamInterface<UniqueLayerTestCPUParams>,
                           virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<UniqueLayerTestCPUParams> obj) {
        std::vector<InputShape> inputShapes;
        std::tuple<bool, int> flatOrAxis;
        bool sorted;
        ElementType dataPrecision;
        CPUSpecificParams cpuParams;
        std::map<std::string, std::string> additionalConfig;

        std::tie(inputShapes, flatOrAxis, sorted, dataPrecision, cpuParams, additionalConfig) = obj.param;

        std::ostringstream result;
        result << "IS=(";
        for (size_t i = 0lu; i < inputShapes.size(); i++) {
            result << CommonTestUtils::partialShape2str({inputShapes[i].first}) << (i < inputShapes.size() - 1lu ? "_" : "");
        }
        result << ")_TS=";
        for (size_t i = 0lu; i < inputShapes.front().second.size(); i++) {
            result << "{";
            for (size_t j = 0lu; j < inputShapes.size(); j++) {
                result << CommonTestUtils::vec2str(inputShapes[j].second[i]) << (j < inputShapes.size() - 1lu ? "_" : "");
            }
            result << "}_";
        }

        if (!std::get<0>(flatOrAxis)) {
            result << "axis=" << std::get<1>(flatOrAxis) << "_";
        } else {
            result << "flattened" << "_";
        }
        result << "sorted=" << (sorted ? "True" : "False") << "_";
        result << "dataPrc=" << dataPrecision;
        result << CPUTestsBase::getTestCaseName(cpuParams);

        if (!additionalConfig.empty()) {
            result << "_PluginConf";
            for (auto &item : additionalConfig) {
                if (item.second == InferenceEngine::PluginConfigParams::YES)
                    result << "_" << item.first << "=" << item.second;
            }
        }

        return result.str();
    }

protected:
    void SetUp() override {
        std::vector<InputShape> inputShapes;
        std::tuple<bool, int> flatOrAxis;
        bool sorted, flattened;
        int axis;
        ElementType dataPrecision;
        CPUSpecificParams cpuParams;
        std::map<std::string, std::string> additionalConfig;

        std::tie(inputShapes, flatOrAxis, sorted, dataPrecision, cpuParams, additionalConfig) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        targetDevice = CommonTestUtils::DEVICE_CPU;
        init_input_shapes(inputShapes);
        configuration.insert(additionalConfig.begin(), additionalConfig.end());
        flattened = std::get<0>(flatOrAxis);

        if (additionalConfig[InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16] == InferenceEngine::PluginConfigParams::YES) {
            selectedType = makeSelectedTypeStr(selectedType, ElementType::bf16);
        } else {
            selectedType = makeSelectedTypeStr(selectedType, dataPrecision);
        }

        auto params = ngraph::builder::makeDynamicParams(dataPrecision, inputDynamicShapes);
        params[0]->set_friendly_name("data");
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ov::op::v0::Parameter>(params));
        std::shared_ptr<ov::Node> uniqueNode;
        if (flattened) {
            uniqueNode = std::make_shared<ov::op::v10::Unique>(paramOuts[0],sorted);
        } else {
            axis = std::get<1>(flatOrAxis);
            uniqueNode = std::make_shared<ov::op::v10::Unique>(paramOuts[0],
                                                               ov::op::v0::Constant::create(ov::element::i64, ov::Shape({1}), {axis}),
                                                               sorted);
        }

        function = makeNgraphFunction(dataPrecision, params, uniqueNode, "UniqueCPU");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        for (int i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::runtime::Tensor tensor;

            if (funcInput.get_node()->get_friendly_name() == "data") {
                int32_t range = std::pow(2, 19);
                tensor = utils::create_and_fill_tensor(
                        funcInput.get_element_type(), targetInputStaticShapes[0], range, -range / 2, 1);
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }
};

TEST_P(UniqueLayerTestCPU, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
    CheckPluginRelatedResults(compiledModel, "Unique");
}

namespace {

const std::vector<ElementType> dataPrecisionSmoke = {
        ElementType::f32,
        ElementType::i32
};
const std::vector<ElementType> dataPrecisionNightly = {
        ElementType::bf16,
        ElementType::i8
};

std::vector<std::tuple<bool, int>> flatOrAxis { {true, 0}, {false, 0}, {false, 1}, {false, 2} };

std::vector<bool> sorted { true, false};

std::vector<std::map<std::string, std::string>> additionalConfig
    = {{{InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16, InferenceEngine::PluginConfigParams::NO}},
       {{InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16, InferenceEngine::PluginConfigParams::YES}}};

std::vector<CPUSpecificParams> getCPUInfo() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (InferenceEngine::with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{}, {}, {"jit_avx512"}, "jit_avx512"});
    } else if (InferenceEngine::with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{}, {}, {"jit_avx2"}, "jit_avx2"});
    } else if (InferenceEngine::with_cpu_x86_avx()) {
        resCPUParams.push_back(CPUSpecificParams{{}, {}, {"jit_avx"}, "jit_avx"});
    } else if (InferenceEngine::with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{}, {}, {"jit_sse42"}, "jit_sse42"});
    }
    return resCPUParams;
}

std::vector<std::vector<InputShape>> getStaticShapes() {
    // SSE42
    std::vector<std::vector<InputShape>> result = {
        { { {}, { {1, 1, 1} } } },    // Static shapes
        { { {}, { {1, 2, 1} } } },    // Static shapes
        { { {}, { {1, 1, 3} } } },    // Static shapes
        { { {}, { {2, 2, 1} } } },    // Static shapes
        { { {}, { {1, 4, 1} } } },    // Static shapes
        { { {}, { {1, 5, 1} } } },    // Static shapes
        { { {}, { {3, 2, 1} } } },    // Static shapes
        { { {}, { {1, 1, 7} } } },    // Static shapes
        { { {}, { {2, 2, 2} } } },    // Static shapes
        { { {}, { {1, 8, 1} } } },    // Static shapes
        { { {}, { {3, 3, 1} } } },    // Static shapes
        { { {}, { {1, 5, 2} } } },    // Static shapes
        { { {}, { {1, 1, 11} } } },   // Static shapes
        { { {}, { {32, 35, 37} } } }  // Static shapes
    };
    // AVX2
    if (InferenceEngine::with_cpu_x86_avx2() || InferenceEngine::with_cpu_x86_avx()) {
        std::vector<std::vector<InputShape>> tmp = {
            { { {}, { {2, 3, 2} } } },    // Static shapes
            { { {}, { {1, 1, 13} } } },   // Static shapes
            { { {}, { {7, 1, 2} } } },    // Static shapes
            { { {}, { {3, 5, 1} } } },    // Static shapes
            { { {}, { {4, 2, 2} } } },    // Static shapes
            { { {}, { {1, 17, 1} } } },   // Static shapes
            { { {}, { {3, 2, 3} } } },    // Static shapes
            { { {}, { {8, 16, 32} } } },  // Static shapes
            { { {}, { {37, 19, 11} } } }  // Static shapes
        };
        result.insert(result.end(), tmp.begin(), tmp.end());
    }
    // AVX512
    if (InferenceEngine::with_cpu_x86_avx512f()) {
        std::vector<std::vector<InputShape>> tmp = {
            { { {}, { {1, 19, 1} } } },   // Static shapes
            { { {}, { {2, 5, 2} } } },    // Static shapes
            { { {}, { {1, 3, 7} } } },    // Static shapes
            { { {}, { {11, 1, 2} } } },   // Static shapes
            { { {}, { {1, 1, 23} } } },   // Static shapes
            { { {}, { {4, 3, 2} } } },    // Static shapes
            { { {}, { {5, 1, 5} } } },    // Static shapes
            { { {}, { {100, 1, 1} } } },  // Static shapes
            { { {}, { {5, 5, 5} } } },     // Static shapes


{ { {}, { {1, 3, 16} } } },     // Static shapes
{ { {}, { {1, 3, 15} } } },     // Static shapes
{ { {}, { {1, 16, 16} } } },     // Static shapes
{ { {}, { {1, 16, 32} } } },     // Static shapes
{ { {}, { {1, 20, 30} } } }     // Static shapes
        };
        result.insert(result.end(), tmp.begin(), tmp.end());
    }

    return result;
}

INSTANTIATE_TEST_SUITE_P(smoke_static, UniqueLayerTestCPU,
                ::testing::Combine(
                        ::testing::ValuesIn(getStaticShapes()),
                        ::testing::ValuesIn(flatOrAxis),
                        ::testing::ValuesIn(sorted),
                        ::testing::ValuesIn(dataPrecisionSmoke),
                        ::testing::ValuesIn(getCPUInfo()),
                        ::testing::Values(additionalConfig[0])),
                UniqueLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_static, UniqueLayerTestCPU,
                ::testing::Combine(
                        ::testing::ValuesIn(getStaticShapes()),
                        ::testing::ValuesIn(flatOrAxis),
                        ::testing::ValuesIn(sorted),
                        ::testing::ValuesIn(dataPrecisionNightly),
                        ::testing::ValuesIn(getCPUInfo()),
                        ::testing::Values(additionalConfig[1])),
                UniqueLayerTestCPU::getTestCaseName);


const std::vector<std::vector<InputShape>> dynamicInSapes = {
//    { { { ov::Dimension(1, 15), -1, -1, -1 },                               // Dynamic shape 0
//        { {1, 1, 1, 1}, {6, 3, 1, 2}, {4, 5, 3, 1}, {2, 7, 2, 2} } },       // Target shapes
//      { { ov::Dimension(1, 16), -1, -1, -1 },                               // Dynamic shape 1
//        { {1, 1, 1, 2}, {6, 2, 2, 2}, {4, 1, 3, 2}, {2, 1, 2, 2} } } },     // Target shapes
//    { { { -1, -1, -1, -1 },                                                 // Dynamic shape 0
//        { {1, 2, 1, 5}, {3, 4, 2, 3}, {5, 6, 7, 1}, {7, 8, 2, 4} } },       // Target shapes
//      { { -1, -1, -1, 2 },                                                  // Dynamic shape 1
//        { {1, 2, 4, 2}, {3, 1, 7, 2}, {5, 2, 3, 2}, {7, 1, 5, 2} } } },     // Target shapes
//    { { { ov::Dimension(2, 15), -1, -1, -1 },                               // Dynamic shape 0
//        { {8, 3, 3, 3}, {6, 5, 2, 5}, {4, 7, 1, 11}, {2, 9, 3, 4} } },      // Target shapes
//      { { -1, 3, 7, 2 },                                                    // Dynamic shape 1
//        { {8, 3, 7, 2}, {6, 3, 7, 2}, {4, 3, 7, 2}, {2, 3, 7, 2} } } },     // Target shapes
//    { { { 3, 4, 4, 5 },                                                     // Dynamic shape 0
//        { {3, 4, 4, 5}, {3, 4, 4, 5}, {3, 4, 4, 5}, {3, 4, 4, 5} } },       // Target shapes
//      { { -1, -1, -1, 2 },                                                  // Dynamic shape 1
//        { {3, 3, 4, 2}, {3, 1, 11, 2}, {3, 2, 5, 2}, {3, 3, 3, 2} } } },    // Target shapes
//    { { { -1, -1, -1, -1 },                                                 // Dynamic shape 0
//        { {1, 2, 1, 13}, {3, 4, 7, 2}, {5, 6, 3, 5}, {7, 8, 4, 4} } },      // Target shapes
//      { { -1, -1, -1, -1 },                                                 // Dynamic shape 1
//        { {1, 4, 4, 2}, {3, 3, 5, 2}, {5, 2, 7, 2}, {7, 1, 13, 2} } } },    // Target shapes
//    { { { -1, -1, -1, -1 },                                                 // Dynamic shape 0
//        { {2, 11, 1, 17}, {4, 9, 6, 3}, {6, 7, 7, 3}, {8, 3, 2, 11} } },    // Target shapes
//      { { -1, -1, -1, 2 },                                                  // Dynamic shape 1
//        { {2, 5, 4, 2}, {4, 1, 19, 2}, {6, 6, 3, 2}, {8, 1, 17, 2} } } },   // Target shapes
//    { { { 3, -1, -1, -1 },                                                  // Dynamic shape 0
//        { {3, 2, 1, 23}, {3, 4, 3, 8}, {3, 6, 5, 5}, {3, 8, 31, 1} } },     // Target shapes
//      { { -1, -1, -1, 2 },                                                  // Dynamic shape 1
//        { {3, 31, 1, 2}, {3, 6, 4, 2}, {3, 23, 1, 2}, {3, 11, 2, 2} } } },  // Target shapes
//    { { { -1, 3, -1, -1 },                                                  // Dynamic shape 0
//        { {8, 3, 8, 4}, {6, 3, 33, 1}, {4, 3, 8, 6}, {2, 3, 8, 8} } },      // Target shapes
//      { { -1, -1, -1, 2 },                                                  // Dynamic shape 1
//        { {8, 8, 8, 2}, {6, 8, 7, 2}, {4, 1, 33, 2}, {2, 4, 8, 2} } } }     // Target shapes
};

INSTANTIATE_TEST_SUITE_P(smoke_dynamic, UniqueLayerTestCPU,
                     ::testing::Combine(
                             ::testing::ValuesIn(dynamicInSapes),
                             ::testing::ValuesIn(flatOrAxis),
                             ::testing::ValuesIn(sorted),
                             ::testing::ValuesIn(dataPrecisionSmoke),
                             ::testing::ValuesIn(getCPUInfo()),
                             ::testing::Values(additionalConfig[0])),
                     UniqueLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_dynamic, UniqueLayerTestCPU,
                         ::testing::Combine(
                                 ::testing::ValuesIn(dynamicInSapes),
                                 ::testing::ValuesIn(flatOrAxis),
                                 ::testing::ValuesIn(sorted),
                                 ::testing::ValuesIn(dataPrecisionNightly),
                                 ::testing::ValuesIn(getCPUInfo()),
                                 ::testing::Values(additionalConfig[1])),
                         UniqueLayerTestCPU::getTestCaseName);
} // namespace
} // namespace CPULayerTestsDefinitions
