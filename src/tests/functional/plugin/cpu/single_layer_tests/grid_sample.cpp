// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>

using namespace CPUTestUtils;
using namespace ov::test;
using ov::op::v9::GridSample;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        std::vector<InputShape>,                 // Input shapes
        GridSample::InterpolationMode,           // Interpolation mode
        GridSample::PaddingMode,                 // Padding mode
        bool,                                    // Align corners
        ElementType,                             // Data precision
        ElementType,                             // Grid precision
        CPUSpecificParams,                       // CPU specific params
        std::map<std::string, std::string>       // Additional config
> GridSampleLayerTestCPUParams;

class GridSampleLayerTestCPU : public testing::WithParamInterface<GridSampleLayerTestCPUParams>,
                           virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<GridSampleLayerTestCPUParams> obj) {
        std::vector<InputShape> inputShapes;
        GridSample::InterpolationMode interpolateMode;
        GridSample::PaddingMode paddingMode;
        bool alignCorners;
        ElementType dataPrecision, gridPrecision;
        CPUSpecificParams cpuParams;
        std::map<std::string, std::string> additionalConfig;

        std::tie(inputShapes, interpolateMode, paddingMode, alignCorners, dataPrecision, gridPrecision, cpuParams, additionalConfig) = obj.param;

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

        result << "interpMode=" << (interpolateMode == GridSample::InterpolationMode::BILINEAR ? "BILINEAR" :
                interpolateMode == GridSample::InterpolationMode::BICUBIC ? "BICUBIC" : "NEAREST") << "_";
        result << "padMode=" << (paddingMode == GridSample::PaddingMode::ZEROS ? "ZEROS" :
                paddingMode == GridSample::PaddingMode::BORDER ? "BORDER" : "REFLECTION") << "_";
        result << "alignCorners=" << (alignCorners ? "True" : "False") << "_";
        result << "dataPrc=" << dataPrecision << "_";
        result << "gridPrc=" << gridPrecision << "_";
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
        GridSample::InterpolationMode interpolateMode;
        GridSample::PaddingMode paddingMode;
        bool alignCorners;
        ElementType dataPrecision, gridPrecision;
        CPUSpecificParams cpuParams;
        std::map<std::string, std::string> additionalConfig;

        std::tie(inputShapes, interpolateMode, paddingMode, alignCorners, dataPrecision, gridPrecision, cpuParams, additionalConfig) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        targetDevice = CommonTestUtils::DEVICE_CPU;
        init_input_shapes(inputShapes);
        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        if (additionalConfig[InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16] == InferenceEngine::PluginConfigParams::YES) {
            selectedType = makeSelectedTypeStr(selectedType, ElementType::bf16);
        } else {
            selectedType = makeSelectedTypeStr(selectedType, dataPrecision);
        }

        auto params = ngraph::builder::makeDynamicParams({dataPrecision, gridPrecision}, inputDynamicShapes);
        params[0]->set_friendly_name("data");
        params[1]->set_friendly_name("grid");
//        params[0]->get_output_tensor(0).set_names({"data"});
//        params[1]->get_output_tensor(0).set_names({"grid"});
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ov::op::v0::Parameter>(params));
        GridSample::Attributes attributes = {alignCorners, interpolateMode, paddingMode};
        auto gridSampleNode = std::make_shared<GridSample>(paramOuts[0], paramOuts[1], attributes);

        function = makeNgraphFunction(dataPrecision, params, gridSampleNode, "GridSampleCPU");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
//        SubgraphBaseTest::generate_inputs(targetInputStaticShapes);
        inputs.clear();
        const auto& funcInputs = function->inputs();

//        auto dataInput = function->input("data");
//        auto dataShape = dataInput.get_shape();
//        int32_t resolution = dataShape[2] * dataShape[3];
//
//        auto gridInput = function->input("grid");
//        ov::runtime::Tensor tensor = utils::create_and_fill_tensor(
//                gridInput.get_element_type(), targetInputStaticShapes[1], 2, -1, resolution);
//        inputs.insert({gridInput.get_node_shared_ptr(), tensor});

        for (int i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::runtime::Tensor tensor;

            if (funcInput.get_node()->get_friendly_name() == "data") {
                int32_t resolution = std::accumulate(targetInputStaticShapes[0].begin(), targetInputStaticShapes[0].end(), 1, std::multiplies<int32_t>());
                tensor = utils::create_and_fill_tensor(
                        funcInput.get_element_type(), targetInputStaticShapes[0], resolution, -resolution / 2, 1);
            } else if (funcInput.get_node()->get_friendly_name() == "grid") {
                int32_t resolution = targetInputStaticShapes[0][2] * targetInputStaticShapes[0][3];
                tensor = utils::create_and_fill_tensor(
                        funcInput.get_element_type(), targetInputStaticShapes[1], resolution, -1, resolution / 2);
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }
};

TEST_P(GridSampleLayerTestCPU, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
    CheckPluginRelatedResults(compiledModel, "GridSample");
}

namespace {

const std::vector<ElementType> dataPrecision = {
        ElementType::f32,
        ElementType::bf16,
        ElementType::i32,
        ElementType::i8
};

const std::vector<ElementType> gridPrecision = {
        ElementType::f32,
        ElementType::bf16,
        ElementType::f16
};

std::vector<std::map<std::string, std::string>> additionalConfig
    = {{{InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16, InferenceEngine::PluginConfigParams::NO}},
       {{InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16, InferenceEngine::PluginConfigParams::YES}}};

std::vector<GridSample::InterpolationMode> interpolateMode {
        GridSample::InterpolationMode::BILINEAR,
        GridSample::InterpolationMode::BICUBIC,
        GridSample::InterpolationMode::NEAREST };

std::vector<GridSample::PaddingMode> paddingMode {
        GridSample::PaddingMode::ZEROS,
        GridSample::PaddingMode::BORDER,
        GridSample::PaddingMode::REFLECTION };

std::vector<bool> alignCorners { true, false };

std::vector<CPUSpecificParams> getCPUInfo() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (InferenceEngine::with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{}, {}, {"jit_avx512"}, "jit_avx512"});
    } else if (InferenceEngine::with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{}, {}, {"jit_avx2"}, "jit_avx2"});
    } else if (InferenceEngine::with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{}, {}, {"jit_sse42"}, "jit_sse42"});
    }
    return resCPUParams;
}

std::vector<std::vector<InputShape>> getStaticShapes() {
    // SSE42
    std::vector<std::vector<InputShape>> result = {
        { { {}, { {1, 5, 1, 1} } },   // Static shapes
          { {}, { {1, 1, 1, 2} } }
        },
        { { {}, { {2, 4, 7, 1} } },   // Static shapes
          { {}, { {2, 1, 2, 2} } }
        },
        { { {}, { {3, 3, 3, 3} } },   // Static shapes
          { {}, { {3, 3, 1, 2} } }
        },
        { { {}, { {4, 2, 5, 4} } },   // Static shapes
          { {}, { {4, 2, 2, 2} } }
        },
        { { {}, { {5, 1, 5, 5} } },   // Static shapes
          { {}, { {5, 1, 5, 2} } }
        },
        { { {}, { {4, 2, 4, 6} } },   // Static shapes
          { {}, { {4, 2, 3, 2} } }
        },
        { { {}, { {3, 3, 5, 7} } },   // Static shapes
          { {}, { {3, 7, 1, 2} } }
        },
        { { {}, { {2, 4, 7, 7} } },   // Static shapes
          { {}, { {2, 2, 4, 2} } }
        },
        { { {}, { {2, 5, 8, 8} } },   // Static shapes
          { {}, { {2, 3, 3, 2} } }
        },
        { { {}, { {2, 6, 9, 8} } },   // Static shapes
          { {}, { {2, 2, 5, 2} } }
        }
    }; // SSE42

    if (InferenceEngine::with_cpu_x86_avx2()) {
        std::vector<std::vector<InputShape>> tmp = {
            { { {}, { {1, 7, 5, 3} } },   // Static shapes
              { {}, { {1, 1, 11, 2} } }
            },
            { { {}, { {2, 6, 7, 2} } },   // Static shapes
              { {}, { {2, 6, 2, 2} } }
            },
            { { {}, { {3, 5, 6, 3} } },   // Static shapes
              { {}, { {3, 1, 13, 2} } }
            },
            { { {}, { {4, 4, 5, 6} } },   // Static shapes
              { {}, { {4, 2, 7, 2} } }
            },
            { { {}, { {5, 3, 4, 5} } },   // Static shapes
              { {}, { {5, 3, 5, 2} } }
            },
            { { {}, { {4, 2, 7, 6} } },   // Static shapes
              { {}, { {4, 4, 4, 2} } }
            },
            { { {}, { {3, 3, 9, 7} } },   // Static shapes
              { {}, { {3, 1, 17, 2} } }
            },
            { { {}, { {2, 4, 9, 8} } },   // Static shapes
              { {}, { {2, 19, 1, 2} } }
            }
        };
        result.insert(result.end(), tmp.begin(), tmp.end());
    } // AVX2

    if (InferenceEngine::with_cpu_x86_avx512f()) {
        std::vector<std::vector<InputShape>> tmp = {
            { { {}, { {1, 7, 2, 9} } },    // Static shapes
              { {}, { {1, 4, 5, 2} } }
            },
            { { {}, { {2, 6, 2, 10} } },   // Static shapes
              { {}, { {2, 3, 7, 2} } },
            },
            { { {}, { {3, 5, 2, 11} } },   // Static shapes
              { {}, { {3, 4, 6, 2} } }
            },
            { { {}, { {4, 4, 2, 12} } },   // Static shapes
              { {}, { {4, 5, 5, 2} } },
            },
            { { {}, { {5, 3, 2, 13} } },   // Static shapes
              { {}, { {5, 1, 31, 2} } },
            },
            { { {}, { {4, 3, 2, 14} } },   // Static shapes
              { {}, { {4, 4, 8, 2} } },
            },
            { { {}, { {3, 2, 2, 15} } },   // Static shapes
              { {}, { {3, 33, 1, 2} } },
            },
            { { {}, { {2, 1, 2, 16} } },   // Static shapes
              { {}, { {2, 8, 8, 2} } },
            }
        };
        result.insert(result.end(), tmp.begin(), tmp.end());
    } // AVX5

    return result;
}

INSTANTIATE_TEST_SUITE_P(smoke_static_jit, GridSampleLayerTestCPU,
                ::testing::Combine(
                    ::testing::ValuesIn(getStaticShapes()),
                    ::testing::ValuesIn(interpolateMode),
                    ::testing::ValuesIn(paddingMode),
                    ::testing::ValuesIn(alignCorners),
                    ::testing::ValuesIn(dataPrecision),
                    ::testing::ValuesIn(gridPrecision),
                    ::testing::ValuesIn(getCPUInfo()),
                    ::testing::ValuesIn(additionalConfig)),
                GridSampleLayerTestCPU::getTestCaseName);

//INSTANTIATE_TEST_SUITE_P(smoke_static_4D_jit16, GridSampleLayerTestCPU,
//                ::testing::Combine(
//                    ::testing::ValuesIn(getStaticShapes()),
//                    ::testing::ValuesIn(get4DAxisBatchJitStat(ElementType::bf16, 2)),
//                    ::testing::Values(ElementType::bf16),
//                    ::testing::Values(true),
//                    ::testing::ValuesIn(getCPUInfo()),
//                    ::testing::Values(additionalConfig[0])),
//                GridSampleLayerTestCPU::getTestCaseName);
//
//INSTANTIATE_TEST_SUITE_P(smoke_static_4D_jit8, GridSampleLayerTestCPU,
//                ::testing::Combine(
//                    ::testing::ValuesIn(getStaticShapes()),
//                    ::testing::ValuesIn(get4DAxisBatchJitStat(ElementType::i8, 2)),
//                    ::testing::Values(ElementType::i8),
//                    ::testing::Values(true),
//                    ::testing::ValuesIn(getCPUInfo()),
//                    ::testing::Values(additionalConfig[0])),
//                GridSampleLayerTestCPU::getTestCaseName);
//
//// batchDims == indicesRank
//INSTANTIATE_TEST_SUITE_P(smoke_static_4D_jit32_Bmax, GridSampleLayerTestCPU,
//                ::testing::Combine(
//                    ::testing::ValuesIn(getStaticShapes()),
//                    ::testing::ValuesIn(get4DAxisBatchJitStat(ElementType::f32, 3)),
//                    ::testing::Values(ElementType::f32),
//                    ::testing::Values(true),
//                    ::testing::ValuesIn(getCPUInfo()),
//                    ::testing::ValuesIn(additionalConfig)),
//                GridSampleLayerTestCPU::getTestCaseName);
//
//INSTANTIATE_TEST_SUITE_P(smoke_static_4D_jit16_Bmax, GridSampleLayerTestCPU,
//                ::testing::Combine(
//                    ::testing::ValuesIn(getStaticShapes()),
//                    ::testing::ValuesIn(get4DAxisBatchJitStat(ElementType::bf16, 3)),
//                    ::testing::Values(ElementType::bf16),
//                    ::testing::Values(true),
//                    ::testing::ValuesIn(getCPUInfo()),
//                    ::testing::Values(additionalConfig[0])),
//                GridSampleLayerTestCPU::getTestCaseName);
//
//INSTANTIATE_TEST_SUITE_P(smoke_static_4D_jit8_Bmax, GridSampleLayerTestCPU,
//                ::testing::Combine(
//                    ::testing::ValuesIn(getStaticShapes()),
//                    ::testing::ValuesIn(get4DAxisBatchJitStat(ElementType::i8, 3)),
//                    ::testing::Values(ElementType::i8),
//                    ::testing::Values(true),
//                    ::testing::ValuesIn(getCPUInfo()),
//                    ::testing::Values(additionalConfig[0])),
//                GridSampleLayerTestCPU::getTestCaseName);
//
//
//std::vector<std::vector<InputShape>> get4DShapesJitDyn(int maxBatchDims) {
//    std::vector<std::vector<InputShape>> result = {};
//    if (InferenceEngine::with_cpu_x86_avx2()) {
//        if (maxBatchDims == 2) {
//            result = {
//                { { { ov::Dimension(5, 15), -1, -1, -1 },                            // Dynamic shape 0
//                    { {8, 2, 2, 1}, {10, 2, 2, 2}, {8, 2, 2, 3}, {10, 2, 2, 4}} },   // Target shapes
//                  { { ov::Dimension(4, 16), -1, -1 },                                // Dynamic shape 1
//                    { {8, 2, 8}, {10, 2, 7}, {8, 2, 6}, {10, 2, 5} } } },            // Target shapes
//                { { { -1, -1, -1, -1 },                                              // Dynamic shape 0
//                    { {8, 2, 2, 5}, {10, 2, 2, 6}, {8, 2, 2, 7}, {10, 2, 2, 8}} },   // Target shapes
//                  { { -1, -1, -1 },                                                  // Dynamic shape 1
//                    { {8, 2, 4}, {10, 2, 3}, {8, 2, 2}, {10, 2, 1} } } },            // Target shapes
//                { { { ov::Dimension(5, 15), -1, -1, -1 },                            // Dynamic shape 0
//                    { {10, 2, 2, 1}, {10, 2, 2, 2}, {10, 2, 2, 3}, {10, 2, 2, 4}} }, // Target shapes
//                  { { 10, 2, 5 },                                                    // Dynamic shape 1
//                    { {10, 2, 5}, {10, 2, 5}, {10, 2, 5}, {10, 2, 5} } } },          // Target shapes
//                { { { 8, 2, 2, 5 },                                                  // Dynamic shape 0
//                    { {8, 2, 2, 5}, {8, 2, 2, 5}, {8, 2, 2, 5}, {8, 2, 2, 5}} },     // Target shapes
//                  { { -1, -1, -1 },                                                  // Dynamic shape 1
//                    { {8, 2, 4}, {8, 2, 3}, {8, 2, 2}, {8, 2, 1} } } }               // Target shapes
//            };
//        } else if (maxBatchDims == 3) {
//            result = {
//                { { { ov::Dimension(5, 15), -1, -1, -1 },                            // Dynamic shape 0
//                    { {8, 2, 8, 1}, {10, 2, 8, 2}, {8, 2, 8, 3}, {10, 2, 5, 4}} },   // Target shapes
//                  { { ov::Dimension(4, 16), -1, -1 },                                // Dynamic shape 1
//                    { {8, 2, 8}, {10, 2, 8}, {8, 2, 8}, {10, 2, 5} } } },            // Target shapes
//                { { { -1, -1, -1, -1 },                                              // Dynamic shape 0
//                    { {8, 2, 4, 5}, {10, 2, 3, 6}, {8, 2, 2, 7}, {10, 2, 1, 8}} },   // Target shapes
//                  { { -1, -1, -1 },                                                  // Dynamic shape 1
//                    { {8, 2, 4}, {10, 2, 3}, {8, 2, 2}, {10, 2, 1} } } },            // Target shapes
//                { { { ov::Dimension(5, 15), -1, -1, -1 },                            // Dynamic shape 0
//                    { {10, 2, 5, 1}, {10, 2, 5, 2}, {10, 2, 5, 3}, {10, 2, 5, 4}} }, // Target shapes
//                  { { 10, 2, 5 },                                                    // Dynamic shape 1
//                    { {10, 2, 5}, {10, 2, 5}, {10, 2, 5}, {10, 2, 5} } } },          // Target shapes
//                { { { 8, 2, 3, 5 },                                                  // Dynamic shape 0
//                    { {8, 2, 3, 5}, {8, 2, 3, 5}, {8, 2, 3, 5}, {8, 2, 3, 5}} },     // Target shapes
//                  { { -1, -1, -1 },                                                  // Dynamic shape 1
//                    { {8, 2, 3}, {8, 2, 3}, {8, 2, 3}, {8, 2, 3} } } }               // Target shapes
//            };
//        } else {
//            throw std::invalid_argument("Invalid test case. Not valid batch dims.");
//        }
//    }
//    if (InferenceEngine::with_cpu_x86_avx512f()) {
//        std::vector<std::vector<InputShape>> tmp;
//        if (maxBatchDims == 2) {
//            tmp = {
//                { { { ov::Dimension(5, 15), -1, -1, -1 },                               // Dynamic shape 0
//                    { {8, 2, 2, 9}, {10, 2, 2, 10}, {8, 2, 2, 11}, {10, 2, 2, 12}} },   // Target shapes
//                  { { ov::Dimension(4, 16), -1, -1 },                                   // Dynamic shape 1
//                    { {8, 2, 16}, {10, 2, 15}, {8, 2, 14}, {10, 2, 13} } } },           // Target shapes
//                { { { -1, -1, -1, -1 },                                                 // Dynamic shape 0
//                    { {8, 2, 2, 13}, {10, 2, 2, 14}, {8, 2, 2, 15}, {10, 2, 2, 16}} },  // Target shapes
//                  { { -1, -1, -1 },                                                     // Dynamic shape 1
//                    { {8, 2, 12}, {10, 2, 11}, {8, 2, 10}, {10, 2, 9} } } },            // Target shapes
//                { { { ov::Dimension(5, 15), -1, -1, -1 },                               // Dynamic shape 0
//                    { {10, 2, 2, 9}, {10, 2, 2, 10}, {10, 2, 2, 11}, {10, 2, 2, 12}} }, // Target shapes
//                  { { 10, 2, 16 },                                                      // Dynamic shape 1
//                    { {10, 2, 16}, {10, 2, 16}, {10, 2, 16}, {10, 2, 16} } } },         // Target shapes
//                { { { 8, 2, 2, 15 },                                                    // Dynamic shape 0
//                    { {8, 2, 2, 15}, {8, 2, 2, 15}, {8, 2, 2, 15}, {8, 2, 2, 15}} },    // Target shapes
//                  { { -1, -1, -1 },                                                     // Dynamic shape 1
//                    { {8, 2, 12}, {8, 2, 11}, {8, 2, 10}, {8, 2, 9} } } }               // Target shapes
//            };
//        } else if (maxBatchDims == 3) {
//            tmp = {
//                { { { ov::Dimension(5, 15), -1, -1, -1 },                                   // Dynamic shape 0
//                    { {8, 2, 16, 9}, {10, 2, 15, 10}, {8, 2, 14, 11}, {10, 2, 13, 12}} },   // Target shapes
//                  { { ov::Dimension(4, 16), -1, -1 },                                       // Dynamic shape 1
//                    { {8, 2, 16}, {10, 2, 15}, {8, 2, 14}, {10, 2, 13} } } },               // Target shapes
//                { { { -1, -1, -1, -1 },                                                     // Dynamic shape 0
//                    { {8, 2, 12, 13}, {10, 2, 11, 14}, {8, 2, 10, 15}, {10, 2, 9, 16}} },   // Target shapes
//                  { { -1, -1, -1 },                                                         // Dynamic shape 1
//                    { {8, 2, 12}, {10, 2, 11}, {8, 2, 10}, {10, 2, 9} } } },                // Target shapes
//                { { { ov::Dimension(5, 15), -1, -1, -1 },                                   // Dynamic shape 0
//                    { {10, 2, 16, 9}, {10, 2, 16, 10}, {10, 2, 16, 11}, {10, 2, 16, 12}} }, // Target shapes
//                  { { 10, 2, 16 },                                                          // Dynamic shape 1
//                    { {10, 2, 16}, {10, 2, 16}, {10, 2, 16}, {10, 2, 16} } } },             // Target shapes
//                { { { 8, 2, 11, 15 },                                                       // Dynamic shape 0
//                    { {8, 2, 11, 15}, {8, 2, 11, 15}, {8, 2, 11, 15}, {8, 2, 11, 15}} },    // Target shapes
//                  { { -1, -1, -1 },                                                         // Dynamic shape 1
//                    { {8, 2, 11}, {8, 2, 11}, {8, 2, 11}, {8, 2, 11} } } }                  // Target shapes
//            };
//        } else {
//            throw std::invalid_argument("Invalid test case. Not valid batch dims.");
//        }
//        result.insert(result.end(), tmp.begin(), tmp.end());
//    }
//
//    return result;
//}
//
//std::vector<std::tuple<int, int>> get4DAxisBatchJitDyn(ov::element::Type type, int maxBatchDims) {
//    std::vector<std::tuple<int, int>> result = {};
//    if (InferenceEngine::with_cpu_x86_avx512f()) {
//        if (type.size() == 4 || type.size() == 2 || type.size() == 1) {
//            if (maxBatchDims == 2)
//                return std::vector<std::tuple<int, int>>{{3, 0}, {3, 1}, {3, 2}};
//            else if (maxBatchDims == 3)
//                return std::vector<std::tuple<int, int>>{{3, 3}};
//            else
//                throw std::invalid_argument("Invalid test case. Not valid batch dims.");
//        }
//    } else if (InferenceEngine::with_cpu_x86_avx2()) {
//        if (type.size() == 4 || type.size() == 2 || type.size() == 1) {
//            if (maxBatchDims == 2)
//                return std::vector<std::tuple<int, int>>{{3, 0}, {3, 1}, {3, 2}};
//            else if (maxBatchDims == 3)
//                return std::vector<std::tuple<int, int>>{{3, 3}};
//            else
//                throw std::invalid_argument("Invalid test case. Not valid batch dims.");
//        }
//    }
//    return {};
//}
//
//INSTANTIATE_TEST_SUITE_P(smoke_dynamic_4D_jit32, GridSampleLayerTestCPU,
//                ::testing::Combine(
//                    ::testing::ValuesIn(get4DShapesJitDyn(2)),
//                    ::testing::ValuesIn(get4DAxisBatchJitDyn(ElementType::f32, 2)),
//                    ::testing::Values(ElementType::f32),
//                    ::testing::ValuesIn(isAxisConst),
//                    ::testing::ValuesIn(getCPUInfo()),
//                    ::testing::ValuesIn(additionalConfig)),
//                GridSampleLayerTestCPU::getTestCaseName);
//
//INSTANTIATE_TEST_SUITE_P(smoke_dynamic_4D_jit16, GridSampleLayerTestCPU,
//                ::testing::Combine(
//                    ::testing::ValuesIn(get4DShapesJitDyn(2)),
//                    ::testing::ValuesIn(get4DAxisBatchJitDyn(ElementType::bf16, 2)),
//                    ::testing::Values(ElementType::bf16),
//                    ::testing::ValuesIn(isAxisConst),
//                    ::testing::ValuesIn(getCPUInfo()),
//                    ::testing::Values(additionalConfig[0])),
//                GridSampleLayerTestCPU::getTestCaseName);
//
//INSTANTIATE_TEST_SUITE_P(smoke_dynamic_4D_jit8, GridSampleLayerTestCPU,
//                ::testing::Combine(
//                    ::testing::ValuesIn(get4DShapesJitDyn(2)),
//                    ::testing::ValuesIn(get4DAxisBatchJitDyn(ElementType::i8, 2)),
//                    ::testing::Values(ElementType::i8),
//                    ::testing::ValuesIn(isAxisConst),
//                    ::testing::ValuesIn(getCPUInfo()),
//                    ::testing::Values(additionalConfig[0])),
//                GridSampleLayerTestCPU::getTestCaseName);
//
//// batchDims == indicesRank
//INSTANTIATE_TEST_SUITE_P(smoke_dynamic_4D_jit32_Bmax, GridSampleLayerTestCPU,
//                ::testing::Combine(
//                    ::testing::ValuesIn(get4DShapesJitDyn(3)),
//                    ::testing::ValuesIn(get4DAxisBatchJitDyn(ElementType::f32, 3)),
//                    ::testing::Values(ElementType::f32),
//                    ::testing::ValuesIn(isAxisConst),
//                    ::testing::ValuesIn(getCPUInfo()),
//                    ::testing::ValuesIn(additionalConfig)),
//                GridSampleLayerTestCPU::getTestCaseName);
//
//INSTANTIATE_TEST_SUITE_P(smoke_dynamic_4D_jit16_Bmax, GridSampleLayerTestCPU,
//                ::testing::Combine(
//                    ::testing::ValuesIn(get4DShapesJitDyn(3)),
//                    ::testing::ValuesIn(get4DAxisBatchJitDyn(ElementType::bf16, 3)),
//                    ::testing::Values(ElementType::bf16),
//                    ::testing::ValuesIn(isAxisConst),
//                    ::testing::ValuesIn(getCPUInfo()),
//                    ::testing::Values(additionalConfig[0])),
//                GridSampleLayerTestCPU::getTestCaseName);
//
//INSTANTIATE_TEST_SUITE_P(smoke_dynamic_4D_jit8_Bmax, GridSampleLayerTestCPU,
//                ::testing::Combine(
//                    ::testing::ValuesIn(get4DShapesJitDyn(3)),
//                    ::testing::ValuesIn(get4DAxisBatchJitDyn(ElementType::i8, 3)),
//                    ::testing::Values(ElementType::i8),
//                    ::testing::ValuesIn(isAxisConst),
//                    ::testing::ValuesIn(getCPUInfo()),
//                    ::testing::Values(additionalConfig[0])),
//                GridSampleLayerTestCPU::getTestCaseName);
//
//
/////// 4D REFERENCE /////
//std::vector<std::vector<InputShape>> get4DShapesRefStat(bool maxBatchDims) {
//    std::vector<std::vector<InputShape>> result = {};
//    if (InferenceEngine::with_cpu_x86_avx2()) {
//        if (!maxBatchDims) {
//            result = {
//                { { {}, { {10, 2, 9, 9} } },   // Static shapes
//                  { {}, { {10, 2, 8} } }
//                },
//                { { {}, { {11, 2, 9, 2} } },   // Static shapes
//                  { {}, { {11, 2, 7} } }
//                },
//                { { {}, { {12, 2, 9, 3} } },   // Static shapes
//                  { {}, { {12, 2, 6} } }
//                },
//                { { {}, { {13, 2, 9, 4} } },   // Static shapes
//                  { {}, { {13, 2, 5} } }
//                },
//                { { {}, { {14, 2, 9, 5} } },   // Static shapes
//                  { {}, { {14, 2, 4} } }
//                },
//                { { {}, { {15, 2, 9, 6} } },   // Static shapes
//                  { {}, { {15, 2, 3} } }
//                },
//                { { {}, { {16, 2, 9, 7} } },   // Static shapes
//                  { {}, { {16, 2, 2} } }
//                },
//                { { {}, { {17, 2, 9, 8} } },   // Static shapes
//                  { {}, { {17, 2, 1} } }
//                }
//            };
//        } else {
//            result = {
//                { { {}, { {10, 8, 2, 39} } },   // Static shapes
//                  { {}, { {10, 8} } }
//                },
//                { { {}, { {11, 7, 2, 42} } },   // Static shapes
//                  { {}, { {11, 7} } }
//                },
//                { { {}, { {12, 6, 2, 43} } },   // Static shapes
//                  { {}, { {12, 6} } }
//                },
//                { { {}, { {13, 5, 2, 44} } },   // Static shapes
//                  { {}, { {13, 5} } }
//                },
//                { { {}, { {14, 4, 2, 45} } },   // Static shapes
//                  { {}, { {14, 4} } }
//                },
//                { { {}, { {15, 3, 2, 46} } },   // Static shapes
//                  { {}, { {15, 3} } }
//                },
//                { { {}, { {16, 2, 2, 47} } },   // Static shapes
//                  { {}, { {16, 2} } }
//                },
//                { { {}, { {17, 1, 2, 38} } },   // Static shapes
//                  { {}, { {17, 1} } }
//                }
//            };
//        }
//    }
//    if (InferenceEngine::with_cpu_x86_avx512f()) {
//        std::vector<std::vector<InputShape>> tmp;
//        if (!maxBatchDims) {
//            tmp = {
//                { { {}, { {25, 4, 4, 17} } },    // Static shapes
//                  { {}, { {25, 4, 16} } }
//                },
//                { { {}, { {24, 4, 4, 18} } },   // Static shapes
//                  { {}, { {24, 4, 15} } },
//                },
//                { { {}, { {23, 4, 4, 19} } },   // Static shapes
//                  { {}, { {23, 4, 14} } }
//                },
//                { { {}, { {22, 4, 4, 20} } },   // Static shapes
//                  { {}, { {22, 4, 13} } },
//                },
//                { { {}, { {21, 4, 4, 21} } },   // Static shapes
//                  { {}, { {21, 4, 12} } },
//                },
//                { { {}, { {20, 4, 4, 22} } },   // Static shapes
//                  { {}, { {20, 4, 11} } },
//                },
//                { { {}, { {19, 4, 4, 23} } },   // Static shapes
//                  { {}, { {19, 4, 10} } },
//                },
//                { { {}, { {18, 4, 4, 24} } },   // Static shapes
//                  { {}, { {18, 4, 9} } },
//                }
//            };
//        } else {
//            tmp = {
//                { { {}, { {25, 16, 4, 65} } },    // Static shapes
//                  { {}, { {25, 16} } }
//                },
//                { { {}, { {24, 15, 4, 66} } },   // Static shapes
//                  { {}, { {24, 15} } },
//                },
//                { { {}, { {23, 14, 4, 67} } },   // Static shapes
//                  { {}, { {23, 14} } }
//                },
//                { { {}, { {22, 13, 4, 68} } },   // Static shapes
//                  { {}, { {22, 13} } },
//                },
//                { { {}, { {21, 12, 4, 69} } },   // Static shapes
//                  { {}, { {21, 12} } },
//                },
//                { { {}, { {20, 11, 4, 70} } },   // Static shapes
//                  { {}, { {20, 11} } },
//                },
//                { { {}, { {19, 10, 4, 71} } },   // Static shapes
//                  { {}, { {19, 10} } },
//                },
//                { { {}, { {18, 9, 4, 72} } },   // Static shapes
//                  { {}, { {18, 9} } },
//                }
//            };
//        }
//        result.insert(result.end(), tmp.begin(), tmp.end());
//    }
//
//    return result;
//}
//
//std::vector<std::tuple<int, int>> get4DAxisBatchRefStat(ov::element::Type type, bool maxBatchDims) {
//    std::vector<std::tuple<int, int>> result = {};
//    if (InferenceEngine::with_cpu_x86_avx512f()) {
//        if (type.size() == 4) {
//            if (!maxBatchDims)
//                return std::vector<std::tuple<int, int>>{{1, 0}, {1, 1}, {0, 0}};
//            else
//                return std::vector<std::tuple<int, int>>{{2, 2}};
//        } else if (type.size() == 2 || type.size() == 1) {
//            if (!maxBatchDims)
//                return std::vector<std::tuple<int, int>>{{0, 0}};
//            else
//                return std::vector<std::tuple<int, int>>{{2, 2}};
//        }
//    } else if (InferenceEngine::with_cpu_x86_avx2()) {
//        if (type.size() == 4) {
//            if (!maxBatchDims)
//                return std::vector<std::tuple<int, int>>{{1, 0}, {1, 1}, {0, 0}};
//            else
//                return std::vector<std::tuple<int, int>>{{2, 2}};
//        } else if (type.size() == 2 || type.size() == 1) {
//            if (!maxBatchDims)
//                return std::vector<std::tuple<int, int>>{{2, 0}, {2, 1}, {2, 2}, {1, 0}, {1, 1}, {0, 0}};
//            else
//                return std::vector<std::tuple<int, int>>{{2, 2}};
//        }
//    }
//    return {};
//}
//
//INSTANTIATE_TEST_SUITE_P(smoke_static_4D_ref32, GridSampleLayerTestCPU,
//                ::testing::Combine(
//                    ::testing::ValuesIn(get4DShapesRefStat(false)),
//                    ::testing::ValuesIn(get4DAxisBatchRefStat(ElementType::f32, false)),
//                    ::testing::Values(ElementType::f32),
//                    ::testing::Values(true),
//                    ::testing::Values(cpuParamsRef),
//                    ::testing::ValuesIn(additionalConfig)),
//                GridSampleLayerTestCPU::getTestCaseName);
//
//INSTANTIATE_TEST_SUITE_P(smoke_static_4D_ref16, GridSampleLayerTestCPU,
//                ::testing::Combine(
//                    ::testing::ValuesIn(get4DShapesRefStat(false)),
//                    ::testing::ValuesIn(get4DAxisBatchRefStat(ElementType::bf16, false)),
//                    ::testing::Values(ElementType::bf16),
//                    ::testing::Values(true),
//                    ::testing::Values(cpuParamsRef),
//                    ::testing::Values(additionalConfig[0])),
//                GridSampleLayerTestCPU::getTestCaseName);
//
//INSTANTIATE_TEST_SUITE_P(smoke_static_4D_ref8, GridSampleLayerTestCPU,
//                ::testing::Combine(
//                    ::testing::ValuesIn(get4DShapesRefStat(false)),
//                    ::testing::ValuesIn(get4DAxisBatchRefStat(ElementType::i8, false)),
//                    ::testing::Values(ElementType::i8),
//                    ::testing::Values(true),
//                    ::testing::Values(cpuParamsRef),
//                    ::testing::Values(additionalConfig[0])),
//                GridSampleLayerTestCPU::getTestCaseName);
//
//// batchDims == indicesRank
//INSTANTIATE_TEST_SUITE_P(smoke_static_4D_ref32_Bmax, GridSampleLayerTestCPU,
//                ::testing::Combine(
//                    ::testing::ValuesIn(get4DShapesRefStat(true)),
//                    ::testing::ValuesIn(get4DAxisBatchRefStat(ElementType::f32, true)),
//                    ::testing::Values(ElementType::f32),
//                    ::testing::Values(true),
//                    ::testing::Values(cpuParamsRef),
//                    ::testing::ValuesIn(additionalConfig)),
//                GridSampleLayerTestCPU::getTestCaseName);
//
//INSTANTIATE_TEST_SUITE_P(smoke_static_4D_ref16_Bmax, GridSampleLayerTestCPU,
//                ::testing::Combine(
//                    ::testing::ValuesIn(get4DShapesRefStat(true)),
//                    ::testing::ValuesIn(get4DAxisBatchRefStat(ElementType::bf16, true)),
//                    ::testing::Values(ElementType::bf16),
//                    ::testing::Values(true),
//                    ::testing::Values(cpuParamsRef),
//                    ::testing::Values(additionalConfig[0])),
//                GridSampleLayerTestCPU::getTestCaseName);
//
//INSTANTIATE_TEST_SUITE_P(smoke_static_4D_ref8_Bmax, GridSampleLayerTestCPU,
//                ::testing::Combine(
//                    ::testing::ValuesIn(get4DShapesRefStat(true)),
//                    ::testing::ValuesIn(get4DAxisBatchRefStat(ElementType::i8, true)),
//                    ::testing::Values(ElementType::i8),
//                    ::testing::Values(true),
//                    ::testing::Values(cpuParamsRef),
//                    ::testing::Values(additionalConfig[0])),
//                GridSampleLayerTestCPU::getTestCaseName);
} // namespace
} // namespace CPULayerTestsDefinitions
