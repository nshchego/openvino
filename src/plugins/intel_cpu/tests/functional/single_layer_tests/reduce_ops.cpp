// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "test_utils/fusing_test_utils.hpp"
#include <cpp_interfaces/interface/ie_internal_plugin_config.hpp>

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        std::vector<int>,               // Axis to reduce order
        CommonTestUtils::OpType,        // Scalar or vector type axis
        bool,                           // Keep dims
        ngraph::helpers::ReductionType, // Reduce operation type
        ElementType,                    // Net precision
        ElementType,                    // Input precision
        ElementType,                    // Output precision
        std::vector<InputShape>,        // Input shapes
        ov::AnyMap                      // Additional network configuration
> basicReduceParams;

typedef std::tuple<
        basicReduceParams,
        CPUSpecificParams,
        fusingSpecificParams> ReduceLayerCPUTestParamSet;

class ReduceCPULayerTest : public testing::WithParamInterface<ReduceLayerCPUTestParamSet>,
                           virtual public SubgraphBaseTest, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ReduceLayerCPUTestParamSet> obj) {
        basicReduceParams basicParams;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::tie(basicParams, cpuParams, fusingParams) = obj.param;

        std::vector<int> axes;
        CommonTestUtils::OpType opType;
        bool keepDims;
        ngraph::helpers::ReductionType reductionType;
        ElementType netPrecision, inPrc, outPrc;
        std::vector<InputShape> inputShapes;
        ov::AnyMap config;

        std::tie(axes, opType, keepDims, reductionType, netPrecision, inPrc, outPrc, inputShapes, config) = basicParams;

        std::ostringstream result;
        result << "IS=(";
        for (const auto& shape : inputShapes) {
            result << CommonTestUtils::partialShape2str({shape.first}) << "_";
        }
        result << ")_TS=(";
        for (const auto& shape : inputShapes) {
            for (const auto& item : shape.second) {
                result << CommonTestUtils::vec2str(item) << "_";
            }
        }
        result << ")_axes=" << CommonTestUtils::vec2str(axes) << "_";
        result << "opType=" << opType << "_";
        result << "type=" << reductionType << "_";
        if (keepDims)
            result << "KeepDims=true_";
        else
            result << "KeepDims=false_";
        result << "netPRC=" << netPrecision << "_";
        result << "inPRC=" << inPrc << "_";
        result << "outPRC=" << outPrc;

        if (!config.empty()) {
            result << "_PluginConf";
            for (const auto& configItem : config) {
                result << "_" << configItem.first << "=";
                configItem.second.print(result);
            }
        }

        result << CPUTestsBase::getTestCaseName(cpuParams);
        result << CpuTestWithFusing::getTestCaseName(fusingParams);

        return result.str();
    }
protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;

        basicReduceParams basicParams;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::tie(basicParams, cpuParams, fusingParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        std::tie(postOpMgrPtr, fusedOps) = fusingParams;

        std::vector<int> axes;
        CommonTestUtils::OpType opType;
        bool keepDims;
        ElementType inPrc, outPrc;
        std::vector<InputShape> inputShapes;

        std::tie(axes, opType, keepDims, reductionType, netPrecision, inPrc, outPrc, inputShapes, configuration) = basicParams;
        inPrc = outPrc = netPrecision;

        init_input_shapes(inputShapes);

        auto params = ngraph::builder::makeDynamicParams(netPrecision, inputDynamicShapes);
        auto paramOuts = ngraph::helpers::convert2OutputVector(
                ngraph::helpers::castOps2Nodes<ov::op::v0::Parameter>(params));

        std::vector<size_t> shapeAxes;
        switch (opType) {
            case CommonTestUtils::OpType::SCALAR:
                if (axes.size() > 1)
                    FAIL() << "In reduce op if op type is scalar, 'axis' input's must contain 1 element";
                break;
            case CommonTestUtils::OpType::VECTOR:
                shapeAxes.push_back(axes.size());
                break;
            default:
                FAIL() << "Reduce op doesn't support operation type: " << opType;
        }
        auto reductionAxesNode = std::dynamic_pointer_cast<ov::Node>(
                std::make_shared<ov::op::v0::Constant>(ElementType::i64, ov::Shape(shapeAxes), axes));

        const auto reduce = ngraph::builder::makeReduce(paramOuts[0], reductionAxesNode, keepDims, reductionType);

        if (inPrc == ElementType::i64 || inPrc == ElementType::u64) {
            auto i64It = configuration.find(PluginConfigInternalParams::KEY_CPU_NATIVE_I64);
            if (i64It == configuration.end() || i64It->second == PluginConfigParams::NO) {
                selectedType = makeSelectedTypeStr(getPrimitiveType(), ElementType::i32);
            } else {
                selectedType = makeSelectedTypeStr(getPrimitiveType(), ElementType::i64);
            }
        } else if (inPrc == ElementType::boolean) {
            selectedType = makeSelectedTypeStr(getPrimitiveType(), ElementType::i8);
        } else {
            selectedType = makeSelectedTypeStr(getPrimitiveType(), inPrc);
        }

        // hybrid layouts
        if (inFmts.size() != 0 && outFmts.size() == 0) {
            size_t outShapeSize = inputDynamicShapes[0].size() - axes.size();
            switch (outShapeSize) {
                case 0:
                case 1:
                    outFmts.push_back(x);
                    break;
                case 2:
                    outFmts.push_back(nc);
                    break;
                case 3:
                    outFmts.push_back(tnc);
                    break;
                case 4:
                    outFmts.push_back(nchw);
                    break;
                default:
                    FAIL() << "Invaid outShapeSize: " << outShapeSize;
            }
        }

        function = makeNgraphFunction(netPrecision, params, reduce, "Reduce");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            if (reductionType == ngraph::helpers::ReductionType::Prod) {
                tensor = utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], 10, 1);
                if (netPrecision == ElementType::f32) {
                    auto *rawBlobDataPtr = static_cast<float *>(tensor.data());
                    for (size_t i = 0; i < tensor.get_size(); ++i) {
                        rawBlobDataPtr[i] /= 10.f;
                    }
                } else if (netPrecision == ElementType::bf16) {
                    auto *rawBlobDataPtr = static_cast<ov::bfloat16 *>(tensor.data());
                    for (size_t i = 0; i < tensor.get_size(); ++i) {
                        rawBlobDataPtr[i] /= 10.f;
                    }
                } else if (netPrecision == ElementType::i64) {
                //     auto *rawBlobDataPtr = static_cast<int64_t *>(tensor.data());
                //     for (size_t i = 0; i < tensor.get_size(); ++i) {
                //         rawBlobDataPtr[i] /= 10;
                //     }
                }
            } else {
                tensor = utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
            }

            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

private:
    ngraph::helpers::ReductionType reductionType;
    ElementType netPrecision;
};

TEST_P(ReduceCPULayerTest, CompareWithRefs) {
    run();

    CheckPluginRelatedResults(compiledModel, "Reduce");
}
namespace {
const std::vector<ElementType> inpOutPrc = {ElementType::bf16, ElementType::f32};

const std::vector<bool> keepDims = {
        true,
        false,
};

const std::vector<std::vector<int>> axes = {
        {0},
        {1},
        {2},
        {3}
};

const std::vector<std::vector<int>> axesND = {
        {0, 1},
        {0, 2},
        {0, 3},
        {1, 2},
        {1, 3},
        {2, 3},
        {0, 1, 2},
        {0, 1, 3},
        {0, 2, 3},
        {1, 2, 3},
        {0, 1, 2, 3}
};

const std::vector<std::vector<int>> axes5D = {
        {2, 4},
        {0, 2, 4},
        {1, 2, 4},
        {0, 1, 2, 3, 4},
};

const std::vector<std::vector<int>> axes6D = {
        {5},
        {4, 5},
        {3, 4, 5},
        {2, 3, 4, 5},
        {1, 2, 3, 4, 5},
        {0, 1, 2, 3, 4, 5}
};

const std::vector<std::vector<int>> axesNDFusing = {
        {0, 1},
        {0, 2},
        {0, 3},
        {1, 2},
        {1, 3},
        {2, 3},
};

const std::vector<std::vector<int>> axes5DFusing = {
        {2, 4},
        {0, 2, 4},
};

const std::vector<std::vector<int>> axesHW = {
        {2, 3}
};

std::vector<CommonTestUtils::OpType> opTypes = {
        CommonTestUtils::OpType::SCALAR,
        CommonTestUtils::OpType::VECTOR,
};

const std::vector<ngraph::helpers::ReductionType> reductionTypes = {
        ngraph::helpers::ReductionType::Mean,
        ngraph::helpers::ReductionType::Max,
        ngraph::helpers::ReductionType::Sum,
        ngraph::helpers::ReductionType::Min,
        ngraph::helpers::ReductionType::Prod,
        ngraph::helpers::ReductionType::L1,
        ngraph::helpers::ReductionType::L2,
};

const std::vector<ngraph::helpers::ReductionType> reductionTypesInt32 = {
        ngraph::helpers::ReductionType::Sum,
        ngraph::helpers::ReductionType::Min,
        ngraph::helpers::ReductionType::Max,
        ngraph::helpers::ReductionType::L1,
};

const std::vector<ngraph::helpers::ReductionType> reductionTypesFusing = {
        ngraph::helpers::ReductionType::Mean,
        ngraph::helpers::ReductionType::Max,
        ngraph::helpers::ReductionType::L2,
};

const std::vector<ngraph::helpers::ReductionType> reductionLogicalTypes = {
        ngraph::helpers::ReductionType::LogicalOr,
        ngraph::helpers::ReductionType::LogicalAnd
};

std::vector<std::vector<InputShape>> inputShapes = {
    {{{}, {{2, 19, 2, 9}}}},
    {{{{1, 5}, 19, {1, 5}, {1, 10}}, {{2, 19, 2, 2}, {2, 19, 2, 9}}}},
};

std::vector<std::vector<InputShape>> inputShapes_5D = {
    {{{}, {{2, 19, 2, 2, 9}}}},
    {{{{1, 5}, 19, {1, 5}, {1, 5}, {1, 5}}, {{2, 19, 2, 2, 2}, {2, 19, 3, 2, 2}}}},
};

std::vector<std::vector<InputShape>> inputShapes_6D = {
    {{{}, {{2, 19, 2, 2, 2, 2}}}},
    {{{{1, 5}, 19, {1, 5}, {1, 5}, {1, 5}, {1, 5}}, {{2, 19, 2, 2, 2, 2}, {2, 19, 2, 2, 3, 2}}}},
};

std::vector<std::vector<InputShape>> inputShapes_Int32 = {
    {{{}, {{2, 19, 2, 3}}}},
    {{{{1, 5}, 19, {1, 5}, {1, 10}}, {{2, 19, 2, 2}, {2, 19, 2, 3}}}},
};

std::vector<std::vector<InputShape>> inputShapes_SmallChannel = {
    {{{}, {{2, 3, 2, 9}}}},
    {{{{1, 5}, 3, {1, 5}, {1, 10}}, {{2, 3, 2, 2}, {2, 3, 2, 9}}}},
};

std::vector<std::vector<InputShape>> inputShapes_SingleBatch = {
    {{{}, {{1, 19, 2, 9}}}},
    {{{{1, 5}, 19, {1, 5}, {1, 10}}, {{1, 19, 2, 2}, {1, 19, 2, 9}}}},
};

std::vector<CPUSpecificParams> cpuParams_4D = {
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
        CPUSpecificParams({nChw16c}, {nChw16c}, {}, {}),
        CPUSpecificParams({nhwc}, {nhwc}, {}, {}),
#endif
        CPUSpecificParams({nchw}, {nchw}, {}, {}),
};

std::vector<CPUSpecificParams> cpuParams_4D_I64 = {
        CPUSpecificParams({nChw16c}, {nChw8c}, {}, {}),
        CPUSpecificParams({nchw}, {nchw}, {}, {}),
        CPUSpecificParams({nhwc}, {nhwc}, {}, {})
};

std::vector<CPUSpecificParams> cpuParams_5D = {
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
        CPUSpecificParams({nCdhw16c}, {nCdhw16c}, {}, {}),
        CPUSpecificParams({ndhwc}, {ndhwc}, {}, {}),
#endif
        CPUSpecificParams({ncdhw}, {ncdhw}, {}, {}),
};

std::vector<CPUSpecificParams> cpuParams_5D_I64 = {
        CPUSpecificParams({nCdhw8c}, {nCdhw8c}, {}, {}),
        CPUSpecificParams({ncdhw}, {ncdhw}, {}, {}),
        CPUSpecificParams({ndhwc}, {ndhwc}, {}, {})
};

std::vector<CPUSpecificParams> cpuParams_HybridLayout_4D = {
        CPUSpecificParams({nChw16c}, {}, {}, {}),
        CPUSpecificParams({nhwc}, {}, {}, {})
};

std::vector<CPUSpecificParams> cpuParams_HybridLayout_4D_I64 = {
        CPUSpecificParams({nChw8c}, {}, {}, {}),
        CPUSpecificParams({nhwc}, {}, {}, {})
};

std::vector<CPUSpecificParams> cpuParams_HybridLayout_5D = {
        CPUSpecificParams({nCdhw16c}, {}, {}, {}),
        CPUSpecificParams({ndhwc}, {}, {}, {})
};

std::vector<CPUSpecificParams> cpuParams_HybridLayout_5D_I64 = {
        CPUSpecificParams({nCdhw16c}, {}, {}, {}),
        CPUSpecificParams({ndhwc}, {}, {}, {})
};

std::vector<CPUSpecificParams> cpuParams_NHWC_4D = {
        CPUSpecificParams({nhwc}, {nhwc}, {}, {})
};

const std::vector<fusingSpecificParams> fusingParamsSet {
        /* activations */
        fusingSwish,

        /* FQ */
        fusingFakeQuantizePerChannelRelu,
        fusingFakeQuantizePerTensorRelu,
        /* another patterns */
        fusingScaleShift
};

// Exclude cases of fusingFakeQuantizePerChannelRelu, where FQ for non-1 channel fallbacks
// to decomposed ngraph reference implementation, so such fusing tests are N/A
const std::vector<fusingSpecificParams> fusingParamsSet_KeepNoDims {
        /* activations */
        fusingSwish,

        /* FQ */
        fusingFakeQuantizePerTensorRelu,
        /* another patterns */
        fusingScaleShift
};

ov::AnyMap additional_config = {};
ov::AnyMap additional_config_i64 = {{PluginConfigInternalParams::KEY_CPU_NATIVE_I64, PluginConfigParams::YES}};

/* ================================ 1.1 No fusion - Arithmetic ================================ */
const auto params_OneAxis = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes),
                testing::ValuesIn(opTypes),
                testing::ValuesIn(keepDims),
                testing::ValuesIn(reductionTypes),
                testing::ValuesIn(inpOutPrc),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes),
                testing::Values(additional_config)),
        testing::Values(emptyCPUSpec),
        testing::Values(emptyFusingSpec));

std::vector<ov::AnyMap> config_i64_ = {{{PluginConfigInternalParams::KEY_CPU_NATIVE_I64, PluginConfigParams::YES}},
        {{PluginConfigInternalParams::KEY_CPU_NATIVE_I64, PluginConfigParams::NO}}};
const auto params_OneAxis_I64 = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes),
                testing::ValuesIn(opTypes),
                testing::ValuesIn(keepDims),
                testing::ValuesIn(reductionTypes),
                testing::Values(ElementType::i64, ElementType::u64),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes),
                testing::Values(additional_config_i64)),
        testing::Values(emptyCPUSpec),
        testing::Values(emptyFusingSpec));

const auto params_MultiAxis_4D = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axesND),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn(reductionTypes),
                testing::ValuesIn(inpOutPrc),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes),
                testing::Values(additional_config)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D)),
        testing::Values(emptyFusingSpec));

 const auto params_MultiAxis_4D_I64 = testing::Combine(
         testing::Combine(
                 testing::ValuesIn(axesND),
                 testing::Values(CommonTestUtils::OpType::VECTOR),
                 testing::Values(true),
                 testing::ValuesIn(reductionTypes),
                 testing::Values(ElementType::i64),
                 testing::Values(ElementType::undefined),
                 testing::Values(ElementType::undefined),
                 testing::ValuesIn(inputShapes),
                 testing::Values(additional_config_i64)),
         testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_I64, ElementType::i64)),
         testing::Values(emptyFusingSpec));

const auto params_MultiAxis_5D = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes5D),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn(reductionTypes),
                testing::ValuesIn(inpOutPrc),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_5D),
                testing::Values(additional_config)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D)),
        testing::Values(emptyFusingSpec));

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)

 const auto params_MultiAxis_5D_I64 = testing::Combine(
         testing::Combine(
                 testing::ValuesIn(axes5D),
                 testing::Values(CommonTestUtils::OpType::VECTOR),
                 testing::Values(true),
                 testing::ValuesIn(reductionTypes),
                 testing::Values(ElementType::i64),
                 testing::Values(ElementType::undefined),
                 testing::Values(ElementType::undefined),
                 testing::ValuesIn(inputShapes_5D),
                 testing::Values(additional_config_i64)),
         testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D_I64, ElementType::i64)),
         testing::Values(emptyFusingSpec));

const auto params_MultiAxis_4D_Hybrid = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axesND),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(false),
                testing::ValuesIn(reductionTypes),
                testing::ValuesIn(inpOutPrc),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes),
                testing::Values(additional_config)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_HybridLayout_4D)),
        testing::Values(emptyFusingSpec));

 const auto params_MultiAxis_4D_Hybrid_I64 = testing::Combine(
         testing::Combine(
                 testing::ValuesIn(axesND),
                 testing::Values(CommonTestUtils::OpType::VECTOR),
                 testing::Values(false),
                 testing::ValuesIn(reductionTypes),
                 testing::Values(ElementType::i64),
                 testing::Values(ElementType::undefined),
                 testing::Values(ElementType::undefined),
                 testing::ValuesIn(inputShapes),
                 testing::Values(additional_config_i64)),
         testing::ValuesIn(filterCPUSpecificParams(cpuParams_HybridLayout_4D_I64, ElementType::i64)),
         testing::Values(emptyFusingSpec));

const auto params_MultiAxis_5D_Hybrid = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes5D),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(false),
                testing::ValuesIn(reductionTypes),
                testing::ValuesIn(inpOutPrc),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_5D),
                testing::Values(additional_config)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_HybridLayout_5D)),
        testing::Values(emptyFusingSpec));
#endif

 const auto params_MultiAxis_5D_Hybrid_I64 = testing::Combine(
         testing::Combine(
                 testing::ValuesIn(axes5D),
                 testing::Values(CommonTestUtils::OpType::VECTOR),
                 testing::Values(false),
                 testing::ValuesIn(reductionTypes),
                 testing::Values(ElementType::i64),
                 testing::Values(ElementType::undefined),
                 testing::Values(ElementType::undefined),
                 testing::ValuesIn(inputShapes_5D),
                 testing::Values(additional_config_i64)),
         testing::ValuesIn(filterCPUSpecificParams(cpuParams_HybridLayout_5D_I64, ElementType::i64)),
         testing::Values(emptyFusingSpec));

const auto params_MultiAxis_6D = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes6D),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::ValuesIn(keepDims),
                testing::ValuesIn(reductionTypes),
                testing::ValuesIn({ElementType::bf16, ElementType::f32}),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_6D),
                testing::Values(additional_config)),
        testing::Values(emptyCPUSpec),
        testing::Values(emptyFusingSpec));

const auto params_Int32 = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::ValuesIn(keepDims),
                testing::ValuesIn(reductionTypesInt32),
                testing::ValuesIn({ElementType::i32, ElementType::i64}),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_Int32),
                testing::Values(additional_config)),
        testing::Values(emptyCPUSpec),
        testing::Values(emptyFusingSpec));

const auto params_NHWC_SmallChannel = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axesHW),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn(reductionTypes),
                testing::ValuesIn(inpOutPrc),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_SmallChannel),
                testing::Values(additional_config)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_NHWC_4D)),
        testing::Values(emptyFusingSpec));

const auto params_SingleBatch = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn(reductionTypes),
                testing::ValuesIn(inpOutPrc),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_SingleBatch),
                testing::Values(additional_config)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_NHWC_4D)),
        testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_OneAxis_CPU,
        ReduceCPULayerTest,
        params_OneAxis,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_OneAxis_CPU_I64,
        ReduceCPULayerTest,
        params_OneAxis_I64,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_4D_CPU,
        ReduceCPULayerTest,
        params_MultiAxis_4D,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_4D_CPU_I64,
        ReduceCPULayerTest,
        params_MultiAxis_4D_I64,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_5D_CPU,
        ReduceCPULayerTest,
        params_MultiAxis_5D,
        ReduceCPULayerTest::getTestCaseName
);

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_5D_CPU_I64,
        ReduceCPULayerTest,
        params_MultiAxis_5D_I64,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_4D_Hybrid_CPU,
        ReduceCPULayerTest,
        params_MultiAxis_4D_Hybrid,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_4D_Hybrid_CPU_I64,
        ReduceCPULayerTest,
        params_MultiAxis_4D_Hybrid_I64,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_5D_Hybrid_CPU,
        ReduceCPULayerTest,
        params_MultiAxis_5D_Hybrid,
        ReduceCPULayerTest::getTestCaseName
);
#endif

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_5D_Hybrid_CPU_I64,
        ReduceCPULayerTest,
        params_MultiAxis_5D_Hybrid_I64,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_6D_CPU,
        ReduceCPULayerTest,
        params_MultiAxis_6D,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_Int32_CPU,
        ReduceCPULayerTest,
        params_Int32,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_NHWC_SmallChannel_CPU,
        ReduceCPULayerTest,
        params_NHWC_SmallChannel,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_SingleBatch_CPU,
        ReduceCPULayerTest,
        params_SingleBatch,
        ReduceCPULayerTest::getTestCaseName
);

/* ================================ 1.2 No fusion - Logical ================================ */
const auto params_OneAxis_Logical = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes),
                testing::ValuesIn(opTypes),
                testing::ValuesIn(keepDims),
                testing::ValuesIn(reductionLogicalTypes),
                testing::Values(ElementType::boolean),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes),
                testing::Values(additional_config)),
        testing::Values(emptyCPUSpec),
        testing::Values(emptyFusingSpec));

const auto params_MultiAxis_4D_Logical = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axesND),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn(reductionLogicalTypes),
                testing::Values(ElementType::boolean),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes),
                testing::Values(additional_config)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D)),
        testing::Values(emptyFusingSpec));

const auto params_MultiAxis_5D_Logical = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes5D),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn(reductionLogicalTypes),
                testing::Values(ElementType::boolean),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_5D),
                testing::Values(additional_config)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D)),
        testing::Values(emptyFusingSpec));

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
const auto params_MultiAxis_4D_Hybrid_Logical = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axesND),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(false),
                testing::ValuesIn(reductionLogicalTypes),
                testing::Values(ElementType::boolean),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes),
                testing::Values(additional_config)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_HybridLayout_4D)),
        testing::Values(emptyFusingSpec));

const auto params_MultiAxis_5D_Hybrid_Logical = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes5D),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(false),
                testing::ValuesIn(reductionLogicalTypes),
                testing::Values(ElementType::boolean),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_5D),
                testing::Values(additional_config)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_HybridLayout_5D)),
        testing::Values(emptyFusingSpec));
#endif

const auto params_MultiAxis_6D_Logical = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes6D),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::ValuesIn(keepDims),
                testing::ValuesIn(reductionLogicalTypes),
                testing::Values(ElementType::boolean),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_6D),
                testing::Values(additional_config)),
        testing::Values(emptyCPUSpec),
        testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_OneAxis_Logical_CPU,
        ReduceCPULayerTest,
        params_OneAxis_Logical,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_4D_Logical_CPU,
        ReduceCPULayerTest,
        params_MultiAxis_4D_Logical,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_5D_Logical_CPU,
        ReduceCPULayerTest,
        params_MultiAxis_5D_Logical,
        ReduceCPULayerTest::getTestCaseName
);

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_4D_Hybrid_Logical_CPU,
        ReduceCPULayerTest,
        params_MultiAxis_4D_Hybrid_Logical,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_5D_Hybrid_Logical_CPU,
        ReduceCPULayerTest,
        params_MultiAxis_5D_Hybrid_Logical,
        ReduceCPULayerTest::getTestCaseName
);
#endif

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_6D_Logical_CPU,
        ReduceCPULayerTest,
        params_MultiAxis_6D_Logical,
        ReduceCPULayerTest::getTestCaseName
);

/* ================================ 2.1 Fusion - KeepDims ================================ */
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)

const std::vector<ElementType> inpOutPrcFusing = {ElementType::bf16, ElementType::f32};

const auto params_OneAxis_fusing = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes),
                testing::ValuesIn(opTypes),
                testing::Values(true),
                testing::ValuesIn(reductionTypesFusing),
                testing::ValuesIn(inpOutPrcFusing),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes),
                testing::Values(additional_config)),
        testing::Values(emptyCPUSpec),
        testing::ValuesIn(fusingParamsSet));

 const auto params_OneAxis_fusing_I64 = testing::Combine(
         testing::Combine(
                 testing::ValuesIn(axes),
                 testing::ValuesIn(opTypes),
                 testing::Values(true),
                 testing::ValuesIn(reductionTypesFusing),
                 testing::Values(ElementType::i64, ElementType::u64),
                 testing::Values(ElementType::undefined),
                 testing::Values(ElementType::undefined),
                 testing::ValuesIn(inputShapes),
                 testing::Values(additional_config_i64)),
         testing::Values(emptyCPUSpec),
         testing::ValuesIn(fusingParamsSet));

const auto params_MultiAxis_4D_fusing = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axesND),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn(reductionTypesFusing),
                testing::ValuesIn(inpOutPrcFusing),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes),
                testing::Values(additional_config)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D)),
        testing::ValuesIn(fusingParamsSet));

 const auto params_MultiAxis_4D_fusing_I64 = testing::Combine(
         testing::Combine(
                 testing::ValuesIn(axesND),
                 testing::Values(CommonTestUtils::OpType::VECTOR),
                 testing::Values(true),
                 testing::ValuesIn(reductionTypesFusing),
                 testing::Values(ElementType::i64),
                 testing::Values(ElementType::undefined),
                 testing::Values(ElementType::undefined),
                 testing::ValuesIn(inputShapes),
                 testing::Values(additional_config_i64)),
         testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_I64, ElementType::i64)),
         testing::ValuesIn(fusingParamsSet));

const auto params_MultiAxis_5D_fusing = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes5D),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn(reductionTypesFusing),
                testing::ValuesIn(inpOutPrcFusing),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_5D),
                testing::Values(additional_config)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D)),
        testing::ValuesIn(fusingParamsSet));

 const auto params_MultiAxis_5D_fusing_I64 = testing::Combine(
         testing::Combine(
                 testing::ValuesIn(axes5D),
                 testing::Values(CommonTestUtils::OpType::VECTOR),
                 testing::Values(true),
                 testing::ValuesIn(reductionTypesFusing),
                 testing::Values(ElementType::i64),
                 testing::Values(ElementType::undefined),
                 testing::Values(ElementType::undefined),
                 testing::ValuesIn(inputShapes_5D),
                 testing::Values(additional_config_i64)),
         testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D_I64, ElementType::i64)),
         testing::ValuesIn(fusingParamsSet));

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_OneAxis_fusing_CPU,
        ReduceCPULayerTest,
        params_OneAxis_fusing,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_OneAxis_fusing_CPU_I64,
        ReduceCPULayerTest,
        params_OneAxis_fusing_I64,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_4D_fusing_CPU,
        ReduceCPULayerTest,
        params_MultiAxis_4D_fusing,
        ReduceCPULayerTest::getTestCaseName
);

 INSTANTIATE_TEST_SUITE_P(
         smoke_Reduce_MultiAxis_4D_fusing_CPU_I64,
         ReduceCPULayerTest,
         params_MultiAxis_4D_fusing_I64,
         ReduceCPULayerTest::getTestCaseName
 );

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_5D_fusing_CPU,
        ReduceCPULayerTest,
        params_MultiAxis_5D_fusing,
        ReduceCPULayerTest::getTestCaseName
);

 INSTANTIATE_TEST_SUITE_P(
         smoke_Reduce_MultiAxis_5D_fusing_CPU_I64,
         ReduceCPULayerTest,
         params_MultiAxis_5D_fusing_I64,
         ReduceCPULayerTest::getTestCaseName
 );

/* ================================ 2.2 Fusion - KeepNoDims ================================ */
const auto params_OneAxis_fusing_KeepNoDims = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes),
                testing::ValuesIn(opTypes),
                testing::Values(false),
                testing::ValuesIn(reductionTypesFusing),
                testing::ValuesIn(inpOutPrcFusing),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes),
                testing::Values(additional_config)),
        testing::Values(emptyCPUSpec),
        testing::ValuesIn(fusingParamsSet_KeepNoDims));

const auto params_OneAxis_fusing_KeepNoDims_I64 = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes),
                testing::ValuesIn(opTypes),
                testing::Values(false),
                testing::ValuesIn(reductionTypesFusing),
                testing::Values(ElementType::i64),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes),
                testing::Values(additional_config_i64)),
        testing::Values(emptyCPUSpec),
        testing::ValuesIn(fusingParamsSet_KeepNoDims));

const auto params_MultiAxis_4D_Hybrid_fusing_KeepNoDims = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axesNDFusing),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(false),
                testing::ValuesIn(reductionTypesFusing),
                testing::ValuesIn(inpOutPrcFusing),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes),
                testing::Values(additional_config)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_HybridLayout_4D)),
        testing::ValuesIn(fusingParamsSet_KeepNoDims));

const auto params_MultiAxis_5D_Hybrid_fusing_KeepNoDims = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes5DFusing),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(false),
                testing::ValuesIn(reductionTypesFusing),
                testing::ValuesIn(inpOutPrcFusing),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_5D),
                testing::Values(additional_config)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_HybridLayout_5D)),
        testing::ValuesIn(fusingParamsSet_KeepNoDims));

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_OneAxis_fusing_KeepNoDims_CPU,
        ReduceCPULayerTest,
        params_OneAxis_fusing_KeepNoDims,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_OneAxis_fusing_KeepNoDims_CPU_I64,
        ReduceCPULayerTest,
        params_OneAxis_fusing_KeepNoDims_I64,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_4D_Hybrid_fusing_KeepNoDims_CPU,
        ReduceCPULayerTest,
        params_MultiAxis_4D_Hybrid_fusing_KeepNoDims,
        ReduceCPULayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_5D_Hybrid_fusing_KeepNoDims_CPU,
        ReduceCPULayerTest,
        params_MultiAxis_5D_Hybrid_fusing_KeepNoDims,
        ReduceCPULayerTest::getTestCaseName
);
#endif

} // namespace
} // namespace CPULayerTestsDefinitions
