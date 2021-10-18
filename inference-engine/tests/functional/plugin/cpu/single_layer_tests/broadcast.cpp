// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

using inputShapesPair = std::pair<std::vector<ov::PartialShape>, std::vector<std::vector<ov::Shape>>>;

using BroadcastLayerTestParamsSet = typename std::tuple<
        inputShapesPair,                       // Shapes
        std::vector<size_t>,                   // Target shapes
        std::vector<size_t>,                   // Axes mapping
        ov::op::BroadcastType,                 // Broadcast mode
        InferenceEngine::Precision,            // Network precision
        std::vector<bool>,                     // Const inputs
        std::string>;                          // Device name

using BroadcastLayerCPUTestParamsSet = typename std::tuple<
        BroadcastLayerTestParamsSet,
        CPUSpecificParams>;

class BroadcastLayerCPUTest : public testing::WithParamInterface<BroadcastLayerCPUTestParamsSet>,
                              virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<BroadcastLayerCPUTestParamsSet> obj) {
        BroadcastLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;

        inputShapesPair inputShapes;
        std::vector<size_t> targetShapes, axesMapping;
        ov::op::BroadcastType mode;
        InferenceEngine::Precision networkPrecision;
        std::vector<bool> isConstInput;
        std::string deviceName;
        std::tie(inputShapes, targetShapes, axesMapping, mode, networkPrecision, isConstInput, deviceName) = basicParamsSet;

        std::ostringstream result;
        result << "Shapes=" << CommonTestUtils::partialShape2str(inputShapes.first) << "_";
        result << "targetShape=" << CommonTestUtils::vec2str(targetShapes)  << "_";
        result << "axesMapping=" << CommonTestUtils::vec2str(axesMapping)  << "_";
        result << "mode=" << mode << "_";
        result << "netPrec=" << networkPrecision << "_";
        result << "constIn=" << CommonTestUtils::vec2str(isConstInput)  << "_";
        result << "trgDev=" << deviceName;

        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

protected:
    void SetUp() override {
        BroadcastLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();

        inputShapesPair inputShapes;
        std::vector<size_t> targetShape, axesMapping;
        ov::op::BroadcastType mode;
        InferenceEngine::Precision networkPrecision;
        std::vector<bool> isConstInput;
        std::tie(inputShapes, targetShape, axesMapping, mode, networkPrecision, isConstInput, targetDevice) = basicParamsSet;
        bool isConstTargetShape = isConstInput[0], isConstAxes = isConstInput[1];
        const auto targetShapeRank = targetShape.size();
        const auto axesMappingRank = axesMapping.size();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        selectedType += std::string("_") + networkPrecision.name();

        targetStaticShapes.reserve(inputShapes.second.size());
        for (const auto& staticShape : inputShapes.second) {
            targetStaticShapes.push_back({staticShape});
        }
        inputDynamicShapes = { inputShapes.first };

        ov::Shape inputDataShape = targetStaticShapes.front().front();

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(networkPrecision);
        ov::ParameterVector functionParams = ngraph::builder::makeParams(ngPrc, { {"data", inputDataShape} });
        if (!isConstTargetShape) {
            functionParams.push_back(ngraph::builder::makeParams(ov::element::i32, { {"targetShape", {targetShapeRank}} })[0]);
            for (auto& stShapes : targetStaticShapes)
                stShapes.push_back({{targetShapeRank}});
            inputDynamicShapes.push_back({static_cast<int64_t>(targetShapeRank)});
        }
        if (!isConstAxes) {
            functionParams.push_back(ngraph::builder::makeParams(ov::element::i32, { {"axesMapping", {axesMappingRank}} })[0]);
            for (auto& stShapes : targetStaticShapes)
                stShapes.push_back({{axesMappingRank}});
            inputDynamicShapes.push_back({static_cast<int64_t>(axesMappingRank)});
        }
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ov::op::v0::Parameter>(functionParams));

        std::shared_ptr<ov::op::v3::Broadcast> broadcastOp;
        if (mode == ov::op::BroadcastType::EXPLICIT) {
            std::shared_ptr<ov::Node> targetShapeOp;
            std::shared_ptr<ov::Node> axesMappingOp;
            if (isConstTargetShape) {
                targetShapeOp = ov::op::v0::Constant::create(ov::element::i64, {targetShapeRank}, targetShape);
            } else {
                targetShapeOp = functionParams[0];
            }
            if (isConstAxes) {
                axesMappingOp = ov::op::v0::Constant::create(ov::element::i64, {axesMappingRank}, axesMapping);
            } else {
                axesMappingOp = functionParams.size() > 2 ? functionParams[2] : functionParams[1];
            }
            broadcastOp = std::make_shared<ov::op::v3::Broadcast>(paramOuts[0],
                                                               targetShapeOp,
                                                               axesMappingOp,
                                                               mode);
        } else if (mode == ov::op::BroadcastType::NUMPY) {
            if (isConstTargetShape) {
                auto targetShapeConst = ov::op::v0::Constant::create(ov::element::i64, {targetShapeRank}, targetShape);
                broadcastOp = std::make_shared<ov::op::v3::Broadcast>(paramOuts[0],
                                                                      targetShapeConst,
                                                                      mode);
            } else {
                broadcastOp = std::make_shared<ov::op::v3::Broadcast>(paramOuts[0],
                                                                      paramOuts[1],
                                                                      mode);
            }
        }

        broadcastOp->get_rt_info() = getCPUInfo();
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(broadcastOp)};
        function = std::make_shared<ov::Function>(results, functionParams, "Broadcast");
    }
};

TEST_P(BroadcastLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "Broadcast");
}

namespace {

/* CPU PARAMS */
const auto cpuParams_nChw16c = CPUSpecificParams {{nChw16c}, {nChw16c}, {}, "ref"};
const auto cpuParams_nCdhw16c = CPUSpecificParams {{nCdhw16c}, {nCdhw16c}, {}, "ref"};

const auto cpuParams_nChw8c = CPUSpecificParams {{nChw8c}, {nChw8c}, {}, "ref"};
const auto cpuParams_nCdhw8c = CPUSpecificParams {{nCdhw8c}, {nCdhw8c}, {}, "ref"};

const auto cpuParams_nhwc = CPUSpecificParams {{nhwc}, {nhwc}, {}, "ref"};
const auto cpuParams_ndhwc = CPUSpecificParams {{ndhwc}, {ndhwc}, {}, "ref"};
/* ========== */

/* COMMON PARAMS */
const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::BF16,
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::I8
};
/* ============= */

/* INSTANCES */
// 4D
const std::vector<CPUSpecificParams> CPUParams4D = {
        cpuParams_nChw16c,
        cpuParams_nChw8c,
        cpuParams_nhwc
};

const std::vector<inputShapesPair>
    staticInputShapes4D = {
        {{}, {{{1, 16, 1, 1}}}}
};
const std::vector<inputShapesPair>
    dynamicInputShapes4D = {
        {{{1, ov::Dimension(1, 16), 1, 1}}, {{{1, 16, 1, 1}}}}
};

const auto staticNumpyBroadcastParams4D = ::testing::Combine(
        ::testing::ValuesIn(staticInputShapes4D),
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 16, 3, 3}, {1, 16, 1, 3}}),
        ::testing::Values(std::vector<size_t>{}),
        ::testing::ValuesIn({ov::op::BroadcastType::NUMPY}),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(std::vector<bool>{true, true}),
        ::testing::Values(CommonTestUtils::DEVICE_CPU));

const auto staticExplicitBroadcastParams4D = ::testing::Combine(
        ::testing::ValuesIn(staticInputShapes4D),
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 16, 3, 3}, {1, 16, 1, 3}}),
        ::testing::Values(std::vector<size_t>{0, 1, 2, 3}),
        ::testing::ValuesIn({ov::op::BroadcastType::EXPLICIT}),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(std::vector<bool>{true, true}),
        ::testing::Values(CommonTestUtils::DEVICE_CPU));

const auto dynamicNumpyBroadcastParams4D = ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes4D),
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 16, 3, 3}, {1, 16, 1, 3}}),
        ::testing::Values(std::vector<size_t>{}),
        ::testing::ValuesIn({ov::op::BroadcastType::NUMPY}),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(std::vector<std::vector<bool>>{{true, true}, {false, true}}),
        ::testing::Values(CommonTestUtils::DEVICE_CPU));


INSTANTIATE_TEST_CASE_P(smoke_StaticShape4D,
                    BroadcastLayerCPUTest,
                    ::testing::Combine(staticNumpyBroadcastParams4D, ::testing::ValuesIn(CPUParams4D)),
                    BroadcastLayerCPUTest::getTestCaseName);
//INSTANTIATE_TEST_CASE_P(smoke_StaticShape4DE,
//                    BroadcastLayerCPUTest,
//                    ::testing::Combine(staticExplicitBroadcastParams4D, ::testing::ValuesIn(CPUParams4D)),
//                    BroadcastLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke_DynamicShape4D,
                    BroadcastLayerCPUTest,
                    ::testing::Combine(dynamicNumpyBroadcastParams4D, ::testing::ValuesIn(std::vector<CPUSpecificParams>{{{}, {}, {}, "ref"}})),
                    BroadcastLayerCPUTest::getTestCaseName);

// 5D
const std::vector<inputShapesPair>
    staticInputShapes5D = {
        {{}, {{{1, 16, 1, 1, 1}}}}
};
const std::vector<inputShapesPair>
    dynamicInputShapes5D = {
        {{{1, ov::Dimension(1, 16), 1, 1, 1}},
        {{{1, 16, 1, 1, 1}}}}
};

const std::vector<CPUSpecificParams> CPUParams5D = {
        cpuParams_nCdhw16c,
        cpuParams_nCdhw8c,
        cpuParams_ndhwc,
};

const auto staticNumpyBroadcastParams5D = ::testing::Combine(
        ::testing::ValuesIn(staticInputShapes5D),
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 16, 1, 1, 3}, {1, 16, 3, 1, 3}}),
        ::testing::Values(std::vector<size_t>{}),
        ::testing::ValuesIn({ov::op::BroadcastType::NUMPY}),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(std::vector<bool>{true, true}),
        ::testing::Values(CommonTestUtils::DEVICE_CPU));

const auto dynamicNumpyBroadcastParams5D = ::testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes5D),
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 16, 1, 1, 3}, {1, 16, 3, 1, 3}}),
        ::testing::Values(std::vector<size_t>{}),
        ::testing::ValuesIn({ov::op::BroadcastType::NUMPY}),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(std::vector<std::vector<bool>>{{true, true}, {false, true}}),
        ::testing::Values(CommonTestUtils::DEVICE_CPU));

INSTANTIATE_TEST_CASE_P(smoke_StaticShape5D,
                    BroadcastLayerCPUTest,
                    ::testing::Combine(staticNumpyBroadcastParams5D, ::testing::ValuesIn(CPUParams5D)),
                    BroadcastLayerCPUTest::getTestCaseName);
//INSTANTIATE_TEST_CASE_P(smoke_StaticShape5DE,
//                    BroadcastLayerCPUTest,
//                    ::testing::Combine(staticExplicitBroadcastParams5D, ::testing::ValuesIn(CPUParams4D)),
//                    BroadcastLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke_DynamicShape5D,
                    BroadcastLayerCPUTest,
                    ::testing::Combine(dynamicNumpyBroadcastParams5D, ::testing::ValuesIn(std::vector<CPUSpecificParams>{{{}, {}, {}, "ref"}})),
                    BroadcastLayerCPUTest::getTestCaseName);
/* ========= */

} // namespace

} // namespace CPULayerTestsDefinitions
