// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

using inputShapesPair = std::pair<std::vector<ov::PartialShape>, std::vector<std::vector<ov::Shape>>>;

using TileLayerTestParamsSet = typename std::tuple<
        inputShapesPair,                       // Input shapes
        std::vector<int64_t>,                  // Repeats
        InferenceEngine::Precision,            // Network precision
        std::vector<bool>,                     // Const inputs
        std::string>;                          // Device name

typedef std::tuple<
        TileLayerTestParamsSet,
        CPUSpecificParams> TileLayerCPUTestParamsSet;

class TileLayerCPUTest : public testing::WithParamInterface<TileLayerCPUTestParamsSet>,
                         virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<TileLayerCPUTestParamsSet> obj) {
        TileLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;

        inputShapesPair inputShapes;
        std::vector<int64_t> repeats;
        InferenceEngine::Precision netPrecision;
        std::vector<bool> isConstInput;
        std::string deviceName;
        std::tie(inputShapes, repeats, netPrecision, isConstInput, deviceName) = basicParamsSet;

        std::ostringstream result;
        result << "DynShapes=" << CommonTestUtils::partialShape2str(inputShapes.first) << "_";
        result << "StatShapes=" << CommonTestUtils::vec2str(inputShapes.second) << "_";
        result << "Repeats=" << CommonTestUtils::vec2str(repeats)  << "_";
        result << "netPrec=" << netPrecision << "_";
        result << "constIn=" << CommonTestUtils::vec2str(isConstInput)  << "_";
        result << "trgDev=" << deviceName;

        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

protected:
    void SetUp() override {
        TileLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

//        TileSpecificParams tileParams;
        inputShapesPair inputShapes;
        std::vector<int64_t> repeats;
        InferenceEngine::Precision netPrecision;
        std::vector<bool> isConstInput;
        std::tie(inputShapes, repeats, netPrecision, isConstInput, targetDevice) = basicParamsSet;

        selectedType += std::string("_") + netPrecision.name();

        targetStaticShapes.reserve(inputShapes.second.size());
        for (const auto& staticShape : inputShapes.second) {
            targetStaticShapes.push_back({staticShape});
        }
        inputDynamicShapes = { inputShapes.first };

        ov::Shape inputDataShape = targetStaticShapes.front().front();

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, { inputDataShape });
        auto repeatsOp = std::make_shared<ov::op::v0::Constant>(ov::element::i64, std::vector<size_t>{repeats.size()}, repeats);
        auto tile = std::make_shared<ov::op::v0::Tile>(params[0], repeatsOp);
        tile->get_rt_info() = getCPUInfo();
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(tile)};
        function = std::make_shared<ov::Function>(results, params, "CPUTile");
    }
};

TEST_P(TileLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "Tile");
}

namespace {

/* CPU PARAMS */
const auto cpuParams_nchw = CPUSpecificParams{{nchw}, {nchw}, {}, "ref"};
const auto cpuParams_ncdhw = CPUSpecificParams{{ncdhw}, {ncdhw}, {}, "ref"};

const auto cpuParams_nChw16c = CPUSpecificParams{{nChw16c}, {nChw16c}, {}, "ref"};
const auto cpuParams_nCdhw16c = CPUSpecificParams{{nCdhw16c}, {nCdhw16c}, {}, "ref"};

const auto cpuParams_nChw8c = CPUSpecificParams{{nChw8c}, {nChw8c}, {}, "ref"};
const auto cpuParams_nCdhw8c = CPUSpecificParams{{nCdhw8c}, {nCdhw8c}, {}, "ref"};

const auto cpuParams_nhwc = CPUSpecificParams{{nhwc}, {nhwc}, {}, "ref"};
const auto cpuParams_ndhwc = CPUSpecificParams{{ndhwc}, {ndhwc}, {}, "ref"};
/* ========== */

/* PARAMS */
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::BF16,
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::I8
};

const std::vector<inputShapesPair>
    staticInputShapes4D = {
        {{}, {{{2, 16, 3, 4}}}},
        {{}, {{{1, 16, 1, 1}}}}
};
const std::vector<inputShapesPair>
    dynamicInputShapes4D = {
        {{{2, ov::Dimension(1, 16), 3, 4}}, {{{2, 16, 3, 4}}}},
        {{{1, ov::Dimension(1, 16), 1, 1}}, {{{1, 16, 1, 1}}}}
};

const std::vector<inputShapesPair>
    staticInputShapes5D = {
        {{}, {{{2, 16, 2, 3, 4}}}}
};
const std::vector<inputShapesPair>
    dynamicInputShapes5D = {
        {{{2, ov::Dimension(1, 16), 2, 3, 4}}, {{{2, 16, 2, 3, 4}}}}
};

const std::vector<std::vector<int64_t>> repeats4D = {
        {2, 3},
        {1, 2, 3},
        {1, 1, 1, 1},
        {1, 1, 2, 3},
        {1, 2, 1, 3},
        {2, 1, 1, 1},
        {2, 3, 1, 1},
};
const std::vector<std::vector<int64_t>> repeats5D = {
        {1, 2, 3},
        {1, 1, 2, 3},
        {1, 1, 1, 2, 3},
        {1, 2, 1, 1, 3},
        {2, 1, 1, 1, 1},
        {2, 3, 1, 1, 1},
};

const std::vector<CPUSpecificParams> CPUParams4D = {
        cpuParams_nchw,
        cpuParams_nChw16c,
        cpuParams_nChw8c,
        cpuParams_nhwc,
};

const std::vector<CPUSpecificParams> CPUParams5D = {
        cpuParams_ncdhw,
        cpuParams_nCdhw16c,
        cpuParams_nCdhw8c,
        cpuParams_ndhwc,
};
/* ============= */

/* INSTANCES */
INSTANTIATE_TEST_CASE_P(smoke_staticShape4D, TileLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        ::testing::ValuesIn(staticInputShapes4D),
                                        ::testing::ValuesIn(repeats4D),
                                        ::testing::ValuesIn(netPrecisions),
                                        ::testing::ValuesIn(std::vector<std::vector<bool>>{{true}, {false}}),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(CPUParams4D)),
                        TileLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_staticShape4D, TileLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        ::testing::ValuesIn(dynsmicInputShapes4D),
                                        ::testing::ValuesIn(repeats4D),
                                        ::testing::ValuesIn(netPrecisions),
                                        ::testing::ValuesIn(std::vector<std::vector<bool>>{{true}, {false}}),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(CPUParams4D)),
                        TileLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_staticShape5D, TileLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        ::testing::ValuesIn(staticInputShapes5D),
                                        ::testing::ValuesIn(repeats5D),
                                        ::testing::ValuesIn(netPrecisions),
                                        ::testing::ValuesIn(std::vector<std::vector<bool>>{{true}, {false}}),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(CPUParams5D)),
                        TileLayerCPUTest::getTestCaseName);
/* ========= */

} // namespace

} // namespace CPULayerTestsDefinitions
