// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/fully_connected_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine::details;

namespace {
const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32,
        // ov::element::f16
};

const std::vector<MatMulShapes> shapes = {
    {
        ov::PartialShape{ 1, 16 },
        ov::PartialShape{ 16, 8 },
        false,
        false
    },
    {
        ov::PartialShape{ 1, 16 },
        ov::PartialShape{ 8, 16 },
        false,
        true
    },
    {
        ov::PartialShape{ 16, 1 },
        ov::PartialShape{ 16, 8 },
        true,
        false
    },
};

const std::vector<ngraph::pass::low_precision::LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParams()
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, FullyConnectedTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(shapes),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::ValuesIn(trasformationParamValues)),
    FullyConnectedTransformation::getTestCaseName);
}  // namespace
