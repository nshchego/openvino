// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/is_inf.hpp"
//#include "common_test_utils/test_constants.hpp"

using namespace ov::test::subgraph;

namespace {
std::vector<std::vector<ov::Shape>> inShapesStatic = {
        {{2}},
        {{2, 200}},
        {{10, 200}},
        {{1, 10, 100}},
        {{4, 4, 16}},
        {{1, 1, 1, 3}},
        {{2, 17, 5, 4}, {1, 17, 1, 1}},
        {{2, 17, 5, 1}, {1, 17, 1, 4}},
        {{1, 2, 4}},
        {{1, 4, 4}},
        {{1, 4, 4, 1}},
        {{16, 16, 16, 16, 16}},
        {{16, 16, 16, 16, 1}},
        {{16, 16, 16, 1, 16}},
        {{16, 32, 1, 1, 1}},
        {{1, 1, 1, 1, 1, 1, 3}},
        {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
};

std::vector<std::vector<ov::test::InputShape>> inShapesDynamic = {
        {{{ngraph::Dimension(1, 10), 200}, {{2, 200}, {1, 200}}},
         {{ngraph::Dimension(1, 10), 200}, {{2, 200}, {5, 200}}}},
};

std::vector<ov::test::ElementType> netPrecisions = {
        ov::element::f32
};

std::vector<bool> detectNegative = {
    true, false
};

std::vector<bool> detectPositive = {
    true, false
};

ov::test::Config additional_config = {};

const auto isInfParams = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapesStatic)),
        ::testing::ValuesIn(detectNegative),
        ::testing::ValuesIn(detectPositive),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(additional_config));

const auto isInfParamsDyn = ::testing::Combine(
        ::testing::ValuesIn(inShapesDynamic),
        ::testing::ValuesIn(detectNegative),
        ::testing::ValuesIn(detectPositive),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(smoke_static, IsInfParams, isInfParams, IsInfParams::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_dynamic, IsInfParams, isInfParamsDyn, IsInfParams::getTestCaseName);
} // namespace