// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/mvn_6.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

std::vector<InferenceEngine::Precision> dataPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::BF16
};

const std::vector<bool> normalizeVariance = {
    true,
    false
};

const std::vector<float> epsilon = {
    0.000000001
};

const std::vector<std::string> epsMode = {
    "inside_sqrt",
    "outside_sqrt"
};

INSTANTIATE_TEST_CASE_P(smoke_Set1, Mvn6LayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 32, 17}, {1, 37, 9}}),
                            ::testing::ValuesIn(dataPrecisions),
                            ::testing::ValuesIn(std::vector<std::vector<int>>{{0, 1, 2}, {1, 2}, {0}, {1}, {2}}),
                            ::testing::ValuesIn(normalizeVariance),
                            ::testing::ValuesIn(epsilon),
                            ::testing::ValuesIn(epsMode),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        Mvn6LayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Set2, Mvn6LayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 16, 5, 8}, {2, 19, 5, 10}}),
                            ::testing::ValuesIn(dataPrecisions),
                            ::testing::ValuesIn(std::vector<std::vector<int>>{{0, 1, 2, 3}, {0, 1, 2}, {0, 3}, {1, 2}, {0}, {1}, {2}, {3}}),
                            ::testing::ValuesIn(normalizeVariance),
                            ::testing::ValuesIn(epsilon),
                            ::testing::ValuesIn(epsMode),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        Mvn6LayerTest::getTestCaseName);
