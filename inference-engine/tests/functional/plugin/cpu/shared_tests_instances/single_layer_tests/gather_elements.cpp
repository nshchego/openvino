// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/gather_elements.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> dPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::I64,
        InferenceEngine::Precision::I16,
        InferenceEngine::Precision::U8,
        InferenceEngine::Precision::I8
};
const std::vector<InferenceEngine::Precision> iPrecisions = {
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::I64
};

INSTANTIATE_TEST_CASE_P(smoke_set1, GatherElLayerTest,
                        ::testing::Combine(
                            ::testing::Values(std::vector<size_t>({2, 2})),     // Data shape
                            ::testing::Values(std::vector<size_t>({2, 2})),     // Indices shape
                            ::testing::ValuesIn(std::vector<int>({0, 1})),      // Axis
                            ::testing::ValuesIn(dPrecisions),
                            ::testing::ValuesIn(iPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        GatherElLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_set2, GatherElLayerTest,
                        ::testing::Combine(
                            ::testing::Values(std::vector<size_t>({2, 2, 1})),  // Data shape
                            ::testing::Values(std::vector<size_t>({4, 2, 1})),  // Indices shape
                            ::testing::Values(0),                               // Axis
                            ::testing::ValuesIn(dPrecisions),
                            ::testing::ValuesIn(iPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        GatherElLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_set3, GatherElLayerTest,
                        ::testing::Combine(
                            ::testing::Values(std::vector<size_t>({2, 2, 3, 5})),   // Data shape
                            ::testing::Values(std::vector<size_t>({2, 2, 3, 7})),   // Indices shape
                            ::testing::Values(3),                                   // Axis
                            ::testing::ValuesIn(dPrecisions),
                            ::testing::ValuesIn(iPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        GatherElLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_set4, GatherElLayerTest,
                        ::testing::Combine(
                            ::testing::Values(std::vector<size_t>({3, 2, 3, 8})),   // Data shape
                            ::testing::Values(std::vector<size_t>({2, 2, 3, 8})),   // Indices shape
                            ::testing::Values(0),                                   // Axis
                            ::testing::ValuesIn(dPrecisions),
                            ::testing::ValuesIn(iPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        GatherElLayerTest::getTestCaseName);
}  // namespace
