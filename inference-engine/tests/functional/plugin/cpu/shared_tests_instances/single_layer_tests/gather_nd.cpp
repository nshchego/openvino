// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/gather_nd.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> dPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::I32,
//        InferenceEngine::Precision::I64,
        InferenceEngine::Precision::U8,
        InferenceEngine::Precision::I8
};
const std::vector<InferenceEngine::Precision> iPrecisions = {
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::I64
};

const auto gatherNDArgsSubset1 = ::testing::Combine(
        ::testing::Values(std::vector<size_t>({2, 2})),         // Data shape
        ::testing::Values(std::vector<size_t>({2, 2})),         // Indices shape
        ::testing::Values(std::vector<int>({0, 0, 1, 0})),   // Indices values
        ::testing::Values(0)                                    // Batch dims
);
INSTANTIATE_TEST_CASE_P(smoke_Set1, GatherNDLayerTest,
                        ::testing::Combine(
                            gatherNDArgsSubset1,
                            ::testing::ValuesIn(dPrecisions),
                            ::testing::ValuesIn(iPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        GatherNDLayerTest::getTestCaseName);

const auto gatherNDArgsSubset2 = ::testing::Combine(
        ::testing::ValuesIn(std::vector<std::vector<size_t>>({{2, 2}, {2, 3, 4}})), // Data shape
        ::testing::Values(std::vector<size_t>({2, 1})),      // Indices shape
        ::testing::Values(std::vector<int>({1, 0})),   // Indices values
        ::testing::ValuesIn(std::vector<int>({0, 1}))        // Batch dims
);
INSTANTIATE_TEST_CASE_P(smoke_Set2, GatherNDLayerTest,
                        ::testing::Combine(
                            gatherNDArgsSubset2,
                            ::testing::ValuesIn(dPrecisions),
                            ::testing::ValuesIn(iPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        GatherNDLayerTest::getTestCaseName);

const auto gatherNDArgsSubset3 = ::testing::Combine(
        ::testing::Values(std::vector<size_t>({2, 2})),                     // Data shape
        ::testing::Values(std::vector<size_t>({2, 1, 1})),                  // Indices shape
        ::testing::Values(std::vector<int>({1, 0})),   // Indices values
        ::testing::Values(0)                                                // Batch dims
);
INSTANTIATE_TEST_CASE_P(smoke_Set3, GatherNDLayerTest,
                        ::testing::Combine(
                            gatherNDArgsSubset3,
                            ::testing::ValuesIn(dPrecisions),
                            ::testing::ValuesIn(iPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        GatherNDLayerTest::getTestCaseName);

const auto gatherNDArgsSubset4 = ::testing::Combine(
        ::testing::Values(std::vector<size_t>({2, 3, 4})),         // Data shape
        ::testing::Values(std::vector<size_t>({2, 3, 1, 1})),      // Indices shape
        ::testing::Values(std::vector<int>({1, 0, 2, 0, 2, 2})),   // Indices values
        ::testing::Values(2)                                       // Batch dims
);
INSTANTIATE_TEST_CASE_P(smoke_Set4, GatherNDLayerTest,
                        ::testing::Combine(
                            gatherNDArgsSubset4,
                            ::testing::ValuesIn(dPrecisions),
                            ::testing::ValuesIn(iPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        GatherNDLayerTest::getTestCaseName);

}  // namespace
