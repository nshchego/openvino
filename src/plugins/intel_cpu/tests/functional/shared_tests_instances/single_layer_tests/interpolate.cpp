// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/interpolate.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::FP32,
};

const std::vector<std::vector<size_t>> inShapes = {
        {1, 4, 6, 6},
};

const  std::vector<ov::op::v4::Interpolate::InterpolateMode> modesWithoutNearest = {
        ov::op::v4::Interpolate::InterpolateMode::LINEAR,
        ov::op::v4::Interpolate::InterpolateMode::LINEAR_ONNX,
        ov::op::v4::Interpolate::InterpolateMode::CUBIC,
};

const  std::vector<ov::op::v4::Interpolate::InterpolateMode> nearestMode = {
        ov::op::v4::Interpolate::InterpolateMode::NEAREST,
};

const std::vector<ov::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModes = {
        ov::op::v4::Interpolate::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN,
        ov::op::v4::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL,
        ov::op::v4::Interpolate::CoordinateTransformMode::HALF_PIXEL,
        ov::op::v4::Interpolate::CoordinateTransformMode::ASYMMETRIC,
        ov::op::v4::Interpolate::CoordinateTransformMode::ALIGN_CORNERS,
};

const std::vector<ov::op::v4::Interpolate::ShapeCalcMode> shapeCalculationMode = {
        ov::op::v4::Interpolate::ShapeCalcMode::SIZES,
        ov::op::v4::Interpolate::ShapeCalcMode::SCALES,
};

const std::vector<ov::op::v4::Interpolate::NearestMode> nearestModes = {
        ov::op::v4::Interpolate::NearestMode::SIMPLE,
        ov::op::v4::Interpolate::NearestMode::ROUND_PREFER_FLOOR,
        ov::op::v4::Interpolate::NearestMode::FLOOR,
        ov::op::v4::Interpolate::NearestMode::CEIL,
        ov::op::v4::Interpolate::NearestMode::ROUND_PREFER_CEIL,
};

const std::vector<ov::op::v4::Interpolate::NearestMode> defaultNearestMode = {
        ov::op::v4::Interpolate::NearestMode::ROUND_PREFER_FLOOR,
};

const std::vector<std::vector<size_t>> pads = {
        {0, 0, 1, 1},
        {0, 0, 0, 0},
};

const std::vector<bool> antialias = {
// Not enabled in Inference Engine
//        true,
        false,
};

const std::vector<double> cubeCoefs = {
        -0.75f,
};

const std::vector<std::vector<int64_t>> defaultAxes = {
    {0, 1, 2, 3}
};

const std::vector<std::vector<size_t>> targetShapes = {
    {1, 4, 8, 8},
};

const std::vector<std::vector<float>> defaultScales = {
    {1.f, 1.f, 1.333333f, 1.333333f}
};

std::map<std::string, std::string> additional_config = {};

const auto interpolateCasesWithoutNearest = ::testing::Combine(
        ::testing::ValuesIn(modesWithoutNearest),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModes),
        ::testing::ValuesIn(defaultNearestMode),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs),
        ::testing::ValuesIn(defaultAxes),
        ::testing::ValuesIn(defaultScales));

const auto interpolateCases = ::testing::Combine(
        ::testing::ValuesIn(nearestMode),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModes),
        ::testing::ValuesIn(nearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs),
        ::testing::ValuesIn(defaultAxes),
        ::testing::ValuesIn(defaultScales));

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Basic, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesWithoutNearest,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(targetShapes),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(additional_config)),
    InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Nearest, InterpolateLayerTest, ::testing::Combine(
        interpolateCases,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(targetShapes),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(additional_config)),
    InterpolateLayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> targetShapesTailTest = {
        {1, 4, 2, 11},  // cover down sample and tails process code path
};

const std::vector<std::vector<float>> defaultScalesTailTest = {
    {1.f, 1.f, 0.333333f, 1.833333f}
};

const auto interpolateCasesWithoutNearestTail = ::testing::Combine(
        ::testing::ValuesIn(modesWithoutNearest),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModes),
        ::testing::ValuesIn(defaultNearestMode),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs),
        ::testing::ValuesIn(defaultAxes),
        ::testing::ValuesIn(defaultScalesTailTest));

const auto interpolateCasesTail = ::testing::Combine(
        ::testing::ValuesIn(nearestMode),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModes),
        ::testing::ValuesIn(nearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs),
        ::testing::ValuesIn(defaultAxes),
        ::testing::ValuesIn(defaultScalesTailTest));

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Basic_Down_Sample_Tail, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesWithoutNearestTail,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(targetShapesTailTest),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(additional_config)),
    InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Nearest_Down_Sample_Tail, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesTail,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(targetShapesTailTest),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(additional_config)),
    InterpolateLayerTest::getTestCaseName);

} // namespace
