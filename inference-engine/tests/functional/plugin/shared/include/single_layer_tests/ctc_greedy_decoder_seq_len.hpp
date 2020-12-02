// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <string>
#include <tuple>
#include <vector>

#include "functional_test_utils/layer_test_utils.hpp"


namespace LayerTestsDefinitions {
typedef std::tuple<
        InferenceEngine::SizeVector,   // Input shape
        InferenceEngine::Precision,    // Probabilities precision
        InferenceEngine::Precision,    // Indices precision
        int,                           // Blank index
        bool,                          // Merge repeated
        std::string                    // Device name
    > ctcGreedyDecoderSeqLenParams;

class CTCGreedyDecoderSeqLenLayerTest
    :  public testing::WithParamInterface<ctcGreedyDecoderSeqLenParams>,
       virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ctcGreedyDecoderSeqLenParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
