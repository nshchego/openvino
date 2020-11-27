// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
        InferenceEngine::SizeVector, // Input shapes
        InferenceEngine::Precision,  // Input precision
        std::vector<int>,            // Axes
        bool,                        // Normalize variance
        float,                       // Epsilon
        std::string,                 // Epsilon mode
        std::string                  // Device name
    > mvn6Params;

class Mvn6LayerTest : public testing::WithParamInterface<mvn6Params>, virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<mvn6Params> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions