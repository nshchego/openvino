// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"

using Config = std::map<std::string, std::string>;

typedef std::tuple<
        std::vector<size_t>,               // Data shapes
        std::vector<size_t>,               // Indices shape
        int,                               // axis
        InferenceEngine::Precision,        // Data precision
        InferenceEngine::Precision,        // Indices precision
        LayerTestsUtils::TargetDevice      // Device name
> GatherElParams;

namespace LayerTestsDefinitions {

class GatherElLayerTest : public testing::WithParamInterface<GatherElParams>,
                          public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GatherElParams> &obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
