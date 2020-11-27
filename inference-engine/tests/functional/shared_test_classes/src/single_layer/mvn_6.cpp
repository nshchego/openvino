// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/mvn_6.hpp"

namespace LayerTestsDefinitions {

std::string Mvn6LayerTest::getTestCaseName(testing::TestParamInfo<mvn6Params> obj) {
    InferenceEngine::SizeVector inputShapes;
    InferenceEngine::Precision dataPrecision;
    std::vector<int> axes;
    bool normalizeVariance;
    float eps;
    std::string epsMode;
    std::string targetDevice;
    std::tie(inputShapes, dataPrecision, axes, normalizeVariance, eps, epsMode, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "DataPrc=" << dataPrecision.name() << "_";
    result << "Ax=" << CommonTestUtils::vec2str(axes) << "_";
    result << "NormVariance=" << (normalizeVariance ? "TRUE" : "FALSE") << "_";
    result << "Eps=" << eps << "_";
    result << "EM=" << epsMode << "_";
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

void Mvn6LayerTest::SetUp() {
    InferenceEngine::SizeVector inputShapes;
    InferenceEngine::Precision dataPrecision;
    std::vector<int> axes;
    bool normalizeVariance;
    float eps;
    std::string epsMode;
    std::tie(inputShapes, dataPrecision, axes, normalizeVariance, eps, epsMode, targetDevice) = this->GetParam();

    auto dataType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(dataPrecision);
    auto param = ngraph::builder::makeParams(dataType, {inputShapes});
    auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(param));
    auto mvn = std::dynamic_pointer_cast<ngraph::op::v6::MVN>(ngraph::builder::makeMVN6(paramOuts[0], axes, normalizeVariance, eps, epsMode));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(mvn)};
    function = std::make_shared<ngraph::Function>(results, param, "MVN6");
}
}  // namespace LayerTestsDefinitions
