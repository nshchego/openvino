// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        InputShape,                        // Output shapes
        std::tuple<double, double>,        // Min and Max values
        ElementType,                       // Shape precision
        ElementType,                       // Output precision
        int,                               // Global seed
        int,                               // Operational seed
        ov::AnyMap,                        // Additional plugin configuration
        CPUTestUtils::CPUSpecificParams
> RandomUniformLayerCPUTestParamSet;

class RandomUniformLayerCPUTest : public testing::WithParamInterface<RandomUniformLayerCPUTestParamSet>,
                                  public SubgraphBaseTest, public CPUTestUtils::CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RandomUniformLayerCPUTestParamSet>& obj) {
        const auto& out_sahpe        = std::get<0>(obj.param);
        const auto& min_max          = std::get<1>(obj.param);
        const auto& shape_prc        = std::get<2>(obj.param);
        const auto& output_prc       = std::get<3>(obj.param);
        const auto& global_seed      = std::get<4>(obj.param);
        const auto& operational_seed = std::get<5>(obj.param);
        const auto& config           = std::get<6>(obj.param);
        const auto& cpu_params       = std::get<7>(obj.param);

        std::ostringstream result;
        result << "IS={" << out_sahpe.second[0].size() << "}_";
        // result << "OS=" << utils::partialShape2str({out_sahpe.second}) << "_";
        result << "OS=(";
        for (const auto& shape : out_sahpe.second) {
            result << utils::vec2str(shape) << "_";
        }
        result << ")_Min=" << std::get<0>(min_max);
        result << "_Max=" << std::get<1>(min_max);
        result << "_ShapePrc=" << shape_prc;
        result << "_OutPrc=" << output_prc;
        result << "_GlobalSeed=" << global_seed;
        result << "_OperationalSeed=" << operational_seed;
        result << CPUTestsBase::getTestCaseName(cpu_params);

        if (!config.empty()) {
            result << "_PluginConf{";
            for (const auto& conf_item : config) {
                result << "_" << conf_item.first << "=";
                conf_item.second.print(result);
            }
            result << "}";
        }

        return result.str();
    }

protected:
    void SetUp() {
        targetDevice = utils::DEVICE_CPU;

        const auto& params           = this->GetParam();
        const auto& out_sahpe        = std::get<0>(params);
        const auto& min_max          = std::get<1>(params);
        const auto& shape_prc        = std::get<2>(params);
        const auto& output_prc       = std::get<3>(params);
        const auto& global_seed      = std::get<4>(params);
        const auto& operational_seed = std::get<5>(params);
        const auto& config           = std::get<6>(params);
        const auto& cpu_params       = std::get<7>(params);

        output_shape = out_sahpe.second[0];
        min_val = std::get<0>(min_max);
        max_val = std::get<1>(min_max);
        std::tie(inFmts, outFmts, priority, selectedType) = cpu_params;

        selectedType = makeSelectedTypeStr("ref_any", shape_prc);

        InputShape in_shape_0 = {{}, {{out_sahpe.second.size()}}},
                   in_shape_1 = {{}, {{1}}},
                   in_shape_2 = {{}, {{1}}};
        init_input_shapes({ in_shape_0, in_shape_1, in_shape_2 });

        ov::ParameterVector inputs;
        inputs.push_back(std::make_shared<ov::op::v0::Parameter>(shape_prc, inputDynamicShapes[0]));
        inputs.push_back(std::make_shared<ov::op::v0::Parameter>(output_prc, inputDynamicShapes[1]));
        inputs.push_back(std::make_shared<ov::op::v0::Parameter>(output_prc, inputDynamicShapes[2]));

        inputs[0]->set_friendly_name("shape");
        inputs[1]->set_friendly_name("minval");
        inputs[2]->set_friendly_name("maxval");

        // const auto inputOrderOp = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
        //                                                                  ov::Shape({inputOrder.size()}),
        //                                                                  inputOrder);
        const auto rnd_op = std::make_shared<ov::op::v8::RandomUniform>(inputs[0], inputs[1], inputs[2], output_prc, global_seed, operational_seed);
        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(rnd_op)};

        function = std::make_shared<ov::Model>(results, inputs, "RandomUniformLayerCPUTest");
        // functionRefs = ngraph::clone_function(*function);
    }

    template<typename TD, typename TS>
    void fill_data(TD* dst, const TS* src, size_t len) {
        for (size_t i = 0llu; i < len; i++) {
            dst[i] = static_cast<TD>(src[i]);
        }
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& func_inputs = function->inputs();

        for (size_t i = 0llu; i < func_inputs.size(); ++i) {
            const auto& func_input = func_inputs[i];
            const auto& name = func_input.get_node()->get_friendly_name();
            const auto& in_prc = func_input.get_element_type();
            auto tensor = ov::Tensor(in_prc, targetInputStaticShapes[i]);

            if (name == "shape") {
                switch (in_prc) {
                    case ElementType::i32:
                        fill_data(tensor.data<int32_t>(), output_shape.data(), output_shape.size()); break;
                    case ElementType::i64:
                        fill_data(tensor.data<int64_t>(), output_shape.data(), output_shape.size()); break;
                    default:
                        OPENVINO_THROW("RandomUniform does not support precision ", in_prc, " for the Shape input.");
                }
            } else if (name == "minval") {
                switch (in_prc) {
                    case ElementType::i32:
                        fill_data(tensor.data<int32_t>(), &min_val, 1); break;
                    case ElementType::f32:
                        fill_data(tensor.data<float>(), &min_val, 1); break;
                    case ElementType::f16:
                        fill_data(tensor.data<int32_t>(), &min_val, 1); break;
                    case ElementType::bf16:
                        fill_data(tensor.data<int32_t>(), &min_val, 1); break;
                    case ElementType::i64:
                        fill_data(tensor.data<int64_t>(), &min_val, 1); break;
                    case ElementType::f64:
                        fill_data(tensor.data<double>(), &min_val, 1); break;
                    default:
                        OPENVINO_THROW("RandomUniform does not support precision ", in_prc, " for the Minval input.");
                }
            } else if (name == "maxval") {
                switch (in_prc) {
                    case ElementType::i32:
                        fill_data(tensor.data<int32_t>(), &max_val, 1); break;
                    case ElementType::f32:
                        fill_data(tensor.data<float>(), &max_val, 1); break;
                    case ElementType::f16:
                        fill_data(tensor.data<int32_t>(), &max_val, 1); break;
                    case ElementType::bf16:
                        fill_data(tensor.data<int32_t>(), &max_val, 1); break;
                    case ElementType::i64:
                        fill_data(tensor.data<int64_t>(), &max_val, 1); break;
                    case ElementType::f64:
                        fill_data(tensor.data<double>(), &max_val, 1); break;
                    default:
                        OPENVINO_THROW("RandomUniform does not support precision ", in_prc, " for the Maxval input.");
                }
            }

            inputs.insert({func_input.get_node_shared_ptr(), tensor});
        }
    }

    ov::Shape output_shape;
    double min_val;
    double max_val;
};

TEST_P(RandomUniformLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "RandomUniform");
}

namespace {

const std::vector<ElementType> shape_prc = {
        ElementType::i32,
        ElementType::i64
};

const std::vector<ElementType> output_prc = {
        ElementType::i32,
        ElementType::f32,
        ElementType::f16,
        ElementType::bf16,
        ElementType::i64,
        ElementType::f64
};

const std::vector<InputShape> output_shapes = {
        {{}, {{2, 3, 5, 7}}},
        {{}, {{3, 4, 8, 64}}},
        {{}, {{1, 16, 27, 55}}}
};

const std::vector<std::tuple<double, double>> min_max = {
        {0, 10},
        {-10, 10},
        {-20, 0}
};

const std::vector<int> global_seed = {
        0, 1, 5
};

const std::vector<int> operational_seed = {
        0, 1, 5
};

const ov::AnyMap empty_plugin_config{};

INSTANTIATE_TEST_SUITE_P(smoke_Static, RandomUniformLayerCPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(output_shapes),
                ::testing::ValuesIn(min_max),
                ::testing::ValuesIn(shape_prc),
                ::testing::ValuesIn(output_prc),
                ::testing::ValuesIn(global_seed),
                ::testing::ValuesIn(operational_seed),
                ::testing::Values(empty_plugin_config),
                ::testing::Values(CPUSpecificParams{})),
        RandomUniformLayerCPUTest::getTestCaseName);
}

} // namespace CPULayerTestsDefinitions
