// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// #include <tuple>
// #include <string>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/builders.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "test_utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/utils/ranges.hpp"
#include "openvino/op/nms_rotated.hpp"

using namespace ov::test;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
    std::vector<InputShape>,   // Input shapes
    ElementType,               // Boxes and scores input precisions
    ElementType,               // Max output boxes input precisions
    ElementType,               // Thresholds precisions
    ElementType,               // Output type
    int64_t,                   // Max output boxes per class
    float,                     // IOU threshold
    float,                     // Score threshold
    bool,                      // Sort result descending
    bool,                      // Clockwise
    bool,                      // Is 1st input constant
    bool,                      // Is 2nd input constant
    bool,                      // Is 3rd input constant
    bool,                      // Is 4th input constant
    bool,                      // Is 5th input constant
    CPUSpecificParams,         // CPU specific params
    ov::AnyMap                 // Additional plugin configuration
> NmsRotatedParams;

class NmsRotatedLayerTestCPU : public testing::WithParamInterface<NmsRotatedParams>,
                               public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<NmsRotatedParams>& obj) {
        const auto& in_shape = std::get<0>(obj.param);

        std::ostringstream result;

        result << "IS=(";
        for (size_t i = 0lu; i < in_shape.size(); i++) {
            result << utils::partialShape2str({in_shape[i].first}) << (i < in_shape.size() - 1lu ? "_" : "");
        }
        result << ")_TS=";
        for (size_t i = 0lu; i < in_shape.front().second.size(); i++) {
            result << "{";
            for (size_t j = 0lu; j < in_shape.size(); j++) {
                result << utils::vec2str(in_shape[j].second[i]) << (j < in_shape.size() - 1lu ? "_" : "");
            }
            result << "}_";
        }
        result << "_BoxPrc="    << std::get<1>(obj.param);
        result << "_MaxPrc="    << std::get<2>(obj.param);
        result << "_ThrPrc="    << std::get<3>(obj.param);
        result << "_OutPrc="    << std::get<4>(obj.param);
        result << "_MaxBox="    << std::get<5>(obj.param);
        result << "_IouThr="    << std::get<6>(obj.param);
        result << "_ScoreThr="  << std::get<7>(obj.param);
        result << "_SortDesc="  << utils::bool2str(std::get<8>(obj.param));
        result << "_Clockwise=" << utils::bool2str(std::get<9>(obj.param));
        result << "_ConstIn={"  << utils::bool2str(std::get<10>(obj.param)) << ","
                                << utils::bool2str(std::get<11>(obj.param)) << ","
                                << utils::bool2str(std::get<12>(obj.param)) << ","
                                << utils::bool2str(std::get<13>(obj.param)) << ","
                                << utils::bool2str(std::get<14>(obj.param)) << "}";

        result << CPUTestsBase::getTestCaseName(std::get<15>(obj.param));

        const auto& config = std::get<16>(obj.param);
        if (!config.empty()) {
            result << "_PluginConf={";
            for (const auto& conf_item : config) {
                result << "_" << conf_item.first << "=";
                conf_item.second.print(result);
            }
            result << "}";
        }

        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = utils::DEVICE_CPU;

        const auto& params          = this->GetParam();
        const auto& in_shape        = std::get<0>(params);
        const auto& boxes_prc       = std::get<1>(params);
        const auto& max_boxes_prc   = std::get<2>(params);
        const auto& thresholds_prc  = std::get<3>(params);
        const auto& out_prc         = std::get<4>(params);
        m_max_out_boxes_per_class   = std::get<5>(params);
        m_iou_threshold             = std::get<6>(params);
        m_score_threshold           = std::get<7>(params);
        const auto& sort_descending = std::get<8>(params);
        const auto& clockwise       = std::get<9>(params);
        const auto& is_0_in_const   = std::get<10>(params);
        const auto& is_1_in_const   = std::get<11>(params);
        const auto& is_2_in_const   = std::get<12>(params);
        const auto& is_3_in_const   = std::get<13>(params);
        const auto& is_4_in_const   = std::get<14>(params);
        const auto& cpu_params      = std::get<15>(params);
        configuration               = std::get<16>(params);

        std::tie(inFmts, outFmts, priority, selectedType) = cpu_params;

        std::vector<InputShape> in_shapes;
        ov::ParameterVector in_params;
        std::vector<std::shared_ptr<ov::Node>> inputs;
        const auto in_shape_1d = InputShape{{1}, {{1}}};

#define CREATE_INPUT(C, P, S, N, H, L)                                                                                    \
        if (C) {                                                                                                          \
            switch (P) {                                                                                                  \
                case ElementType::f32:                                                                                    \
                    inputs.push_back(ngraph::builder::makeConstant(P, S.second[0], std::vector<float>{}, true, float(H), float(L)));    \
                    break;                                                                                                \
                case ElementType::i32:                                                                                    \
                    inputs.push_back(ngraph::builder::makeConstant(P, S.second[0], std::vector<int32_t>{}, true, int32_t(H), int32_t(L)));  \
                    break;                                                                                \
            }                                                                                             \
        } else {                                                                                          \
            in_shapes.push_back(S);                                                                       \
            in_params.push_back(std::make_shared<ov::op::v0::Parameter>(P, S.first));                     \
            in_params.back()->set_friendly_name(N);                                                       \
            inputs.push_back(in_params.back());                                                           \
        }

        CREATE_INPUT(is_0_in_const, boxes_prc,      in_shape[0], "Boxes", 30, 10)
        CREATE_INPUT(is_1_in_const, boxes_prc,      in_shape[1], "Scores", 1, 0)
        CREATE_INPUT(is_2_in_const, max_boxes_prc,  in_shape_1d, "MaxOutputBoxesPerClass", m_max_out_boxes_per_class, m_max_out_boxes_per_class)
        CREATE_INPUT(is_3_in_const, thresholds_prc, in_shape_1d, "IouThreshold", m_iou_threshold, m_iou_threshold)
        CREATE_INPUT(is_4_in_const, thresholds_prc, in_shape_1d, "ScoreThreshold", m_score_threshold, m_score_threshold)

#undef CREATE_INPUT

        init_input_shapes(in_shapes);

        const auto nms_op = std::make_shared<ov::op::v13::NMSRotated>(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
                                                                      sort_descending, out_prc, clockwise);
        ov::ResultVector results;
        for (size_t i = 0lu; i < nms_op->get_output_size(); i++) {
            results.push_back(std::make_shared<ov::op::v0::Result>(nms_op->output(i)));
        }

        function = std::make_shared<ov::Model>(results, in_params, "NMSRotated");
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

#define FILL_DATA(P, S, L)                                                          \
case P :                                                                            \
    fill_data(tensor.data<ov::element_type_traits<P>::value_type>(), S, L); break;

            if (name == "Boxes") {
                switch (in_prc) {
                    case ElementType::f32:
                        utils::fill_data_random(tensor.data<float>(), shape_size(targetInputStaticShapes[i]), 30, 20);
                        break;
                    default:
                        OPENVINO_THROW("NmsRotated does not support precision ", in_prc, " for the Scores input.");
                }
            } else if (name == "Scores") {
                switch (in_prc) {
                    case ElementType::f32:
                        utils::fill_data_random(tensor.data<float>(), shape_size(targetInputStaticShapes[i]), 1, 0, 100);
                        break;
                    default:
                        OPENVINO_THROW("NmsRotated does not support precision ", in_prc, " for the Scores input.");
                }
            } else if (name == "MaxOutputBoxesPerClass") {
                switch (in_prc) {
                    FILL_DATA(ElementType::i64, &m_max_out_boxes_per_class, 1)
                    FILL_DATA(ElementType::i32, &m_max_out_boxes_per_class, 1)
                    default:
                        OPENVINO_THROW("NmsRotated does not support precision ", in_prc, " for the MaxOutputBoxesPerClass input.");
                }
            } else if (name == "IouThreshold") {
                switch (in_prc) {
                    FILL_DATA(ElementType::f32,  &m_iou_threshold, 1)
                    FILL_DATA(ElementType::f16,  &m_iou_threshold, 1)
                    FILL_DATA(ElementType::bf16, &m_iou_threshold, 1)
                    default:
                        OPENVINO_THROW("NmsRotated does not support precision ", in_prc, " for the IouThreshold input.");
                }
            } else if (name == "ScoreThreshold") {
                switch (in_prc) {
                    FILL_DATA(ElementType::f32,  &m_score_threshold, 1)
                    FILL_DATA(ElementType::f16,  &m_score_threshold, 1)
                    FILL_DATA(ElementType::bf16, &m_score_threshold, 1)
                    default:
                        OPENVINO_THROW("NmsRotated does not support precision ", in_prc, " for the ScoreThreshold input.");
                }
            }

#undef FILL_DATA
        }
    }

private:
    int64_t m_max_out_boxes_per_class;
    float m_iou_threshold;
    float m_score_threshold;
};

TEST_P(NmsRotatedLayerTestCPU, CompareWithRefs) {
    run();
};

static const std::vector<std::vector<InputShape>> input_shapes = {
    {
        { {}, {{1, 5, 5}} },
        { {}, {{1, 7, 5}} }
    },
    {
        { {}, {{2, 9, 5}} },
        { {}, {{2, 15, 9}} }
    },
    {
        { {}, {{5, 17, 5}} },
        { {}, {{5, 7, 17}} }
    },
    {
        { {}, {{9, 75, 5}} },
        { {}, {{9, 55, 75}} }
    },
    {
        { {-1, -1,  5}, {{5, 20, 5},  {3, 50,  5},  {2, 99,  5}} },
        { {-1, -1, -1}, {{5, 30, 20}, {3, 100, 50}, {2, 133, 99}} }
    }
};

static const std::vector<ElementType> boxes_prc = { ElementType::f32, ElementType::f16, ElementType::bf16 };
static const std::vector<ElementType> max_out_boxes_prc = { ElementType::i32, ElementType::i64 };
static const std::vector<ElementType> threshods_prc = { ElementType::f32, ElementType::f16, ElementType::bf16 };
static const std::vector<ElementType> output_prc = { ElementType::i32, ElementType::i64 };
static const std::vector<int64_t> max_out_box_per_class = { 5, 20 };
static const std::vector<float> iou_threshold = { 0.3f, 0.7f };
static const std::vector<float> score_threshold = { 0.3f, 0.7f };


INSTANTIATE_TEST_SUITE_P(smoke_Param, NmsRotatedLayerTestCPU,
        ::testing::Combine(
                ::testing::ValuesIn(input_shapes),
                ::testing::ValuesIn(boxes_prc),
                ::testing::ValuesIn(max_out_boxes_prc),
                ::testing::ValuesIn(threshods_prc),
                ::testing::ValuesIn(output_prc),
                ::testing::ValuesIn(max_out_box_per_class),
                ::testing::ValuesIn(iou_threshold),
                ::testing::ValuesIn(score_threshold),
                ::testing::Values(true, false),
                ::testing::Values(true, false),
                ::testing::Values(true, false),
                ::testing::Values(true, false),
                ::testing::Values(true, false),
                ::testing::Values(true, false),
                ::testing::Values(true, false),
                ::testing::Values(emptyCPUSpec),
                ::testing::Values(empty_plugin_config)),
        NmsRotatedLayerTestCPU::getTestCaseName);

} // namespace CPULayerTestsDefinitions
