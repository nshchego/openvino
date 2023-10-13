// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "non_max_suppression.h"

#include "ie_parallel.hpp"
#include "utils/general_utils.h"
#include "shape_inference/shape_inference_internal_dyn.hpp"
#include "openvino/op/nms_rotated.hpp"
#include "openvino/op/non_max_suppression.hpp"
#include "ov_ops/nms_ie_internal.hpp"

#include <queue>


#include <chrono>

using namespace InferenceEngine;
using namespace dnnl::impl::cpu;

namespace ov {
namespace intel_cpu {
namespace node {

bool NonMaxSuppression::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(), op::v9::NonMaxSuppression::get_type_info_static(),
                                         op::internal::NonMaxSuppressionIEInternal::get_type_info_static(),
                                         op::v13::NMSRotated::get_type_info_static())) {
            errorMessage = "Only NonMaxSuppression from opset9, NonMaxSuppressionIEInternal and NMSRotated from opset13 are supported.";
            return false;
        }

        if (auto nms9 = as_type<const op::v9::NonMaxSuppression>(op.get())) {
            const auto boxEncoding = nms9->get_box_encoding();
            if (!one_of(boxEncoding, op::v9::NonMaxSuppression::BoxEncodingType::CENTER, op::v9::NonMaxSuppression::BoxEncodingType::CORNER)) {
                errorMessage = "Supports only CENTER and CORNER box encoding type";
                return false;
            }
        }
    } catch (...) {
        return false;
    }
    return true;
}

NonMaxSuppression::NonMaxSuppression(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
        : Node(op, context, InternalDynShapeInferFactory()),
          m_is_soft_suppressed_by_iou(false) {
std::cout << "[CPU] NonMaxSuppression Ctr +" << std::endl;
std::cout << "[CPU] NMS_SOFT_NMS_SIGMA size: " << sizeof(NMS_SOFT_NMS_SIGMA) << std::endl;
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW(errorMessage);
    }

    if (one_of(op->get_type_info(), op::internal::NonMaxSuppressionIEInternal::get_type_info_static())) {
        m_out_static_shape = true;
    }

    if (getOriginalInputsNumber() < 2 || getOriginalInputsNumber() > NMS_SOFT_NMS_SIGMA + 1) {
        THROW_CPU_NODE_ERR("has incorrect number of input edges: ", getOriginalInputsNumber());
    }
    if (getOriginalOutputsNumber() != 3) {
        THROW_CPU_NODE_ERR("has incorrect number of output edges: ", getOriginalOutputsNumber());
    }

    if (auto nms9 = as_type<const op::v9::NonMaxSuppression>(op.get())) {
        boxEncodingType = static_cast<NMSBoxEncodeType>(nms9->get_box_encoding());
        m_sort_result_descending = nms9->get_sort_result_descending();
        m_coord_num = 4lu;
    } else if (auto nmsIe = as_type<const op::internal::NonMaxSuppressionIEInternal>(op.get())) {
        boxEncodingType = nmsIe->m_center_point_box ? NMSBoxEncodeType::CENTER : NMSBoxEncodeType::CORNER;
        m_sort_result_descending = nmsIe->m_sort_result_descending;
        m_coord_num = 4lu;
    } else if (auto nms = as_type<const op::v13::NMSRotated>(op.get())) {
        m_sort_result_descending = nms->get_sort_result_descending();
        m_clockwise = nms->get_clockwise();
        m_rotated_boxes = true;
        m_coord_num = 5lu;
    } else {
        const auto &typeInfo = op->get_type_info();
        THROW_CPU_NODE_ERR("doesn't support NMS: ", typeInfo.name, " v", typeInfo.version_id);
    }

    const auto &boxes_dims = getInputShapeAtPort(NMS_BOXES).getDims();
    if (boxes_dims.size() != 3) {
        THROW_CPU_NODE_ERR("has unsupported 'boxes' input rank: ", boxes_dims.size());
    }
    if (boxes_dims[2] != m_coord_num) {
        THROW_CPU_NODE_ERR("has unsupported 'boxes' input 3rd dimension size: ", boxes_dims[2]);
    }

    const auto &scores_dims = getInputShapeAtPort(NMS_SCORES).getDims();
    if (scores_dims.size() != 3) {
        THROW_CPU_NODE_ERR("has unsupported 'scores' input rank: ", scores_dims.size());
    }

    const auto& valid_outputs_shape = getOutputShapeAtPort(NMS_VALID_OUTPUTS);
    if (valid_outputs_shape.getRank() != 1) {
        THROW_CPU_NODE_ERR("has unsupported 'valid_outputs' output rank: ", valid_outputs_shape.getRank());
    }
    if (valid_outputs_shape.getDims()[0] != 1) {
        THROW_CPU_NODE_ERR("has unsupported 'valid_outputs' output 1st dimension size: ", valid_outputs_shape.getDims()[1]);
    }

    const auto op_inputs_num = op->get_input_size();
    for (size_t i = 0lu; i < op_inputs_num; i++) {
        m_const_inputs[i] = is_type<op::v0::Constant>(op->get_input_node_ptr(i));
    }
    for (size_t i = 0lu; i < op->get_output_size(); i++) {
        m_defined_outputs[i] = !op->get_output_target_inputs(i).empty();
    }

    if (m_const_inputs[NMS_MAX_OUTPUT_BOXES_PER_CLASS] && op_inputs_num > NMS_MAX_OUTPUT_BOXES_PER_CLASS) {
        int64_t val = 0l;
        switch (op->get_input_element_type(NMS_MAX_OUTPUT_BOXES_PER_CLASS)) {
            case element::i64: val = as_type<op::v0::Constant>(op->get_input_node_ptr(NMS_MAX_OUTPUT_BOXES_PER_CLASS))->get_data_ptr<int64_t>()[0]; break;
            case element::i32: val = as_type<op::v0::Constant>(op->get_input_node_ptr(NMS_MAX_OUTPUT_BOXES_PER_CLASS))->get_data_ptr<int32_t>()[0]; break;
            default: THROW_CPU_NODE_ERR("has input with unsupported precision.");
        }
        m_max_output_boxes_per_class = val < 0l ? 0lu : static_cast<size_t>(val);
    }
    if (m_const_inputs[NMS_IOU_THRESHOLD] && op_inputs_num > NMS_IOU_THRESHOLD) {
        switch (op->get_input_element_type(NMS_IOU_THRESHOLD)) {
            case element::f32: m_iou_threshold = as_type<op::v0::Constant>(op->get_input_node_ptr(NMS_IOU_THRESHOLD))->get_data_ptr<float>()[0]; break;
            default: THROW_CPU_NODE_ERR("has input with unsupported precision.");
        }
    }
    if (m_const_inputs[NMS_SCORE_THRESHOLD] && op_inputs_num > NMS_SCORE_THRESHOLD) {
        switch (op->get_input_element_type(NMS_SCORE_THRESHOLD)) {
            case element::f32: m_score_threshold = as_type<op::v0::Constant>(op->get_input_node_ptr(NMS_SCORE_THRESHOLD))->get_data_ptr<float>()[0]; break;
            default: THROW_CPU_NODE_ERR("has input with unsupported precision.");
        }
        // m_score_threshold = 0.0001f; // TODO: REMOVE
    }
    if (m_const_inputs[NMS_SOFT_NMS_SIGMA] && op_inputs_num > NMS_SOFT_NMS_SIGMA) {
        switch (op->get_input_element_type(NMS_SOFT_NMS_SIGMA)) {
            case element::f32: m_soft_nms_sigma = as_type<op::v0::Constant>(op->get_input_node_ptr(NMS_SOFT_NMS_SIGMA))->get_data_ptr<float>()[0]; break;
            default: THROW_CPU_NODE_ERR("has input with unsupported precision.");
        }
        if (m_soft_nms_sigma > 0.f) {
            m_scale = -0.5f / m_soft_nms_sigma;
        }
    }
std::cout << "[CPU] NonMaxSuppression Ctr -" << std::endl;
}

void NonMaxSuppression::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    const std::vector<Precision> supportedFloatPrecision = {Precision::FP32, Precision::BF16};
    const std::vector<Precision> supportedIntOutputPrecision = {Precision::I32, Precision::I64};

    checkPrecision(getOriginalInputPrecisionAtPort(NMS_BOXES), supportedFloatPrecision, "boxes", inType);
    checkPrecision(getOriginalInputPrecisionAtPort(NMS_SCORES), supportedFloatPrecision, "scores", inType);
    checkPrecision(getOriginalOutputPrecisionAtPort(NMS_VALID_OUTPUTS), supportedIntOutputPrecision, "valid_outputs", outType);

    const std::vector<Precision> supportedPrecision = {Precision::I16, Precision::U8, Precision::I8, Precision::U16, Precision::I32,
                                                       Precision::U32, Precision::I64, Precision::U64};

    if (inputShapes.size() > NMS_MAX_OUTPUT_BOXES_PER_CLASS)
        check1DInput(getInputShapeAtPort(NMS_MAX_OUTPUT_BOXES_PER_CLASS), supportedPrecision, "max_output_boxes_per_class", NMS_MAX_OUTPUT_BOXES_PER_CLASS);
    if (inputShapes.size() > NMS_IOU_THRESHOLD)
        check1DInput(getInputShapeAtPort(NMS_IOU_THRESHOLD), supportedFloatPrecision, "iou_threshold", NMS_IOU_THRESHOLD);
    if (inputShapes.size() > NMS_SCORE_THRESHOLD)
        check1DInput(getInputShapeAtPort(NMS_SCORE_THRESHOLD), supportedFloatPrecision, "score_threshold", NMS_SCORE_THRESHOLD);
    if (inputShapes.size() > NMS_SOFT_NMS_SIGMA)
        check1DInput(getInputShapeAtPort(NMS_SCORE_THRESHOLD), supportedFloatPrecision, "soft_nms_sigma", NMS_SCORE_THRESHOLD);

    checkOutput(getOutputShapeAtPort(NMS_SELECTED_INDICES), supportedIntOutputPrecision, "selected_indices", NMS_SELECTED_INDICES);
    checkOutput(getOutputShapeAtPort(NMS_SELECTED_SCORES), supportedFloatPrecision, "selected_scores", NMS_SELECTED_SCORES);

    std::vector<PortConfigurator> inDataConf;
    inDataConf.reserve(inputShapes.size());
    for (size_t i = 0; i < inputShapes.size(); ++i) {
        Precision inPrecision = i == NMS_MAX_OUTPUT_BOXES_PER_CLASS ? Precision::I32 : Precision::FP32;
        inDataConf.emplace_back(LayoutType::ncsp, inPrecision);
    }

    std::vector<PortConfigurator> outDataConf;
    outDataConf.reserve(outputShapes.size());
    for (size_t i = 0; i < outputShapes.size(); ++i) {
        Precision outPrecision = i == NMS_SELECTED_SCORES ? Precision::FP32 : Precision::I32;
        outDataConf.emplace_back(LayoutType::ncsp, outPrecision);
    }

    // as only FP32 and ncsp is supported, and kernel is shape agnostic, we can create here. There is no need to recompilation.
    createJitKernel();

    x64::cpu_isa_t actual_isa = x64::isa_undef;
    if (m_jit_kernel) {
        actual_isa = m_jit_kernel->getIsa();
    }

    impl_desc_type impl_type;
    switch (actual_isa) {
        case x64::avx512_core: impl_type = impl_desc_type::jit_avx512; break;
        case x64::avx2:        impl_type = impl_desc_type::jit_avx2;   break;
        case x64::sse41:       impl_type = impl_desc_type::jit_sse42;  break;
        default:               impl_type = impl_desc_type::ref;
    }

    addSupportedPrimDesc(inDataConf, outDataConf, impl_type);
}

void NonMaxSuppression::prepareParams() {
printf("[CPU] NonMaxSuppression::prepareParams\n");
    const auto& boxesDims = isDynamicNode() ? getParentEdgesAtPort(NMS_BOXES)[0]->getMemory().getStaticDims() :
                                               getInputShapeAtPort(NMS_BOXES).getStaticDims();
    const auto& scoresDims = isDynamicNode() ? getParentEdgesAtPort(NMS_SCORES)[0]->getMemory().getStaticDims() :
                                                getInputShapeAtPort(NMS_SCORES).getStaticDims();

    m_batches_num = boxesDims[0];
    m_boxes_num = boxesDims[1];
    m_classes_num = scoresDims[1];
    if (m_batches_num != scoresDims[0]) {
        THROW_CPU_NODE_ERR("Batches number is different in 'boxes' and 'scores' inputs");
    }
    if (m_boxes_num != scoresDims[2]) {
        THROW_CPU_NODE_ERR("Boxes number is different in 'boxes' and 'scores' inputs");
    }

    if (m_const_inputs[NMS_MAX_OUTPUT_BOXES_PER_CLASS]) {
        m_max_output_boxes_per_class = std::min(m_max_output_boxes_per_class, m_boxes_num);
    }

    m_num_filtered_boxes.resize(m_batches_num);
    for (auto & i : m_num_filtered_boxes) {
        i.resize(m_classes_num);
    }
}

void NonMaxSuppression::createJitKernel() {
#if defined(OPENVINO_ARCH_X86_64)
    if (!m_rotated_boxes) {
        auto jcp = kernel::NmsCompileParams();
        jcp.box_encode_type = boxEncodingType;
        jcp.is_soft_suppressed_by_iou = m_is_soft_suppressed_by_iou;

        m_jit_kernel = kernel::JitKernel<kernel::NmsCompileParams, kernel::NmsCallArgs>::createInstance<kernel::NonMaxSuppression>(jcp);
    } else {
        auto jcp = kernel::NmsRotatedCompileParams();

        m_jit_kernel = kernel::JitKernel<kernel::NmsRotatedCompileParams, kernel::NmsRotatedCallArgs>::createInstance<kernel::NmsRotated>(jcp);
    }
#endif // OPENVINO_ARCH_X86_64
}

void NonMaxSuppression::executeDynamicImpl(dnnl::stream strm) {
// printf("[CPU] NonMaxSuppression::executeDynamicImpl\n");
    if (hasEmptyInputTensors() || (inputShapes.size() > NMS_MAX_OUTPUT_BOXES_PER_CLASS &&
            reinterpret_cast<int *>(getParentEdgeAt(NMS_MAX_OUTPUT_BOXES_PER_CLASS)->getMemoryPtr()->getData())[0] == 0)) {
        redefineOutputMemory({{0, 3}, {0, 3}, {1}});
        *reinterpret_cast<int *>(getChildEdgesAtPort(NMS_VALID_OUTPUTS)[0]->getMemoryPtr()->getData()) = 0;
        return;
    }
    execute(strm);
}

void NonMaxSuppression::execute(dnnl::stream strm) {
static double elps = 0lu;
static uint64_t counter = 0lu;
auto t1 = std::chrono::high_resolution_clock::now();

// printf("[CPU] NonMaxSuppression::execute +\n");
// auto box_shape = getParentEdgeAt(NMS_BOXES)->getMemoryPtr()->getShape().getStaticDims();
// printf("[CPU] execute box_shape: {%lu; %lu; %lu}\n", box_shape[0lu], box_shape[1], box_shape[2]);

    const auto inputs_num = inputShapes.size();

    if (!m_const_inputs[NMS_MAX_OUTPUT_BOXES_PER_CLASS] && inputs_num > NMS_MAX_OUTPUT_BOXES_PER_CLASS) {
        m_max_output_boxes_per_class = reinterpret_cast<int32_t *>(getParentEdgeAt(NMS_MAX_OUTPUT_BOXES_PER_CLASS)->getMemoryPtr()->getData())[0];
        m_max_output_boxes_per_class = std::min(m_max_output_boxes_per_class, m_boxes_num);
    }
    if (m_max_output_boxes_per_class == 0lu) {
        return;
    }

    if (!m_const_inputs[NMS_IOU_THRESHOLD] && inputs_num > NMS_IOU_THRESHOLD) {
        m_iou_threshold = reinterpret_cast<float *>(getParentEdgeAt(NMS_IOU_THRESHOLD)->getMemoryPtr()->getData())[0];
    }
    if (!m_const_inputs[NMS_SCORE_THRESHOLD] && inputs_num > NMS_SCORE_THRESHOLD) {
        m_score_threshold = reinterpret_cast<float *>(getParentEdgeAt(NMS_SCORE_THRESHOLD)->getMemoryPtr()->getData())[0];
    }
    if (!m_const_inputs[NMS_SOFT_NMS_SIGMA] && inputs_num > NMS_SOFT_NMS_SIGMA) {
        m_soft_nms_sigma = reinterpret_cast<float *>(getParentEdgeAt(NMS_SOFT_NMS_SIGMA)->getMemoryPtr()->getData())[0];
        m_scale = (m_soft_nms_sigma > 0.f) ? (-0.5f / m_soft_nms_sigma) : 0.f;
    }

    // auto boxes_memory = getParentEdgeAt(NMS_BOXES)->getMemoryPtr();
    const auto boxes = reinterpret_cast<const float *>(getParentEdgeAt(NMS_BOXES)->getMemoryPtr()->getData());
    const auto scores = reinterpret_cast<const float *>(getParentEdgeAt(NMS_SCORES)->getMemoryPtr()->getData());

    auto boxesStrides = getParentEdgeAt(NMS_BOXES)->getMemory().getDescWithType<BlockedMemoryDesc>()->getStrides();
    auto scoresStrides = getParentEdgeAt(NMS_SCORES)->getMemory().getDescWithType<BlockedMemoryDesc>()->getStrides();

    const auto maxNumberOfBoxes = m_max_output_boxes_per_class * m_batches_num * m_classes_num;
    // std::vector<FilteredBox> filtered_boxes(maxNumberOfBoxes);

//     // if (m_rotated_boxes) {
//     //     nmsRotated(boxes, scores, boxesStrides, scoresStrides, filtered_boxes);
//     // } else
//     if (m_soft_nms_sigma == 0.f) {
//         nmsWithoutSoftSigma(boxes, scores, boxesStrides, scoresStrides, filtered_boxes);
//     } else {
//         nmsWithSoftSigma(boxes, scores, boxesStrides, scoresStrides, filtered_boxes);
//     }
// // printf("rotatedBoxesIntersection: %lu; getRotatedVertices: %lu\n", counter_0, counter_1);

//     size_t startOffset = m_num_filtered_boxes[0][0];
//     for (size_t b = 0; b < m_num_filtered_boxes.size(); b++) {
//         size_t batchOffset = b * m_classes_num * m_max_output_boxes_per_class;
//         for (size_t c = (b == 0 ? 1 : 0); c < m_num_filtered_boxes[b].size(); c++) {
//             size_t offset = batchOffset + c * m_max_output_boxes_per_class;
//             for (size_t i = 0; i < m_num_filtered_boxes[b][c]; i++) {
//                 filtered_boxes[startOffset + i] = filtered_boxes[offset + i];
//             }
//             startOffset += m_num_filtered_boxes[b][c];
//         }
//     }
//     filtered_boxes.resize(startOffset);

//     // need more particular comparator to get deterministic behaviour
//     // escape situation when filtred boxes with same score have different position from launch to launch
//     if (m_sort_result_descending) {
//         parallel_sort(filtered_boxes.begin(), filtered_boxes.end(),
//                       [](const FilteredBox& l, const FilteredBox& r) {
//                           return (l.score > r.score) ||
//                                  (l.score == r.score && l.batch_index < r.batch_index) ||
//                                  (l.score == r.score && l.batch_index == r.batch_index && l.class_index < r.class_index) ||
//                                  (l.score == r.score && l.batch_index == r.batch_index && l.class_index == r.class_index && l.box_index < r.box_index);
//                       });
//     }
// // for (const auto& box_info : filtered_boxes) {
// //     printf("[CPU] batch_index: %ld; class_index: %ld; score: %f\n",
// //         box_info.batch_index, box_info.class_index, box_info.score);
// // }

//     auto indicesMemPtr = getChildEdgesAtPort(NMS_SELECTED_INDICES)[0]->getMemoryPtr();
//     const size_t validOutputs = std::min(filtered_boxes.size(), maxNumberOfBoxes);
// printf("validOutputs: %lu\n", validOutputs);

static bool call_mem = true;
    // if (!m_out_static_shape) {
    // if (!getChildEdges()[0].lock()->getMemory().getDesc().getShape().isStatic()) {
    if (call_mem) {
        call_mem = false;
        const size_t validOutputs = 220;
    // if (!getChildEdgesAtPort(0)[0]->getMemory().getDesc().getShape().isStatic()) {
        VectorDims newDims{validOutputs, 3};
        redefineOutputMemory({newDims, newDims, {1}});
    }

    // const auto selectedIndicesStride = indicesMemPtr->getDescWithType<BlockedMemoryDesc>()->getStrides()[0];

    // if (m_defined_outputs[NMS_SELECTED_INDICES]) {
    //     auto selectedIndicesPtr = reinterpret_cast<int32_t *>(indicesMemPtr->getData());
    //     int* boxes_ptr = &(filtered_boxes[0].batch_index);

    //     size_t idx = 0lu;
    //     for (; idx < validOutputs; idx++) {
    //         // selectedIndicesPtr[0] = filtered_boxes[idx].batch_index;
    //         // selectedIndicesPtr[1] = filtered_boxes[idx].class_index;
    //         // selectedIndicesPtr[2] = filtered_boxes[idx].box_index;
    //         // memcpy(selectedIndicesPtr, &(filtered_boxes[idx].batch_index), 12);
    //         memcpy(selectedIndicesPtr, boxes_ptr, 12);
    //         selectedIndicesPtr += selectedIndicesStride;
    //         boxes_ptr += 4;
    //     }

    //     if (m_out_static_shape) {
    //         std::fill(selectedIndicesPtr, selectedIndicesPtr + (maxNumberOfBoxes - idx) * selectedIndicesStride, -1);
    //     }
    // }

    // if (m_defined_outputs[NMS_SELECTED_SCORES]) {
    //     auto selectedScoresPtr = reinterpret_cast<float *>(getChildEdgesAtPort(NMS_SELECTED_SCORES)[0]->getMemoryPtr()->getData());

    //     size_t idx = 0lu;
    //     for (; idx < validOutputs; idx++) {
    //         selectedScoresPtr[0] = static_cast<float>(filtered_boxes[idx].batch_index);
    //         selectedScoresPtr[1] = static_cast<float>(filtered_boxes[idx].class_index);
    //         selectedScoresPtr[2] = static_cast<float>(filtered_boxes[idx].score);
    //         selectedScoresPtr += selectedIndicesStride;
    //     }

    //     if (m_out_static_shape) {
    //         std::fill(selectedScoresPtr, selectedScoresPtr + (maxNumberOfBoxes - idx) * selectedIndicesStride, -1.f);
    //     }
    // }

    // if (m_defined_outputs[NMS_VALID_OUTPUTS]) {
    //     auto valid_outputs = reinterpret_cast<int *>(getChildEdgesAtPort(NMS_VALID_OUTPUTS)[0]->getMemoryPtr()->getData());
    //     *valid_outputs = static_cast<int>(validOutputs);
    // }

auto t2 = std::chrono::high_resolution_clock::now();
if (counter > 0lu) {
    std::chrono::duration<double, std::nano> ms_double = t2 - t1;
    double elps_cur = ms_double.count();
    elps += elps_cur;
    printf("[CPU] execute elps cur: %f; avg: %f nsec\n", elps_cur, elps/counter);
}
counter++;
}

void NonMaxSuppression::nmsWithSoftSigma(const float *boxes, const float *scores, const VectorDims &boxesStrides,
                                                             const VectorDims &scoresStrides, std::vector<FilteredBox> &filtBoxes) {
    auto less = [](const boxInfo& l, const boxInfo& r) {
        return l.score < r.score || ((l.score == r.score) && (l.idx > r.idx));
    };

    // update score, if iou is 0, weight is 1, score does not change
    // if is_soft_suppressed_by_iou is false, apply for all iou, including iou>iou_threshold, soft suppressed when score < score_threshold
    // if is_soft_suppressed_by_iou is true, hard suppressed by iou_threshold, then soft suppress
    auto coeff = [&](float iou) {
        if (m_is_soft_suppressed_by_iou && iou > m_iou_threshold)
            return 0.0f;
        return std::exp(m_scale * iou * iou);
    };

    parallel_for2d(m_batches_num, m_classes_num, [&](int batch_idx, int class_idx) {
        std::vector<FilteredBox> selectedBoxes;
        const float *boxesPtr = boxes + batch_idx * boxesStrides[0];
        const float *scoresPtr = scores + batch_idx * scoresStrides[0] + class_idx * scoresStrides[1];

        std::priority_queue<boxInfo, std::vector<boxInfo>, decltype(less)> sorted_boxes(less);  // score, box_id, suppress_begin_index
        for (int box_idx = 0; box_idx < static_cast<int>(m_boxes_num); box_idx++) {
            if (scoresPtr[box_idx] > m_score_threshold)
                sorted_boxes.emplace(boxInfo({scoresPtr[box_idx], box_idx, 0}));
        }
        size_t sorted_boxes_size = sorted_boxes.size();
        size_t maxSeletedBoxNum = std::min(sorted_boxes_size, m_max_output_boxes_per_class);
        selectedBoxes.reserve(maxSeletedBoxNum);
        if (maxSeletedBoxNum > 0) {
            // include first directly
            boxInfo candidateBox = sorted_boxes.top();
            sorted_boxes.pop();
            selectedBoxes.push_back({ candidateBox.score, batch_idx, class_idx, candidateBox.idx });
            if (maxSeletedBoxNum > 1) {
                if (m_jit_kernel) {
                    std::vector<float> boxCoord0(maxSeletedBoxNum, 0.0f);
                    std::vector<float> boxCoord1(maxSeletedBoxNum, 0.0f);
                    std::vector<float> boxCoord2(maxSeletedBoxNum, 0.0f);
                    std::vector<float> boxCoord3(maxSeletedBoxNum, 0.0f);

                    boxCoord0[0] = boxesPtr[candidateBox.idx * m_coord_num];
                    boxCoord1[0] = boxesPtr[candidateBox.idx * m_coord_num + 1];
                    boxCoord2[0] = boxesPtr[candidateBox.idx * m_coord_num + 2];
                    boxCoord3[0] = boxesPtr[candidateBox.idx * m_coord_num + 3];

                    auto arg = kernel::NmsCallArgs();
                    arg.iou_threshold = static_cast<float*>(&m_iou_threshold);
                    arg.score_threshold = static_cast<float*>(&m_score_threshold);
                    arg.scale = static_cast<float*>(&m_scale);
                    while (selectedBoxes.size() < m_max_output_boxes_per_class && !sorted_boxes.empty()) {
                        boxInfo candidateBox = sorted_boxes.top();
                        float origScore = candidateBox.score;
                        sorted_boxes.pop();

                        int candidateStatus = NMSCandidateStatus::SELECTED; // 0 for suppressed, 1 for selected, 2 for updated
                        arg.score = static_cast<float*>(&candidateBox.score);
                        arg.selected_boxes_num = selectedBoxes.size() - candidateBox.suppress_begin_index;
                        arg.selected_boxes_coord[0] = static_cast<float*>(&boxCoord0[candidateBox.suppress_begin_index]);
                        arg.selected_boxes_coord[1] = static_cast<float*>(&boxCoord1[candidateBox.suppress_begin_index]);
                        arg.selected_boxes_coord[2] = static_cast<float*>(&boxCoord2[candidateBox.suppress_begin_index]);
                        arg.selected_boxes_coord[3] = static_cast<float*>(&boxCoord3[candidateBox.suppress_begin_index]);
                        arg.candidate_box = static_cast<const float*>(&boxesPtr[candidateBox.idx * m_coord_num]);
                        arg.candidate_status = static_cast<int*>(&candidateStatus);
                        (*m_jit_kernel)(&arg);

                        if (candidateStatus == NMSCandidateStatus::SUPPRESSED) {
                            continue;
                        } else {
                            if (candidateBox.score == origScore) {
                                selectedBoxes.push_back({ candidateBox.score, batch_idx, class_idx, candidateBox.idx });
                                int selectedSize = selectedBoxes.size();
                                boxCoord0[selectedSize - 1] = boxesPtr[candidateBox.idx * m_coord_num];
                                boxCoord1[selectedSize - 1] = boxesPtr[candidateBox.idx * m_coord_num + 1];
                                boxCoord2[selectedSize - 1] = boxesPtr[candidateBox.idx * m_coord_num + 2];
                                boxCoord3[selectedSize - 1] = boxesPtr[candidateBox.idx * m_coord_num + 3];
                            } else {
                                candidateBox.suppress_begin_index = selectedBoxes.size();
                                sorted_boxes.push(candidateBox);
                            }
                        }
                    }
                } else {
                    while (selectedBoxes.size() < m_max_output_boxes_per_class && !sorted_boxes.empty()) {
                        boxInfo candidateBox = sorted_boxes.top();
                        float origScore = candidateBox.score;
                        sorted_boxes.pop();

                        int candidateStatus = NMSCandidateStatus::SELECTED; // 0 for suppressed, 1 for selected, 2 for updated
                        for (int selected_idx = static_cast<int>(selectedBoxes.size()) - 1; selected_idx >= candidateBox.suppress_begin_index; selected_idx--) {
                            float iou = intersectionOverUnion(&boxesPtr[candidateBox.idx * m_coord_num],
                                                              &boxesPtr[selectedBoxes[selected_idx].box_index * m_coord_num]);

                            // when is_soft_suppressed_by_iou is true, score is decayed to zero and implicitely suppressed if iou > iou_threshold.
                            candidateBox.score *= coeff(iou);
                            // soft suppressed
                            if (candidateBox.score <= m_score_threshold) {
                                candidateStatus = NMSCandidateStatus::SUPPRESSED;
                                break;
                            }
                        }

                        if (candidateStatus == NMSCandidateStatus::SUPPRESSED) {
                            continue;
                        } else {
                            if (candidateBox.score == origScore) {
                                selectedBoxes.push_back({ candidateBox.score, batch_idx, class_idx, candidateBox.idx });
                            } else {
                                candidateBox.suppress_begin_index = selectedBoxes.size();
                                sorted_boxes.push(candidateBox);
                            }
                        }
                    }
                }
            }
        }
        m_num_filtered_boxes[batch_idx][class_idx] = selectedBoxes.size();
        size_t offset = batch_idx * m_classes_num * m_max_output_boxes_per_class + class_idx * m_max_output_boxes_per_class;
        for (size_t i = 0; i < selectedBoxes.size(); i++) {
            filtBoxes[offset + i] = selectedBoxes[i];
        }
    });
}

void NonMaxSuppression::nmsWithoutSoftSigma(const float *boxes, const float *scores, const VectorDims &boxesStrides,
                                                                const VectorDims &scoresStrides, std::vector<FilteredBox> &filtBoxes) {
// printf("[CPU] NonMaxSuppression::nmsWithoutSoftSigma +\n");
// printf("[CPU] FilteredBox size: %ld\n", sizeof(FilteredBox));
    const auto max_out_boxes = static_cast<int64_t>(m_max_output_boxes_per_class);
    parallel_for2d(m_batches_num, m_classes_num, [&](int64_t batch_idx, int64_t class_idx) {
// for (int64_t batch_idx = 0l; batch_idx < m_batches_num; batch_idx++) {
// for (int64_t class_idx = 0l; class_idx < m_classes_num; class_idx++) {
        const float *boxesPtr = boxes + batch_idx * boxesStrides[0];
        const float *scoresPtr = scores + batch_idx * scoresStrides[0] + class_idx * scoresStrides[1];

        std::vector<std::pair<float, int>> sorted_boxes;  // score, box_idx
        sorted_boxes.reserve(m_boxes_num);
        for (size_t box_idx = 0; box_idx < m_boxes_num; box_idx++) {
            if (scoresPtr[box_idx] > m_score_threshold) {
                sorted_boxes.emplace_back(std::make_pair(scoresPtr[box_idx], box_idx));
// printf("[CPU] sorted_boxes_1 score: %f; box_idx: %d\n", sorted_boxes.back().first, sorted_boxes.back().second);
// auto data_f = reinterpret_cast<float *>(sorted_boxes.data());
// auto data_i = reinterpret_cast<int32_t *>(sorted_boxes.data());
// printf("[CPU] sorted_boxes_2 score: %f; box_idx: %d\n", data_f[(sorted_boxes.size() - 1) * 2], data_i[(sorted_boxes.size() - 1) * 2 + 1]);
            }
        }

        int64_t io_selection_size = 0l;
        const size_t sorted_boxes_size = sorted_boxes.size();
// printf("[CPU] sorted_boxes_size: %lu; max_out_boxes: %ld\n", sorted_boxes_size, max_out_boxes);
        if (sorted_boxes_size > 0lu) {
            parallel_sort(sorted_boxes.begin(), sorted_boxes.end(),
                          [](const std::pair<float, int>& l, const std::pair<float, int>& r) {
                              return (l.first > r.first || ((l.first == r.first) && (l.second < r.second)));
                          });
            int64_t offset = batch_idx * m_classes_num * m_max_output_boxes_per_class + class_idx * m_max_output_boxes_per_class;
            filtBoxes[offset + 0] = FilteredBox(sorted_boxes[0].first, batch_idx, class_idx, sorted_boxes[0].second);
// printf("[CPU] selected class_index: %d; score: %f; box_index: %d\n", filtBoxes[offset + io_selection_size].class_index,
//     filtBoxes[offset + io_selection_size].score, filtBoxes[offset + io_selection_size].box_index);
            io_selection_size++;
            if (sorted_boxes_size > 1lu) {
                if (m_jit_kernel) {
                    std::vector<float> boxCoord0(sorted_boxes_size, 0.0f);
                    std::vector<float> boxCoord1(sorted_boxes_size, 0.0f);
                    std::vector<float> boxCoord2(sorted_boxes_size, 0.0f);
                    std::vector<float> boxCoord3(sorted_boxes_size, 0.0f);

                    boxCoord0[0] = boxesPtr[sorted_boxes[0].second * m_coord_num];
                    boxCoord1[0] = boxesPtr[sorted_boxes[0].second * m_coord_num + 1];
                    boxCoord2[0] = boxesPtr[sorted_boxes[0].second * m_coord_num + 2];
                    boxCoord3[0] = boxesPtr[sorted_boxes[0].second * m_coord_num + 3];

                    auto arg = kernel::NmsCallArgs();
                    arg.iou_threshold = static_cast<float*>(&m_iou_threshold);
                    arg.score_threshold = static_cast<float*>(&m_score_threshold);
                    arg.scale = static_cast<float*>(&m_scale);
                    // box start index do not change for hard supresion
                    arg.selected_boxes_coord[0] = static_cast<float*>(&boxCoord0[0]);
                    arg.selected_boxes_coord[1] = static_cast<float*>(&boxCoord1[0]);
                    arg.selected_boxes_coord[2] = static_cast<float*>(&boxCoord2[0]);
                    arg.selected_boxes_coord[3] = static_cast<float*>(&boxCoord3[0]);

                    for (size_t candidate_idx = 1; (candidate_idx < sorted_boxes_size) && (io_selection_size < max_out_boxes); candidate_idx++) {
                        int candidateStatus = NMSCandidateStatus::SELECTED; // 0 for suppressed, 1 for selected
                        arg.selected_boxes_num = io_selection_size;
                        arg.candidate_box = static_cast<const float*>(&boxesPtr[sorted_boxes[candidate_idx].second * m_coord_num]);
                        arg.candidate_status = static_cast<int*>(&candidateStatus);
                        (*m_jit_kernel)(&arg);
                        if (candidateStatus == NMSCandidateStatus::SELECTED) {
                            boxCoord0[io_selection_size] = boxesPtr[sorted_boxes[candidate_idx].second * m_coord_num];
                            boxCoord1[io_selection_size] = boxesPtr[sorted_boxes[candidate_idx].second * m_coord_num + 1];
                            boxCoord2[io_selection_size] = boxesPtr[sorted_boxes[candidate_idx].second * m_coord_num + 2];
                            boxCoord3[io_selection_size] = boxesPtr[sorted_boxes[candidate_idx].second * m_coord_num + 3];
                            filtBoxes[offset + io_selection_size] =
                                FilteredBox(sorted_boxes[candidate_idx].first, batch_idx, class_idx, sorted_boxes[candidate_idx].second);
                            io_selection_size++;
                        }
                    }
                } else {
                    // TODO: Declare function ptr rotatedIntersectionOverUnion
                    for (size_t candidate_idx = 1lu; (candidate_idx < sorted_boxes_size) && (io_selection_size < max_out_boxes); candidate_idx++) {
// printf("[CPU] candidate_idx: %lu; io_selection_size: %ld\n", candidate_idx, io_selection_size);
                        NMSCandidateStatus candidateStatus = NMSCandidateStatus::SELECTED;
                        // auto ref_box = boxesPtr[sorted_boxes[candidate_idx].second * m_coord_num];
                        for (int64_t selected_idx = io_selection_size - 1l; selected_idx >= 0l; selected_idx--) {
// printf("[CPU] selected_idx: %ld\n", selected_idx);
                            float iou = 0.f;
                            if (!m_rotated_boxes) {
                                iou = intersectionOverUnion(&boxesPtr[sorted_boxes[candidate_idx].second * m_coord_num],
                                    &boxesPtr[filtBoxes[offset + selected_idx].box_index * m_coord_num]);
                            } else {
                                iou = rotatedIntersectionOverUnion_2(&boxesPtr[sorted_boxes[candidate_idx].second * m_coord_num],
                                    &boxesPtr[filtBoxes[offset + selected_idx].box_index * m_coord_num]);
// printf("[CPU] nmsWithoutSoftSigma iou: %f\n", iou);
                            }
                            if (iou > m_iou_threshold) {
// printf("[CPU] nmsWithoutSoftSigma (iou > m_iou_threshold) iou: %f; m_iou_threshold: %f\n", iou, m_iou_threshold);
                                candidateStatus = NMSCandidateStatus::SUPPRESSED;
                                break;
                            }
                        }

                        if (candidateStatus == NMSCandidateStatus::SELECTED) {
                            filtBoxes[offset + io_selection_size] =
                                FilteredBox(sorted_boxes[candidate_idx].first, batch_idx, class_idx, sorted_boxes[candidate_idx].second);
// printf("[CPU] selected class_index: %d; score: %f; box_index: %d\n", filtBoxes[offset + io_selection_size].class_index,
//     filtBoxes[offset + io_selection_size].score, filtBoxes[offset + io_selection_size].box_index);
                            io_selection_size++;
                        }
                    }
                }
            }
        }

        m_num_filtered_boxes[batch_idx][class_idx] = io_selection_size;
    });
// }}
}

////////// Rotated boxes //////////

struct RotatedBox {
    float x_ctr, y_ctr, w, h, a;
};

inline float dot_2d(const Point2D& A, const Point2D& B) {
    return A.x * B.x + A.y * B.y;
}

inline float cross_2d(const Point2D& A, const Point2D& B) {
    return A.x * B.y - B.x * A.y;
}

inline void getRotatedVertices(const float* box, Point2D (&pts)[4], bool clockwise) {
// static size_t counter_1 = 0lu;
// printf("getRotatedVertices: %lu\n", ++counter_1);
    auto theta = clockwise ? box[4] : -box[4];

    auto cos_theta = std::cos(theta) * 0.5f;
    auto sin_theta = std::sin(theta) * 0.5f;

    // y: top --> down; x: left --> right
    // Left-Down
    pts[0].x = box[0] - sin_theta * box[3] - cos_theta * box[2];
    pts[0].y = box[1] + cos_theta * box[3] - sin_theta * box[2];
    // Left-Top
    pts[1].x = box[0] + sin_theta * box[3] - cos_theta * box[2];
    pts[1].y = box[1] - cos_theta * box[3] - sin_theta * box[2];
    // Right-Top
    pts[2].x = 2 * box[0] - pts[0].x;
    pts[2].y = 2 * box[1] - pts[0].y;
    // Right-Down
    pts[3].x = 2 * box[0] - pts[1].x;
    pts[3].y = 2 * box[1] - pts[1].y;
}

inline void getRotatedVertices_2(const RotatedBox& box, Point2D (&pts)[4], bool clockwise) {
// static size_t counter_1 = 0lu;
// printf("getRotatedVertices_2: %lu\n", ++counter_1);
    // M_PI / 180. == 0.01745329251
    auto theta = clockwise ? box.a : -box.a;  // angle already in radians
// printf("[CPU] theta: %f\n", theta);
    // auto theta = box.a;
    auto cos_theta = std::cos(theta) * 0.5f;
    auto sin_theta = std::sin(theta) * 0.5f;

    // y: top --> down; x: left --> right
    // Left-Down
    pts[0].x = box.x_ctr - sin_theta * box.h - cos_theta * box.w;
    pts[0].y = box.y_ctr + cos_theta * box.h - sin_theta * box.w;
    // Left-Top
    pts[1].x = box.x_ctr + sin_theta * box.h - cos_theta * box.w;
    pts[1].y = box.y_ctr - cos_theta * box.h - sin_theta * box.w;
    // Right-Top
    pts[2].x = 2 * box.x_ctr - pts[0].x;
    pts[2].y = 2 * box.y_ctr - pts[0].y;
    // Right-Down
    pts[3].x = 2 * box.x_ctr - pts[1].x;
    pts[3].y = 2 * box.y_ctr - pts[1].y;
}

inline float polygon_area(const Point2D (&q)[24], const int64_t& m) {
    if (m <= 2l) {
        return 0.f;
    }

    float area = 0.f;
    size_t mlu = static_cast<size_t>(m - 1l);
    for (size_t i = 1lu; i < mlu; i++) {
        area += std::abs(cross_2d(q[i] - q[0], q[i + 1] - q[0]));
    }

    return area / 2.f;
}

inline int convexHullGraham(const Point2D (&p)[24],
                            const size_t num_in,
                            Point2D (&q)[24],
                            bool shift_to_zero = false) {
    assert(num_in >= 2lu);

    // Step 1:
    // Find point with minimum y
    // if more than 1 points have the same minimum y,
    // pick the one with the minimum x.
    size_t t = 0;
    for (size_t i = 1lu; i < num_in; i++) {
        if (p[i].y < p[t].y || (p[i].y == p[t].y && p[i].x < p[t].x)) {
            t = i;
        }
    }
    auto& start = p[t];  // starting point

    // Step 2:
    // Subtract starting point from every points (for sorting in the next step)
    for (size_t i = 0lu; i < num_in; i++) {
        q[i] = p[i] - start;
    }

    // Swap the starting point to position 0
    std::swap(q[t], q[0]);

    // Step 3:
    // Sort point 1 ~ num_in according to their relative cross-product values
    // (essentially sorting according to angles)
    // If the angles are the same, sort according to their distance to origin
    float dist[24];
    for (size_t i = 0; i < num_in; i++) {
        dist[i] = dot_2d(q[i], q[i]);
    }

    std::sort(q + 1, q + num_in, [](const Point2D& A, const Point2D& B) -> bool {
        float temp = cross_2d(A, B);
        if (std::abs(temp) < 1e-6f) {
            return dot_2d(A, A) < dot_2d(B, B);
        } else {
            return temp > 0.f;
        }
    });
    // compute distance to origin after sort, since the points are now different.
    for (size_t i = 0lu; i < num_in; i++) {
        dist[i] = dot_2d(q[i], q[i]);
    }

    // Step 4:
    // Make sure there are at least 2 points (that don't overlap with each other)
    // in the stack
    size_t k;  // index of the non-overlapped second point
    for (k = 1lu; k < num_in; k++) {
        if (dist[k] > 1e-8f) {
            break;
        }
    }
    if (k == num_in) {
        // We reach the end, which means the convex hull is just one point
        q[0] = p[t];
        return 1;
    }
    q[1] = q[k];
    int64_t m = 2;  // 2 points in the stack
    // Step 5:
    // Finally we can start the scanning process.
    // When a non-convex relationship between the 3 points is found
    // (either concave shape or duplicated points),
    // we pop the previous point from the stack
    // until the 3-point relationship is convex again, or
    // until the stack only contains two points
    for (size_t i = k + 1; i < num_in; i++) {
        while (m > 1 && cross_2d(q[i] - q[m - 2], q[m - 1] - q[m - 2]) >= 0) {
            m--;
        }
        q[m++] = q[i];
    }

    // Step 6 (Optional):
    // In general sense we need the original coordinates, so we
    // need to shift the points back (reverting Step 2)
    // But if we're only interested in getting the area/perimeter of the shape
    // We can simply return.
    if (!shift_to_zero) {
        size_t mlu = static_cast<size_t>(m);
        for (size_t i = 0lu; i < mlu; i++) {
            q[i] += start;
        }
    }

    return m;
}

inline size_t getIntersectionPoints(const Point2D (&pts1)[4],
                                    const Point2D (&pts2)[4],
                                    Point2D (&intersections)[24]) {
    // Line vector
    // A line from p1 to p2 is: p1 + (p2-p1)*t, t=[0,1]
    Point2D vec1[4], vec2[4];
    for (size_t i = 0lu; i < 4lu; i++) {
        vec1[i] = pts1[(i + 1lu) % 4lu] - pts1[i];
        vec2[i] = pts2[(i + 1lu) % 4lu] - pts2[i];
    }

    // Line test - test all line combos for intersection
    size_t num = 0lu;  // number of intersections
    for (size_t i = 0lu; i < 4lu; i++) {
        for (size_t j = 0lu; j < 4lu; j++) {
            // Solve for 2x2 Ax=b
            float det = cross_2d(vec2[j], vec1[i]);

            // This takes care of parallel lines
            if (std::abs(det) <= 1e-14f) {
                continue;
            }

            auto vec12 = pts2[j] - pts1[i];

            auto t1 = cross_2d(vec2[j], vec12) / det;
            auto t2 = cross_2d(vec1[i], vec12) / det;

            if (t1 >= 0.f && t1 <= 1.f && t2 >= 0.f && t2 <= 1.f) {
                intersections[num++] = pts1[i] + vec1[i] * t1;
            }
        }
    }

    // Check for vertices of rect1 inside rect2
    {
        const auto& AB = vec2[0];
        const auto& DA = vec2[3];
        auto ABdotAB = dot_2d(AB, AB);
        auto ADdotAD = dot_2d(DA, DA);
        for (size_t i = 0lu; i < 4lu; i++) {
            // assume ABCD is the rectangle, and P is the point to be judged
            // P is inside ABCD iff. P's projection on AB lies within AB
            // and P's projection on AD lies within AD

            auto AP = pts1[i] - pts2[0];

            auto APdotAB = dot_2d(AP, AB);
            auto APdotAD = -dot_2d(AP, DA);

            if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) && (APdotAD <= ADdotAD)) {
                intersections[num++] = pts1[i];
            }
        }
    }

    // Reverse the check - check for vertices of rect2 inside rect1
    {
        const auto& AB = vec1[0];
        const auto& DA = vec1[3];
        auto ABdotAB = dot_2d(AB, AB);
        auto ADdotAD = dot_2d(DA, DA);
        for (size_t i = 0lu; i < 4lu; i++) {
            auto AP = pts2[i] - pts1[0];

            auto APdotAB = dot_2d(AP, AB);
            auto APdotAD = -dot_2d(AP, DA);

            if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) && (APdotAD <= ADdotAD)) {
                intersections[num++] = pts2[i];
            }
        }
    }

    return num;
}

inline float rotatedBoxesIntersection(const Point2D (&vertices_0)[4], const float* box_1, const bool clockwise) {
// static size_t counter_1 = 0lu;
// printf("rotatedBoxesIntersection: %lu\n", ++counter_1);
    // There are up to 4 x 4 + 4 + 4 = 24 intersections (including dups) returned
    Point2D intersect_pts[24], ordered_pts[24];

    Point2D vertices_1[4];
    getRotatedVertices(box_1, vertices_1, clockwise);

    auto num = getIntersectionPoints(vertices_0, vertices_1, intersect_pts);

    if (num <= 2lu) {
        return 0.f;
    }

    auto num_convex = convexHullGraham(intersect_pts, num, ordered_pts, true);
    return polygon_area(ordered_pts, num_convex);
}

inline float rotatedBoxesIntersection_2(const RotatedBox& box_0, const RotatedBox& box_1, bool clockwise) {
// static size_t counter_1 = 0lu;
// printf("rotatedBoxesIntersection_2: %lu\n", ++counter_1);
    // There are up to 4 x 4 + 4 + 4 = 24 intersections (including dups) returned
    Point2D intersect_pts[24], ordered_pts[24];

    Point2D pts1[4];
    Point2D pts2[4];
    getRotatedVertices_2(box_0, pts1, clockwise);
    getRotatedVertices_2(box_1, pts2, clockwise);

    auto num = getIntersectionPoints(pts1, pts2, intersect_pts);

    if (num <= 2lu) {
        return 0.f;
    }

    auto num_convex = convexHullGraham(intersect_pts, num, ordered_pts, true);
    return polygon_area(ordered_pts, num_convex);
}

inline float NonMaxSuppression::rotatedIntersectionOverUnion(const Point2D (&vertices_0)[4], const float area_0, const float* box_1) {
    const auto area_1 = box_1[2] * box_1[3]; // W x H
    if (area_1 <= 0.f) {
        return 0.f;
    }

    const auto intersection = rotatedBoxesIntersection(vertices_0, box_1, m_clockwise);

// printf("[CPU] NonMaxSuppression::rotatedIntersectionOverUnion -\n");
    return intersection / (area_0 + area_1 - intersection);
}

inline float NonMaxSuppression::rotatedIntersectionOverUnion_2(const float* box_0, const float* box_1) {
    auto boxI = *(reinterpret_cast<const RotatedBox*>(box_0));
    auto boxJ = *(reinterpret_cast<const RotatedBox*>(box_1)); // TODO: leave in float

    const auto intersection = rotatedBoxesIntersection_2(boxI, boxJ, m_clockwise);
    const auto area_0 = boxI.w * boxI.h;
    const auto area_1 = boxJ.w * boxJ.h;

    if (area_0 <= 0.f || area_1 <= 0.f) {
        return 0.f;
    }

    const auto union_area = area_0 + area_1 - intersection;
    return intersection / union_area;
}

/////////////// End of Rotated boxes ///////////////

void NonMaxSuppression::nmsRotated(const float *boxes, const float *scores, const VectorDims &boxesStrides,
                                                                const VectorDims &scoresStrides, std::vector<FilteredBox> &filtBoxes) {
// printf("[CPU] NonMaxSuppression::nmsRotated +\n");
// printf("[CPU] FilteredBox size: %ld\n", sizeof(FilteredBox));
    const auto max_out_boxes = static_cast<int64_t>(m_max_output_boxes_per_class);
    parallel_for2d(m_batches_num, m_classes_num, [&](int64_t batch_idx, int64_t class_idx) {
// for (int64_t batch_idx = 0l; batch_idx < m_batches_num; batch_idx++) {
// for (int64_t class_idx = 0l; class_idx < m_classes_num; class_idx++) {
        const float *boxesPtr = boxes + batch_idx * boxesStrides[0];
        const float *scoresPtr = scores + batch_idx * scoresStrides[0] + class_idx * scoresStrides[1];

        std::vector<std::pair<float, int>> sorted_boxes;  // score, box_idx
        sorted_boxes.reserve(m_boxes_num);
        for (size_t box_idx = 0; box_idx < m_boxes_num; box_idx++) {
            if (scoresPtr[box_idx] > m_score_threshold) {
                sorted_boxes.emplace_back(std::make_pair(scoresPtr[box_idx], box_idx));
// printf("[CPU] sorted_boxes_1 score: %f; box_idx: %d\n", sorted_boxes.back().first, sorted_boxes.back().second);
// auto data_f = reinterpret_cast<float *>(sorted_boxes.data());
// auto data_i = reinterpret_cast<int32_t *>(sorted_boxes.data());
// printf("[CPU] sorted_boxes_2 score: %f; box_idx: %d\n", data_f[(sorted_boxes.size() - 1) * 2], data_i[(sorted_boxes.size() - 1) * 2 + 1]);
            }
        }

        int64_t io_selection_size = 0l;
        const size_t sorted_boxes_size = sorted_boxes.size();
// printf("[CPU] sorted_boxes_size: %lu; max_out_boxes: %ld\n", sorted_boxes_size, max_out_boxes);
        if (sorted_boxes_size > 0lu) {
            parallel_sort(sorted_boxes.begin(), sorted_boxes.end(),
                          [](const std::pair<float, int>& l, const std::pair<float, int>& r) {
                              return (l.first > r.first || ((l.first == r.first) && (l.second < r.second)));
                          });
            int64_t offset = batch_idx * m_classes_num * m_max_output_boxes_per_class + class_idx * m_max_output_boxes_per_class;
            filtBoxes[offset + 0] = FilteredBox(sorted_boxes[0].first, batch_idx, class_idx, sorted_boxes[0].second);
// printf("[CPU] selected class_index: %d; score: %f; box_index: %d\n", filtBoxes[offset + io_selection_size].class_index,
//     filtBoxes[offset + io_selection_size].score, filtBoxes[offset + io_selection_size].box_index);
            io_selection_size++;
            if (sorted_boxes_size > 1lu) {
                // TODO: Declare function ptr rotatedIntersectionOverUnion
                for (size_t candidate_idx = 1lu; (candidate_idx < sorted_boxes_size) && (io_selection_size < max_out_boxes); candidate_idx++) {
// printf("[CPU] candidate_idx: %lu; io_selection_size: %ld\n", candidate_idx, io_selection_size);
                    NMSCandidateStatus candidateStatus = NMSCandidateStatus::SELECTED;
                    // auto ref_box = boxesPtr[sorted_boxes[candidate_idx].second * m_coord_num];
                    for (int64_t selected_idx = io_selection_size - 1l; selected_idx >= 0l; selected_idx--) {
// printf("[CPU] selected_idx: %ld\n", selected_idx);
                        float iou = rotatedIntersectionOverUnion_2(&boxesPtr[sorted_boxes[candidate_idx].second * m_coord_num],
                            &boxesPtr[filtBoxes[offset + selected_idx].box_index * m_coord_num]);
// printf("[CPU] nmsWithoutSoftSigma iou: %f\n", iou);
                        if (iou > m_iou_threshold) {
// printf("[CPU] nmsWithoutSoftSigma (iou > m_iou_threshold) iou: %f; m_iou_threshold: %f\n", iou, m_iou_threshold);
                            candidateStatus = NMSCandidateStatus::SUPPRESSED;
                            break;
                        }
                    }

                    if (candidateStatus == NMSCandidateStatus::SELECTED) {
                        filtBoxes[offset + io_selection_size] =
                            FilteredBox(sorted_boxes[candidate_idx].first, batch_idx, class_idx, sorted_boxes[candidate_idx].second);
// printf("[CPU] selected class_index: %d; score: %f; box_index: %d\n", filtBoxes[offset + io_selection_size].class_index,
//     filtBoxes[offset + io_selection_size].score, filtBoxes[offset + io_selection_size].box_index);
                        io_selection_size++;
                    }
                }
            }
        }

        m_num_filtered_boxes[batch_idx][class_idx] = io_selection_size;
    });
// }}
}

// void NonMaxSuppression::nmsRotated(const float* boxes, const float* scores, const VectorDims& boxes_strides,
//                                    const VectorDims& scores_strides, std::vector<FilteredBox>& filtered_boxes) {
// // printf("[CPU] NonMaxSuppression::nmsRotated +\n");
// // printf("[CPU] FilteredBox size: %ld\n", sizeof(FilteredBox));
//     const auto max_out_boxes = static_cast<int64_t>(m_max_output_boxes_per_class);
//     const auto class_box_offset = m_classes_num * m_max_output_boxes_per_class;

//     parallel_for2d(m_batches_num, m_classes_num, [&](size_t batch_idx, size_t class_idx) {
// // for (int64_t batch_idx = 0l; batch_idx < m_batches_num; batch_idx++) {
// // for (int64_t class_idx = 0l; class_idx < m_classes_num; class_idx++) {
//         const float *boxes_ptr = boxes + batch_idx * boxes_strides[0];
//         const float *scores_ptr = scores + batch_idx * scores_strides[0] + class_idx * scores_strides[1];

//         std::vector<std::pair<float, size_t>> sorted_indices;  // score, box_idx
//         sorted_indices.reserve(m_boxes_num);
//         for (size_t box_idx = 0lu; box_idx < m_boxes_num; box_idx++) {
//             if (scores_ptr[box_idx] > m_score_threshold) {
//                 sorted_indices.emplace_back(std::make_pair(scores_ptr[box_idx], box_idx));
// // printf("[CPU] sorted_boxes_1 score: %f; box_idx: %d\n", sorted_indices.back().first, sorted_indices.back().second);
// // auto data_f = reinterpret_cast<float *>(sorted_indices.data());
// // auto data_i = reinterpret_cast<int32_t *>(sorted_indices.data());
// // printf("[CPU] sorted_boxes_2 score: %f; box_idx: %d\n", data_f[(sorted_indices.size() - 1) * 2], data_i[(sorted_indices.size() - 1) * 2 + 1]);
//             }
//         }

//         // size_t io_selection_size = 0lu;
//         int64_t io_selection_size = 0l;
//         const size_t sorted_boxes_size = sorted_indices.size();
// // printf("[CPU] sorted_boxes_size: %lu; max_out_boxes: %ld\n", sorted_boxes_size, max_out_boxes);
//         if (sorted_boxes_size > 0lu) {
//             parallel_sort(sorted_indices.begin(), sorted_indices.end(),
//                           [](const std::pair<float, size_t>& l, const std::pair<float, size_t>& r) {
//                               return (l.first > r.first || ((l.first == r.first) && (l.second < r.second)));
//                           });
//             size_t offset = class_box_offset * batch_idx + m_max_output_boxes_per_class * class_idx;
//             filtered_boxes[offset] = FilteredBox(sorted_indices[0].first, batch_idx, class_idx, sorted_indices[0].second);
// // printf("[CPU] selected class_index: %d; score: %f; box_index: %d\n", filtered_boxes[offset + io_selection_size].class_index,
// //     filtered_boxes[offset + io_selection_size].score, filtered_boxes[offset + io_selection_size].box_index);
//             io_selection_size++;
//             if (sorted_boxes_size > 1lu) {
//                 // if (m_jit_kernel) {
//                 //     auto arg = kernel::NmsRotatedCallArgs();

//                 //     arg.boxes_ptr = boxes_ptr;
//                 //     arg.sorted_indices_ptr = sorted_indices.data();
//                 //     arg.sorted_boxes_size = sorted_boxes_size;
//                 //     arg.max_out_boxes = max_out_boxes;
//                 //     arg.io_selection_size = io_selection_size;

//                 //     (*m_jit_kernel)(&arg);
//                 // } else {
//                     NMSCandidateStatus candidate_status;

//                     for (size_t candidate_idx = 1lu; (candidate_idx < sorted_boxes_size) && (io_selection_size < max_out_boxes); candidate_idx++) {
// // printf("[CPU] candidate_idx: %lu; io_selection_size: %ld\n", candidate_idx, io_selection_size);
//                         candidate_status = NMSCandidateStatus::SELECTED;
//                         auto box_0 = boxes_ptr + sorted_indices[candidate_idx].second * m_coord_num;
//                         Point2D vertices_0[4];
//                         getRotatedVertices(box_0, vertices_0, m_clockwise);
//                         const auto area_0 = box_0[2] * box_0[3]; // W x H
// // printf("[CPU] nmsRotated area_0: %f\n", area_0);

//                         if (area_0 > 0.f) {
//                             // size_t indices_num = io_selection_size - 1lu;
//                             // for (size_t selected_idx = 0lu; selected_idx < indices_num; selected_idx++) {
//                             for (int64_t selected_idx = io_selection_size - 1l; selected_idx >= 0l; selected_idx--) {
// // printf("[CPU] selected_idx: %ld\n", selected_idx);
//                                 auto iou = rotatedIntersectionOverUnion(vertices_0, area_0,
//                                         boxes_ptr + filtered_boxes[offset + selected_idx].box_index * m_coord_num);
// // printf("[CPU] nmsRotated iou: %f\n", iou);
//                                 if (iou > m_iou_threshold) {
// // printf("[CPU] nmsRotated (iou > m_iou_threshold) iou: %f; m_iou_threshold: %f\n", iou, m_iou_threshold);
//                                     candidate_status = NMSCandidateStatus::SUPPRESSED;
//                                     break;
//                                 }
//                             }
//                         } else if (0.f > m_iou_threshold) {
// // printf("[CPU] nmsRotated (iou > m_iou_threshold) iou: %f; m_iou_threshold: %f\n", iou, m_iou_threshold);
//                             candidate_status = NMSCandidateStatus::SUPPRESSED;
//                         }

//                         if (candidate_status == NMSCandidateStatus::SELECTED) {
//                             filtered_boxes[offset + io_selection_size] =
//                                 FilteredBox(sorted_indices[candidate_idx].first, batch_idx, class_idx, sorted_indices[candidate_idx].second);
// // printf("[CPU] selected class_index: %d; score: %f; box_index: %d\n", filtered_boxes[offset + io_selection_size].class_index,
// //     filtered_boxes[offset + io_selection_size].score, filtered_boxes[offset + io_selection_size].box_index);
//                             io_selection_size++;
//                         }
//                     }
//                 // }
//             }
//         }

//         m_num_filtered_boxes[batch_idx][class_idx] = io_selection_size;
//     });
// // }}
// }

float NonMaxSuppression::intersectionOverUnion(const float *boxesI, const float *boxesJ) {
std::cout << "[CPU] NonMaxSuppression::intersectionOverUnion +" << std::endl;
    float yminI, xminI, ymaxI, xmaxI, yminJ, xminJ, ymaxJ, xmaxJ;
    if (boxEncodingType == NMSBoxEncodeType::CENTER) {
        //  box format: x_center, y_center, width, height
        yminI = boxesI[1] - boxesI[3] / 2.f;
        xminI = boxesI[0] - boxesI[2] / 2.f;
        ymaxI = boxesI[1] + boxesI[3] / 2.f;
        xmaxI = boxesI[0] + boxesI[2] / 2.f;
        yminJ = boxesJ[1] - boxesJ[3] / 2.f;
        xminJ = boxesJ[0] - boxesJ[2] / 2.f;
        ymaxJ = boxesJ[1] + boxesJ[3] / 2.f;
        xmaxJ = boxesJ[0] + boxesJ[2] / 2.f;
    } else {
        //  box format: y1, x1, y2, x2
        yminI = (std::min)(boxesI[0], boxesI[2]);
        xminI = (std::min)(boxesI[1], boxesI[3]);
        ymaxI = (std::max)(boxesI[0], boxesI[2]);
        xmaxI = (std::max)(boxesI[1], boxesI[3]);
        yminJ = (std::min)(boxesJ[0], boxesJ[2]);
        xminJ = (std::min)(boxesJ[1], boxesJ[3]);
        ymaxJ = (std::max)(boxesJ[0], boxesJ[2]);
        xmaxJ = (std::max)(boxesJ[1], boxesJ[3]);
    }

    float areaI = (ymaxI - yminI) * (xmaxI - xminI);
    float areaJ = (ymaxJ - yminJ) * (xmaxJ - xminJ);
    if (areaI <= 0.f || areaJ <= 0.f)
        return 0.f;

    float intersection_area =
            (std::max)((std::min)(ymaxI, ymaxJ) - (std::max)(yminI, yminJ), 0.f) *
            (std::max)((std::min)(xmaxI, xmaxJ) - (std::max)(xminI, xminJ), 0.f);
    return intersection_area / (areaI + areaJ - intersection_area);
}

void NonMaxSuppression::checkPrecision(const Precision& prec, const std::vector<Precision>& precList,
                                                           const std::string& name, const std::string& type) {
    if (std::find(precList.begin(), precList.end(), prec) == precList.end()) {
        THROW_CPU_NODE_ERR("has unsupported '", name, "' ", type, " precision: ", prec);
    }
}

void NonMaxSuppression::check1DInput(const Shape& shape, const std::vector<Precision>& precList,
                                                         const std::string& name, const size_t port) {
    checkPrecision(getOriginalInputPrecisionAtPort(port), precList, name, inType);

    if (shape.getRank() != 0 && shape.getRank() != 1)
        THROW_CPU_NODE_ERR("has unsupported '", name, "' input rank: ", shape.getRank());
    if (shape.getRank() == 1)
        if (shape.getDims()[0] != 1)
            THROW_CPU_NODE_ERR("has unsupported '", name, "' input 1st dimension size: ", MemoryDescUtils::dim2str(shape.getDims()[0]));
}

void NonMaxSuppression::checkOutput(const Shape& shape, const std::vector<Precision>& precList,
                                                        const std::string& name, const size_t port) {
    checkPrecision(getOriginalOutputPrecisionAtPort(port), precList, name, outType);

    if (shape.getRank() != 2)
        THROW_CPU_NODE_ERR("has unsupported '", name, "' output rank: ", shape.getRank());
    if (shape.getDims()[1] != 3)
        THROW_CPU_NODE_ERR("has unsupported '", name, "' output 2nd dimension size: ", MemoryDescUtils::dim2str(shape.getDims()[1]));
}

bool NonMaxSuppression::isExecutable() const {
    return isDynamicNode() || Node::isExecutable();
}

bool NonMaxSuppression::created() const {
    return getType() == Type::NonMaxSuppression;
}

void NonMaxSuppression::executeDynamic(dnnl::stream strm) {
// static double elps = 0lu;
// static uint64_t counter = 0lu;
// auto t1 = std::chrono::high_resolution_clock::now();

    if (isExecutable()) {
        executeDynamicImpl(strm);
    }
    updateLastInputDims();

// auto t2 = std::chrono::high_resolution_clock::now();
// if (counter > 0lu) {
//     std::chrono::duration<double, std::nano> ms_double = t2 - t1;
//     double elps_cur = ms_double.count();
//     elps += elps_cur;
//     printf("[CPU] executeDynamic elps cur: %f; avg: %f nsec\n", elps_cur, elps/counter);
// }
// counter++;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
