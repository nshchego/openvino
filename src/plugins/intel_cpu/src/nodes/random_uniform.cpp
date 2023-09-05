// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "random_uniform.hpp"

#include "ie_parallel.hpp"
#include "ie_ngraph_utils.hpp"
#include <openvino/op/constant.hpp>
#include <openvino/op/random_uniform.hpp>

// using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

bool RandomUniform::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (op->get_type_info() != op::v8::RandomUniform::get_type_info_static()) {
            errorMessage = "Only RandomUniform operation from the opset8 is supported by the CPU plugin.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

RandomUniform::RandomUniform(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
        : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    m_shape_prc = op->get_input_element_type(SHAPE);
    if (!one_of(m_shape_prc, element::i32, element::i64)) {
        m_shape_prc = element::i32;
    }

    auto rnd_op = as_type_ptr<op::v8::RandomUniform>(op);
    // m_output_prc = rnd_op->get_out_type();
    m_global_seed = rnd_op->get_global_seed();
    m_op_seed = rnd_op->get_op_seed();

    m_output_prc = op->get_output_element_type(0);
    if (m_output_prc.is_real() && !one_of(m_output_prc, element::f32, element::f64)) {
        m_output_prc = element::f32;
    }
    if (m_output_prc.is_integral() && !one_of(m_output_prc, element::i32, element::i64)) {
        m_output_prc = element::i32;
    }

    for (int i = 0; i < op->get_input_size(); i++) {
        if (is_type<op::v0::Constant>(op->get_input_node_ptr(i))) {
            m_const_inputs[i] = true;
        }
    }

    if (m_const_inputs[SHAPE]) {
        initOutShape(m_out_shape, as_type<op::v0::Constant>(op->get_input_node_ptr(SHAPE))->get_data_ptr(), m_shape_prc,
                op->get_input_shape(SHAPE)[0]);
    }
    if (m_const_inputs[MIN_VAL]) {
        initEdge(m_min_val, as_type<op::v0::Constant>(op->get_input_node_ptr(MIN_VAL))->get_data_ptr(), m_output_prc);
    }
    if (m_const_inputs[MAX_VAL]) {
        initEdge(m_max_val, as_type<op::v0::Constant>(op->get_input_node_ptr(MAX_VAL))->get_data_ptr(), m_output_prc);
    }

    m_generator = std::default_random_engine{m_op_seed};
}

void RandomUniform::getSupportedDescriptors() {
    if (getParentEdges().size() != 3) {
        THROW_CPU_NODE_ERR << "has incorrect number of input edges.";
    }
    if (getChildEdges().empty()) {
        THROW_CPU_NODE_ERR << "has incorrect number of output edges.";
    }
}

void RandomUniform::initSupportedPrimitiveDescriptors() {
    auto shape_prc = InferenceEngine::details::convertPrecision(m_shape_prc);
    auto out_prc = InferenceEngine::details::convertPrecision(m_output_prc);

    addSupportedPrimDesc({{LayoutType::ncsp, shape_prc, m_const_inputs[SHAPE]},
                          {LayoutType::ncsp, out_prc, m_const_inputs[MIN_VAL]},
                          {LayoutType::ncsp, out_prc, m_const_inputs[MAX_VAL]}},
                         {{LayoutType::ncsp, out_prc}},
                         ref_any);
}

void RandomUniform::execute(dnnl::stream strm) {
    if (!m_const_inputs[SHAPE]) {
        auto memPtr = getParentEdgeAt(SHAPE)->getMemoryPtr();
        // VectorDims out_shape;
        // if (memPtr->getDesc().getPrecision() == Precision::I64) {
        //     auto data = reinterpret_cast<int64_t*>(memPtr->getData());
        //     out_shape.assign(data, data + memPtr->getShape()->getElementsCount());
        // } else if (memPtr->getDesc().getPrecision() == Precision::I32) {
        //     auto data = reinterpret_cast<int32_t*>(memPtr->getData());
        //     out_shape.assign(data, data + memPtr->getShape()->getElementsCount());
        // }
        initOutShape(m_out_shape, memPtr->getData(), m_shape_prc, memPtr->getShape().getElementsCount());

        redefineOutputMemory({m_out_shape});
    }
    if (!m_const_inputs[MIN_VAL]) {
        // m_min_val = getParentEdgeAt(MIN_VAL)->getMemoryPtr()->getData();
        initEdge(m_min_val, getParentEdgeAt(MIN_VAL)->getMemoryPtr()->getData(), m_output_prc);
    }
    if (!m_const_inputs[MAX_VAL]) {
        // m_max_val = getParentEdgeAt(MAX_VAL)->getMemoryPtr()->getData();
        initEdge(m_max_val, getParentEdgeAt(MAX_VAL)->getMemoryPtr()->getData(), m_output_prc);
    }

    auto dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    compute(dstMemPtr->getData(), dstMemPtr->getShape().getElementsCount());

    // auto dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    // auto indicesMemPtr = getParentEdgeAt(INDICES_ID)->getMemoryPtr();
    // auto updateMemPtr = getParentEdgeAt(UPDATE_ID)->getMemoryPtr();

    // uint8_t *dstPtr = reinterpret_cast<uint8_t*>(dstMemPtr->getData());
    // uint8_t *srcPtr = reinterpret_cast<uint8_t*>(srcMemPtr->getData());
    // uint8_t *indicesPtr = reinterpret_cast<uint8_t*>(indicesMemPtr->getData());
    // uint8_t *updatePtr = reinterpret_cast<uint8_t*>(updateMemPtr->getData());

    // const auto& srcDataDim = getParentEdgeAt(DATA_ID)->getMemory().getStaticDims();
    // const auto& indicesDim = getParentEdgeAt(INDICES_ID)->getMemory().getStaticDims();

    // if (axisRelaxed) {
    //     parallel_nt(0, [&](const int ithr, const int nthr) {
    //         size_t start = 0, end = 0;
    //         splitter(indicesBlockND[0], nthr, ithr, start, end);
    //         for (size_t i = start; i < end; i++) {
    //             int64_t idxValue =  getIndicesValue(indicesPtr, i);
    //             if (idxValue >= static_cast<int64_t>(srcDimAxis) ||
    //                 (idxValue < 0 && scatterUpdateMode != ScatterUpdateMode::ScatterElementsUpdate)) {
    //                 IE_THROW() << errorPrefix
    //                            << " have indices value that points to non-existing output tensor element";
    //             }
    //         }
    //     });
    // }

    // if (isInputTensorAtPortEmpty(INDICES_ID)) {
    //     return;
    // }
}

void RandomUniform::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

void RandomUniform::compute(void* out, size_t work_amount) {
    switch (m_output_prc) {
        case element::f32: {
            generateData<float, std::uniform_real_distribution<float>>(
                    std::uniform_real_distribution<float>{m_min_val.f32, m_max_val.f32}, out, work_amount);
        } break;
        case element::i32: {
            generateData<int32_t, std::uniform_int_distribution<int32_t>>(
                    std::uniform_int_distribution<int32_t>{m_min_val.i32, m_max_val.i32}, out, work_amount);
        } break;
        case element::i64: {
            generateData<int64_t, std::uniform_int_distribution<int64_t>>(
                    std::uniform_int_distribution<int64_t>{m_min_val.i64, m_max_val.i64}, out, work_amount);
        } break;
        case element::f64: {
            generateData<double, std::uniform_real_distribution<double>>(
                    std::uniform_real_distribution<double>{m_min_val.f64, m_max_val.f64}, out, work_amount);
        } break;
        default:
            THROW_CPU_NODE_ERR << "has unsupported output type: " << m_output_prc;
    }
}

template <typename T, typename DISTR_TYPE>
void RandomUniform::generateData(DISTR_TYPE distribution, void* out, size_t work_amount) {
    auto dst = reinterpret_cast<T*>(out);
    for (size_t i = 0; i < work_amount; i++) {
        *dst = distribution(m_generator);
        ++dst;
    }
}

void RandomUniform::initOutShape(VectorDims& dst, const void* src, const element::Type& shape_type, size_t len) {
    switch (shape_type) {
        case element::i32: {
            auto data = reinterpret_cast<const int32_t*>(src);
            dst.assign(data, data + len);
        } break;
        case element::i64: {
            auto data = reinterpret_cast<const int64_t*>(src);
            dst.assign(data, data + len);
        } break;
        default:
            THROW_CPU_NODE_ERR << "has unsupported shape precision: " << m_output_prc;
    }
}

void RandomUniform::initEdge(edge& dst, const void* src, const element::Type& output_type) {
    switch (output_type) {
        case element::f32:
            dst.f32 = *reinterpret_cast<const float*>(src);
            break;
        case element::i32:
            dst.i32 = *reinterpret_cast<const int32_t*>(src);
            break;
        case element::i64:
            dst.i64 = *reinterpret_cast<const int64_t*>(src);
            break;
        case element::f64:
            dst.f64 = *reinterpret_cast<const double*>(src);
            break;
        default:
            THROW_CPU_NODE_ERR << "has unsupported output precision: " << output_type;
    }
}

bool RandomUniform::needPrepareParams() const {
    return false;
}

bool RandomUniform::isExecutable() const {
    return !isInputTensorAtPortEmpty(SHAPE);
}

bool RandomUniform::created() const {
    return getType() == Type::RandomUniform;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
