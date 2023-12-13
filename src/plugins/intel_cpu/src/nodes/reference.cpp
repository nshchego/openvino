// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reference.h"
#include "common/cpu_memcpy.h"

using OvString = ov::element_type_traits<ov::element::string>::value_type;

namespace ov {
namespace intel_cpu {
namespace node {

Reference::Reference(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context,
                                         const std::string& errorMessage) :
        Node(op, context, NgraphShapeInferFactory(op, FULL_PORT_MASK)), ovCoreNode(op), additionalErrorMessage(errorMessage) {
std::cout << "[CPU] Reference prc: " << op->get_input_element_type(0) << std::endl;
    if (!op->has_evaluate()) {
        OPENVINO_THROW_NOT_IMPLEMENTED(
            "Cannot fallback on ngraph reference implementation (Ngraph::Node::evaluate() is not implemented");
    }

    setType(Type::Reference);
    setTypeStr("Reference");
}

void Reference::getSupportedDescriptors() {}

void Reference::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    std::vector<PortConfigurator> inputConfigurators;
    inputConfigurators.reserve(inputShapes.size());
    for (size_t i = 0; i < inputShapes.size(); i++) {
        inputConfigurators.emplace_back(LayoutType::ncsp, ovCoreNode->get_input_element_type(i), inputShapes[i]);
    }

    std::vector<PortConfigurator> outputConfigurators;
    outputConfigurators.reserve(inputShapes.size());
    for (size_t i = 0; i < outputShapes.size(); i++) {
        outputConfigurators.emplace_back(LayoutType::ncsp, ovCoreNode->get_output_element_type(i), outputShapes[i]);
    }

    addSupportedPrimDesc(inputConfigurators, outputConfigurators, impl_desc_type::ref);
}

void Reference::createPrimitive() {}

void Reference::execute(dnnl::stream strm) {
std::cout << "[CPU] Reference execute prc: " << getOriginalOutputPrecisionAtPort(0) << std::endl;
    auto inputs = prepareInputs();
    auto outputs = prepareOutputs();
    if (!ovCoreNode->evaluate(outputs, inputs)) {
        THROW_CPU_NODE_ERR("evaluation failed for core operation: ", std::string(ovCoreNode->get_type_name()));
    }
}

void Reference::executeDynamicImpl(dnnl::stream strm) {
std::cout << "[CPU] Reference executeDynamicImpl prc: " << getOriginalOutputPrecisionAtPort(0) << std::endl;
    auto inputs = prepareInputs();
    ov::TensorVector outputs;
    auto result = Node::shapeInfer();
    if (ShapeInferStatus::success == result.status) {
        Node::redefineOutputMemory(result.dims);
        outputs = prepareOutputs();
    } else if (ShapeInferStatus::skip == result.status) {
        outputs.reserve(outputShapes.size());
        for (size_t i = 0; i < outputShapes.size(); ++i) {
            auto mem_desc = getBaseMemDescAtOutputPort(i);
            if (mem_desc->isDefined()) {
                outputs.emplace_back(ovCoreNode->get_output_element_type(i), mem_desc->getShape().getStaticDims());
            } else {
                outputs.emplace_back(ovCoreNode->get_output_element_type(i), ov::Shape{0});
            }
        }
    } else {
         THROW_CPU_NODE_ERR("got unexpected shape infer result status during the inference.");
    }
    if (!ovCoreNode->evaluate(outputs, inputs)) {
        THROW_CPU_NODE_ERR("evaluation failed for core operation: ", std::string(ovCoreNode->get_type_name()));
    }
    if (ShapeInferStatus::skip == result.status) {
        std::vector<VectorDims> newOutputDims;
        newOutputDims.reserve(outputs.size());
        for (auto& tensor : outputs) {
            newOutputDims.emplace_back(tensor.get_shape());
        }
        Node::redefineOutputMemory(newOutputDims);
        for (size_t i = 0lu; i < outputShapes.size(); ++i) {
            auto memory = getChildEdgesAtPort(i)[0]->getMemoryPtr();
            auto& tensor = outputs[i];
            if (memory->getSize() != tensor.get_byte_size()) {
                THROW_CPU_NODE_ERR("output tensor data size mismatch occurred during the inference on output port number ", i);
            }
            if (tensor.get_element_type() == element::string) {
                auto srcPtr = tensor.data<OvString>();
                auto dstPtr = reinterpret_cast<OvString *>(memory->getData());
                std::copy(srcPtr, srcPtr + tensor.get_size(), dstPtr);
            } else {
                cpu_memcpy(memory->getData(), tensor.data(), tensor.get_byte_size());
            }
        }
    }
}

bool Reference::created() const {
    return getType() == Type::Reference;
}

bool Reference::needShapeInfer() const {
    return false;
}

ov::TensorVector Reference::prepareInputs() const {
std::cout << "[CPU] Reference::prepareInputs" << std::endl;
    ov::TensorVector inputs;
    for (size_t i = 0lu; i < inputShapes.size(); i++) {
        auto srcDataPtr = getParentEdgesAtPort(i)[0]->getMemory().getData();
std::cout << "    In prc: " << getParentEdgesAtPort(i)[0]->getMemory().getDesc().getPrecision() <<
    "; in ptr: " << srcDataPtr << std::endl;

// auto InData = reinterpret_cast<std::string *>(srcDataPtr);
// std::cout << "[CPU] Reference::prepareInputs input " << getParentEdgesAtPort(i)[0]->getMemory().getSize() << std::endl;
// for (size_t i = 0lu; i < getParentEdgesAtPort(i)[0]->getMemory().getSize() / sizeof(std::string *); i++) {
// for (size_t i = 0lu; i < 5lu; i++) {
//     std::cout << "    InData: \"" << InData[i] << "\"; ptr: " << (InData + i) << std::endl;
// }

// auto OutData = op_outputs[0].data<ov::element_type_traits<ov::element::string>::value_type>();
// std::cout << "[TEST] INTExecutable::call expected out: " << std::endl;
// for (size_t i = 0lu; i < op_outputs[0].get_size(); i++) {
//     std::cout << "    OutData: \"" << OutData[i] << "\"" << std::endl;
// }

        ov::Shape shape = ovCoreNode->get_input_partial_shape(i).rank().get_length() == 0 ?
                ov::Shape{} : getParentEdgesAtPort(i)[0]->getMemory().getStaticDims();

        if (std::any_of(shape.begin(), shape.end(), [](const size_t dim) { return dim == 0lu; } )) {
            inputs.push_back(ov::Tensor(ovCoreNode->get_input_element_type(i), shape));
        } else {
            CPU_NODE_ASSERT(srcDataPtr, "has empty input data on port ", i);
            inputs.push_back(ov::Tensor(ovCoreNode->get_input_element_type(i), shape, srcDataPtr));
        }
    }
    return inputs;
}

ov::TensorVector Reference::prepareOutputs() const {
std::cout << "[CPU] Reference::prepareOutputs" << std::endl;
    ov::TensorVector outputs;
    for (size_t i = 0lu; i < outputShapes.size(); i++) {
        void *dstDataPtr = getChildEdgesAtPort(i)[0]->getMemory().getData();
        ov::Shape shape = ovCoreNode->get_output_partial_shape(i).rank().get_length() == 0 ?
                ov::Shape{} : getChildEdgesAtPort(i)[0]->getMemory().getStaticDims();

        if (std::any_of(shape.begin(), shape.end(), [](const size_t dim) { return dim == 0lu; } )) {
            outputs.push_back(ov::Tensor(ovCoreNode->get_output_element_type(i), shape));
        } else {
            CPU_NODE_ASSERT(dstDataPtr, "has empty output data on port ", i);
            outputs.push_back(ov::Tensor(ovCoreNode->get_output_element_type(i), shape, dstDataPtr));
        }
    }
    return outputs;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
