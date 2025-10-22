// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert.h"

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "common/blocked_desc_creator.h"
#include "cpu_memory.h"
#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "graph_context.h"
#include "memory_desc/blocked_memory_desc.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/executors/common/ref_convert.hpp"
#include "nodes/executors/convert_list.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/node_config.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/convert.hpp"
#include "shape_inference/shape_inference_pass_through.hpp"
#include "utils/general_utils.h"
#include "utils/serialization/internal_types.hpp"

using namespace dnnl;

namespace ov::intel_cpu::node {

bool Convert::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto convert = ov::as_type_ptr<const ov::op::v0::Convert>(op);
        if (!convert) {
            errorMessage = "Only opset1 Convert operation is supported";
            return false;
        }

        auto srcPrc = op->get_input_element_type(0);
        auto dstPrc = op->get_output_element_type(0);
        if (!CommonConvertExecutor::isSupported(srcPrc, dstPrc)) {
            errorMessage =
                "cpu_convert can't convert from: " + srcPrc.to_string() + " precision to: " + dstPrc.to_string();
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Convert::Convert(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, PassThroughShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    auto convert = ov::as_type_ptr<const ov::op::v0::Convert>(op);
    m_convert_params.origPrc = convert->get_destination_type();
}

Convert::Convert(BinaryInputBuffer& in_buf, const GraphContext::CPtr& context)
    : Node(in_buf, context) {
    load(in_buf);
}

Convert::Convert(const Shape& shape,
                 const ov::element::Type& inPrc,
                 const ov::element::Type& outPrc,
                 const std::string& nodeName,
                 const GraphContext::CPtr& context)
    : Node("Convert", {shape}, {shape}, {inPrc}, {outPrc}, nodeName, context) {
    m_convert_params.origPrc = outPrc;

    isDynamic = shape.isDynamic();
    if (isDynamicNode()) {
        shapeInference = std::make_shared<ShapeInferPassThrough>();
    }
}

void Convert::getSupportedDescriptors() {
    // if tensor descriptors are set via setDescs method we need to update the inDims/outDims data
    // from correspond tensor descriptors.
    if (m_output_shapes.empty()) {
        m_output_shapes.push_back(output->getShape());
    }
    if (m_input_shapes.empty()) {
        m_input_shapes.push_back(input->getShape());
    }
    CPU_NODE_ASSERT(getParentEdges().size() == 1, "has incorrect number of input edges");
    CPU_NODE_ASSERT(!getChildEdges().empty(), "has incorrect number of output edges");
}

bool Convert::isSupportedDesc(const MemoryDesc& desc) {
    bool isSupported = (desc.getType() & MemoryDescType::Blocked) != 0;
    if (desc.getType() == MemoryDescType::DnnlBlocked) {
        isSupported &= desc.as<const DnnlMemoryDesc>()->hasEmptyExtraData();
    }
    return isSupported;
}

void Convert::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    NodeConfig config;
    PortConfig dataIn;
    PortConfig dataConfigOut;

    bool canInitExternalDesc = false;
    if (input && output) {
        canInitExternalDesc = true;
        canInitExternalDesc &= isSupportedDesc(*input);
        canInitExternalDesc &= isSupportedDesc(*output);
    }

    auto executor_context = std::make_shared<ExecutorContext>(m_context, getImplPriority());

    auto supportedPrimitiveDescriptorsBuilder = [this, executor_context](NodeConfig config) {
        MemoryDescPtr srcMemoryDesc = config.inConfs[0].getMemDesc();
        MemoryDescPtr dstMemoryDesc = config.outConfs[0].getMemDesc();
        m_convert_params.srcPrc = srcMemoryDesc->getPrecision();
        m_convert_params.dstPrc = dstMemoryDesc->getPrecision();
        auto factory =
            std::make_shared<ConvertExecutorFactory>(m_convert_params,
                                                     srcMemoryDesc,
                                                     dstMemoryDesc,
                                                     executor_context);
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown, factory);
    };

    // if input and output pointers are not null and not contain extra data, then the inp/output tensor descriptors were
    // set using setDescs method, so they should be used as the actual descriptors.
    if (canInitExternalDesc) {
        dataIn.setMemDesc(input);
        config.inConfs.push_back(dataIn);

        // inp/out layouts must be the same
        dataConfigOut.setMemDesc(config.inConfs[0].getMemDesc());
        dataConfigOut.setMemDesc(dataConfigOut.getMemDesc()->cloneWithNewPrecision(output->getPrecision()));
        config.outConfs.push_back(dataConfigOut);
        supportedPrimitiveDescriptorsBuilder(config);
        return;
    }

    CPU_NODE_ASSERT(all_of(1U, m_input_shapes.size(), m_output_shapes.size()), "has incorrect number of input/output edges");

    const Shape& insShape = getInputShapeAtPort(0);
    auto insPrecision = getOriginalInputPrecisionAtPort(0);
    const Shape& outputShape = getOutputShapeAtPort(0);
    auto outPrecision = getOriginalOutputPrecisionAtPort(0);

    config.inConfs.push_back(dataIn);
    config.outConfs.push_back(dataConfigOut);

    auto creators = BlockedDescCreator::getCommonCreators();

    // As long as convert is placed right before the output, only planar layout makes sense since the output tensor
    // is always in a planar layout (ngraph limitation), so there is no reason to convert in any other layout.
    bool hasOutputChild = false;
    for (auto& childEdge : getChildEdgesAtPort(0)) {
        if (Type::Output == childEdge->getChild()->getType()) {
            hasOutputChild = true;
            break;
        }
    }
    auto range = hasOutputChild
                     ? BlockedDescCreator::makeFilteredRange(creators, insShape.getRank(), {LayoutType::ncsp})
                     : BlockedDescCreator::makeFilteredRange(creators, insShape.getRank());

    for (auto itr = range.first; itr != range.second; ++itr) {
        config.inConfs[0].setMemDesc(
            std::make_shared<CpuBlockedMemoryDesc>(itr->second->createDesc(insPrecision, insShape)));
        config.outConfs[0].setMemDesc(
            std::make_shared<CpuBlockedMemoryDesc>(itr->second->createDesc(outPrecision, outputShape)));

        supportedPrimitiveDescriptorsBuilder(config);
    }
}

void Convert::prepareParams() {
    const auto& parentMem = getParentEdgeAt(0)->getMemory();
    m_convert_params.size = parentMem.getDescWithType<BlockedMemoryDesc>()->getPaddedElementsCount();

    auto* selectedPD = getSelectedPrimitiveDescriptor();
    MemoryDescPtr srcDesc = getSrcMemoryAtPort(0)->getDescPtr();
    MemoryDescPtr dstDesc = getDstMemoryAtPort(0)->getDescPtr();
    execPtr =
        selectedPD->getExecutorFactoryAs<ConvertExecutorFactory>()->makeExecutor(m_convert_params, srcDesc, dstDesc, {});
    selectedPD->setImplementationType(execPtr->implType());
}

void Convert::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

void Convert::execute([[maybe_unused]] const dnnl::stream& strm) {
    const auto& parentMem = getParentEdgeAt(0)->getMemory();
    const auto& childMem = getChildEdgeAt(0)->getMemory();

    const auto parentPaddElemCount = parentMem.getDescWithType<BlockedMemoryDesc>()->getPaddedElementsCount();
    const auto childPaddElemCount = childMem.getDescWithType<BlockedMemoryDesc>()->getPaddedElementsCount();

    CPU_NODE_ASSERT(parentPaddElemCount == childPaddElemCount,
                    "has different elements number in input and output buffers");

    MemoryCPtr srcMemory = getSrcMemoryAtPort(0);
    MemoryPtr dstMemory = getDstMemoryAtPort(0);
    execPtr->exec({srcMemory}, {dstMemory});
}

bool Convert::created() const {
    return getType() == Type::Convert;
}

void Convert::save(BinaryOutputBuffer& ob) const {
    Node::save(ob);

ob.dump_position();  // TODO: remove

    // ob << input;
    // ob << output;
    ob << m_convert_params;
    
ob.dump_position();  // TODO: remove
}

void Convert::load(BinaryInputBuffer& in_buf) {
in_buf.check_position();  // TODO: remove

    // in_buf >> input;
    // in_buf >> output;
    in_buf >> m_convert_params;

    for (auto& desc : supportedPrimitiveDescriptors) {
        const auto& config = desc.getConfig();
        auto src_memory_desc = config.inConfs[0].getMemDesc();
        auto dst_memory_desc = config.outConfs[0].getMemDesc();
        auto factory =
            std::make_shared<ConvertExecutorFactory>(m_convert_params,
                                                     src_memory_desc,
                                                     dst_memory_desc,
                                                     std::make_shared<ExecutorContext>(m_context, getImplPriority()));
        desc.setExecutorFactory(factory);
    }

in_buf.check_position();  // TODO: remove
}

}  // namespace ov::intel_cpu::node
