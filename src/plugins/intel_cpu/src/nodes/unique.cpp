// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "unique.hpp"
#include "ie_parallel.hpp"
#include <ngraph/opsets/opset1.hpp>

using namespace InferenceEngine;
using namespace dnnl::impl::cpu;
using namespace ov::intel_cpu;
using namespace ov::intel_cpu::node;

#define THROW_ERROR IE_THROW() << getTypeStr() << " node with name '" << getName() << "' "

bool Unique::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!ov::is_type<op::v10::Unique>(op)) {
            errorMessage = "Not supported Unique operation version. CPU plug-in supports only 10th version.";
            return false;
        }
        if (op->get_input_size() > AXIS && !ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(AXIS))) { // TODO: check get_input_size
            errorMessage = "CPU plug-in supports only constant Axis input.";
            return false;
        }
        if (!x64::mayiuse(x64::avx512_core)) {
            errorMessage = "Not supported CPU instructions set.";
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}

Unique::Unique(const std::shared_ptr<ov::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache) :
        Node(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    if (!one_of(op->get_input_size(), 1, 2) || op->get_output_size() != 4)
        THROW_ERROR << "has incorrect number of input/output edges.";

    for (int i = 0; i < 4; i++) {
        jcp.definedOutputs[i] = !op->get_output_target_inputs(i).empty();
    }

    jcp.sorted = ov::as_type_ptr<ov::op::v10::Unique>(op)->get_sorted();
    if (op->get_input_size() > AXIS) {
        jcp.flattened = false;
        jcp.axis = ov::as_type<ov::op::v0::Constant>(op->get_input_node_ptr(AXIS))->cast_vector<int>()[0];
        if (jcp.axis < 0) {
            jcp.axis += op->get_input_partial_shape(IN_DATA).rank().get_length();
        }
        if (jcp.axis < 0) {
            THROW_ERROR << "has invalid axis value " << ov::as_type<ov::op::v0::Constant>(op->get_input_node_ptr(AXIS))->cast_vector<int>()[0];
        }
    } else {
        jcp.flattened = true;
    }
    jcp.dynamicShapes = isDynamicNode();
}

void Unique::initSupportedPrimitiveDescriptors() {
    jcp.dataPrc = getOriginalInputPrecisionAtPort(IN_DATA);
    if (jcp.dataPrc != Precision::I32 && jcp.dataPrc != Precision::I8 && jcp.dataPrc != Precision::U8) {
        jcp.dataPrc = Precision::FP32;
    }
    const InferenceEngine::Precision axisPrecision = Precision::I32;

    impl_desc_type implType = jit_sse42;
    if (x64::mayiuse(x64::avx512_core)) {
        implType = jit_avx512;
    } else if (x64::mayiuse(x64::avx2)) {
        implType = jit_avx2;
    } else if (x64::mayiuse(x64::avx)) {
        implType = jit_avx;
    }

    std::vector<PortConfigurator> inPortConfigs = { {LayoutType::ncsp, jcp.dataPrc} };
    if (!jcp.flattened) {
        inPortConfigs.push_back( {LayoutType::ncsp, axisPrecision} );
    }
    std::vector<PortConfigurator> outPortConfigs;
    for (int i = 0; i < 4; i++) {
        if (jcp.definedOutputs[i]) {
            outPortConfigs.push_back({LayoutType::ncsp, i == 0 ? jcp.dataPrc : axisPrecision});
        }
    }

    addSupportedPrimDesc(inPortConfigs, outPortConfigs, implType, isDynamicNode());
}

void Unique::createPrimitive() {
//    UniqueKernelConfParams jcp;

//    jcp.sorted = sorted;
//    jcp.flattened = flattened;
//    jcp.axis   = axis;
//    for (int i = 0; i < 4; i++) {
//        jcp.definedOutputs[i] = definedOutputs[i];
//    }
//    jcp.dynamicShapes = isDynamicNode();
//    jcp.inDataPrc     = dataPrecision;

//    const auto& srcDataDims = getInputShapeAtPort(IN_DATA).getDims();
//    if (getInputShapeAtPort(IN_DATA).isStatic()) {
//
//    }
//    if (!jcp.dynamicShapes) {
//    } else {
//    }

    if (x64::mayiuse(x64::avx512_core)) {
        kernel.reset(new UniqueKernel<x64::avx512_core>(jcp));
    } else if (x64::mayiuse(x64::avx2)) {
        kernel.reset(new UniqueKernel<x64::avx2>(jcp));
    } else if (x64::mayiuse(x64::avx)) {
        kernel.reset(new UniqueKernel<x64::avx>(jcp));
    } else if (x64::mayiuse(x64::sse41)) {
        kernel.reset(new UniqueKernel<x64::sse41>(jcp));
    }
    if (!kernel) {
        THROW_ERROR << " could not create JIT kernel.";
    }
    kernel->create_ker();

    threadsNum = parallel_get_max_threads();
    execArgsPerThread.resize(threadsNum);
//    if (!x64::mayiuse(x64::avx512_core)) {
//        const auto dataElPerVec = kernel->getDataElPerVec();
//        parallel_nt(threadsNum, [&](const int ithr, const int nthr) {
//            auto& p = execArgsPerThread[ithr];
//
//        });
//    }

    Node::createPrimitive();
}

void Unique::prepareParams() {
    auto& dataMemPtr = getParentEdgeAt(IN_DATA)->getMemoryPtr();
    if (!dataMemPtr || !dataMemPtr->isAllocated()) {
        THROW_ERROR << " has not allocated input data memory.";
    }
    for (int i = 0; i < 4; i++) {
        if (jcp.definedOutputs[i]) {
            auto& dstMemPtr = getChildEdgeAt(i)->getMemoryPtr();
            if (!dstMemPtr || !dstMemPtr->isAllocated()) {
                THROW_ERROR << " has not allocated output memory at port " << i;
            }
        }
    }
    if (getSelectedPrimitiveDescriptor() == nullptr) {
        THROW_ERROR << " has unidentified preferable primitive descriptor.";
    }

    const uint64_t dataElPerVec = kernel->getDataElPerVec();
    const auto& srcDataShape = dataMemPtr->getStaticDims();
    const uint64_t totalWork = jcp.flattened ? std::accumulate(srcDataShape.begin(), srcDataShape.end(), 1, std::multiplies<Dim>()) : srcDataShape[jcp.axis];
    const uint64_t wpt = ((totalWork / dataElPerVec) / threadsNum + 1) * dataElPerVec;

    parallel_nt(threadsNum, [&](const int ithr, const int nthr) {
        const uint64_t dstStart = std::min(wpt * ithr, totalWork);
        const uint64_t dstEnd = std::min(wpt * (ithr + 1), totalWork);
printf("[%d] start: %lu; end: %lu; wa: %lu\n", ithr, dstStart, dstEnd, dstEnd - dstStart);

        auto& arg = execArgsPerThread[ithr];

        arg.workAmount = dstEnd - dstStart;
        if (arg.workAmount == 0)
            return;

        for (int i = 0; i < 4; i++) {
            if (jcp.definedOutputs[i]) {
                arg.dstPtr[i] = getChildEdgeAt(i)->getMemoryPtr()->GetPtr();
            }
        }
    });
}

void Unique::execute(dnnl::stream strm) {
    const void* srcDataPtr = getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetPtr();

// DEBUG
std::cout << "\nINPUT DATA: " << std::endl;
//float* srcDataF = reinterpret_cast<float*>(getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetPtr());
int* srcDataF = reinterpret_cast<int*>(getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetPtr());
for (int i = 0; i < getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetSize() / sizeof(float); i++) {
    if (i % kernel->getDataElPerVec() == 0)
        std::cout << "| ";
    std::cout << srcDataF[i] << "; ";
}
std::cout << std::endl;
// DEBUG

    parallel_nt(threadsNum,  [&](const int ithr, const int nthr) {
        auto& arg = execArgsPerThread[ithr];
        if (arg.workAmount == 0lu) {
            return;
        }
        arg.srcPtr = srcDataPtr;

        (*kernel)(&arg);
    });

//    VectorDims newDims{validOutputs, 3};
//    redefineOutputMemory( {newDims, newDims, {1}} );

// DEBUG
std::cout << "OUTPUT: " << std::endl;
//float* dstDataF = reinterpret_cast<float*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
int* dstDataF = reinterpret_cast<int*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
//char* dstDataF = reinterpret_cast<char*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
for (int i = 0; i < getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetSize() / sizeof(float); i++) {
//for (int i = 0; i < getChildEdgeAt(0)->getMemoryPtr()->GetSize() / sizeof(float); i++) {
    if (i % 4 == 0)
        std::cout << "| ";
    std::cout << dstDataF[i] << "; ";
}
std::cout << std::endl << std::endl;
// DEBUG
}

void Unique::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

void Unique::flattenTensorExec() {
    // Covers both sorted and unsorted.
    // Per thread sorting.
    parallel_nt(threadsNum, [&](const int ithr, const int nthr) {
        const auto& arg = execArgsPerThread[ithr];
        if (arg.workAmount == 0lu) {
            return;
        }

//        arg.src                = srcData;
//        arg.grid               = gridData + p.gridStartB;
//        arg.dst                = dstData  + p.dstStartB;
//        arg.buffer             = p.buffer.data();
//        arg.workAmount         = p.workAmount;

        (*kernel)(&arg);
    });

    // Sort chosen points.
//    (*kernel)(&arg);

    // Choose elements.

    // Blocks exchange

    // Parallel sorting

    // Calculate indices.
}

void Unique::slicedTensorExec() {

}

std::vector<VectorDims> Unique::shapeInfer() const {
    return Node::shapeInferGeneric(PortMask(1)); // TODO: check number
}

bool Unique::created() const {
    return getType() == Type::Unique;
}
