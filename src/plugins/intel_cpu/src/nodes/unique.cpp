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

    threadsNum = 1;//parallel_get_max_threads();
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
//int8_t * srcDataF = reinterpret_cast<int8_t*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
for (int i = 0; i < getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetSize() / sizeof(int); i++) {
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

//int* dstDataI = reinterpret_cast<int*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
//const int N = getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetSize() / sizeof(int);
//const int P = N / 16;
//std::vector<int> samples1;
//std::vector<int> aux1idx;
//
//for (int j = 0; j < P; j++) {
//    int k = j * (N / P);
//    for (int i = 0; i < P; i++) {
//        aux1idx.push_back(k + i * N / (P * P));
//        samples1.push_back(dstDataI[k + i * N / (P * P)]);
//    }
//}
//std::vector<int> aux1idxidx(P * P, 0);
//for (int i = 1; i < aux1idxidx.size(); i ++) {
//    aux1idxidx[i] = aux1idxidx[i - 1] + 1;
//}
//
//// sort
//for (int j = 1; j < samples1.size(); j++) {
//    int key = samples1[j];
//    int keyIdx = aux1idxidx[j];
//    int i = j - 1;
//    while (i >= 0 && samples1[i] > key) {
//        samples1[i + 1] = samples1[i];
//        aux1idxidx[i + 1] = aux1idxidx[i];
//        i = i - 1;
//    }
//    samples1[i + 1] = key;
//    aux1idxidx[i + 1] = keyIdx;
//}
//
//std::vector<int> aux2;
//std::vector<int> aux2idx;
//for (int i = 1; i < P; i++) {
//    int idx = i * P + P / 2 - 1;
//    if (idx >= samples1.size())
//        break;
//    aux2idx.push_back(aux1idxidx[i * P + P / 2 - 1]);
//    aux2.push_back(samples1[i * P + P / 2 - 1]);
//}
//
//std::vector<int> sorted(getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetSize() / sizeof(int));
////int k = 0, l = 0, m = 0;
////for (int i = 0; i < aux1idxidx.size(); i++) {
////    auto start = aux1idx[aux1idxidx[i]];
////    auto end = aux1idxidx[i] + 1 < aux1idx.size() ? aux1idx[aux1idxidx[i] + 1] : sorted.size() - 1;
////    if (l < aux2idx.size() && aux1idxidx.size() < (i + 1) && aux2idx[l] == aux1idxidx[i + 1]) {
////        end++;
////        l++;
////    }
////    if (m < aux2idx.size() && aux1idxidx.size() < (i + 1) && aux2idx[m] == aux1idxidx[i]) {
////        start++;
////        m++;
////    }
////    for (int j = start; j < end; j++, k++) {
////        if (j >= sorted.size() || k >= sorted.size()) {
////            break;
////        }
////        sorted[k] = dstDataI[j];
////    }
////}
////for (int l = 0; l < aux2.size(); l++)
//std::vector<std::vector<int>> bounds(std::vector<std::vector<int>>(P, std::vector<int>(P, 0)));
//for (int ps = 0; ps < P; ps++) {
//    bounds[0][ps] = ps * N / P;
//}
//std::vector<std::vector<int>> finalBounds(std::vector<std::vector<int>>(P, std::vector<int>(2, 0)));

//int s = 0, k = 0, ps = 0;
//for (int pd = 0; pd < P; pd++) {
////    int start = 0, end = aux2[l];
////    int l = 0;
//    ps = 0;
//    s = bounds[pd][0];
//std::cout << "s[" << pd << "][" << 0 << "]=" << s << std::endl;
//    finalBounds[pd][0] = k;
//    for (int j = 0; j < N && s < sorted.size() && k < sorted.size(); j++) {
//        if (pd < aux2.size() && dstDataI[s] > aux2[pd]) {
//            ps++;
//            bounds[pd + 1][ps - 1] = s;
//std::cout << "bounds[" << pd + 1 << "][" << ps - 1 << "]=" <<  bounds[pd + 1][ps - 1] << std::endl;
//            if (ps == P) {
////                sorted[k++] = dstDataI[s++];
//                break;
//            }
//            s = bounds[pd][ps];
//std::cout << "s[" << pd << "][" << ps << "]=" << s << std::endl;
//        }
//        if (pd == P - 1 && s == (ps + 1) * N / P) {
//            ps++;
//            s = bounds[pd][ps];
//std::cout << "s[" << pd << "][" << ps << "]=" << s << std::endl;
//        }
//        sorted[k++] = dstDataI[s++];
//    }
//    finalBounds[pd][1] = k;
//}

//for (int p = 0; p < P; p++) {
////    const int start = i * N / (P * P);
////    const int end = (i + 1) * N / (P * P);
//    for (int j = finalBounds[p][0] + 1; j < finalBounds[p][1]; j++) {
//        int key = sorted[j];
//        int i = j - 1;
//        while (i >= finalBounds[p][0] && sorted[i] > key) {
//            sorted[i + 1] = sorted[i];
//            i = i - 1;
//        }
//        sorted[i + 1] = key;
//    }
//}

//    VectorDims newDims{validOutputs, 3};
//    redefineOutputMemory( {newDims, newDims, {1}} );

// DEBUG
std::cout << "OUTPUT: " << std::endl;
//float* dstDataF = reinterpret_cast<float*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
int* dstDataF = reinterpret_cast<int*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
//int8_t * dstDataF = reinterpret_cast<int8_t*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
for (int i = 0; i < getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetSize() / sizeof(int); i++) {
//for (int i = 0; i < getChildEdgeAt(0)->getMemoryPtr()->GetSize() / sizeof(float); i++) {
    if (i % 4 == 0)
        std::cout << "| ";
    std::cout << dstDataF[i] << "; ";
//    std::cout << sorted[i] << "; ";
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
