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
    dataTypeSize = jcp.dataPrc.size();
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
//        if (jcp.definedOutputs[i]) {
            outPortConfigs.push_back({LayoutType::ncsp, i == 0 ? jcp.dataPrc : axisPrecision});
//        }
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
    } else if (x64::mayiuse(x64::sse41)) {
        kernel.reset(new UniqueKernel<x64::sse41>(jcp));
    }
    if (!kernel) {
        THROW_ERROR << " could not create JIT kernel.";
    }
    kernel->create_ker();

    threadsNum = 1;//parallel_get_max_threads();
    execArgsPerThread.resize(threadsNum);
    blockLen.resize(threadsNum);
    samples.resize(threadsNum);
    pivots.resize(threadsNum);
    samplesIdx.resize(threadsNum);
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
//        if (jcp.definedOutputs[i]) {
            auto& dstMemPtr = getChildEdgeAt(i)->getMemoryPtr();
            if (!dstMemPtr || !dstMemPtr->isAllocated()) {
                THROW_ERROR << " has not allocated output memory at port " << i;
            }
//        }
    }
    if (getSelectedPrimitiveDescriptor() == nullptr) {
        THROW_ERROR << " has unidentified preferable primitive descriptor.";
    }

//    const uint64_t dataElPerVec = kernel->getDataElPerVec();
    const auto& srcDataShape = dataMemPtr->getStaticDims();
    const int64_t totalWork = jcp.flattened ? std::accumulate(srcDataShape.begin(), srcDataShape.end(), 1, std::multiplies<Dim>()) : srcDataShape[jcp.axis];
    const int64_t blNum = (totalWork - 1) / kernel->getDataElPerBlock() + 1;
    const int64_t wpt = blNum < threadsNum ? totalWork / blNum : totalWork / threadsNum;

    parallel_nt(threadsNum, [&](const int ithr, const int nthr) {
        const uint64_t dstStart = std::min(wpt * ithr, totalWork);
        const uint64_t dstEnd = std::min(wpt * (ithr + 1), totalWork);
printf("[%d] start: %lu; end: %lu; wa: %lu\n", ithr, dstStart, dstEnd, dstEnd - dstStart);

        auto& arg = execArgsPerThread[ithr];

        arg.workAmount = dstEnd - dstStart;
        if (arg.workAmount == 0)
            return;

        for (int i = 0; i < 4; i++) {
//            if (jcp.definedOutputs[i]) {
                arg.dstPtr[i] = getChildEdgeAt(i)->getMemoryPtr()->GetPtr();
//            }
        }

std::string blockLenStr;
        if (arg.workAmount <= kernel->getDataElPerBlock()) {
            arg.blocksNum = 1;
            blockLen[ithr].resize(1, arg.workAmount);
blockLenStr += std::to_string(blockLen[ithr][0]) + "; ";
        } else {
            arg.blocksNum = arg.workAmount / kernel->getDataElPerBlock() + 1;

            const auto minWork = arg.workAmount / arg.blocksNum;
            auto restWork = arg.workAmount % arg.blocksNum;
            blockLen[ithr].resize(arg.blocksNum, minWork);
            for (int i = 0; restWork > 0; restWork--, i++) {
                blockLen[ithr][i]++;
            }
for (int i = 0; i < arg.blocksNum; i++) {
blockLenStr += std::to_string(blockLen[ithr][i]) + "; ";
}
            // SAMPLES
            arg.samplesLen = arg.blocksNum * arg.blocksNum;
samples[ithr].resize(std::max(kernel->getDataElPerVec(), uint64_t(arg.samplesLen)), 0);
arg.samplesPtr = samples[ithr].data();
            // Store samples if not fitted to block.
            if (arg.samplesLen > kernel->getDataElPerBlock()) {
                samples[ithr].resize(arg.samplesLen);
                arg.samplesPtr = samples[ithr].data();
            }
            samplesIdx[ithr].resize(kernel->getDataElPerVec(), 0);
            const auto inc = blockLen[ithr][0] / arg.blocksNum * dataTypeSize;
            for (int i = 1; i < arg.samplesLen && i < samplesIdx[ithr].size(); i++) {
                samplesIdx[ithr][i] = samplesIdx[ithr][i - 1] + inc;
            }
            arg.samplesIdxPtr = samplesIdx[ithr].data();
            arg.samplesIdxStep = inc * kernel->getDataElPerVec();

            // PIVOTS
            const auto pivotsLen = arg.blocksNum;
            // Store pivots if not fitted to block.
            if (pivotsLen > kernel->getDataElPerBlock()) {
                pivots[ithr].resize(pivotsLen);
                arg.pivotsPtr = pivots[ithr].data();
            }
        }
        arg.blockLen = blockLen[ithr].data();

printf("[%d] blocksNum: %lu; blockLen {%s}\n", ithr, arg.blocksNum, blockLenStr.c_str());
    });
}

void Unique::execute(dnnl::stream strm) {
//    const void* srcDataPtr = getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetPtr();
//    const int* srcDataPtr = reinterpret_cast<const int *>(getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetPtr());

// DEBUG
std::cout << "\nINPUT DATA: " << std::endl;
//float* srcDataF = reinterpret_cast<float*>(getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetPtr());
int* srcDataF = reinterpret_cast<int*>(getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetPtr());
//int8_t * srcDataF = reinterpret_cast<int8_t*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
for (int i = 0; i < getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetSize() / sizeof(int); i++) {
    if (i > 0 && i % 4 == 0)
        std::cout << "| ";
    if (i > 0 && i % 16 == 0)
        std::cout << std::endl;
    std::cout << srcDataF[i] << "; ";
}
std::cout << std::endl;
// DEBUG

    if (jcp.flattened) {
        size_t uniqueLen = 1lu;
        if (jcp.dataPrc == Precision::FP32) {
            uniqueLen = flattenTensorExec<PrecisionTrait<Precision::FP32>::value_type>();
        } else if (jcp.dataPrc == Precision::I32) {
            uniqueLen = flattenTensorExec<PrecisionTrait<Precision::I32>::value_type>();
        } else if (jcp.dataPrc == Precision::I8) {
            uniqueLen = flattenTensorExec<PrecisionTrait<Precision::I8>::value_type>();
        } else if (jcp.dataPrc == Precision::U8) {
            uniqueLen = flattenTensorExec<PrecisionTrait<Precision::U8>::value_type>();
        }

        const auto& srcDataShape = getParentEdgeAt(IN_DATA)->getMemoryPtr()->getStaticDims();
        const size_t totalWork = jcp.flattened ? std::accumulate(srcDataShape.begin(), srcDataShape.end(), 1, std::multiplies<Dim>()) : srcDataShape[jcp.axis];
        redefineOutputMemory({ {uniqueLen}, {uniqueLen}, {totalWork}, {uniqueLen}});
    } else {
        size_t uniqueLen = 0lu;
        if (jcp.dataPrc == Precision::FP32) {
            uniqueLen = slicedTensorExec<PrecisionTrait<Precision::FP32>::value_type>();
        } else if (jcp.dataPrc == Precision::I32) {
            uniqueLen = slicedTensorExec<PrecisionTrait<Precision::I32>::value_type>();
        } else if (jcp.dataPrc == Precision::I8) {
            uniqueLen = slicedTensorExec<PrecisionTrait<Precision::I8>::value_type>();
        } else if (jcp.dataPrc == Precision::U8) {
            uniqueLen = slicedTensorExec<PrecisionTrait<Precision::U8>::value_type>();
        }

        const auto& srcDataShape = getParentEdgeAt(IN_DATA)->getMemoryPtr()->getStaticDims();
        const size_t totalWork = jcp.flattened ? std::accumulate(srcDataShape.begin(), srcDataShape.end(), 1, std::multiplies<Dim>()) : srcDataShape[jcp.axis];
//        redefineOutputMemory({ {uniqueLen}, {uniqueLen}, {totalWork}, {uniqueLen}});
    }

// DEBUG
std::cout << "OUTPUT_0: " << std::endl;
//float* dstDataF = reinterpret_cast<float*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
int* dstDataF = reinterpret_cast<int*>(getChildEdgesAtPort(UNIQUE_DATA)[0]->getMemoryPtr()->GetPtr());
//int* dstDataF = reinterpret_cast<int*>(getChildEdgeAt(FIRST_UNIQUE_IDX)->getMemoryPtr()->GetPtr());
//int8_t * dstDataF = reinterpret_cast<int8_t*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
for (int i = 0; i < getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetSize() / sizeof(int); i++) {
//for (int i = 0; i < getChildEdgeAt(0)->getMemoryPtr()->GetSize() / sizeof(float); i++) {
    if (i > 0 && i % 4 == 0)
        std::cout << "| ";
    if (i > 0 && i % 16 == 0)
        std::cout << std::endl;
    std::cout << dstDataF[i] << "; ";
//    std::cout << sorted[i] << "; ";
}
std::cout << std::endl << std::endl;
for (int o = 1; o < 4; o++) {
    if (jcp.definedOutputs[o]) {
        std::cout << "OUTPUT_: " << o << std::endl;
        int *dst1 = reinterpret_cast<int *>(getChildEdgesAtPort(o)[0]->getMemoryPtr()->GetPtr());
        for (int i = 0; i < getChildEdgesAtPort(o)[0]->getMemoryPtr()->GetSize() / sizeof(int); i++) {
            if (i > 0 && i % 4 == 0)
                std::cout << "| ";
            if (i > 0 && i % 16 == 0)
                std::cout << std::endl;
            std::cout << dst1[i] << "; ";
        }
        std::cout << std::endl << std::endl;
    }
}
//for (int ithr = 0; ithr < threadsNum; ithr++) {
//    std::string res;
//    for (int i = 0; i < samples[ithr].size(); i++) {
//        res += std::to_string(samples[ithr][i]) + ";";
//    }
//    printf("[%d] Samples {%s}\n", ithr, res.c_str());
//}
// DEBUG

}

void Unique::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

template <typename T>
size_t Unique::flattenTensorExec() {
    const T* srcDataPtr = reinterpret_cast<const T*>(getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetPtr());
    const auto inputLen = getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetSize() / sizeof(T);
    T* uniqueData = reinterpret_cast<T*>(getChildEdgesAtPort(UNIQUE_DATA)[0]->getMemoryPtr()->GetPtr());

    std::memcpy(uniqueData, srcDataPtr, inputLen * sizeof(T));
    std::sort(uniqueData, uniqueData + inputLen);
    auto last = std::unique(uniqueData, uniqueData + inputLen);
    const int64_t uniqLen = reinterpret_cast<int64_t>(last - uniqueData);

    if (!jcp.sorted) {
        T* first = uniqueData;
        for (int i = 0; i < inputLen; i++) {
            if (i > 0 && srcDataPtr[i - 1] == srcDataPtr[i]) {
                continue;
            }
            for (T* it = first; it < last; it++) {
                if (srcDataPtr[i] == *it) {
                    T val = *first;
                    *first++ = *it;
                    *it = val;
                    break;
                }
            }
            if (first >= last) {
                break;
            }
        }
    }
    if (jcp.definedOutputs[FIRST_UNIQUE_IDX]) {
        int* idxPtr = reinterpret_cast<int*>(getChildEdgesAtPort(FIRST_UNIQUE_IDX)[0]->getMemoryPtr()->GetPtr());
        T* first = uniqueData;
        for (int i = 0; i < inputLen; i++) {
            if (i > 0 && srcDataPtr[i - 1] == srcDataPtr[i]) {
                continue;
            }
            for (T* it = first; it < last; it++) {
                if (srcDataPtr[i] == *it) {
                    *idxPtr++ = i;
                    first++;
                    break;
                }
            }
            if (first >= last) {
                break;
            }
        }
    }
    if (jcp.definedOutputs[INPUT_TO_UNIQ_IDX]) {
        int* idxPtr = reinterpret_cast<int*>(getChildEdgesAtPort(INPUT_TO_UNIQ_IDX)[0]->getMemoryPtr()->GetPtr());
        for (int i = 0; i < inputLen; i++) {
            if (i > 0 && srcDataPtr[i] == srcDataPtr[i - 1]) {
                idxPtr[i] = idxPtr[i - 1];
                continue;
            }
            for (int j = 0; j < uniqLen; j++) {
                if (srcDataPtr[i] == uniqueData[j]) {
                    idxPtr[i] = j;
                    break;
                }
            }
        }
    }
    if (jcp.definedOutputs[OCCURRENCES_NUM]) {
        int* idxPtr = reinterpret_cast<int*>(getChildEdgesAtPort(OCCURRENCES_NUM)[0]->getMemoryPtr()->GetPtr());
        std::fill(idxPtr, idxPtr + uniqLen, 0);
        for (int j = 0; j < uniqLen; j++) {
            for (int i = 0; i < inputLen; i++) {
                if (srcDataPtr[i] == uniqueData[j]) {
                    idxPtr[j]++;
                }
            }
        }
    }

    return uniqLen;
}

template <typename T>
size_t Unique::slicedTensorExec() {
    const T* srcDataPtr = reinterpret_cast<const T*>(getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetPtr());
    const auto inputLen = getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetSize() / sizeof(T);
    T* uniqueData = reinterpret_cast<T*>(getChildEdgesAtPort(UNIQUE_DATA)[0]->getMemoryPtr()->GetPtr());
    int *firstPtr, *inToOutPtr, *occurPtr;
    if (jcp.definedOutputs[FIRST_UNIQUE_IDX]) {
        firstPtr = reinterpret_cast<int*>(getChildEdgesAtPort(FIRST_UNIQUE_IDX)[0]->getMemoryPtr()->GetPtr());
    }
    if (jcp.definedOutputs[INPUT_TO_UNIQ_IDX]) {
        inToOutPtr = reinterpret_cast<int*>(getChildEdgesAtPort(INPUT_TO_UNIQ_IDX)[0]->getMemoryPtr()->GetPtr());
    }
    if (jcp.definedOutputs[OCCURRENCES_NUM]) {
        occurPtr = reinterpret_cast<int*>(getChildEdgesAtPort(OCCURRENCES_NUM)[0]->getMemoryPtr()->GetPtr());
    }

    const auto& srcDataShape = getParentEdgeAt(IN_DATA)->getMemoryPtr()->getStaticDims();

    const auto cmpBlNum = srcDataShape[jcp.axis]; // Blocks to compare.
    const auto partsInBl = std::accumulate(srcDataShape.begin(), srcDataShape.begin() + jcp.axis - 1, 1, std::multiplies<Dim>()); // Parts in block.
    const auto elPerPart = std::accumulate(srcDataShape.begin() + jcp.axis + 1, srcDataShape.end(), 1, std::multiplies<Dim>()); // Elements in part.
    const auto partLen = elPerPart * jcp.dataPrc.size();
    const auto partStep = elPerPart * cmpBlNum;

    auto first1 = srcDataPtr;
    auto first2 = uniqueData;
    for (int p = 0; p < partsInBl; p++) {
        memcpy(first2, first1, partLen);
        first1 += partStep;
        first2 += partStep;
    }
    if (jcp.definedOutputs[FIRST_UNIQUE_IDX]) {
        firstPtr[0] = 0;
    }
    if (jcp.definedOutputs[INPUT_TO_UNIQ_IDX]) {
        inToOutPtr[0] = 0;
    }
    if (jcp.definedOutputs[OCCURRENCES_NUM]) {
        occurPtr[0] = 1;
        std::fill(occurPtr, occurPtr + inputLen, 1);
    }

    const T* last1;
    size_t uniqLen = 1;
    for (int b1 = 1; b1 < cmpBlNum; b1++) {
        first1 = srcDataPtr + b1 * partLen;
        last1 = srcDataPtr + (b1 + 1) * partLen;
        bool equal = true;
        int b2 = 0;
        for (; b2 < uniqLen; b2++) {
            first2 = uniqueData + b2 * partLen;
            equal = true;
            for (int p = 0; p < partsInBl; p++) {
                equal = std::equal(first1, last1, first2);
                if (!equal) {
                    break;
                }
            }
            if (equal) {
                break;
            }
        }
        if (!equal) {
            first2 = uniqueData + uniqLen * partLen;
            for (int p = 0; p < partsInBl; p++) {
                memcpy(first2, first1, partLen);
                first1 += partStep;
                first2 += partStep;
            }

            if (jcp.definedOutputs[FIRST_UNIQUE_IDX]) {
                firstPtr[uniqLen ] = b1;
            }
            if (jcp.definedOutputs[INPUT_TO_UNIQ_IDX]) {
                inToOutPtr[b1] = uniqLen;
            }

            uniqLen++;
        } else if (jcp.definedOutputs[OCCURRENCES_NUM]) {
            occurPtr[b2]++;
        }
    }

    if (jcp.sorted) {
        for (int p = 0; p < partsInBl; p++) {
            for (int e = 0; e < elPerPart; e++) {
                for (int e = 0; e < elPerPart; e++) {

                }
            }
        }
    }

    return uniqLen;
}

std::vector<VectorDims> Unique::shapeInfer() const {
    return Node::shapeInferGeneric(PortMask(1)); // TODO: check number
}

bool Unique::created() const {
    return getType() == Type::Unique;
}
