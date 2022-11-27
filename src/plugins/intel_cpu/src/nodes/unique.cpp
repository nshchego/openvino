// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "unique.hpp"
#include <ngraph/opsets/opset1.hpp>

using namespace InferenceEngine;
using namespace ov::intel_cpu;
using namespace ov::intel_cpu::node;

#define THROW_ERROR IE_THROW() << getTypeStr() << " node with name '" << getName() << "' "

bool Unique::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!ov::is_type<op::v10::Unique>(op)) {
            errorMessage = "Not supported Unique operation version. CPU plug-in supports only 10th version.";
            return false;
        }
        if (op->get_input_size() > AXIS && !ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(AXIS))) {
            errorMessage = "CPU plug-in supports only constant Axis input.";
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
        definedOutputs[i] = !op->get_output_target_inputs(i).empty();
    }

    sorted = ov::as_type_ptr<ov::op::v10::Unique>(op)->get_sorted();
    if (op->get_input_size() > AXIS) {
        flattened = false;
        axis = ov::as_type<ov::op::v0::Constant>(op->get_input_node_ptr(AXIS))->cast_vector<int>()[0];
        if (axis < 0) {
            axis += op->get_input_partial_shape(IN_DATA).rank().get_length();
        }
        if (axis < 0 || axis >= op->get_input_partial_shape(IN_DATA).rank().get_length()) {
            THROW_ERROR << "has invalid axis value: " << ov::as_type<ov::op::v0::Constant>(op->get_input_node_ptr(AXIS))->cast_vector<int>()[0];
        }
    } else {
        flattened = true;
    }
}

void Unique::initSupportedPrimitiveDescriptors() {
    dataPrecision = getOriginalInputPrecisionAtPort(IN_DATA);
    if (dataPrecision != Precision::I32 && dataPrecision != Precision::I8 && dataPrecision != Precision::U8) {
        dataPrecision = Precision::FP32;
    }
    dataTypeSize = dataPrecision.size();
    const InferenceEngine::Precision axisPrecision = Precision::I32;

    impl_desc_type implType = ref;

    std::vector<PortConfigurator> inPortConfigs = { {LayoutType::ncsp, dataPrecision} };
    if (!flattened) {
        inPortConfigs.push_back({LayoutType::ncsp, axisPrecision});
    }
    std::vector<PortConfigurator> outPortConfigs;
    for (int i = 0; i < 4; i++) {
        outPortConfigs.push_back({LayoutType::ncsp, i == 0 ? dataPrecision : axisPrecision});
    }

    addSupportedPrimDesc(inPortConfigs, outPortConfigs, implType, isDynamicNode());
}

void Unique::createPrimitive() {
    Node::createPrimitive();
}

void Unique::prepareParams() {
    auto& dataMemPtr = getParentEdgeAt(IN_DATA)->getMemoryPtr();
    if (!dataMemPtr || !dataMemPtr->isAllocated()) {
        THROW_ERROR << " has not allocated input data memory.";
    }
    for (int i = 0; i < 4; i++) {
        if (definedOutputs[i]) {
            auto& dstMemPtr = getChildEdgeAt(i)->getMemoryPtr();
            if (!dstMemPtr || !dstMemPtr->isAllocated()) {
                THROW_ERROR << " has not allocated output memory at port " << i;
            }
        }
    }
    if (getSelectedPrimitiveDescriptor() == nullptr) {
        THROW_ERROR << " has unidentified preferable primitive descriptor.";
    }
}

void Unique::execute(dnnl::stream strm) {
// DEBUG
std::cout << "\nINPUT DATA: " << std::endl;
float* srcDataF = reinterpret_cast<float*>(getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetPtr());
// int* srcDataF = reinterpret_cast<int*>(getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetPtr());
// int8_t * srcDataF = reinterpret_cast<int8_t*>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
for (int i = 0; i < getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetSize() / sizeof(int); i++) {
    if (i > 0 && i % 4 == 0)
        std::cout << "| ";
    if (i > 0 && i % 16 == 0)
        std::cout << std::endl;
    std::cout << int(srcDataF[i]) << "; ";
}
std::cout << std::endl;
// DEBUG

    if (flattened) {
        size_t uniqueLen = 1lu;
        if (dataPrecision == Precision::FP32) {
            uniqueLen = flattenTensorExec<PrecisionTrait<Precision::FP32>::value_type>();
        } else if (dataPrecision == Precision::I32) {
            uniqueLen = flattenTensorExec<PrecisionTrait<Precision::I32>::value_type>();
        } else if (dataPrecision == Precision::I8) {
            uniqueLen = flattenTensorExec<PrecisionTrait<Precision::I8>::value_type>();
        } else if (dataPrecision == Precision::U8) {
            uniqueLen = flattenTensorExec<PrecisionTrait<Precision::U8>::value_type>();
        }

        const size_t srcLen = getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetSize() / dataPrecision.size();
        redefineOutputMemory({ {uniqueLen}, {uniqueLen}, {srcLen}, {uniqueLen}});
    } else {
        size_t uniqueLen = 1lu;
        if (dataPrecision == Precision::FP32) {
            uniqueLen = slicedTensorExec<PrecisionTrait<Precision::FP32>::value_type>();
        } else if (dataPrecision == Precision::I32) {
            uniqueLen = slicedTensorExec<PrecisionTrait<Precision::I32>::value_type>();
        } else if (dataPrecision == Precision::I8) {
            uniqueLen = slicedTensorExec<PrecisionTrait<Precision::I8>::value_type>();
        } else if (dataPrecision == Precision::U8) {
            uniqueLen = slicedTensorExec<PrecisionTrait<Precision::U8>::value_type>();
        }

        auto dstDataShape = getParentEdgeAt(IN_DATA)->getMemoryPtr()->getStaticDims();
        const size_t srcLen = dstDataShape[axis];
        dstDataShape[axis] = uniqueLen;
        redefineOutputMemory({ dstDataShape, {uniqueLen}, {srcLen}, {uniqueLen}});
    }

// DEBUG
std::cout << "OUTPUT_0: " << std::endl;
float* dstDataF = reinterpret_cast<float*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
// int* dstDataF = reinterpret_cast<int*>(getChildEdgesAtPort(UNIQUE_DATA)[0]->getMemoryPtr()->GetPtr());
//int* dstDataF = reinterpret_cast<int*>(getChildEdgeAt(FIRST_UNIQUE_IDX)->getMemoryPtr()->GetPtr());
// int8_t * dstDataF = reinterpret_cast<int8_t*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
for (int i = 0; i < getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetSize() / sizeof(int); i++) {
//for (int i = 0; i < getChildEdgeAt(0)->getMemoryPtr()->GetSize() / sizeof(float); i++) {
    if (i > 0 && i % 4 == 0)
        std::cout << "| ";
    if (i > 0 && i % 16 == 0)
        std::cout << std::endl;
    std::cout << int(dstDataF[i]) << "; ";
//    std::cout << sorted[i] << "; ";
}
std::cout << std::endl << std::endl;
for (int o = 1; o < 4; o++) {
   if (definedOutputs[o]) {
       std::cout << "OUTPUT_" << o << ": " << std::endl;
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
// DEBUG
}

void Unique::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

template <typename T>
size_t Unique::flattenTensorExec() {
    const T* srcDataPtr = reinterpret_cast<const T*>(getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetPtr());
    const int64_t inputLen = getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetSize() / sizeof(T);
    T* uniqueData = reinterpret_cast<T*>(getChildEdgesAtPort(UNIQUE_DATA)[0]->getMemoryPtr()->GetPtr());
    int *firstPtr, *inToOutPtr, *occurPtr;
    if (definedOutputs[FIRST_UNIQUE_IDX]) {
        firstPtr = reinterpret_cast<int*>(getChildEdgesAtPort(FIRST_UNIQUE_IDX)[0]->getMemoryPtr()->GetPtr());
    }
    if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
        inToOutPtr = reinterpret_cast<int*>(getChildEdgesAtPort(INPUT_TO_UNIQ_IDX)[0]->getMemoryPtr()->GetPtr());
    }
    if (definedOutputs[OCCURRENCES_NUM]) {
        occurPtr = reinterpret_cast<int*>(getChildEdgesAtPort(OCCURRENCES_NUM)[0]->getMemoryPtr()->GetPtr());
    }
    int64_t uniqLen = inputLen;

    if (sorted) {
        std::memcpy(uniqueData, srcDataPtr, inputLen * sizeof(T));
        std::sort(uniqueData, uniqueData + inputLen);
        auto last = std::unique(uniqueData, uniqueData + inputLen);
        uniqLen = last - uniqueData;

        if (definedOutputs[FIRST_UNIQUE_IDX]) {
            T* first = uniqueData;
            for (T* it = first; it < last; it++) {
                for (int i = 0; i < inputLen; i++) {
                    if (srcDataPtr[i] == *it) {
                        *firstPtr++ = i;
                        first++;
                        break;
                    }
                }
            }
        }
        if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
            for (int i = 0; i < inputLen; i++) {
                if (i > 0 && srcDataPtr[i] == srcDataPtr[i - 1]) {
                    inToOutPtr[i] = inToOutPtr[i - 1];
                    continue;
                }
                for (int j = 0; j < uniqLen; j++) {
                    if (srcDataPtr[i] == uniqueData[j]) {
                        inToOutPtr[i] = j;
                        break;
                    }
                }
            }
        }
        if (definedOutputs[OCCURRENCES_NUM]) {
            std::fill(occurPtr, occurPtr + uniqLen, 0);
            for (int j = 0; j < uniqLen; j++) {
                for (int i = 0; i < inputLen; i++) {
                    if (srcDataPtr[i] == uniqueData[j]) {
                        occurPtr[j]++;
                    }
                }
            }
        }
    } else {
        uniqueData[0] = srcDataPtr[0];
        if (definedOutputs[FIRST_UNIQUE_IDX]) {
            firstPtr[0] = 0;
        }
        if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
            inToOutPtr[0] = 0;
        }
        if (definedOutputs[OCCURRENCES_NUM]) {
            std::fill(occurPtr, occurPtr + inputLen, 1);
        }
        uniqLen = 1;

         for (int i = 1; i < inputLen; i++) {
             bool found = false;
             int j = 0;
             for (; j < uniqLen; j++) {
                 if (uniqueData[j] == srcDataPtr[i]) {
                     found = true;
                     break;
                 }
             }
             if (!found) {
                 uniqueData[uniqLen] = srcDataPtr[i];

                 if (definedOutputs[FIRST_UNIQUE_IDX]) {
                     firstPtr[uniqLen] = i;
                 }
                 if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
                     inToOutPtr[i] = j;
                 }

                 uniqLen++;
             } else {
                 if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
                     inToOutPtr[i] = j;
                 }
                 if (definedOutputs[OCCURRENCES_NUM]) {
                     occurPtr[j]++;
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
    if (definedOutputs[FIRST_UNIQUE_IDX]) {
        firstPtr = reinterpret_cast<int*>(getChildEdgesAtPort(FIRST_UNIQUE_IDX)[0]->getMemoryPtr()->GetPtr());
    }
    if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
        inToOutPtr = reinterpret_cast<int*>(getChildEdgesAtPort(INPUT_TO_UNIQ_IDX)[0]->getMemoryPtr()->GetPtr());
    }
    if (definedOutputs[OCCURRENCES_NUM]) {
        occurPtr = reinterpret_cast<int*>(getChildEdgesAtPort(OCCURRENCES_NUM)[0]->getMemoryPtr()->GetPtr());
    }

    const auto& srcDataShape = getParentEdgeAt(IN_DATA)->getMemoryPtr()->getStaticDims();

    const auto cmpBlNum = srcDataShape[axis]; // Blocks to compare.
    int64_t partsInBl = 1; // Parts in block
    if (axis > 0) {
        partsInBl = std::accumulate(srcDataShape.begin(), srcDataShape.begin() + axis, 1, std::multiplies<Dim>());
    }
    int64_t elPerPart = 1; // Elements number in part.
    if (axis < srcDataShape.size() - 1) {
        elPerPart = std::accumulate(srcDataShape.begin() + axis + 1, srcDataShape.end(), 1, std::multiplies<Dim>());
    }
    const auto partLenB = elPerPart * dataPrecision.size();
    const auto partStep = elPerPart * cmpBlNum;
std::cout << "cmpBlNum: " << cmpBlNum << "; partsInBl: " << partsInBl << "; elPerPart: " << elPerPart
<< "; partLenB: " << partLenB << "; partStep: " << partStep << std::endl;

    if (definedOutputs[FIRST_UNIQUE_IDX]) {
        firstPtr[0] = 0;
    }
    if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
        inToOutPtr[0] = 0;
    }
    if (definedOutputs[OCCURRENCES_NUM]) {
        occurPtr[0] = 1;
        std::fill(occurPtr, occurPtr + inputLen, 1);
    }

    size_t uniqLen = 1;
    std::vector<int64_t> uniqIdx(cmpBlNum, 0);
    for (int b1 = 1; b1 < cmpBlNum; b1++) {
        auto first1 = srcDataPtr + b1 * elPerPart;
        auto last1 = srcDataPtr + (b1 + 1) * elPerPart;
        bool equal = true;
        int b2 = 0;
        // Compare with unique blocks.
        for (; b2 < uniqLen; b2++) {
            auto first2 = srcDataPtr + uniqIdx[b2] * elPerPart;
            equal = true;
            for (int p = 0; p < partsInBl; p++) {
                equal = std::equal(first1, last1, first2);
                if (!equal) {
                    break;
                }
                first2 += partStep;
            }
            if (equal) {
                break;
            }
        }
        if (!equal) {
            if (definedOutputs[FIRST_UNIQUE_IDX]) {
                firstPtr[uniqLen] = b1;
            }
            if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
                inToOutPtr[b1] = b2;
            }

            uniqIdx[uniqLen++] = b1;
        } else {
            if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
                inToOutPtr[b1] = b2;
            }
            if (definedOutputs[OCCURRENCES_NUM]) {
                occurPtr[b2]++;
            }
        }
    }

    const auto dstPrtStep = elPerPart * uniqLen;
    for (int b1 = 0; b1 < uniqLen; b1++) {
        auto first1 = srcDataPtr + uniqIdx[b1] * elPerPart;
        auto first2 = uniqueData + b1 * elPerPart;
        for (int p = 0; p < partsInBl; p++) {
            memcpy(first2, first1, partLenB);
            first1 += partStep;
            first2 += dstPrtStep;
        }
    }

    if (sorted) {
        const auto elInBl = elPerPart * partsInBl;
        struct OrdEl {
            T val;
            int64_t idx;
        };

        std::vector<OrdEl> colToSort(cmpBlNum);
        std::vector<T> buff(elPerPart);
        for (int64_t p = partsInBl - 1; p >= 0; p--) {
            for (int64_t e = elPerPart - 1; e >= 0 ; e--) {
                int64_t pos1 = p * dstPrtStep + e;
                for (int64_t i = 0; i < cmpBlNum; i++) {
                    int64_t pos2 = i * elInBl + pos1;
                    colToSort[i] = {uniqueData[pos2], i};
                }
                std::stable_sort(colToSort.begin(), colToSort.end(), [](const OrdEl &el1, const OrdEl &el2) { return el1.val < el2.val; });

                // perm
                for (int64_t i = 0; i < cmpBlNum; i++) {
                    if (colToSort[i].idx == i) {
                        continue;
                    }
                    T* dst = uniqueData + i * elPerPart;
                    T* src = uniqueData + colToSort[i].idx * elPerPart;
                    for (int p = 0; p < partsInBl; p++) {
                        memcpy(buff.data(), dst, partLenB);
                        memcpy(dst, src, partLenB);
                        memcpy(src, buff.data(), partLenB);
                        dst += dstPrtStep;
                        src += dstPrtStep;
                    }
                    colToSort[colToSort[i].idx].idx = colToSort[i].idx;

                    if (definedOutputs[FIRST_UNIQUE_IDX]) {
                        auto idx = firstPtr[colToSort[i].idx];
                        firstPtr[colToSort[i].idx] = firstPtr[i];
                        firstPtr[i] = idx;
                    }
                    if (definedOutputs[INPUT_TO_UNIQ_IDX]) {
                        auto idx = inToOutPtr[colToSort[i].idx];
                        inToOutPtr[colToSort[i].idx] = inToOutPtr[i];
                        inToOutPtr[i] = idx;
                    }
                    if (definedOutputs[OCCURRENCES_NUM]) {
                        auto idx = occurPtr[colToSort[i].idx];
                        occurPtr[colToSort[i].idx] = occurPtr[i];
                        occurPtr[i] = idx;
                    }
                }
            }
        }
    }

    return uniqLen;
}
