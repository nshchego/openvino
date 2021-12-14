// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "ie_parallel.hpp"
#include "mkldnn_gather_node.h"
#include <ngraph/opsets/opset1.hpp>
#include "common/cpu_memcpy.h"
#include <utils/general_utils.h>
#include "kernels/gather_uni_kernel.hpp"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn::impl::cpu;

bool MKLDNNGatherNode::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(),
                ov::op::v7::Gather::get_type_info_static(),
                ov::op::v8::Gather::get_type_info_static())) {
            errorMessage = "Not supported Gather operation version. CPU plug-in supports only 7 and 8 versions.";
            return false;
        }

        if (!isDynamicNgraphNode(op) && !ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(GATHER_AXIS))) {
            errorMessage = "Only Constant operation on 'axis' input is supported for static node.";
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}

MKLDNNGatherNode::MKLDNNGatherNode(const std::shared_ptr<ov::Node>& op, const mkldnn::engine& eng,
        MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache), batchDims(0) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }
    errorPrefix = std::string("Layer Gather with name '") + op->get_friendly_name() + "' ";

    if (op->get_input_size() != 3 || op->get_output_size() != 1)
        IE_THROW() << errorPrefix << "has incorrect number of input/output edges!";

    dataSrcRank = inputShapes[GATHER_DATA].getRank(); // TODO: class member?
    const auto indicesRank = inputShapes[GATHER_INDICES].getRank();
    if (dataSrcRank == 0lu || indicesRank == 0lu)
        IE_THROW() << errorPrefix << "has incorrect input parameters ranks.";

    if (ov::is_type<ov::op::v8::Gather>(op)) {
        batchDims = static_cast<int>(ov::as_type_ptr<ov::op::v8::Gather>(op)->get_batch_dims());
        reverseIndexing = true;
        // TODO: remove this WA when NMS & Gather will support dynamic shape.
        if (!op->get_input_element_type(1).is_signed())
            reverseIndexing = false;
    } else if (ov::is_type<ov::op::v7::Gather>(op)) {
        batchDims = static_cast<int>(ov::as_type_ptr<ov::op::v7::Gather>(op)->get_batch_dims());
        reverseIndexing = false;
    }

    if (batchDims < 0)
        batchDims += indicesRank;
    if (batchDims < 0 || batchDims >= std::min(static_cast<int>(dataSrcRank), static_cast<int>(indicesRank)))
        IE_THROW() << errorPrefix << "has incorrect batch_dims " << batchDims << "!";

    if (ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(GATHER_AXIS))) {
        isAxisInputConst = true;
        axis = ov::as_type<ov::op::v0::Constant>(op->get_input_node_ptr(GATHER_AXIS))->cast_vector<int>()[0];
        if (axis < 0)
            axis += dataSrcRank;
        if (axis < 0 || axis >= dataSrcRank || batchDims > axis)
            IE_THROW() << errorPrefix << "has incorrect input parameter axis value: " << axis;
    }

    dataTypeSize = getOriginalInputPrecisionAtPort(GATHER_DATA).size();
}

void MKLDNNGatherNode::initSupportedPrimitiveDescriptors() {
std::cout << "MKLDNNGatherNode::initSupportedPrimitiveDescriptors()" << std::endl;
    if (!supportedPrimitiveDescriptors.empty())
        return;

    Precision dataPrecision = getOriginalInputPrecisionAtPort(GATHER_DATA);
    addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                          {LayoutType::ncsp, Precision::I32},
                          {LayoutType::ncsp, Precision::I32, isAxisInputConst}},
                         {{LayoutType::ncsp, dataPrecision}},
                         impl_desc_type::ref_any);
}

void MKLDNNGatherNode::prepareParams() {
std::cout << "MKLDNNGatherNode::prepareParams()" << std::endl;
    auto& srcMemPtr = getParentEdgeAt(GATHER_DATA)->getMemoryPtr();
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        IE_THROW() << errorPrefix << " has not allocated input memory.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW() << errorPrefix << " has unidentified preferable primitive descriptor.";

    const auto& dataDims = srcMemPtr->getStaticDims();
    const auto& idxDims = getParentEdgeAt(GATHER_INDICES)->getMemory().getStaticDims();

    if (!isAxisInputConst) {
        axis = (reinterpret_cast<const int32_t*>(getParentEdgeAt(GATHER_AXIS)->getMemoryPtr()->GetPtr()))[0];
        if (axis < 0)
            axis += dataSrcRank;
        if (axis < 0 || axis >= dataSrcRank || batchDims > axis)
            IE_THROW() << errorPrefix << "has incorrect input parameter axis value: " << axis;
    }

    axisDim = dataDims[axis];
    beforeBatchSize = std::accumulate(dataDims.begin(), dataDims.begin() + batchDims, 1lu, std::multiplies<uint64_t>());
    betweenBatchAndAxisSize = std::accumulate(dataDims.begin() + batchDims, dataDims.begin() + axis, 1lu, std::multiplies<uint64_t>());
    afterAxisSize = std::accumulate(dataDims.begin() + axis + 1, dataDims.end(), 1lu, std::multiplies<uint64_t>());
    specIndicesSize = std::accumulate(idxDims.begin() + batchDims, idxDims.end(), 1lu, std::multiplies<uint64_t>());
    afterAxisSizeInBytes = afterAxisSize * dataTypeSize;
    axisAndAfterAxisSizeInBytes = axisDim * afterAxisSizeInBytes;
    srcAfterBatchSizeInBytes = betweenBatchAndAxisSize * axisAndAfterAxisSizeInBytes;
    totalWork = beforeBatchSize * betweenBatchAndAxisSize * specIndicesSize;

//    if (!isDynamic) { // && afterAxisSize == 1 && specIndicesSize < idxElPerVec) {
    if (jitKernel && !isDynamicNode()) {
        const uint64_t dataElPerVec = jitKernel->getDataElPerVec();
        const uint64_t nthr = parallel_get_max_threads();
        const uint64_t wpt = ((totalWork / dataElPerVec) / nthr + 1) * dataElPerVec;
        shortPermIdx        = std::vector<std::vector<int>>(nthr, std::vector<int>(dataElPerVec));
        shortBeforeAxisDiff = std::vector<std::vector<int>>(nthr, std::vector<int>(dataElPerVec));
        specIndicesInBytes  = std::vector<std::vector<int>>(nthr, std::vector<int>(dataElPerVec));
        idxBatchSumInBytes  = std::vector<std::vector<int>>(nthr, std::vector<int>(dataElPerVec));
        dataBeforeAxisSumBPerTr = std::vector<std::vector<int>>(nthr, std::vector<int>(dataElPerVec));
        betweenBatchAndAxisIters = std::vector<int>(nthr, 0);
        for (uint64_t ithr = 0lu; ithr < nthr; ithr++) {
            uint64_t start = std::min(wpt * ithr, totalWork);
            initShortParams(shortPermIdx[ithr], shortBeforeAxisDiff[ithr], start);
            betweenBatchAndAxisIters[ithr] = (start / specIndicesSize) % betweenBatchAndAxisSize;
            for (uint64_t j = 0lu; j < dataElPerVec; j++, start++) {
                specIndicesInBytes[ithr][j] = (start % specIndicesSize) * idxTypeSize;
                idxBatchSumInBytes[ithr][j] = ((start / specIndicesSize) / betweenBatchAndAxisSize) * specIndicesSize * idxTypeSize;
                dataBeforeAxisSumBPerTr[ithr][j] = ((start / specIndicesSize) % betweenBatchAndAxisSize) * axisAndAfterAxisSizeInBytes +
                        srcAfterBatchSizeInBytes * idxBatchSumInBytes[ithr][j] / idxTypeSize / specIndicesSize;
            }
        }
    }
}

bool MKLDNNGatherNode::needPrepareParams() const {
    bool result = inputShapesModified();
    if (!isAxisInputConst)
        result = result || axis != (reinterpret_cast<const int32_t*>(getParentEdgeAt(GATHER_AXIS)->getMemoryPtr()->GetPtr()))[0];
    return result;
}

void MKLDNNGatherNode::createPrimitive() {
std::cout << "MKLDNNGatherNode::createPrimitive()" << std::endl;
    // Gather instruction is not supported by SSE.
    if ((x64::mayiuse(x64::avx512_common) || x64::mayiuse(x64::avx2))) {
        jGatherConfParams jcp;
        jcp.dataTypeSize = dataTypeSize;
        jcp.reverseIndexing = reverseIndexing;
        jcp.dynamicShapes = isDynamicNode();
        if (!jcp.dynamicShapes) {
            auto dataDims = getInputShapeAtPort(GATHER_DATA).getDims();
            afterAxisSize = std::accumulate(dataDims.begin() + axis + 1, dataDims.end(), 1lu, std::multiplies<Dim>());
            jcp.dataAfterAxisSize = afterAxisSize;
        }
        if (x64::mayiuse(x64::avx512_common)) {
            jitKernel.reset(new jitUniGatherKernel<x64::avx512_common>(jcp));
        } else if (x64::mayiuse(x64::avx2)) {
            jitKernel.reset(new jitUniGatherKernel<x64::avx2>(jcp));
        }
        if (jitKernel) {
            jitKernel->create_ker();
        }
    }
    if (inputShapesDefined()) {
        if (needPrepareParams())
            prepareParams();
        updateLastInputDims();
    }
}

void MKLDNNGatherNode::execute(mkldnn::stream strm) {
    if (jitKernel && afterAxisSize == 1) {
//    if (jitKernel) {
        const void* srcIndices = getParentEdgeAt(GATHER_INDICES)->getMemoryPtr()->GetPtr();
        const void* srcData = getParentEdgeAt(GATHER_DATA)->getMemoryPtr()->GetPtr();
        uint8_t* dstData = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

const int* srcIndicesInt = reinterpret_cast<int*>(getParentEdgeAt(GATHER_INDICES)->getMemoryPtr()->GetPtr());
//const int* srcDataInt = reinterpret_cast<int*>(getParentEdgeAt(GATHER_DATA)->getMemoryPtr()->GetPtr());
const char* srcDataInt = reinterpret_cast<char*>(getParentEdgeAt(GATHER_DATA)->getMemoryPtr()->GetPtr());
    std::string srcIndicesIntStr = "srcIndicesInt {", srcDataIntStr = "srcDataInt {";
for (int i = 0; i < getParentEdgeAt(GATHER_DATA)->getMemoryPtr()->GetShape().getElementsCount(); i++) {
    srcDataIntStr += std::to_string(srcDataInt[i]) + "; ";
}
for (int i = 0; i < getParentEdgeAt(GATHER_INDICES)->getMemoryPtr()->GetShape().getElementsCount(); i++) {
    srcIndicesIntStr += std::to_string(srcIndicesInt[i]) + "; ";
}
srcIndicesIntStr += "}\n";
srcDataIntStr += "}\n";
//printf("%s%s", srcDataIntStr.c_str(), srcIndicesIntStr.c_str());

        const uint64_t dataElPerVec = jitKernel->getDataElPerVec();

        auto threadBody = [&](const int ithr, const int nthr) {
            const uint64_t wpt = ((totalWork / dataElPerVec) / nthr + 1) * dataElPerVec;
            const uint64_t start = std::min(wpt * ithr, totalWork);
            const uint64_t end = std::min(wpt * (ithr + 1), totalWork);
//            const uint64_t start = 0;
//            const uint64_t end = totalWork;
            const uint64_t workAmount = end - start;

            auto arg = gatherJitExecArgs();

            arg.src = srcData;
            arg.dst = dstData + afterAxisSizeInBytes * start;
            arg.indices = srcIndices;
            arg.start = &start;
            arg.axisDim = &axisDim;
            arg.afterAxSize = afterAxisSize;
            arg.axisAndAfterAxisSizeB = &axisAndAfterAxisSizeInBytes;
            arg.srcAfterBatchSizeB = &srcAfterBatchSizeInBytes;
            arg.betweenBatchAndAxisSize = &betweenBatchAndAxisSize;
            arg.specIndicesSize = &specIndicesSize;
            arg.workAmount = workAmount;
            arg.specIdxB = specIndicesInBytes[ithr].data();
            arg.idxBatchSumB = idxBatchSumInBytes[ithr].data();
            arg.dataBeforeAxisSumB = dataBeforeAxisSumBPerTr[ithr].data();
            arg.betweenBatchAndAxisIter = betweenBatchAndAxisIters[ithr];
    std::string seqStr = std::string("[") + std::to_string(ithr) + "] TW: " + std::to_string(totalWork) + " start: " + std::to_string(start) +
        "; end: " + std::to_string(end) + "\n";
//printf("%s\n", seqStr.c_str());

            const uint64_t idxElPerVec = jitKernel->getIdxElPerVec();
            const uint64_t dataElPerVec = jitKernel->getDataElPerVec();

    std::string thrIdx = "[" + std::to_string(ithr) + "] ";
    std::string specIndicesInBytesStr = thrIdx + "specIndicesInBytes {", idxBatchSumInBytesStr = thrIdx + "idxBatchSumInBytes {",
        srcBeforeAxisSumStr = thrIdx + "dataBeforeAxisSumB {", betweenBatchAndAxisIterStr = thrIdx + "betweenBatchAndAxisIter: " +
            std::to_string(betweenBatchAndAxisIters[ithr]) + "; betweenBatchAndAxisSize: " + std::to_string(betweenBatchAndAxisSize) + "\n";
for (int i = 0; i < dataElPerVec; i++) {
    specIndicesInBytesStr += std::to_string(specIndicesInBytes[ithr][i]) + "; ";
    idxBatchSumInBytesStr += std::to_string(idxBatchSumInBytes[ithr][i]) + "; ";
    srcBeforeAxisSumStr += std::to_string(dataBeforeAxisSumBPerTr[ithr][i]) + "; ";
}
specIndicesInBytesStr += "}\n";
idxBatchSumInBytesStr += "}\n";
srcBeforeAxisSumStr += "}\n";
printf("%s%s%s%s%s", seqStr.c_str(), specIndicesInBytesStr.c_str(), idxBatchSumInBytesStr.c_str(), srcBeforeAxisSumStr.c_str(),
    betweenBatchAndAxisIterStr.c_str());

            if (afterAxisSize == 1 && specIndicesSize < idxElPerVec) {
                arg.permIdx = shortPermIdx[ithr].data();
                arg.beforeAxisDiff = shortBeforeAxisDiff[ithr].data();
            } else if (afterAxisSize > 1 && afterAxisSize < dataElPerVec) {
                int beforeBlockDiff[16];
                int div = idxElPerVec / afterAxisSize;
                int remainder = idxElPerVec % afterAxisSize;
                int blockIndices[16] = {0};
                for (int i = 0; i < idxElPerVec; i++) {
                    blockIndices[i] = (start + i) % afterAxisSize;
                }
                for (int i = 0; i < idxElPerVec; i++) {
                    if (blockIndices[i] < afterAxisSize - remainder)
                        beforeBlockDiff[i] = div;//axisDim * div;
                    else
                        beforeBlockDiff[i] = div + 1;//axisDim * (div + 1);
                }
                arg.beforeAxisDiff = beforeBlockDiff;
            }

            (*jitKernel)(&arg);

//int* tmpDst = reinterpret_cast<int*>(arg.dst);
//    std::string outData = "[" + std::to_string(ithr) + "] OUT DATA: ";
//for (int i = 0; i < 4; i++) {
//    outData += std::to_string(tmpDst[i]) + ";";
//}
//outData += "\n";
//printf("%s", outData.c_str());
        };

        parallel_nt(0, threadBody);
//char* tmpDst = reinterpret_cast<char*>(dstData);
//std::cout << "\nOUT DATA:\n";
//for (int i = 0; i < getChildEdgeAt(0)->getMemoryPtr()->GetShape().getElementsCount(); i++) {
//    if (i % 8 == 0)
//        std::cout << "_";
//    std::cout << std::to_string(tmpDst[i]) << ";";
//}
//std::cout << std::endl;

//int* tmpDst = reinterpret_cast<int*>(dstData);
//std::cout << "\nOUT DATA:\n";
//for (int i = 0; i < getOutputShapeAtPort(0).getElementsCount(); i++) {
//    if (i % 8 == 0)
//        std::cout << "_";
////    if (i % 16 == 0)
////        std::cout << std::endl;
//    std::cout << std::to_string(tmpDst[i]) << ";";
//}
//std::cout << std::endl;

    } else {
        execReference();
    }
}

void MKLDNNGatherNode::executeDynamicImpl(mkldnn::stream strm) {
//    initParams();

    if (jitKernel && afterAxisSize == 1) {
//    if (jitKernel) {
std::cout << "Dyn kernel" << std::endl;
        const void* srcIndices = getParentEdgeAt(GATHER_INDICES)->getMemoryPtr()->GetPtr();
        const void* srcData = getParentEdgeAt(GATHER_DATA)->getMemoryPtr()->GetPtr();
        uint8_t* dstData = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

//        const uint64_t totalWork = beforeBatchSize * betweenBatchAndAxisSize * specIndicesSize;
        const uint64_t dataElPerVec = jitKernel->getDataElPerVec();

        auto threadBody = [&](const int ithr, const int nthr) {
            const uint64_t wpt = ((totalWork / dataElPerVec) / nthr + 1) * dataElPerVec;
            const uint64_t start = std::min(wpt * ithr, totalWork);
            const uint64_t end = std::min(wpt * (ithr + 1), totalWork);
            const uint64_t workAmount = end - start;

            auto arg = gatherJitExecArgs();

            arg.src = srcData;
            arg.dst = dstData + afterAxisSizeInBytes * start;
            arg.indices = srcIndices;
            arg.start = &start;
            arg.axisDim = &axisDim;
            arg.afterAxSize = afterAxisSize;
            arg.axisAndAfterAxisSizeB = &axisAndAfterAxisSizeInBytes;
            arg.srcAfterBatchSizeB = &srcAfterBatchSizeInBytes;
            arg.betweenBatchAndAxisSize = &betweenBatchAndAxisSize;
            arg.specIndicesSize = &specIndicesSize;
            arg.workAmount = workAmount;
//    std::string seqStr = std::string("[") + std::to_string(ithr) + "] TW: " + std::to_string(totalWork) + " start: " + std::to_string(start) +
//        "; end: " + std::to_string(end);
//printf("%s\n", seqStr.c_str());

            const uint64_t idxElPerVec = jitKernel->getIdxElPerVec();
            const uint64_t dataElPerVec = jitKernel->getDataElPerVec();
            if (afterAxisSize == 1 && specIndicesSize < idxElPerVec) {
                int permIdx[16];
                int beforeAxisDiff[16];
                permIdx[0] = idxElPerVec - specIndicesSize;
                int div = idxElPerVec / specIndicesSize;
                int remainder = idxElPerVec % specIndicesSize;
                for (int i = 1; i < idxElPerVec; i++) {
                    permIdx[i] = permIdx[i - 1] + 1;
                    if (permIdx[i] == idxElPerVec)
                        permIdx[i] = idxElPerVec - specIndicesSize;
                }
                int specIndices[16] = {0};
                for (int i = 0; i < idxElPerVec; i++) {
                    specIndices[i] = (start + i) % specIndicesSize;
                }
                for (int i = 0; i < idxElPerVec; i++) {
                    if (specIndices[i] < specIndicesSize - remainder)
                        beforeAxisDiff[i] = axisDim * div;
                    else
                        beforeAxisDiff[i] = axisDim * (div + 1);
                }
                arg.permIdx = permIdx;
                arg.beforeAxisDiff = beforeAxisDiff;
            } else if (afterAxisSize > 1 && afterAxisSize < dataElPerVec) {
                int beforeBlockDiff[16];
                int div = idxElPerVec / afterAxisSize;
                int remainder = idxElPerVec % afterAxisSize;
                int blockIndices[16] = {0};
                for (int i = 0; i < idxElPerVec; i++) {
                    blockIndices[i] = (start + i) % afterAxisSize;
                }
                for (int i = 0; i < idxElPerVec; i++) {
                    if (blockIndices[i] < afterAxisSize - remainder)
                        beforeBlockDiff[i] = div;//axisDim * div;
                    else
                        beforeBlockDiff[i] = div + 1;//axisDim * (div + 1);
                }
                arg.beforeAxisDiff = beforeBlockDiff;
            }

            (*jitKernel)(&arg);
        };

        parallel_nt(0, threadBody);
//char* tmpDst = reinterpret_cast<char*>(dstData);
//std::cout << "\nOUT DATA:\n";
//for (int i = 216 * 4; i < getChildEdgeAt(0)->getShape().getElementsCount() * 4; i++) {
//    if (i % 4 == 0)
//        std::cout << "_";
//    std::cout << std::to_string(tmpDst[i]) << ";";
//}
//std::cout << std::endl;
//int* tmpDst = reinterpret_cast<int*>(dstData);
//std::cout << "\nOUT DATA:\n";
//for (int i = 0; i < getChildEdgeAt(0)->getShape().getElementsCount(); i++) {
//    if (i % 8 == 0)
//        std::cout << "_";
//    std::cout << std::to_string(tmpDst[i]) << ";";
//}
//std::cout << std::endl;
    } else {
        execReference();
    }
}

void MKLDNNGatherNode::initShortParams(std::vector<int>& shortPermIdx, std::vector<int>& shortBeforeAxisDiff, uint64_t start) {
    const uint64_t idxElPerVec = jitKernel->getIdxElPerVec();

    shortPermIdx[0] = idxElPerVec - specIndicesSize;
    int div = idxElPerVec / specIndicesSize;
    int remainder = idxElPerVec % specIndicesSize;
    for (int i = 1; i < idxElPerVec; i++) {
        shortPermIdx[i] = shortPermIdx[i - 1] + 1;
        if (shortPermIdx[i] == idxElPerVec)
            shortPermIdx[i] = idxElPerVec - specIndicesSize;
    }
    int specIndices[16] = {0};
    for (int i = 0; i < idxElPerVec; i++) {
        specIndices[i] = (start + i) % specIndicesSize;
    }
    for (int i = 0; i < idxElPerVec; i++) {
        if (specIndices[i] < specIndicesSize - remainder)
            shortBeforeAxisDiff[i] = axisDim * div;
        else
            shortBeforeAxisDiff[i] = axisDim * (div + 1);
    }
}

void MKLDNNGatherNode::execReference() {
    const int32_t* srcIndices = reinterpret_cast<const int32_t*>(getParentEdgeAt(GATHER_INDICES)->getMemoryPtr()->GetPtr());
    const uint8_t* srcData = reinterpret_cast<const uint8_t*>(getParentEdgeAt(GATHER_DATA)->getMemoryPtr()->GetPtr());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    const size_t dstIdxAndAfterAxisSize = afterAxisSizeInBytes * specIndicesSize;
    const size_t dstAfterBatchSize = betweenBatchAndAxisSize * dstIdxAndAfterAxisSize;
    parallel_for2d(beforeBatchSize, specIndicesSize, [&](const size_t b, const size_t j) {
        int ii = srcIndices[b * specIndicesSize + j];
        if (ii < 0)
            ii += axisDim;
        size_t idx = ii;
        size_t c2 = dstAfterBatchSize * b + afterAxisSizeInBytes * j;
        if (idx < axisDim) {
            size_t c1 = srcAfterBatchSizeInBytes * b + afterAxisSizeInBytes * idx;
            for (size_t i = 0; i < betweenBatchAndAxisSize; i++) {
                size_t srcIdx = c1 + axisAndAfterAxisSizeInBytes * i;
                size_t dstIdx = c2 + dstIdxAndAfterAxisSize * i;

                cpu_memcpy(&dstData[dstIdx], &srcData[srcIdx], afterAxisSizeInBytes);
            }
        } else {
            for (size_t i = 0; i < betweenBatchAndAxisSize; i++) {
                memset(&dstData[c2 + dstIdxAndAfterAxisSize * i], 0, afterAxisSizeInBytes);
            }
        }
    });
}

std::vector<VectorDims> MKLDNNGatherNode::shapeInfer() const {
    ngraph::OutputVector inputsForShapeInfer {
        std::make_shared<ov::op::v0::Parameter>(opToShapeInfer->get_input_element_type(GATHER_DATA),
                                                getParentEdgesAtPort(GATHER_DATA)[0]->getMemory().GetShape().toPartialShape()),
        std::make_shared<ov::op::v0::Parameter>(opToShapeInfer->get_input_element_type(GATHER_INDICES),
                                                getParentEdgesAtPort(GATHER_INDICES)[0]->getMemory().GetShape().toPartialShape()),
        std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i32,
                                               getParentEdgesAtPort(GATHER_AXIS)[0]->getMemory().GetShape().getStaticDims(),
                                               getParentEdgesAtPort(GATHER_AXIS)[0]->getMemory().GetPtr())
    };
    const auto localShapeInferOp = opToShapeInfer->clone_with_new_inputs(inputsForShapeInfer);

    localShapeInferOp->validate_and_infer_types();

    std::vector<VectorDims> newOutputShapes(outputShapes.size());
    for (size_t i = 0lu; i < newOutputShapes.size(); i++) {
        const auto &partShape = localShapeInferOp->get_output_partial_shape(i);
        newOutputShapes[i] = partShape.get_shape();
    }
    return newOutputShapes;
}

bool MKLDNNGatherNode::created() const {
    return getType() == Gather;
}

REG_MKLDNN_PRIM_FOR(MKLDNNGatherNode, Gather)
