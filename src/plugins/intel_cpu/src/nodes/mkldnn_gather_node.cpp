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

#define THROW_ERROR IE_THROW() << NameFromType(getType()) << " node with name '" << getName() << "' "

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

    if (op->get_input_size() != 3 || op->get_output_size() != 1)
        THROW_ERROR << "has incorrect number of input/output edges!";

    const auto& dataShape = getInputShapeAtPort(GATHER_DATA);
    isDataShapeStat = dataShape.isStatic();
    dataSrcRank = dataShape.getRank();

    const auto& idxShape = getInputShapeAtPort(GATHER_INDICES);
    isIdxShapeStat = idxShape.isStatic();
    const auto indicesRank = idxShape.getRank();
    if (dataSrcRank == 0lu || indicesRank == 0lu)
        THROW_ERROR << "has incorrect input parameters ranks.";

    if (ov::is_type<ov::op::v8::Gather>(op)) {
        batchDims = static_cast<int>(ov::as_type_ptr<ov::op::v8::Gather>(op)->get_batch_dims());
        reverseIndexing = true;
    } else if (ov::is_type<ov::op::v7::Gather>(op)) {
        batchDims = static_cast<int>(ov::as_type_ptr<ov::op::v7::Gather>(op)->get_batch_dims());
        reverseIndexing = false;
    }

    if (batchDims < 0)
        batchDims += indicesRank;
    if (batchDims < 0 || batchDims >= std::min(static_cast<int>(dataSrcRank), static_cast<int>(indicesRank)))
        THROW_ERROR << "has incorrect batch_dims " << batchDims << "!";

    dataTypeSize = getOriginalInputPrecisionAtPort(GATHER_DATA).size();

    if (ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(GATHER_AXIS))) {
        isAxisInputConst = true;
        axis = ov::as_type<ov::op::v0::Constant>(op->get_input_node_ptr(GATHER_AXIS))->cast_vector<int>()[0];
        if (axis < 0)
            axis += dataSrcRank;
        if (axis < 0 || axis >= dataSrcRank || batchDims > axis)
            THROW_ERROR << "has incorrect input parameter axis value: " << axis;

        if (isDataShapeStat) {
            const auto& dataDims = dataShape.getDims();
            axisDim = dataDims[axis];
            beforeAxisSize = std::accumulate(dataDims.begin(), dataDims.begin() + axis, 1lu, std::multiplies<Dim>());
            betweenBatchAndAxisSize = std::accumulate(dataDims.begin() + batchDims, dataDims.begin() + axis, 1lu, std::multiplies<Dim>());
            afterAxisSize = std::accumulate(dataDims.begin() + axis + 1, dataDims.end(), 1lu, std::multiplies<Dim>());

            afterAxisSizeInBytes = afterAxisSize * dataTypeSize;
            axisAndAfterAxisSizeInBytes = axisDim * afterAxisSizeInBytes;
            srcAfterBatchSizeInBytes = betweenBatchAndAxisSize * axisAndAfterAxisSizeInBytes;
        }
    }

    if (isDataShapeStat) {
        const auto& dataDims = dataShape.getDims();
        beforeBatchSize = std::accumulate(dataDims.begin(), dataDims.begin() + batchDims, 1lu, std::multiplies<Dim>());
    }
    if (isIdxShapeStat) {
        const auto& idxDims = idxShape.getDims();
        specIndicesSize = std::accumulate(idxDims.begin() + batchDims, idxDims.end(), 1lu, std::multiplies<Dim>());

        if (isDataShapeStat) {
            specIdxAndAfterAxSizeB = specIndicesSize * afterAxisSizeInBytes;
            totalWork = beforeBatchSize * betweenBatchAndAxisSize * specIndicesSize * afterAxisSize;
        }
    }
}

void MKLDNNGatherNode::initSupportedPrimitiveDescriptors() {
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
    auto& dataMemPtr = getParentEdgeAt(GATHER_DATA)->getMemoryPtr();
    if (!dataMemPtr || !dataMemPtr->GetPrimitivePtr())
        THROW_ERROR << " has not allocated input data memory.";
    auto& idxMemPtr = getParentEdgeAt(GATHER_INDICES)->getMemoryPtr();
    if (!idxMemPtr || !idxMemPtr->GetPrimitivePtr())
        THROW_ERROR << " has not allocated input indices memory.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_ERROR << " has unidentified preferable primitive descriptor.";

    const auto& dataDims = dataMemPtr->getStaticDims();
    const auto& idxDims = idxMemPtr->getStaticDims();

    if (!isAxisInputConst) {
        axis = (reinterpret_cast<const int32_t*>(getParentEdgeAt(GATHER_AXIS)->getMemoryPtr()->GetPtr()))[0];
        if (axis < 0)
            axis += dataSrcRank;
        if (axis < 0 || axis >= dataSrcRank || batchDims > axis)
            THROW_ERROR << "has incorrect input parameter axis value: " << axis;
    }

    if (!isDataShapeStat || !isAxisInputConst) {
        axisDim = dataDims[axis];
        beforeBatchSize = std::accumulate(dataDims.begin(), dataDims.begin() + batchDims, 1lu, std::multiplies<uint64_t>());
        betweenBatchAndAxisSize = std::accumulate(dataDims.begin() + batchDims, dataDims.begin() + axis, 1lu, std::multiplies<uint64_t>());
        afterAxisSize = std::accumulate(dataDims.begin() + axis + 1, dataDims.end(), 1lu, std::multiplies<uint64_t>());

        afterAxisSizeInBytes = afterAxisSize * dataTypeSize;
        axisAndAfterAxisSizeInBytes = axisDim * afterAxisSizeInBytes;
        srcAfterBatchSizeInBytes = betweenBatchAndAxisSize * axisAndAfterAxisSizeInBytes;
    }
    if (!isIdxShapeStat) {
        specIndicesSize = std::accumulate(idxDims.begin() + batchDims, idxDims.end(), 1lu, std::multiplies<uint64_t>());

        specIdxAndAfterAxSizeB = specIndicesSize * afterAxisSizeInBytes;
        totalWork = beforeBatchSize * betweenBatchAndAxisSize * specIndicesSize * afterAxisSize;
    }

    if (jitKernel && !isDynamicNode()) {
        const uint64_t dataElPerVec = jitKernel->getDataElPerVec();
        const uint64_t nthr = parallel_get_max_threads();
        const uint64_t wpt = ((totalWork / dataElPerVec) / nthr + 1) * dataElPerVec;
        permIdxMaskPerThr         = std::vector<std::vector<int>>(nthr);
        srcBeforeAxisDiffPerThr      = std::vector<std::vector<int>>(nthr);
        specIdxInBytesPerThr      = std::vector<std::vector<int>>(nthr, std::vector<int>(dataElPerVec));
        idxBatchSumInBytes        = std::vector<std::vector<int>>(nthr, std::vector<int>(dataElPerVec));
        dataBeforeAxisSumInBytesPerThr = std::vector<std::vector<int>>(nthr, std::vector<int>(dataElPerVec));

        afterAxIdxInBytesPerThr   = std::vector<std::vector<int>>(nthr);
        specIdxDiffPerThr         = std::vector<std::vector<int>>(nthr);
        beforeAxPermMaskPerThr    = std::vector<std::vector<int>>(nthr);
        afterAxPermMaskPerThr     = std::vector<std::vector<int>>(nthr);

        betweenBatchAndAxisIters  = std::vector<int>(nthr, 0);
        specIdxAndAfterAxIterBPerThr = std::vector<int>(nthr, 0);
        parallel_nt(nthr, [&](const int ithr, const int nthr) {
            uint64_t dstStart = std::min(wpt * ithr, totalWork);
            betweenBatchAndAxisIters[ithr] = (dstStart / specIndicesSize) % betweenBatchAndAxisSize;
            for (uint64_t j = 0lu; j < dataElPerVec; j++) {
                specIdxInBytesPerThr[ithr][j] = (((dstStart + j) / afterAxisSize) % specIndicesSize) * idxTypeSize;
                idxBatchSumInBytes[ithr][j] = ((dstStart + j) / (betweenBatchAndAxisSize * specIndicesSize * afterAxisSize)) *
                        specIndicesSize * idxTypeSize;
                dataBeforeAxisSumInBytesPerThr[ithr][j] = ((dstStart + j) / (specIndicesSize * afterAxisSize)) * axisAndAfterAxisSizeInBytes;
            }
            initShortParams(ithr, dstStart);
        });
    }
}

bool MKLDNNGatherNode::needPrepareParams() const {
    bool result = inputShapesModified();
    if (!isAxisInputConst)
        result = result || axis != (reinterpret_cast<const int32_t*>(getParentEdgeAt(GATHER_AXIS)->getMemoryPtr()->GetPtr()))[0];
    return result;
}

void MKLDNNGatherNode::createPrimitive() {
    uint64_t dataElPerVec = 1;
    if (!isDynamicNode()) {
        dataElPerVec = x64::mayiuse(x64::avx512_common) ? x64::cpu_isa_traits<x64::avx512_common>::vlen / dataTypeSize :
            x64::mayiuse(x64::avx2) ? x64::cpu_isa_traits<x64::avx2>::vlen / dataTypeSize : 1;
    }
    // Gather instruction is not supported by SSE.
    if ((x64::mayiuse(x64::avx512_common) || x64::mayiuse(x64::avx2)) &&
            (isDynamicNode() || afterAxisSize < dataElPerVec)) {
        jGatherConfParams jcp;
        jcp.dataTypeSize = dataTypeSize;
        jcp.reverseIndexing = reverseIndexing;
        jcp.dynamicShapes = isDynamicNode();
        jcp.batchDims = batchDims;
        if (!jcp.dynamicShapes) {
            jcp.beforeAxisSize = beforeAxisSize;
            jcp.specIdxSize = specIndicesSize;
            jcp.afterAxisSize = afterAxisSize;
        } else if (getInputShapeAtPort(GATHER_DATA).isStatic() && isAxisInputConst) {
            jcp.beforeAxisSize = beforeAxisSize;
            jcp.afterAxisSize = afterAxisSize;
        } else if (getInputShapeAtPort(GATHER_INDICES).isStatic()) {
            jcp.specIdxSize = specIndicesSize;
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
//    if (jitKernel && afterAxisSize == 1) {
    if (jitKernel && (
            afterAxisSize < jitKernel->getDataElPerVec())) {
//            afterAxisSize == 1 ||
//            afterAxisSize > 1 && specIndicesSize * afterAxisSize < jitKernel->getDataElPerVec())) {
//    if (jitKernel) {
        const void* srcIndices = getParentEdgeAt(GATHER_INDICES)->getMemoryPtr()->GetPtr();
        const void* srcData = getParentEdgeAt(GATHER_DATA)->getMemoryPtr()->GetPtr();
        uint8_t* dstData = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

//const int* srcIndicesInt = reinterpret_cast<int*>(getParentEdgeAt(GATHER_INDICES)->getMemoryPtr()->GetPtr());
//const int* srcDataInt = reinterpret_cast<int*>(getParentEdgeAt(GATHER_DATA)->getMemoryPtr()->GetPtr());
//////const char* srcDataInt = reinterpret_cast<char*>(getParentEdgeAt(GATHER_DATA)->getMemoryPtr()->GetPtr());
//    std::string srcIndicesIntStr = "srcIndicesInt {", srcDataIntStr = "srcDataInt {";
//for (int i = 0; i < getParentEdgeAt(GATHER_DATA)->getMemoryPtr()->GetShape().getElementsCount(); i++) {
//    srcDataIntStr += std::to_string(srcDataInt[i]) + "; ";
//}
//for (int i = 0; i < getParentEdgeAt(GATHER_INDICES)->getMemoryPtr()->GetShape().getElementsCount(); i++) {
//    srcIndicesIntStr += std::to_string(srcIndicesInt[i]) + "; ";
//}
//srcIndicesIntStr += "}\n";
//srcDataIntStr += "}\n";
//printf("%s%s", srcDataIntStr.c_str(), srcIndicesIntStr.c_str());

        const uint64_t dataElPerVec = jitKernel->getDataElPerVec();

        auto threadBody = [&](const int ithr, const int nthr) {
            const uint64_t wpt = ((totalWork / dataElPerVec) / nthr + 1) * dataElPerVec;
            const uint64_t start = std::min(wpt * ithr, totalWork);
            const uint64_t end = std::min(wpt * (ithr + 1), totalWork);
            const uint64_t workAmount = end - start;

            auto arg = gatherJitExecArgs();

            arg.src = srcData;
            arg.dst = dstData + start * dataTypeSize;
            arg.indices = srcIndices;
            arg.start = &start;
            arg.axisDim = &axisDim;
            arg.afterAxSize = afterAxisSize;
            arg.axisAndAfterAxisSizeB = &axisAndAfterAxisSizeInBytes;
            arg.srcAfterBatchSizeB = &srcAfterBatchSizeInBytes;
            arg.betweenBatchAndAxisSize = &betweenBatchAndAxisSize;
            arg.specIndicesSize = &specIndicesSize;
            arg.workAmount = workAmount;
            arg.specIdxB = specIdxInBytesPerThr[ithr].data();
            arg.idxBatchSumB = idxBatchSumInBytes[ithr].data();
            arg.dataBeforeAxisSumB = dataBeforeAxisSumInBytesPerThr[ithr].data();
            arg.betweenBatchAndAxisIter = betweenBatchAndAxisIters[ithr];
    std::string seqStr = std::string("[") + std::to_string(ithr) + "] TW: " + std::to_string(totalWork) + " start: " + std::to_string(start) +
        "; end: " + std::to_string(end) + "\n";

            const uint64_t idxElPerVec = jitKernel->getIdxElPerVec();
            const uint64_t dataElPerVec = jitKernel->getDataElPerVec();

    std::string thrIdx = "[" + std::to_string(ithr) + "] ";
    std::string specIndicesInBytesStr = thrIdx + "specIndicesInBytes {",
            idxBatchSumInBytesStr = thrIdx + "idxBatchSumInBytes {",
            srcBeforeAxisSumStr = thrIdx + "dataBeforeAxisSumB {",
            beforeAxisDiffStr = thrIdx + "beforeAxisDiff {",
            beforeAxPermMaskStr = thrIdx + "beforeAxPermMask {",
            specIdxDiffStr = thrIdx + "specIdxDiff {",
            afterAxisPermStr = thrIdx + "afterAxisPermMask {",
            afterAxIdxBStr = thrIdx + "afterAxIdxB {",
            betweenBatchAndAxisIterStr = thrIdx + "betweenBatchAndAxisIter: " + std::to_string(betweenBatchAndAxisIters[ithr]) +
                "; betweenBatchAndAxisSize: " + std::to_string(betweenBatchAndAxisSize) +
                "; srcAfterBatchSizeInBytes: " + std::to_string(srcAfterBatchSizeInBytes) +
                "; specIdxAndAfterAxIterB: " + std::to_string(specIdxAndAfterAxIterBPerThr[ithr]) +
                "; specIdxAndAfterAxSizeB: " + std::to_string(specIdxAndAfterAxSizeB) + "\n";
for (int i = 0; i < dataElPerVec; i++) {
    specIndicesInBytesStr += std::to_string(specIdxInBytesPerThr[ithr][i]) + "; ";
    idxBatchSumInBytesStr += std::to_string(idxBatchSumInBytes[ithr][i]) + "; ";
    srcBeforeAxisSumStr += std::to_string(dataBeforeAxisSumInBytesPerThr[ithr][i]) + "; ";
    if (i < afterAxIdxInBytesPerThr[ithr].size())
        afterAxIdxBStr += std::to_string(afterAxIdxInBytesPerThr[ithr][i]) + "; ";
    if (i < srcBeforeAxisDiffPerThr[ithr].size())
        beforeAxisDiffStr += std::to_string(srcBeforeAxisDiffPerThr[ithr][i]) + "; ";
    if (i < beforeAxPermMaskPerThr[ithr].size())
        beforeAxPermMaskStr += std::to_string(beforeAxPermMaskPerThr[ithr][i]) + "; ";
    if (i < specIdxDiffPerThr[ithr].size())
        specIdxDiffStr += std::to_string(specIdxDiffPerThr[ithr][i]) + "; ";
    if (i < afterAxPermMaskPerThr[ithr].size())
        afterAxisPermStr += std::to_string(afterAxPermMaskPerThr[ithr][i]) + "; ";
}
specIndicesInBytesStr += "}\n";
idxBatchSumInBytesStr += "}\n";
srcBeforeAxisSumStr += "}\n";
afterAxIdxBStr += "}\n";
beforeAxisDiffStr += "}\n";
beforeAxPermMaskStr += "}\n";
specIdxDiffStr += "}\n";
afterAxisPermStr += "}\n";

            if (afterAxisSize == 1 && specIndicesSize < idxElPerVec) { // Elementwise short case.
                arg.permIdxMask = permIdxMaskPerThr[ithr].data();
                arg.beforeAxisDiff = srcBeforeAxisDiffPerThr[ithr].data();
            } else if (afterAxisSize > 1 && afterAxisSize < dataElPerVec) { // Blocked short case.
                arg.afterAxIdxB = afterAxIdxInBytesPerThr[ithr].data();
                arg.specIdxDiff = specIdxDiffPerThr[ithr].data();
                arg.beforeAxisDiff = srcBeforeAxisDiffPerThr[ithr].data();
                arg.beforeAxisPermMask = beforeAxPermMaskPerThr[ithr].data();
                arg.afterAxisPermMask = afterAxPermMaskPerThr[ithr].data();
                const int afterAxSizeB = afterAxisSize;
                arg.afterAxSizePtr = &afterAxSizeB;
                arg.specIdxAndAfterAxIterB = specIdxAndAfterAxIterBPerThr[ithr];
                arg.specIdxAndAfterAxSizeB = specIdxAndAfterAxSizeB;
            }

printf("%s%s%s%s%s%s%s%s%s%s", seqStr.c_str(), specIndicesInBytesStr.c_str(), idxBatchSumInBytesStr.c_str(), srcBeforeAxisSumStr.c_str(),
    afterAxIdxBStr.c_str(), beforeAxisDiffStr.c_str(), beforeAxPermMaskStr.c_str(),
    specIdxDiffStr.c_str(), afterAxisPermStr.c_str(), betweenBatchAndAxisIterStr.c_str());

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

int* tmpDst = reinterpret_cast<int*>(dstData);
std::cout << "\nOUT DATA:\n";
for (int i = 0; i < getOutputShapeAtPort(0).getElementsCount(); i++) {
    if (i % 8 == 0)
        std::cout << "_";
    if (i % 96 == 0)
        std::cout << std::endl;
    std::cout << std::to_string(tmpDst[i]) << ";";
}
std::cout << std::endl;

    } else {
        execReference();
    }
}

void MKLDNNGatherNode::executeDynamicImpl(mkldnn::stream strm) {
    if (jitKernel && afterAxisSize == 1) {
//    if (jitKernel) {
//std::cout << "Dyn kernel" << std::endl;
        const void* srcIndices = getParentEdgeAt(GATHER_INDICES)->getMemoryPtr()->GetPtr();
        const void* srcData = getParentEdgeAt(GATHER_DATA)->getMemoryPtr()->GetPtr();
        uint8_t* dstData = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

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
    std::string seqStr = std::string("[") + std::to_string(ithr) + "] TW: " + std::to_string(totalWork) + " start: " + std::to_string(start) +
        "; end: " + std::to_string(end);
printf("%s\n", seqStr.c_str());

            const uint64_t idxElPerVec = jitKernel->getIdxElPerVec();
            const uint64_t dataElPerVec = jitKernel->getDataElPerVec();
            if (afterAxisSize == 1 && specIndicesSize < idxElPerVec) {
                int permIdxMask[16];
                int beforeAxisDiff[16];
                permIdxMask[0] = idxElPerVec - specIndicesSize;
                int div = idxElPerVec / specIndicesSize;
                int remainder = idxElPerVec % specIndicesSize;
                for (int i = 1; i < idxElPerVec; i++) {
                    permIdxMask[i] = permIdxMask[i - 1] + 1;
                    if (permIdxMask[i] == idxElPerVec)
                        permIdxMask[i] = idxElPerVec - specIndicesSize;
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
                arg.permIdxMask = permIdxMask;
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
                        beforeBlockDiff[i] = div; // axisDim * div;
                    else
                        beforeBlockDiff[i] = div + 1; // axisDim * (div + 1);
                }
                arg.beforeAxisDiff = beforeBlockDiff;
            }
//    std::string thrIdx = "[" + std::to_string(ithr) + "] ";
//    std::string specIndicesInBytesStr = thrIdx + "specIndicesInBytes {",
//            idxBatchSumInBytesStr = thrIdx + "idxBatchSumInBytes {",
//            srcBeforeAxisSumStr = thrIdx + "dataBeforeAxisSumB {",
//            beforeAxisDiffStr = thrIdx + "beforeAxisDiff {",
//            beforeAxPermMaskStr = thrIdx + "beforeAxPermMask {",
//            specIdxDiffStr = thrIdx + "specIdxDiff {",
//            afterAxisPermStr = thrIdx + "afterAxisPermMask {",
//            afterAxIdxBStr = thrIdx + "afterAxIdxB {",
//            betweenBatchAndAxisIterStr = thrIdx + "betweenBatchAndAxisIter: " + std::to_string(betweenBatchAndAxisIters[ithr]) +
//                "; betweenBatchAndAxisSize: " + std::to_string(betweenBatchAndAxisSize) +
//                "; srcAfterBatchSizeInBytes: " + std::to_string(srcAfterBatchSizeInBytes) +
//                "; specIdxAndAfterAxIterB: " + std::to_string(specIdxAndAfterAxIterBPerThr[ithr]) +
//                "; specIdxAndAfterAxSizeB: " + std::to_string(specIdxAndAfterAxSizeB) + "\n";
//for (int i = 0; i < dataElPerVec; i++) {
//    specIndicesInBytesStr += std::to_string(specIdxInBytesPerThr[ithr][i]) + "; ";
//    idxBatchSumInBytesStr += std::to_string(idxBatchSumInBytes[ithr][i]) + "; ";
//    srcBeforeAxisSumStr += std::to_string(dataBeforeAxisSumInBytesPerThr[ithr][i]) + "; ";
//    if (i < afterAxIdxInBytesPerThr[ithr].size())
//        afterAxIdxBStr += std::to_string(afterAxIdxInBytesPerThr[ithr][i]) + "; ";
//    if (i < srcBeforeAxisDiffPerThr[ithr].size())
//        beforeAxisDiffStr += std::to_string(srcBeforeAxisDiffPerThr[ithr][i]) + "; ";
//    if (i < beforeAxPermMaskPerThr[ithr].size())
//        beforeAxPermMaskStr += std::to_string(beforeAxPermMaskPerThr[ithr][i]) + "; ";
//    if (i < specIdxDiffPerThr[ithr].size())
//        specIdxDiffStr += std::to_string(specIdxDiffPerThr[ithr][i]) + "; ";
//    if (i < afterAxPermMaskPerThr[ithr].size())
//        afterAxisPermStr += std::to_string(afterAxPermMaskPerThr[ithr][i]) + "; ";
//}
//specIndicesInBytesStr += "}\n";
//idxBatchSumInBytesStr += "}\n";
//srcBeforeAxisSumStr += "}\n";
//afterAxIdxBStr += "}\n";
//beforeAxisDiffStr += "}\n";
//beforeAxPermMaskStr += "}\n";
//specIdxDiffStr += "}\n";
//afterAxisPermStr += "}\n";
//printf("%s%s%s%s%s%s%s%s%s%s", seqStr.c_str(), specIndicesInBytesStr.c_str(), idxBatchSumInBytesStr.c_str(), srcBeforeAxisSumStr.c_str(),
//    afterAxIdxBStr.c_str(), beforeAxisDiffStr.c_str(), beforeAxPermMaskStr.c_str(),
//    specIdxDiffStr.c_str(), afterAxisPermStr.c_str(), betweenBatchAndAxisIterStr.c_str());

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

int* tmpDst = reinterpret_cast<int*>(dstData);
std::cout << "\nOUT DATA:\n";
for (int i = 0; i < beforeBatchSize * betweenBatchAndAxisSize * specIndicesSize * afterAxisSize; i++) {
    if (i % 8 == 0)
        std::cout << "_";
    std::cout << std::to_string(tmpDst[i]) << ";";
}
std::cout << std::endl;
    } else {
        execReference();
    }
}

void MKLDNNGatherNode::initShortParams(const uint64_t ithr, const uint64_t start) {
    if (!jitKernel)
        THROW_ERROR << "has uninitialized kernel in function initShortParams.";
    const uint64_t idxElPerVec = jitKernel->getIdxElPerVec();

    if (afterAxisSize == 1) { // Elementwise gather.
        if (specIndicesSize >= idxElPerVec)
            return; // Is not a short case.
        if (ithr >= permIdxMaskPerThr.size())
            THROW_ERROR << ". Thread index exceeds permutation array size.";
        if (ithr >= srcBeforeAxisDiffPerThr.size())
            THROW_ERROR << ". Thread index exceeds before axis diff array size.";

        permIdxMaskPerThr[ithr].resize(idxElPerVec);
        srcBeforeAxisDiffPerThr[ithr].resize(idxElPerVec);
        auto& permIdxMask = permIdxMaskPerThr[ithr];
        auto& beforeAxisDiff = srcBeforeAxisDiffPerThr[ithr];

        permIdxMask[0] = idxElPerVec - specIndicesSize;
        int div = idxElPerVec / specIndicesSize;
        int remainder = idxElPerVec % specIndicesSize;
        for (int i = 1; i < idxElPerVec; i++) {
            permIdxMask[i] = permIdxMask[i - 1] + 1;
            if (permIdxMask[i] == idxElPerVec)
                permIdxMask[i] = idxElPerVec - specIndicesSize;
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
    } else { // Blocked gather.
        if (afterAxisSize > idxElPerVec)
            return; // Is not a short case.
        if (ithr >= specIdxDiffPerThr.size())
            THROW_ERROR << ". Thread index exceeds spec indices diff array size.";
        if (ithr >= afterAxPermMaskPerThr.size())
            THROW_ERROR << ". Thread index exceeds permutation array size.";

        afterAxIdxInBytesPerThr[ithr].resize(idxElPerVec);
        beforeAxPermMaskPerThr[ithr].resize(idxElPerVec);
        afterAxPermMaskPerThr[ithr].resize(idxElPerVec);
        specIdxDiffPerThr[ithr].resize(idxElPerVec);
        srcBeforeAxisDiffPerThr[ithr].resize(idxElPerVec);
        auto& afterAxIdxInBytes = afterAxIdxInBytesPerThr[ithr];
        auto& beforeAxPermMask = beforeAxPermMaskPerThr[ithr];
        auto& afterAxPermMask = afterAxPermMaskPerThr[ithr];
        auto& specIdxDiff = specIdxDiffPerThr[ithr];
        auto& srcBeforeAxisDiff = srcBeforeAxisDiffPerThr[ithr];

//        int divAA = idxElPerVec / afterAxisSize;
//        int divBA = idxElPerVec / (axis * afterAxisSize);
//        int remainder = idxElPerVec % afterAxisSize;
        int secondStart = start + idxElPerVec;
//        int beforeAxSize = beforeBatchSize * betweenBatchAndAxisSize;
//        beforeAxPermMask[0] = idxElPerVec - axisDim * afterAxisSize;
        for (int i = 0; i < idxElPerVec; i++) {
            afterAxIdxInBytes[i] = (start + i) % afterAxisSize;
            specIdxDiff[i] = (((secondStart + i) / afterAxisSize) % specIndicesSize) * idxTypeSize - specIdxInBytesPerThr[ithr][i];
            if (specIdxDiff[i] < 0)
                specIdxDiff[i] += specIndicesSize * idxTypeSize;
//            if (afterAxIdxInBytes[i] < afterAxisSize - remainder) {
////                specIdxDiff[i] = (secondStart + i)//div * idxTypeSize;
//                srcBeforeAxisDiff[i] = divAA * afterAxisSize * dataTypeSize;
//            } else {
////                specIdxDiff[i] = //(div + 1) * idxTypeSize;
//                srcBeforeAxisDiff[i] = (divAA + 1) * afterAxisSize * dataTypeSize;
//            }
//            if (srcBeforeAxisDiff[i] >= beforeAxSize) {
//                srcBeforeAxisDiff[i] = 0; // TODO: fix
//            }
            srcBeforeAxisDiff[i] = ((start + i + idxElPerVec) / (specIndicesSize * afterAxisSize)) * axisAndAfterAxisSizeInBytes -
                    ((start + i) / (specIndicesSize * afterAxisSize)) * axisAndAfterAxisSizeInBytes;

//            if (i > 0) {
//                beforeAxPermMask[i] = beforeAxPermMask[i - 1] + 1;
//                if (beforeAxPermMask[i] == idxElPerVec)
//                    beforeAxPermMask[i] = idxElPerVec - axisDim * afterAxisSize;
//            }
            afterAxIdxInBytes[i] *= dataTypeSize;
            afterAxPermMask[i] = idxElPerVec - afterAxisSize + i;
            for (size_t j = 0lu; j < 3lu; j++) {
                if (afterAxPermMask[i] >= idxElPerVec)
                    afterAxPermMask[i] -= afterAxisSize;
            }
        }
        if (specIndicesSize * afterAxisSize < idxElPerVec) {
            beforeAxPermMask[0] = idxElPerVec - specIndicesSize * afterAxisSize;
            for (int i = 1; i < idxElPerVec; i++) {
                beforeAxPermMask[i] = beforeAxPermMask[i - 1] + 1;
                if (beforeAxPermMask[i] == idxElPerVec)
                    beforeAxPermMask[i] = idxElPerVec - specIndicesSize * afterAxisSize;
            }
        }

        specIdxAndAfterAxIterBPerThr[ithr] = (start * dataTypeSize) % specIdxAndAfterAxSizeB;
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
