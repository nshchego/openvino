// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "grid_sample.hpp"
#include "ie_parallel.hpp"
#include <ngraph/opsets/opset1.hpp>
//#include "common/cpu_memcpy.h"
//#include <utils/general_utils.h>
//#include "kernels/grid_sample_kernel.hpp"

using namespace InferenceEngine;
using namespace dnnl::impl::cpu;
using namespace ov::intel_cpu;
using namespace ov::intel_cpu::node;

#define THROW_ERROR IE_THROW() << getTypeStr() << " node with name '" << getName() << "' "


bool GridSample::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
std::cout << "GridSample::isSupportedOperation+" << std::endl;
printf("GridSample::isSupportedOperation+");
    try {
        if (!ov::is_type<op::v9::GridSample>(op)) {
            errorMessage = "Not supported GridSample operation version. CPU plug-in supports only 9th version.";
            return false;
        }
        if (!x64::mayiuse(x64::sse41)) {
            errorMessage = "Not supported CPU instruction set.";
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}

GridSample::GridSample(const std::shared_ptr<ov::Node>& op, const dnnl::engine& eng,
        WeightsSharing::Ptr &cache) : Node(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    if (op->get_input_size() != 2 || op->get_output_size() != 1)
        THROW_ERROR << "has incorrect number of input/output edges.";

    const auto& dataShape = getInputShapeAtPort(IN_DATA);
    if (dataShape.getRank() != 4)
        THROW_ERROR << "has incorrect rank of the Data input.";

    const auto& gridShape = getInputShapeAtPort(IN_GRID);
    if (gridShape.getRank() != 4)
        THROW_ERROR << "has incorrect rank of the Grid input.";
    if (gridShape.getDims()[3] != 2)
        THROW_ERROR << "has incorrect shape of the Grid input. The 4th dimension should be equal to 2.";

    const auto& attributes = ov::as_type_ptr<ov::op::v9::GridSample>(op)->get_attributes();
    alignCorners = attributes.align_corners;
    switch (attributes.mode) {
        case op::v9::GridSample::InterpolationMode::BILINEAR:
            interpolationMode = InterpolationMode::BILINEAR;
            break;
        case op::v9::GridSample::InterpolationMode::BICUBIC:
            interpolationMode = InterpolationMode::BICUBIC;
            break;
        case op::v9::GridSample::InterpolationMode::NEAREST:
            interpolationMode = InterpolationMode::NEAREST;
            break;
    }
    switch (attributes.padding_mode) {
        case op::v9::GridSample::PaddingMode::ZEROS:
            paddingMode = PaddingMode::ZEROS;
            break;
        case op::v9::GridSample::PaddingMode::BORDER:
            paddingMode = PaddingMode::BORDER;
            break;
        case op::v9::GridSample::PaddingMode::REFLECTION:
            paddingMode = PaddingMode::REFLECTION;
            break;
    }
}

void GridSample::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    const auto& dataDims = getInputShapeAtPort(IN_DATA).getDims();

    // Implementation desc type will be redefined in the fn prepareParams if a kernel will be created.
    Precision dataPrecision = getOriginalInputPrecisionAtPort(IN_DATA);
    Precision gridPrecision = Precision::FP32;
    // TODO: Extend kernel to support other precisions.
    if (dataPrecision.is_float()) {
        dataPrecision = Precision::FP32;
    } else {
        dataPrecision = Precision::I32;
    }
    dataTypeSize = dataPrecision.size();
    gridTypeSize = gridPrecision.size();

    impl_desc_type implType = jit_sse42;
    if (x64::mayiuse(x64::avx512_core)) {
        implType = jit_avx512;
    } else if (x64::mayiuse(x64::avx2)) {
        implType = jit_avx2;
    }

    addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                          {LayoutType::ncsp, gridPrecision}},
                         {{LayoutType::ncsp, dataPrecision}},
                         implType,
                         isDynamicNode());
}

void GridSample::createPrimitive() {
    jGridSampleConfParams jcp;

//    jcp.dataTypeSize = dataTypeSize;
    jcp.inDataPrc = getRuntimePrecision();
    jcp.dynamicShapes = isDynamicNode();
    jcp.alignCorners = alignCorners;
    jcp.interpolationMode = interpolationMode;
    jcp.paddingMode = paddingMode;

    if (!jcp.dynamicShapes) {
        const auto& srcDataShape = getInputShapeAtPort(IN_DATA).getDims();
        const auto& gridShape = getInputShapeAtPort(IN_GRID).getDims();
        const auto& dstShape = getOutputShapeAtPort(0).getDims();
        jcp.srcBatchStepB = std::accumulate(srcDataShape.begin() + 1, srcDataShape.end(), 1, std::multiplies<Dim>());
        jcp.gridBatchStepB = std::accumulate(gridShape.begin() + 1, gridShape.end(), 1, std::multiplies<Dim>());
        jcp.dstBatchStepB = std::accumulate(dstShape.begin() + 1, dstShape.end(), 1, std::multiplies<Dim>());
        totalWork = dstShape[2] * dstShape[3];
    } else {
    }

    if (x64::mayiuse(x64::avx512_core)) {
        jitKernel.reset(new jitUniGridSampleKernel<x64::avx512_core>(jcp));
    } else if (x64::mayiuse(x64::avx2)) {
        jitKernel.reset(new jitUniGridSampleKernel<x64::avx2>(jcp));
    } else {
        jitKernel.reset(new jitUniGridSampleKernel<x64::sse41>(jcp));
    }
    if (jitKernel) {
        jitKernel->create_ker();
    } else {
        THROW_ERROR << " could not create JIT kernel.";
    }

    if (!isDynamicNode()) {
        const uint64_t dataElPerVec = jitKernel->getDataElPerVec();
        const uint64_t nthr = parallel_get_max_threads();
        const uint64_t wpt = ((totalWork / dataElPerVec) / nthr + 1) * dataElPerVec;
        execParamsPerThread.resize(nthr);

        parallel_nt(nthr, [&](const int ithr, const int nthr) {
            const uint64_t dstStart = std::min(wpt * ithr, totalWork);
            const uint64_t dstEnd = std::min(wpt * (ithr + 1), totalWork);

            auto& p = execParamsPerThread[ithr];
            p.workAmount = dstEnd - dstStart;
            p.dstStartB = dstStart * dataTypeSize;
            p.gridStartB = dstStart * 2 * gridTypeSize;
            const auto& srcDataShape = getInputShapeAtPort(IN_DATA).getDims();
            p.channelsNum = srcDataShape[1];
            p.dstChStepB = srcDataShape[2] * srcDataShape[3] * dataTypeSize;
            p.srcWidthFl = srcDataShape[2];
            p.srcHeightFl = srcDataShape[3];
        });
    }

    Node::createPrimitive();
}

//bool GridSample::needPrepareParams() const {
//    bool result = inputShapesModified();
//    if (!isAxisInputConst)
//        result = result || axis != (reinterpret_cast<const int32_t*>(getParentEdgeAt(GATHER_AXIS)->getMemoryPtr()->GetPtr()))[0];
//    return result;
//}

void GridSample::prepareParams() {
    auto& dataMemPtr = getParentEdgeAt(IN_DATA)->getMemoryPtr();
    if (!dataMemPtr || !dataMemPtr->isAllocated())
        THROW_ERROR << " has not allocated input data memory.";
    auto& idxMemPtr = getParentEdgeAt(IN_GRID)->getMemoryPtr();
    if (!idxMemPtr || !idxMemPtr->isAllocated())
        THROW_ERROR << " has not allocated input grid memory.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_ERROR << " has unidentified preferable primitive descriptor.";

//    if (!isAxisInputConst) {
//        axis = (reinterpret_cast<const int32_t*>(getParentEdgeAt(GATHER_AXIS)->getMemoryPtr()->GetPtr()))[0];
//        if (axis < 0)
//            axis += dataSrcRank;
//        if (axis < 0 || axis >= dataSrcRank || batchDims > axis)
//            THROW_ERROR << "has incorrect input parameter axis value: " << axis;
//    }
//
//    if (!isDataShapeStat || !isAxisInputConst) {
//        const auto& dataDims = dataMemPtr->getStaticDims();
//        axisDim = dataDims[axis];
//        beforeBatchSize = std::accumulate(dataDims.begin(), dataDims.begin() + batchDims, 1lu, std::multiplies<uint64_t>());
//        betweenBatchAndAxisSize = std::accumulate(dataDims.begin() + batchDims, dataDims.begin() + axis, 1lu, std::multiplies<uint64_t>());
//        afterAxisSize = std::accumulate(dataDims.begin() + axis + 1, dataDims.end(), 1lu, std::multiplies<uint64_t>());
//
//        afterAxisSizeInBytes = afterAxisSize * dataTypeSize;
//        axisAndAfterAxisSizeInBytes = axisDim * afterAxisSizeInBytes;
//        srcAfterBatchSizeInBytes = betweenBatchAndAxisSize * axisAndAfterAxisSizeInBytes;
//
//        if (isIdxShapeStat) {
//            specIdxAndAfterAxSizeB = specIndicesSize * afterAxisSizeInBytes;
//            totalWork = beforeBatchSize * betweenBatchAndAxisSize * specIndicesSize * afterAxisSize;
//        }
//    }
//
//    if (!isIdxShapeStat) {
//        const auto& idxDims = idxMemPtr->getStaticDims();
//        specIndicesSize = std::accumulate(idxDims.begin() + batchDims, idxDims.end(), 1lu, std::multiplies<uint64_t>());
//
//        specIdxAndAfterAxSizeB = specIndicesSize * afterAxisSizeInBytes;
//        totalWork = beforeBatchSize * betweenBatchAndAxisSize * specIndicesSize * afterAxisSize;
//    }

//    const auto& selectedPD = getSelectedPrimitiveDescriptor();
//    if (x64::mayiuse(x64::avx512_core)) {
//        selectedPD->setImplementationType(jit_avx512);
//    } else if (x64::mayiuse(x64::avx2)) {
//        selectedPD->setImplementationType(jit_avx2);
//    } else {
//        selectedPD->setImplementationType(jit_sse42);
//    }
}

void GridSample::execute(dnnl::stream strm) {
    const void* srcData = getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetPtr();
    const uint8_t* gridData = reinterpret_cast<uint8_t*>(getParentEdgeAt(IN_GRID)->getMemoryPtr()->GetPtr());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    auto threadBody = [&](const int ithr, const int nthr) {
        const auto& p = execParamsPerThread[ithr];
        auto arg = jGridSamplesExecArgs();

        arg.src = srcData;
        arg.grid = gridData + p.gridStartB;
        arg.dst = dstData + p.dstStartB;
        arg.workAmount = p.workAmount;
        arg.channelsNum = p.channelsNum;
        arg.dstChStepB = p.dstChStepB;
        arg.srcWidthFl = &p.srcWidthFl;
        arg.srcHeightFl = &p.srcHeightFl;

        (*jitKernel)(&arg);
    };

    parallel_nt(0, threadBody);
}

void GridSample::executeDynamicImpl(dnnl::stream strm) {
    const void* srcData = getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetPtr();
    const uint8_t* gridData = reinterpret_cast<uint8_t*>(getParentEdgeAt(IN_GRID)->getMemoryPtr()->GetPtr());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    const uint64_t dataElPerVec = jitKernel->getDataElPerVec();

    auto threadBody = [&](const int ithr, const int nthr) {
//        const uint64_t wpt = ((totalWork / dataElPerVec) / nthr + 1) * dataElPerVec;
//        const uint64_t start = std::min(wpt * ithr, totalWork);
//        const uint64_t end = std::min(wpt * (ithr + 1), totalWork);
//        const uint64_t workAmount = end - start;
//
//        auto arg = gatherJitExecArgs();
//
//        arg.src = srcData;
//        arg.dst = dstData + afterAxisSizeInBytes * start;
//        arg.indices = gridData;
//        arg.start = &start;
//        arg.axisDim = &axisDim;
//        arg.afterAxSize = afterAxisSize;
//        arg.axisAndAfterAxisSizeB = &axisAndAfterAxisSizeInBytes;
//        arg.srcAfterBatchSizeB = &srcAfterBatchSizeInBytes;
//        arg.betweenBatchAndAxisSize = &betweenBatchAndAxisSize;
//        arg.specIndicesSize = &specIndicesSize;
//        arg.workAmount = workAmount;
//
//        const uint64_t idxElPerVec = jitKernel->getIdxElPerVec();
//        int permIdxMask[16];
//        int beforeAxisDiff[16];
//        if (afterAxisSize == 1 && specIndicesSize < idxElPerVec) {
//            permIdxMask[0] = idxElPerVec - specIndicesSize;
//            int div = idxElPerVec / specIndicesSize;
//            int remainder = idxElPerVec % specIndicesSize;
//            for (int i = 1; i < idxElPerVec; i++) {
//                permIdxMask[i] = permIdxMask[i - 1] + 1;
//                if (permIdxMask[i] == idxElPerVec)
//                    permIdxMask[i] = idxElPerVec - specIndicesSize;
//            }
//            for (int i = 0; i < idxElPerVec; i++) {
//                if (((start + i) % specIndicesSize) < (specIndicesSize - remainder))
//                    beforeAxisDiff[i] = axisDim * div;
//                else
//                    beforeAxisDiff[i] = axisDim * (div + 1);
//            }
//            arg.permIdxMask = permIdxMask;
//            arg.beforeAxisDiff = beforeAxisDiff;
//        }
//
//        (*jitKernel)(&arg);
    };

    parallel_nt(0, threadBody);
}

std::vector<VectorDims> GridSample::shapeInfer() const {
    return Node::shapeInferGeneric(PortMask(1, 2, 3));
}

bool GridSample::created() const {
    return getType() == Type::GridSample;
}
