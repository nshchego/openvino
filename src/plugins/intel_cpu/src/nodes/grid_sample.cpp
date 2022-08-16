// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "grid_sample.hpp"
#include "ie_parallel.hpp"
#include <ngraph/opsets/opset1.hpp>

using namespace InferenceEngine;
using namespace dnnl::impl::cpu;
using namespace ov::intel_cpu;
using namespace ov::intel_cpu::node;

#define THROW_ERROR IE_THROW() << getTypeStr() << " node with name '" << getName() << "' "


bool GridSample::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
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

    Precision dataPrecision = getOriginalInputPrecisionAtPort(IN_DATA);
    Precision gridPrecision = Precision::FP32;
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

    jcp.inDataPrc = getRuntimePrecision(); // TODO: fix
    jcp.gridPrc = getRuntimePrecision(); // TODO: fix
    jcp.dynamicShapes = isDynamicNode();
    jcp.alignCorners = alignCorners;
    jcp.interpolationMode = interpolationMode;
    jcp.paddingMode = paddingMode;

    if (!jcp.dynamicShapes) {
        const auto& srcDataShape = getInputShapeAtPort(IN_DATA).getDims();
        const auto& dstShape = getOutputShapeAtPort(0).getDims();
        jcp.batchNum = srcDataShape[0];
        jcp.srcBatchStepB = std::accumulate(srcDataShape.begin() + 1, srcDataShape.end(), dataTypeSize, std::multiplies<Dim>());
//        jcp.dstBatchStepB = (dstShape[1] - 1) * dstShape[2] * dstShape[3] * dataTypeSize;
    }

    if (x64::mayiuse(x64::avx512_core)) {
        jitKernel.reset(new jitGridSampleKernel<x64::avx512_core>(jcp));
    } else if (x64::mayiuse(x64::avx2)) {
        jitKernel.reset(new jitGridSampleKernel<x64::avx2>(jcp));
    } else if (x64::mayiuse(x64::sse41)) {
        jitKernel.reset(new jitGridSampleKernel<x64::sse41>(jcp));
    }
    if (jitKernel) {
        jitKernel->create_ker();
    } else {
        THROW_ERROR << " could not create JIT kernel.";
    }

    if (!isDynamicNode()) {
        const uint64_t dataElPerVec = jitKernel->getDataElPerVec();
        const auto& srcDataShape = getInputShapeAtPort(IN_DATA).getDims();
        const auto& dstShape = getOutputShapeAtPort(0).getDims();
        const uint64_t totalWork = dstShape[2] * dstShape[3];
printf("TotalWork: %lu\n", totalWork);
        const uint64_t nthr = parallel_get_max_threads();
        const uint64_t wpt = ((totalWork / dataElPerVec) / nthr + 1) * dataElPerVec;
        execParamsPerThread.resize(nthr);

        parallel_nt(nthr, [&](const int ithr, const int nthr) {
            const uint64_t dstStart = std::min(wpt * ithr, totalWork);
            const uint64_t dstEnd = std::min(wpt * (ithr + 1), totalWork);
printf("[%d] Start: %lu; End: %lu\n", ithr, dstStart, dstEnd);

            auto& p = execParamsPerThread[ithr];

            p.workAmount = dstEnd - dstStart;
            p.dstStartB = dstStart * dataTypeSize;
            p.gridStartB = dstStart * 2 * gridTypeSize;
            p.dstBatchStepB = (dstShape[1] * dstShape[2] * dstShape[3] - p.workAmount) * dataTypeSize;
            p.gridBatchStepB = (dstShape[2] * dstShape[3] - p.workAmount) * 2 * gridTypeSize;

            p.channelsNum = srcDataShape[1];
            p.srcHeightF = srcDataShape[2];
            p.srcWidthF = srcDataShape[3];
            if (interpolationMode == InterpolationMode::BICUBIC && srcDataShape[3] >= 4) {
                p.srcWidthB = (srcDataShape[3] - 3) * dataTypeSize;
            } else {
                p.srcWidthB = srcDataShape[3] * dataTypeSize;
            }
            p.srcHeightSub1F = p.srcHeightF - 1.f;
            p.srcWidthSub1F = p.srcWidthF - 1.f;
            p.srcHeightMul2F = p.srcHeightF * 2.f;
            p.srcWidthMul2F = p.srcWidthF * 2.f;
            if (alignCorners) {
                p.srcHeightMul2Sub1F = p.srcHeightSub1F * 2.f;
                p.srcWidthMul2Sub1F = p.srcWidthSub1F * 2.f;
                p.wDenormCoef = (p.srcWidthF - 1.f) / 2.f;
                p.hDenormCoef = (p.srcHeightF - 1.f) / 2.f;
            } else {
                p.srcHeightMul2Sub1F = p.srcHeightMul2F - 1.f;
                p.srcWidthMul2Sub1F = p.srcWidthMul2F - 1.f;
            }
            p.srcChannelStepB = srcDataShape[2] * srcDataShape[3] * dataTypeSize;
            p.dstChannelStepB = dstShape[2] * dstShape[3] * dataTypeSize;
        });
    }

    Node::createPrimitive();
}

void GridSample::prepareParams() {
    auto& dataMemPtr = getParentEdgeAt(IN_DATA)->getMemoryPtr();
    if (!dataMemPtr || !dataMemPtr->isAllocated())
        THROW_ERROR << " has not allocated input data memory.";
    auto& idxMemPtr = getParentEdgeAt(IN_GRID)->getMemoryPtr();
    if (!idxMemPtr || !idxMemPtr->isAllocated())
        THROW_ERROR << " has not allocated input grid memory.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_ERROR << " has unidentified preferable primitive descriptor.";
}

void GridSample::execute(dnnl::stream strm) {
    const void* srcData = getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetPtr();
    const uint8_t* gridData = reinterpret_cast<uint8_t*>(getParentEdgeAt(IN_GRID)->getMemoryPtr()->GetPtr());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

// DEBUG
std::cout << "\nINPUT DATA: " << std::endl;
float* srcDataF = reinterpret_cast<float*>(getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetPtr());
for (int i = 0; i < getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetSize() / sizeof(float); i++) {
    if (i % jitKernel->getDataElPerVec() == 0)
        std::cout << "| ";
    std::cout << srcDataF[i] << "; ";
}
std::cout << std::endl;

std::cout << "GRID DATA: " << std::endl;
float* gridDataF = reinterpret_cast<float*>(getParentEdgeAt(IN_GRID)->getMemoryPtr()->GetPtr());
for (int i = 0; i < getParentEdgeAt(IN_GRID)->getMemoryPtr()->GetSize() / gridTypeSize; i++) {
    if (i % jitKernel->getGridElPerVec() == 0)
        std::cout << "| ";
    std::cout << gridDataF[i] << "; ";
}
std::cout << std::endl;
// DEBUG

    auto threadBody = [&](const int ithr, const int nthr) {
        const auto& p = execParamsPerThread[ithr];
        auto arg = jGridSamplesExecArgs();

        arg.src                 = srcData;
        arg.grid                = gridData + p.gridStartB;
        arg.dst                 = dstData + p.dstStartB;
        arg.channelsNum         = p.channelsNum;
        arg.srcHeightF         = &p.srcHeightF;
        arg.srcWidthF          = &p.srcWidthF;
        arg.srcWidthB           = &p.srcWidthB;
        arg.srcChannelStepB     = p.srcChannelStepB;
        arg.dstChannelStepB     = p.dstChannelStepB;
        arg.gridBatchStepB      = p.gridBatchStepB;
        arg.dstBatchStepB       = p.dstBatchStepB;
        arg.srcHeightSub1F     = &p.srcHeightSub1F;
        arg.srcWidthSub1F      = &p.srcWidthSub1F;
        arg.srcWidthMul2F      = &p.srcWidthMul2F;
        arg.srcHeightMul2F     = &p.srcHeightMul2F;
        arg.srcHeightMul2Sub1F = &p.srcHeightMul2Sub1F;
        arg.srcWidthMul2Sub1F  = &p.srcWidthMul2Sub1F;
        arg.wDenormCoef         = &p.wDenormCoef;
        arg.hDenormCoef         = &p.hDenormCoef;
        arg.one                 = &p.one;
//        arg.halfVal             = &p.halfVal;
        arg.workAmount          = p.workAmount;

        (*jitKernel)(&arg);
    };

    parallel_nt(0, threadBody);

// DEBUG
std::cout << "OUTPUT: " << std::endl;
float* dstDataF = reinterpret_cast<float*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
//int* dstDataF = reinterpret_cast<int*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
for (int i = 0; i < getChildEdgeAt(0)->getMemoryPtr()->GetSize() / sizeof(float); i++) {
    if (i % jitKernel->getDataElPerVec() == 0)
        std::cout << "| ";
    std::cout << dstDataF[i] << "; ";
}
std::cout << std::endl;
// DEBUG
}

void GridSample::executeDynamicImpl(dnnl::stream strm) {
    const void* srcData = getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetPtr();
    const uint8_t* gridData = reinterpret_cast<uint8_t*>(getParentEdgeAt(IN_GRID)->getMemoryPtr()->GetPtr());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    const auto& srcDataShape = getInputShapeAtPort(IN_DATA).getDims();
    const auto& dstShape = getOutputShapeAtPort(0).getDims();
    const uint64_t totalWork = dstShape[2] * dstShape[3];
printf("TotalWork: %lu\n", totalWork);
    const uint64_t dataElPerVec = jitKernel->getDataElPerVec();

    auto threadBody = [&](const int ithr, const int nthr) {
        const uint64_t wpt = ((totalWork / dataElPerVec) / nthr + 1) * dataElPerVec;
        const uint64_t start = std::min(wpt * ithr, totalWork);
        const uint64_t end = std::min(wpt * (ithr + 1), totalWork);

        const float srcHeight = srcDataShape[2], srcWidth = srcDataShape[3];
        const float srcHeightSub1F = srcHeight - 1.f, srcWidthSub1F = srcWidth - 1.f;
        const float srcHeightMul2F = srcHeight * 2.f;
        const float srcWidthMul2F  = srcWidth * 2.f;
        float srcHeightMul2Sub1F, srcWidthMul2Sub1F;
        if (alignCorners) {
            srcHeightMul2Sub1F = srcHeightSub1F * 2.f;
            srcWidthMul2Sub1F  = srcWidthSub1F * 2.f;
        } else {
            srcHeightMul2Sub1F = srcHeightMul2F - 1.f;
            srcWidthMul2Sub1F  = srcWidthMul2F - 1.f;
        };
        const float hDenormCoef = (srcHeight - 1.f) / 2.f;
        const float wDenormCoef = (srcWidth  - 1.f) / 2.f;
        const float halfVal = 0.5f, oneVal = 1.f;
        const uint64_t srcWidthB = srcDataShape[3] * dataTypeSize;
        const uint64_t srcChannelStepB = srcDataShape[2] * srcWidthB;
        const uint64_t srcBatchStepB = srcChannelStepB * srcDataShape[1];
        const uint64_t dstBatchStepB = (dstShape[1] - 1) * dstShape[2] * dstShape[3] * dataTypeSize;

        auto arg = jGridSamplesExecArgs();

        arg.src                 = srcData;
        arg.grid                = gridData + start * 2 * dataTypeSize;
        arg.dst                 = dstData + start * dataTypeSize;
        arg.batchNum            = srcDataShape[0];
        arg.channelsNum         = srcDataShape[1];
        arg.srcHeightF         = &srcHeight;
        arg.srcWidthF          = &srcWidth;
        arg.srcWidthB           = &srcWidthB;
        arg.srcChannelStepB     = srcChannelStepB;
        arg.dstChannelStepB     = dstShape[2] * dstShape[3] * dataTypeSize;
        arg.srcBatchStepB       = &srcBatchStepB;
        arg.dstBatchStepB       = dstBatchStepB;
        arg.srcHeightSub1F     = &srcHeightSub1F;
        arg.srcWidthSub1F      = &srcWidthSub1F;
        arg.srcHeightMul2F     = &srcHeightMul2F;
        arg.srcWidthMul2F      = &srcWidthMul2F;
        arg.srcHeightMul2Sub1F = &srcHeightMul2Sub1F;
        arg.srcWidthMul2Sub1F  = &srcWidthMul2Sub1F;
        arg.hDenormCoef         = &hDenormCoef;
        arg.wDenormCoef         = &wDenormCoef;
        arg.one                 = &oneVal;
//        arg.halfVal             = &halfVal;
        arg.workAmount          = end - start;

        (*jitKernel)(&arg);
    };

    parallel_nt(0, threadBody);
}

std::vector<VectorDims> GridSample::shapeInfer() const {
    return Node::shapeInferGeneric(PortMask(1, 2));
}

bool GridSample::created() const {
    return getType() == Type::GridSample;
}
