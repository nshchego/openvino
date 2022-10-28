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
            errorMessage = "Not supported CPU instructions set.";
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
    if (gridShape.isStatic() && gridShape.getDims()[3] != 2)
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
        default:
            THROW_ERROR << "supports only BILINEAR, BICUBIC, NEAREST interpolation modes.";
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
        default:
            THROW_ERROR << "supports only BORDER, REFLECTION, ZEROS paddings modes.";
    }
}

void GridSample::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    const auto& dataDims = getInputShapeAtPort(IN_DATA).getDims();

    dataPrecision = getOriginalInputPrecisionAtPort(IN_DATA);
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
    } else if (x64::mayiuse(x64::avx)) {
        implType = jit_avx;
    }

    addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                          {LayoutType::ncsp, gridPrecision}},
                         {{LayoutType::ncsp, dataPrecision}},
                         implType,
                         isDynamicNode());
}

void GridSample::createPrimitive() {
    jGridSampleConfParams jcp;

    jcp.inDataPrc     = dataPrecision;
    jcp.gridPrc       = gridPrecision;
    jcp.dynamicShapes = isDynamicNode();
    jcp.alignCorners  = alignCorners;
    jcp.interpolationMode = interpolationMode;
    jcp.paddingMode   = paddingMode;

    const auto& srcDataDims = getInputShapeAtPort(IN_DATA).getDims();
    if (!jcp.dynamicShapes) {
        const auto& dstShape     = getOutputShapeAtPort(0).getDims();
        jcp.batchNum      = srcDataDims[0];
        jcp.cannelNum     = srcDataDims[1];
        jcp.srcBatchStepB = std::accumulate(srcDataDims.begin() + 1, srcDataDims.end(), dataTypeSize, std::multiplies<Dim>());
    } else {
        jcp.batchNum  = srcDataDims[0] == Shape::UNDEFINED_DIM ? 1lu : srcDataDims[0];
        jcp.cannelNum = srcDataDims[1] == Shape::UNDEFINED_DIM ? 1lu : srcDataDims[1];
    }

    if (x64::mayiuse(x64::avx512_core)) {
        jitKernel.reset(new JitGridSampleKernel<x64::avx512_core>(jcp));
    } else if (x64::mayiuse(x64::avx2)) {
        jitKernel.reset(new JitGridSampleKernel<x64::avx2>(jcp));
    } else if (x64::mayiuse(x64::avx)) {
        jitKernel.reset(new JitGridSampleKernel<x64::avx>(jcp));
    } else if (x64::mayiuse(x64::sse41)) {
        jitKernel.reset(new JitGridSampleKernel<x64::sse41>(jcp));
    }
    if (!jitKernel) {
        THROW_ERROR << " could not create JIT kernel.";
    }
    jitKernel->create_ker();

    nthr = parallel_get_max_threads();
    execParamsPerThread.resize(nthr);
    if (!x64::mayiuse(x64::avx512_core)) {
        const auto dataElPerVec = jitKernel->getDataElPerVec();
        parallel_nt(nthr, [&](const int ithr, const int nthr) {
            auto& p = execParamsPerThread[ithr];

            p.srcHeightF.resize(dataElPerVec);
            p.srcWidthF.resize(dataElPerVec);
            p.srcWidthB.resize(dataElPerVec);
            p.dataTypeSize.resize(dataElPerVec);
            p.srcHeightSub1F.resize(dataElPerVec);
            p.srcWidthSub1F.resize(dataElPerVec);
            p.srcHeightMul2F.resize(dataElPerVec);
            p.srcWidthMul2F.resize(dataElPerVec);
            p.srcHeightMul2Sub1F.resize(dataElPerVec);
            p.srcWidthMul2Sub1F.resize(dataElPerVec);
            if (alignCorners) {
                p.wDenormCoefF.resize(dataElPerVec);
                p.hDenormCoefF.resize(dataElPerVec);
            }
            if (interpolationMode == InterpolationMode::BICUBIC) {
                p.buffer.resize(dataElPerVec * dataTypeSize * 20); // TODO: reduce
            }
        });
    }

    Node::createPrimitive();
}

void GridSample::prepareParams() {
    auto& dataMemPtr = getParentEdgeAt(IN_DATA)->getMemoryPtr();
    if (!dataMemPtr || !dataMemPtr->isAllocated())
        THROW_ERROR << " has not allocated input data memory.";
    auto& gridMemPtr = getParentEdgeAt(IN_GRID)->getMemoryPtr();
    if (!gridMemPtr || !gridMemPtr->isAllocated())
        THROW_ERROR << " has not allocated input grid memory.";
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->isAllocated())
        THROW_ERROR << " has not allocated output memory.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_ERROR << " has unidentified preferable primitive descriptor.";

    const uint64_t dataElPerVec = jitKernel->getDataElPerVec();
    const auto& srcDataShape = dataMemPtr->getStaticDims();
    const auto& dstShape     = dstMemPtr->getStaticDims();
    const uint64_t totalWork = dstShape[2] * dstShape[3];
    const uint64_t wpt = ((totalWork / dataElPerVec) / nthr + 1) * dataElPerVec;

    parallel_nt(nthr, [&](const int ithr, const int nthr) {
        const uint64_t dstStart = std::min(wpt * ithr, totalWork);
        const uint64_t dstEnd = std::min(wpt * (ithr + 1), totalWork);
printf("[%d] start: %lu; end: %lu; wa: %lu\n", ithr, dstStart, dstEnd, dstEnd - dstStart);

        auto& p = execParamsPerThread[ithr];

        p.batchNum      = srcDataShape[0];
        p.channelsNum   = srcDataShape[1];
        p.srcHeightF[0] = srcDataShape[2];
        p.srcWidthF[0]  = srcDataShape[3];

        p.workAmount = dstEnd - dstStart;
        p.gridStartB = dstStart * 2 * gridTypeSize;
        p.dstStartB  = dstStart * dataTypeSize;

        p.srcBatchStepB  = std::accumulate(srcDataShape.begin() + 1, srcDataShape.end(), dataTypeSize, std::multiplies<Dim>());
        p.gridBatchStepB = (dstShape[2] * dstShape[3] - p.workAmount) * 2 * gridTypeSize;
        p.dstBatchStepB  = (dstShape[1] * dstShape[2] * dstShape[3] - p.workAmount) * dataTypeSize;

        p.srcChannelStepB = srcDataShape[2] * srcDataShape[3] * dataTypeSize;
        p.dstChannelStepB = dstShape[2] * dstShape[3] * dataTypeSize;
        p.dataTypeSize[0] = dataTypeSize;

        p.srcHeightSub1F[0] = p.srcHeightF[0] - 1.f;
        p.srcWidthSub1F[0]  = p.srcWidthF[0]  - 1.f;
        p.srcHeightMul2F[0] = p.srcHeightF[0] * 2.f;
        p.srcWidthMul2F[0]  = p.srcWidthF[0]  * 2.f;
        if (interpolationMode == InterpolationMode::BICUBIC && srcDataShape[3] >= 4) {
            p.srcWidthB[0] = (srcDataShape[3] - 3) * dataTypeSize;
        } else {
            p.srcWidthB[0] = srcDataShape[3] * dataTypeSize;
        }
        if (alignCorners) {
            p.srcHeightMul2Sub1F[0] = p.srcHeightF[0] == 1.f ? 1.f : p.srcHeightSub1F[0] * 2.f;
            p.srcWidthMul2Sub1F[0]  = p.srcWidthF[0]  == 1.f ? 1.f : p.srcWidthSub1F[0]  * 2.f;
            p.wDenormCoefF[0] = (p.srcWidthF[0]  - 1.f) / 2.f;
            p.hDenormCoefF[0] = (p.srcHeightF[0] - 1.f) / 2.f;
        } else {
            p.srcHeightMul2Sub1F[0] = p.srcHeightMul2F[0] - 1.f;
            p.srcWidthMul2Sub1F[0]  = p.srcWidthMul2F[0]  - 1.f;
        }
        if (!x64::mayiuse(x64::avx512_core)) {
            std::fill(p.srcHeightF.begin(),         p.srcHeightF.end(),         p.srcHeightF[0]);
            std::fill(p.srcWidthF.begin(),          p.srcWidthF.end(),          p.srcWidthF[0]);
            std::fill(p.dataTypeSize.begin(),       p.dataTypeSize.end(),       p.dataTypeSize[0]);
            std::fill(p.srcHeightSub1F.begin(),     p.srcHeightSub1F.end(),     p.srcHeightSub1F[0]);
            std::fill(p.srcWidthSub1F.begin(),      p.srcWidthSub1F.end(),      p.srcWidthSub1F[0]);
            std::fill(p.srcHeightMul2F.begin(),     p.srcHeightMul2F.end(),     p.srcHeightMul2F[0]);
            std::fill(p.srcWidthMul2F.begin(),      p.srcWidthMul2F.end(),      p.srcWidthMul2F[0]);
            std::fill(p.srcWidthB.begin(),          p.srcWidthB.end(),          p.srcWidthB[0]);
            std::fill(p.srcHeightMul2Sub1F.begin(), p.srcHeightMul2Sub1F.end(), p.srcHeightMul2Sub1F[0]);
            std::fill(p.srcWidthMul2Sub1F.begin(),  p.srcWidthMul2Sub1F.end(),  p.srcWidthMul2Sub1F[0]);
            if (alignCorners) {
                std::fill(p.wDenormCoefF.begin(), p.wDenormCoefF.end(), p.wDenormCoefF[0]);
                std::fill(p.hDenormCoefF.begin(), p.hDenormCoefF.end(), p.hDenormCoefF[0]);
            }
        }
    });
}

void GridSample::execute(dnnl::stream strm) {
    const void*    srcData = getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetPtr();
    const uint8_t* gridData = reinterpret_cast<uint8_t*>(getParentEdgeAt(IN_GRID)->getMemoryPtr()->GetPtr());
    uint8_t*       dstData = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

// DEBUG
//std::cout << "\nINPUT DATA: " << std::endl;
//float* srcDataF = reinterpret_cast<float*>(getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetPtr());
//for (int i = 0; i < getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetSize() / sizeof(float); i++) {
//    if (i % jitKernel->getDataElPerVec() == 0)
//        std::cout << "| ";
//    std::cout << srcDataF[i] << "; ";
//}
//std::cout << std::endl;
//
//std::cout << "GRID DATA: " << std::endl;
//float* gridDataF = reinterpret_cast<float*>(getParentEdgeAt(IN_GRID)->getMemoryPtr()->GetPtr());
//for (int i = 0; i < getParentEdgeAt(IN_GRID)->getMemoryPtr()->GetSize() / gridTypeSize; i++) {
//    if (i % jitKernel->getGridElPerVec() == 0)
//        std::cout << "| ";
//    std::cout << gridDataF[i] << "; ";
//}
//std::cout << std::endl;
//std::cout << "batchNum: " << execParamsPerThread[0].batchNum << std::endl;
// DEBUG

    auto threadBody = [&](const int ithr, const int nthr) {
        const auto& p = execParamsPerThread[ithr];
        auto arg = jGridSamplesExecArgs();
        if (p.workAmount == 0lu) {
            return;
        }

        arg.src                = srcData;
        arg.grid               = gridData + p.gridStartB;
        arg.dst                = dstData  + p.dstStartB;
        arg.batchNum           = p.batchNum;
        arg.channelsNum        = p.channelsNum;
        arg.srcHeightF         = p.srcHeightF.data();
        arg.srcWidthF          = p.srcWidthF.data();
        arg.srcWidthB          = p.srcWidthB.data();
        arg.srcChannelStepB    = p.srcChannelStepB;
        arg.dstChannelStepB    = p.dstChannelStepB;
        arg.srcBatchStepB      = p.srcBatchStepB;
        arg.gridBatchStepB     = p.gridBatchStepB;
        arg.dstBatchStepB      = p.dstBatchStepB;
        arg.srcHeightSub1F     = p.srcHeightSub1F.data();
        arg.srcWidthSub1F      = p.srcWidthSub1F.data();
        arg.srcWidthMul2F      = p.srcWidthMul2F.data();
        arg.srcHeightMul2F     = p.srcHeightMul2F.data();
        arg.srcHeightMul2Sub1F = p.srcHeightMul2Sub1F.data();
        arg.srcWidthMul2Sub1F  = p.srcWidthMul2Sub1F.data();
        arg.wDenormCoefF       = p.wDenormCoefF.data();
        arg.hDenormCoefF       = p.hDenormCoefF.data();
        arg.dataTypeSize       = p.dataTypeSize.data();
        arg.buffer             = p.buffer.data();
        arg.workAmount         = p.workAmount;

        (*jitKernel)(&arg);
    };

    parallel_nt(nthr, threadBody);

// DEBUG
std::cout << "OUTPUT: " << std::endl;
float* dstDataF = reinterpret_cast<float*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
//int* dstDataF = reinterpret_cast<int*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
//char* dstDataF = reinterpret_cast<char*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
for (int i = 0; i < getChildEdgeAt(0)->getMemoryPtr()->GetSize() / sizeof(float); i++) {
    if (i % jitKernel->getDataElPerVec() == 0)
        std::cout << "| ";
    std::cout << dstDataF[i] << "; ";
}
std::cout << std::endl << std::endl;
// DEBUG
}

void GridSample::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

std::vector<VectorDims> GridSample::shapeInfer() const {
    return Node::shapeInferGeneric(PortMask(1));
}

bool GridSample::created() const {
    return getType() == Type::GridSample;
}
