// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather.hpp"

#include "partitioned_mem_mgr.h"

// #include <cstdint>
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
// #include <openvino/opsets/opset1.hpp>
// #include <string>
// #include <vector>

#include "common/cpu_memcpy.h"
#include "kernels/x64/gather.hpp"
#include "openvino/core/parallel.hpp"
#include "ov_ops/gather_compressed.hpp"
#include "selective_build.h"
#include "shape_inference/custom/gather.hpp"
// #include "utils/general_utils.h"
#include "utils/ngraph_utils.hpp"

using namespace dnnl::impl::cpu;

namespace ov {
namespace intel_cpu {
namespace node {

bool Gather::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (op->get_output_element_type(0) == element::string) {
            return false;
        }
        if (!one_of(op->get_type_info(),
                    op::v7::Gather::get_type_info_static(),
                    op::v8::Gather::get_type_info_static(),
                    op::internal::GatherCompressed::get_type_info_static())) {
            errorMessage = "Not supported Gather operation version. CPU plug-in supports only 7, 8 and GatherCompressed versions.";
            return false;
        }

        if (!isDynamicNgraphNode(op) && !is_type<op::v0::Constant>(op->get_input_node_ptr(GATHER_AXIS))) {
            errorMessage = "Only Constant operation on 'axis' input is supported for static node.";
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}

Gather::Gather(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, GatherShapeInferFactory(op)),
      m_batch_dims(0) {
    std::string error_message;
    if (!isSupportedOperation(op, error_message)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(error_message);
    }

    if (!one_of(op->get_input_size(), 3lu, 4lu, 5lu) || op->get_output_size() != 1lu) {
        THROW_CPU_NODE_ERR("has incorrect number of input/output edges.");
    }

    const auto& dataShape = getInputShapeAtPort(GATHER_DATA);
    m_data_shape_static = dataShape.isStatic();
    dataSrcRank = dataShape.getRank();

    const auto& idxShape = getInputShapeAtPort(GATHER_INDICES);
    m_idx_shape_static = idxShape.isStatic();
    const auto indicesRank = idxShape.getRank();
    if (dataSrcRank == 0lu || indicesRank == 0lu) {
        THROW_CPU_NODE_ERR("has incorrect input parameters ranks.");
    }

    if (auto gather = as_type<op::v8::Gather>(op.get())) {
        m_batch_dims = static_cast<int>(gather->get_batch_dims());
        // WA for NMS->Gather construction. NMS fills part of the output blob by the -1 if these values
        // must not be taken into account. There is appropriate pass that looks for such subgraphs
        // and sets the dontReverseIndices flag.
        const auto& rti = op->get_rt_info();
        const auto& reverse = rti.find("dontReverseIndices");
        if (reverse == rti.end()) {
            m_reverse_indexing = true;
        } else {
            m_reverse_indexing = false;
        }
    } else if (auto gather = as_type<op::v7::Gather>(op.get())) {
        m_batch_dims = static_cast<int>(gather->get_batch_dims());
        m_reverse_indexing = false;
    } else if (auto gather = as_type<op::internal::GatherCompressed>(op.get())) {
        m_compressed = true;
        m_batch_dims = static_cast<int>(gather->get_batch_dims());
        m_reverse_indexing = true;
    }

    if (m_batch_dims < 0) {
        m_batch_dims += indicesRank;
    }
    if (m_batch_dims < 0 || m_batch_dims > std::min(static_cast<int>(dataSrcRank), static_cast<int>(indicesRank))) {
        THROW_CPU_NODE_ERR("has incorrect batch_dims ", m_batch_dims, "!");
    }

    if (auto axis_op = as_type<op::v0::Constant>(op->get_input_node_ptr(GATHER_AXIS))) {
        m_axis_input_const = true;
        m_axis = axis_op->cast_vector<int>()[0];
        if (m_axis < 0) {
            m_axis += dataSrcRank;
        }
        if (m_axis < 0 || m_axis >= dataSrcRank || m_batch_dims > m_axis) {
            THROW_CPU_NODE_ERR("has incorrect input parameter axis value: ", m_axis);
        }
    }

    if (auto indices = as_type<op::v0::Constant>(op->get_input_node_ptr(GATHER_INDICES))) {
        constIndices = indices->cast_vector<int>();
    }
}

void Gather::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    m_data_et_size = getOriginalInputPrecisionAtPort(GATHER_DATA).size();
    m_idx_et_size = getOriginalInputPrecisionAtPort(GATHER_INDICES).size();
    // if () {
    //     m_idx_et_size = 4;
    // }

    const auto& dataDims = getInputShapeAtPort(GATHER_DATA).getDims();
    if (m_axis_input_const && m_data_shape_static) {
        axisDim = dataDims[m_axis];
        beforeAxisSize = std::accumulate(dataDims.begin(), dataDims.begin() + m_axis, 1lu, std::multiplies<Dim>());
        betweenBatchAndAxisSize =
            std::accumulate(dataDims.begin() + m_batch_dims, dataDims.begin() + m_axis, 1lu, std::multiplies<Dim>());
        afterAxisSize = std::accumulate(dataDims.begin() + m_axis + 1, dataDims.end(), 1lu, std::multiplies<Dim>());

        afterAxisSizeInBytes = afterAxisSize * m_data_et_size;
        axisAndAfterAxisSize = axisDim * afterAxisSize;
        axisAndAfterAxisSizeInBytes = axisDim * afterAxisSizeInBytes;
        srcAfterBatchSize = betweenBatchAndAxisSize * axisAndAfterAxisSize;
        srcAfterBatchSizeInBytes = betweenBatchAndAxisSize * axisAndAfterAxisSizeInBytes;
    }
    if (m_data_shape_static) {
        beforeBatchSize = std::accumulate(dataDims.begin(), dataDims.begin() + m_batch_dims, 1lu, std::multiplies<Dim>());
    }
    if (m_idx_shape_static) {
        const auto& idxDims = getInputShapeAtPort(GATHER_INDICES).getDims();
        specIndicesSize = std::accumulate(idxDims.begin() + m_batch_dims, idxDims.end(), 1lu, std::multiplies<Dim>());

        if (m_data_shape_static) {
            specIdxAndAfterAxSize = specIndicesSize * afterAxisSize;
            specIdxAndAfterAxSizeB = specIndicesSize * afterAxisSizeInBytes;
            totalWork = beforeBatchSize * betweenBatchAndAxisSize * specIndicesSize * afterAxisSize;
        }
    }

    auto dataPrecision = getOriginalInputPrecisionAtPort(GATHER_DATA);
    if (m_compressed) {
        if (!one_of(dataPrecision, element::u8, element::u4, element::i8, element::i4)) {
            dataPrecision = element::f32;
        }

        auto scalePrecision = getOriginalInputPrecisionAtPort(GATHER_SCALE);
        if (scalePrecision != element::f32) {
            scalePrecision = element::f32;
        }

        auto outPrecision = getOriginalOutputPrecisionAtPort(0);
        if (!one_of(outPrecision, element::f32, element::f16, element::bf16)) {
            outPrecision = element::f32;
        }
        scale_group_size =
            getInputShapeAtPort(GATHER_DATA).getElementsCount() / getInputShapeAtPort(GATHER_SCALE).getElementsCount();
        have_scalar_scale = getInputShapeAtPort(GATHER_SCALE).getElementsCount() == 1u;

        if (getOriginalInputsNumber() == 5u) {
            auto zpPrecision = getOriginalInputPrecisionAtPort(GATHER_ZP);
            if (zpPrecision != element::f32) {
                zpPrecision = element::f32;
            }

            have_zp = true;
            have_scalar_zp = getInputShapeAtPort(GATHER_ZP).getElementsCount() == 1u;
            zp_group_size =
                getInputShapeAtPort(GATHER_DATA).getElementsCount() / getInputShapeAtPort(GATHER_ZP).getElementsCount();
            addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                                  {LayoutType::ncsp, element::i32},
                                  {LayoutType::ncsp, element::i32},
                                  {LayoutType::ncsp, scalePrecision},
                                  {LayoutType::ncsp, zpPrecision}},
                                 {{LayoutType::ncsp, outPrecision}},
                                 ref_any);
        } else {
            addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                                  {LayoutType::ncsp, element::i32},
                                  {LayoutType::ncsp, element::i32},
                                  {LayoutType::ncsp, scalePrecision}},
                                 {{LayoutType::ncsp, outPrecision}},
                                 ref_any);
        }
        return;
    } else {
        // Implementation desc type will be redefined in the fn prepareParams if a kernel will be created.
        addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                              {LayoutType::ncsp, element::i32},
                              {LayoutType::ncsp, element::i32, m_axis_input_const}},
                             {{LayoutType::ncsp, dataPrecision}},
                             ref_any);
    }

    // Let's check for the special inPlace memory use case
    // in place only makes sense when we split by dense blocks since strided tensors are not supported by most nodes

    if (!m_axis_input_const) {
        return;
    }

    if (m_batch_dims != 0) {
        return;
    }

    if (constIndices.size() != 1) {
        return;
    }

    const auto& parentDims = inputShapes[0].getDims();
    const auto axisDim = parentDims[m_axis];
    if (Shape::UNDEFINED_DIM == axisDim) {
        return;
    }

    const auto indx = constIndices.front();
    const auto normIndex = indx < 0 ? static_cast<int64_t>(axisDim) + indx : indx;

    if (normIndex < 0 || normIndex >= static_cast<int64_t>(axisDim)) {
        return;
    }

    if (std::any_of(parentDims.begin(), parentDims.begin() + m_axis, [](size_t dim) {
            return dim != 1;
        })) {
        return;
    }

    addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                          {LayoutType::ncsp, element::i32},
                          {LayoutType::ncsp, element::i32, m_axis_input_const}},
                         {{LayoutType::ncsp, dataPrecision, false, GATHER_DATA}},
                         unknown);
}

void Gather::createPrimitive() {
    if (isInPlace()) {
        return;
    }
#if defined(OPENVINO_ARCH_X86_64)
    // Gather instruction is not supported by SSE.
    if (x64::mayiuse(x64::avx2)) {
        kernel::GatherCompileParams jcp;

        jcp.data_et_size = m_data_et_size;
        jcp.idx_et_size = m_idx_et_size;
        jcp.reverse_indexing = m_reverse_indexing;
        jcp.dynamic_shapes = isDynamicNode();
        jcp.batch_dims = m_batch_dims;
        if (!jcp.dynamic_shapes) {
            jcp.before_axis_size = beforeAxisSize;
            jcp.spec_idx_size = specIndicesSize;
            jcp.after_axis_size = afterAxisSize;
        } else {
            if (m_data_shape_static && m_axis_input_const) {
                jcp.before_axis_size = beforeAxisSize;
                jcp.after_axis_size = afterAxisSize;
            }
            if (m_idx_shape_static) {
                jcp.spec_idx_size = specIndicesSize;
            }
        }

        m_jit_kernel = kernel::JitKernel<kernel::GatherCompileParams, kernel::GatherCallArgs>::createInstance<kernel::Gather>(jcp);

        // if (m_jit_kernel) {
        //     if (auto selected_pd = getSelectedPrimitiveDescriptor()) {
        //         using namespace dnnl::impl::cpu;
        //         if (m_jit_kernel->getIsa() == x64::avx512_core) {
        //             selected_pd->setImplementationType(jit_avx512);
        //         } else if (m_jit_kernel->getIsa() == x64::avx2) {
        //             selected_pd->setImplementationType(jit_avx2);
        //         } else if (m_jit_kernel->getIsa() == x64::sse41) {
        //             selected_pd->setImplementationType(jit_sse42);
        //         }
        //     }
        // }
        if (m_jit_kernel) {
            if (!isDynamicNode()) {
                const uint64_t dataElPerVec = m_jit_kernel->getVectorLen() / m_data_et_size;
                const uint64_t nthr = parallel_get_max_threads();
                const uint64_t wpt = ((totalWork / dataElPerVec) / nthr + 1) * dataElPerVec;
                execParamsPerThread.resize(nthr);

                parallel_nt(nthr, [&](const int ithr, const int nthr) {
                    const uint64_t dst_start = std::min(wpt * ithr, totalWork);
                    const uint64_t dst_end = std::min(wpt * (ithr + 1), totalWork);

                    auto& p = execParamsPerThread[ithr];
                    p.work_amount = dst_end - dst_start;
                    p.dst_start = dst_start;
                    p.specIdxInBytes.resize(dataElPerVec);
                    p.idxBatchSumInBytes.resize(dataElPerVec);
                    p.dataBeforeAxisSumInBytes.resize(dataElPerVec);
                    p.betweenBatchAndAxisIter = (dst_start / specIndicesSize) % betweenBatchAndAxisSize;
                    for (uint64_t j = 0lu; j < dataElPerVec; j++) {
                        p.specIdxInBytes[j] = (((dst_start + j) / afterAxisSize) % specIndicesSize) * m_idx_et_size;
                        p.idxBatchSumInBytes[j] =
                            ((dst_start + j) / (betweenBatchAndAxisSize * specIndicesSize * afterAxisSize)) *
                            specIndicesSize * m_idx_et_size;
                        p.dataBeforeAxisSumInBytes[j] =
                            ((dst_start + j) / (specIndicesSize * afterAxisSize)) * axisAndAfterAxisSizeInBytes;
                    }
                    initShortParams(p, dst_start);
                });
            }
        }
    }
#endif
    Node::createPrimitive();
}

bool Gather::needPrepareParams() const {
    if (isInPlace()) {
        return false;
    }
    bool result = inputShapesModified();
    if (!m_axis_input_const)
        result = result || m_axis != (getSrcDataAtPortAs<const int32_t>(GATHER_AXIS))[0];
    return result;
}

void Gather::prepareParams() {
    auto dataMemPtr = getSrcMemoryAtPort(GATHER_DATA);
    if (!dataMemPtr || !dataMemPtr->isAllocated())
        THROW_CPU_NODE_ERR("has not allocated input data memory.");
    auto idxMemPtr = getSrcMemoryAtPort(GATHER_INDICES);
    if (!idxMemPtr || !idxMemPtr->isAllocated())
        THROW_CPU_NODE_ERR("has not allocated input indices memory.");
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_CPU_NODE_ERR("has unidentified preferable primitive descriptor.");

    // short 1D vector fast execution impl (typical in shape infer subgraph)
    canOptimize1DCase = false;
    if (dataSrcRank <= 1 && dataMemPtr->getDesc().getPrecision() == element::i32) {
        const auto& dataDims = dataMemPtr->getStaticDims();
        const auto& idxDims = idxMemPtr->getStaticDims();
        if ((dataDims.size() == 0 || (dataDims.size() == 1 && dataDims[0] <= 64)) &&
            (idxDims.size() == 0 || (idxDims.size() == 1 && idxDims[0] <= 64))) {
            canOptimize1DCase = true;
            return;
        }
    }

    if (!m_axis_input_const) {
        m_axis = (getSrcDataAtPortAs<const int32_t>(GATHER_AXIS))[0];
        if (m_axis < 0) {
            m_axis += dataSrcRank;
        }
        if (m_axis < 0 || m_axis >= dataSrcRank || m_batch_dims > m_axis) {
            THROW_CPU_NODE_ERR("has incorrect input parameter axis value: ", m_axis);
        }
    }

    if (!m_data_shape_static || !m_axis_input_const) {
        const auto& dataDims = dataMemPtr->getStaticDims();
        axisDim = dataDims[m_axis];
        beforeBatchSize =
            std::accumulate(dataDims.begin(), dataDims.begin() + m_batch_dims, 1lu, std::multiplies<uint64_t>());
        betweenBatchAndAxisSize =
            std::accumulate(dataDims.begin() + m_batch_dims, dataDims.begin() + m_axis, 1lu, std::multiplies<uint64_t>());
        afterAxisSize = std::accumulate(dataDims.begin() + m_axis + 1, dataDims.end(), 1lu, std::multiplies<uint64_t>());

        afterAxisSizeInBytes = afterAxisSize * m_data_et_size;
        axisAndAfterAxisSize = axisDim * afterAxisSize;
        axisAndAfterAxisSizeInBytes = axisDim * afterAxisSizeInBytes;
        srcAfterBatchSize = betweenBatchAndAxisSize * axisAndAfterAxisSize;
        srcAfterBatchSizeInBytes = betweenBatchAndAxisSize * axisAndAfterAxisSizeInBytes;

        if (m_idx_shape_static) {
            specIdxAndAfterAxSize = specIndicesSize * afterAxisSize;
            specIdxAndAfterAxSizeB = specIndicesSize * afterAxisSizeInBytes;
            totalWork = beforeBatchSize * betweenBatchAndAxisSize * specIndicesSize * afterAxisSize;
        }
    }

    if (!m_idx_shape_static) {
        const auto& idxDims = idxMemPtr->getStaticDims();
        specIndicesSize = std::accumulate(idxDims.begin() + m_batch_dims, idxDims.end(), 1lu, std::multiplies<uint64_t>());

        specIdxAndAfterAxSize = specIndicesSize * afterAxisSize;
        specIdxAndAfterAxSizeB = specIndicesSize * afterAxisSizeInBytes;
        totalWork = beforeBatchSize * betweenBatchAndAxisSize * specIndicesSize * afterAxisSize;
    }

#if defined(OPENVINO_ARCH_X86_64)
    const auto& selectedPD = getSelectedPrimitiveDescriptor();
    if (m_jit_kernel) {
        if (x64::mayiuse(x64::avx512_core)) {
            selectedPD->setImplementationType(jit_avx512);
        } else if (x64::mayiuse(x64::avx2)) {
            selectedPD->setImplementationType(jit_avx2);
        }
    }
#endif
}

void Gather::execute(dnnl::stream strm) {
    if (isInPlace()) {
        return;
    }

    if (canOptimize1DCase) {
        exec1DCase();
        return;
    }

    if (m_compressed) {
        return execCompressed();
    }
#if defined(OPENVINO_ARCH_X86_64)
    if (m_jit_kernel) {
        const void* srcIndices = getSrcDataAtPort(GATHER_INDICES);
        const void* srcData = getSrcDataAtPort(GATHER_DATA);
        uint8_t* dstData = getDstDataAtPortAs<uint8_t>(0);

        const uint64_t dataElPerVec = m_jit_kernel->getVectorLen() / m_data_et_size;

        auto threadBody = [&](const int ithr, const int nthr) {
            auto& p = execParamsPerThread[ithr];
            auto arg = kernel::GatherCallArgs();

            arg.src = srcData;
            arg.dst = dstData + p.dst_start * m_data_et_size;
            arg.indices = srcIndices;
            arg.start = &p.dst_start;
            arg.axisDim = &axisDim;
            arg.afterAxSize = afterAxisSize;
            arg.axisAndAfterAxisSizeB = &axisAndAfterAxisSizeInBytes;
            arg.srcAfterBatchSizeB = &srcAfterBatchSizeInBytes;
            arg.betweenBatchAndAxisSize = &betweenBatchAndAxisSize;
            arg.specIndicesSize = &specIndicesSize;
            arg.work_amount = p.work_amount;
            arg.specIdxB = p.specIdxInBytes.data();
            arg.idxBatchSumB = p.idxBatchSumInBytes.data();
            arg.dataBeforeAxisSumB = p.dataBeforeAxisSumInBytes.data();
            arg.betweenBatchAndAxisIter = p.betweenBatchAndAxisIter;

            const uint64_t idxElPerVec = m_jit_kernel->getVectorLen() / m_idx_et_size;

            if (afterAxisSize == 1 && specIndicesSize < idxElPerVec) {  // Elementwise short case.
                arg.permIdxMask = p.permIdxMask.data();
                arg.beforeAxisDiff = p.srcBeforeAxisDiff.data();
            } else if (afterAxisSize > 1 && afterAxisSize <= dataElPerVec) {  // Blocked short case.
                arg.afterAxIdxB = p.afterAxIdxInBytes.data();
                arg.specIdxDiff = p.specIdxDiff.data();
                arg.beforeAxisDiff = p.srcBeforeAxisDiff.data();
                arg.beforeAxisPermMask = p.beforeAxPermMask.data();
                arg.afterAxisPermMask = p.afterAxPermMask.data();
                arg.afterAxisSize = &afterAxisSize;
                arg.specIdxAndAfterAxIterB = p.specIdxAndAfterAxIterB;
                arg.specIdxAndAfterAxSizeB = specIdxAndAfterAxSizeB;
            }

            (*m_jit_kernel)(&arg);
        };

        parallel_nt(0, threadBody);

        return;
    }
#endif
    execReference();
}

void Gather::executeDynamicImpl(dnnl::stream strm) {
    if (isInPlace()) {
        return;
    }
    if (canOptimize1DCase) {
        exec1DCase();
        return;
    }

    if (m_compressed) {
        return execCompressed();
    }

#if defined(OPENVINO_ARCH_X86_64)
    if (m_jit_kernel) {
        const void* srcIndices = getSrcDataAtPort(GATHER_INDICES);
        const void* srcData = getSrcDataAtPort(GATHER_DATA);
        uint8_t* dstData = getDstDataAtPortAs<uint8_t>(0);

        const uint64_t dataElPerVec = m_jit_kernel->getVectorLen() / m_data_et_size;

        auto threadBody = [&](const int ithr, const int nthr) {
            const uint64_t wpt = ((totalWork / dataElPerVec) / nthr + 1) * dataElPerVec;
            const uint64_t start = std::min(wpt * ithr, totalWork);
            const uint64_t end = std::min(wpt * (ithr + 1), totalWork);
            const uint64_t work_amount = end - start;

            auto arg = kernel::GatherCallArgs();

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
            arg.work_amount = work_amount;

            const uint64_t idxElPerVec = m_jit_kernel->getVectorLen() / m_idx_et_size;
            int permIdxMask[16];
            int beforeAxisDiff[16];
            if (afterAxisSize == 1 && specIndicesSize < idxElPerVec) {
                permIdxMask[0] = idxElPerVec - specIndicesSize;
                int div = idxElPerVec / specIndicesSize;
                int remainder = idxElPerVec % specIndicesSize;
                for (uint64_t i = 1; i < idxElPerVec; i++) {
                    permIdxMask[i] = permIdxMask[i - 1] + 1;
                    if (static_cast<uint64_t>(permIdxMask[i]) == idxElPerVec)
                        permIdxMask[i] = idxElPerVec - specIndicesSize;
                }
                for (uint64_t i = 0; i < idxElPerVec; i++) {
                    if (((start + i) % specIndicesSize) < (specIndicesSize - remainder))
                        beforeAxisDiff[i] = axisDim * div;
                    else
                        beforeAxisDiff[i] = axisDim * (div + 1);
                }
                arg.permIdxMask = permIdxMask;
                arg.beforeAxisDiff = beforeAxisDiff;
            }

            (*m_jit_kernel)(&arg);
        };

        parallel_nt(0, threadBody);

        return;
    }
#endif
    execReference();
}

void Gather::initShortParams(threadExecParams& p, const uint64_t start) {
    if (!m_jit_kernel) {
        THROW_CPU_NODE_ERR("has uninitialized kernel in function initShortParams.");
    }
    const uint64_t idxElPerVec = m_jit_kernel->getVectorLen() / m_idx_et_size;

    if (afterAxisSize == 1) {  // Elementwise gather.
        if (specIndicesSize >= idxElPerVec)
            return;  // Is not a short case.

        p.permIdxMask.resize(idxElPerVec);
        p.srcBeforeAxisDiff.resize(idxElPerVec);

        p.permIdxMask[0] = idxElPerVec - specIndicesSize;
        for (uint64_t i = 1; i < idxElPerVec; i++) {
            p.permIdxMask[i] = p.permIdxMask[i - 1] + 1;
            if (static_cast<uint64_t>(p.permIdxMask[i]) == idxElPerVec)
                p.permIdxMask[i] = idxElPerVec - specIndicesSize;
        }

        const int div = idxElPerVec / specIndicesSize;
        const int remainder = idxElPerVec % specIndicesSize;
        for (uint64_t i = 0; i < idxElPerVec; i++) {
            if (((start + i) % specIndicesSize) < (specIndicesSize - remainder)) {
                p.srcBeforeAxisDiff[i] = axisDim * div;
            } else {
                p.srcBeforeAxisDiff[i] = axisDim * (div + 1);
            }
        }
    } else {  // Blocked gather.
        if (afterAxisSize > idxElPerVec)
            return;  // Is not a short case.

        p.afterAxIdxInBytes.resize(idxElPerVec);
        p.afterAxPermMask.resize(idxElPerVec);
        p.beforeAxPermMask.resize(idxElPerVec);
        p.specIdxDiff.resize(idxElPerVec);
        p.srcBeforeAxisDiff.resize(idxElPerVec);

        int secondStart = start + idxElPerVec;
        for (uint64_t i = 0; i < idxElPerVec; i++) {
            p.afterAxIdxInBytes[i] = (start + i) % afterAxisSize;
            p.specIdxDiff[i] =
                (((secondStart + i) / afterAxisSize) % specIndicesSize) * m_idx_et_size - p.specIdxInBytes[i];
            if (p.specIdxDiff[i] < 0)
                p.specIdxDiff[i] += specIndicesSize * m_idx_et_size;
            p.srcBeforeAxisDiff[i] =
                ((start + i + idxElPerVec) / (specIndicesSize * afterAxisSize)) * axisAndAfterAxisSizeInBytes -
                ((start + i) / (specIndicesSize * afterAxisSize)) * axisAndAfterAxisSizeInBytes;

            p.afterAxIdxInBytes[i] *= m_data_et_size;
            p.afterAxPermMask[i] = idxElPerVec - afterAxisSize + i;
            for (size_t j = 0lu; j < 6lu; j++) {
                if (static_cast<uint64_t>(p.afterAxPermMask[i]) >= idxElPerVec)
                    p.afterAxPermMask[i] -= afterAxisSize;
            }
        }
        if (specIndicesSize * afterAxisSize < idxElPerVec) {
            p.beforeAxPermMask[0] = idxElPerVec - specIndicesSize * afterAxisSize;
            for (uint64_t i = 1; i < idxElPerVec; i++) {
                p.beforeAxPermMask[i] = p.beforeAxPermMask[i - 1] + 1;
                if (static_cast<uint64_t>(p.beforeAxPermMask[i]) == idxElPerVec)
                    p.beforeAxPermMask[i] = idxElPerVec - specIndicesSize * afterAxisSize;
            }
        }

        p.specIdxAndAfterAxIterB = (start * m_data_et_size) % specIdxAndAfterAxSizeB;
    }
}

template <typename OUT_TYPE, int8_t get4Bit(const uint8_t&, bool)>
void Gather::execCompressed4Bit() {
    const int32_t* srcIndices = getSrcDataAtPortAs<const int32_t>(GATHER_INDICES);
    const uint8_t* srcData = getSrcDataAtPortAs<const uint8_t>(GATHER_DATA);
    OUT_TYPE* dstData = getDstDataAtPortAs<OUT_TYPE>(0);

    // zp/scale
    float const_zp = 0;
    const auto* zp = have_zp ? getSrcDataAtPortAs<float_t>(GATHER_ZP) : &const_zp;
    const auto* scale = getSrcDataAtPortAs<float_t>(GATHER_SCALE);

    const size_t dstAfterBatchSize = betweenBatchAndAxisSize * specIdxAndAfterAxSize;
    parallel_for2d(beforeBatchSize, specIndicesSize, [&](const size_t b, const size_t j) {
        int ii = srcIndices[b * specIndicesSize + j];
        if (ii < 0) {
            if (m_reverse_indexing)
                ii += axisDim;
            else
                ii = axisDim;
        }
        const size_t idx = ii;
        const size_t c2 = dstAfterBatchSize * b + afterAxisSize * j;
        if (idx < static_cast<size_t>(axisDim)) {
            size_t c1 = srcAfterBatchSize * b + afterAxisSize * idx;
            for (size_t i = 0; i < betweenBatchAndAxisSize; i++) {
                size_t srcIdx = c1 + axisAndAfterAxisSize * i;
                size_t dstIdx = c2 + specIdxAndAfterAxSize * i;

                OUT_TYPE* pdst = &dstData[dstIdx];

                size_t p = srcIdx;
                size_t dst_idx = 0;

                // heuristic:
                // ((m_axis_input_const && axis == 0) && (cond1 || cond2)) take >99% probability
                bool processed = false;
                if (m_axis_input_const && m_axis == 0) {
                    bool cond1 = have_zp && zp_group_size == scale_group_size;
                    bool cond2 = (!have_zp) || have_scalar_zp;
                    bool cond3 = have_scalar_scale && cond2;
                    if (cond3) {
                        processed = true;
                        for (; p < srcIdx + afterAxisSize; p++) {
                            auto val = srcData[p >> 1];
                            pdst[dst_idx] = static_cast<OUT_TYPE>((get4Bit(val, p % 2) - zp[0]) * scale[0]);
                            dst_idx++;
                        }
                    } else if (cond1 || cond2) {
                        processed = true;
                        for (; p < srcIdx + afterAxisSize; p += scale_group_size) {
                            const auto& cur_scale = scale[p / scale_group_size];
                            const auto& cur_zp = cond2 ? zp[0] : zp[p / zp_group_size];
                            for (size_t g = p; g < p + scale_group_size; g++) {
                                auto val = srcData[g >> 1];
                                pdst[dst_idx] = static_cast<OUT_TYPE>((get4Bit(val, g % 2) - cur_zp) * cur_scale);
                                dst_idx++;
                            }
                        }
                    }
                }

                // Reference
                if (!processed) {
                    for (; p < srcIdx + afterAxisSize; p++) {
                        auto val = srcData[p >> 1];
                        const size_t scale_offset = p / scale_group_size;
                        auto cur_zp = have_zp ? zp[p / zp_group_size] : 0;
                        pdst[dst_idx] = static_cast<OUT_TYPE>((get4Bit(val, p % 2) - cur_zp) * scale[scale_offset]);
                        dst_idx++;
                    }
                }
            }
        } else {
            for (size_t i = 0; i < betweenBatchAndAxisSize; i++) {
                size_t dstIdx = c2 + specIdxAndAfterAxSize * i;
                for (size_t p = 0; p < afterAxisSize; p++)
                    dstData[dstIdx] = 0;
            }
        }
    });
}

template <typename OUT_TYPE, typename IN_TYPE>
void Gather::execCompressed8Bit() {
    const int32_t* srcIndices = getSrcDataAtPortAs<const int32_t>(GATHER_INDICES);
    const IN_TYPE* srcData = getSrcDataAtPortAs<const IN_TYPE>(GATHER_DATA);
    OUT_TYPE* dstData = getDstDataAtPortAs<OUT_TYPE>(0);

    // zp/scale
    float const_zp = 0;
    const auto* zp = have_zp ? getSrcDataAtPortAs<float_t>(GATHER_ZP) : &const_zp;
    const auto* scale = getSrcDataAtPortAs<float_t>(GATHER_SCALE);

    const size_t dstAfterBatchSize = betweenBatchAndAxisSize * specIdxAndAfterAxSize;

    parallel_for2d(beforeBatchSize, specIndicesSize, [&](const size_t b, const size_t j) {
        int ii = srcIndices[b * specIndicesSize + j];
        if (ii < 0) {
            if (m_reverse_indexing)
                ii += axisDim;
            else
                ii = axisDim;
        }
        const size_t idx = ii;
        const size_t c2 = dstAfterBatchSize * b + afterAxisSize * j;
        if (idx < static_cast<size_t>(axisDim)) {
            size_t c1 = srcAfterBatchSize * b + afterAxisSize * idx;
            for (size_t i = 0; i < betweenBatchAndAxisSize; i++) {
                size_t srcIdx = c1 + axisAndAfterAxisSize * i;
                size_t dstIdx = c2 + specIdxAndAfterAxSize * i;

                OUT_TYPE* pdst = &dstData[dstIdx];

                size_t p = srcIdx;
                size_t dst_idx = 0;

                // heuristic:
                // ((m_axis_input_const && axis == 0) && (cond1 || cond2)) take >99% probability
                bool processed = false;
                if (m_axis_input_const && m_axis == 0) {
                    bool cond1 = have_zp && zp_group_size == scale_group_size;
                    bool cond2 = (!have_zp) || have_scalar_zp;
                    bool cond3 = have_scalar_scale && cond2;
                    if (cond3) {
                        processed = true;
                        for (; p < srcIdx + afterAxisSize; p++) {
                            pdst[dst_idx] = static_cast<OUT_TYPE>((static_cast<float>(srcData[p]) - zp[0]) * scale[0]);
                            dst_idx++;
                        }
                    } else if (cond1 || cond2) {
                        processed = true;
                        for (; p < srcIdx + afterAxisSize; p += scale_group_size) {
                            const auto& cur_scale = scale[p / scale_group_size];
                            const auto& cur_zp = cond2 ? zp[0] : zp[p / zp_group_size];
                            for (size_t g = p; g < p + scale_group_size; g++) {
                                pdst[dst_idx] =
                                    static_cast<OUT_TYPE>((static_cast<float>(srcData[g]) - cur_zp) * cur_scale);
                                dst_idx++;
                            }
                        }
                    }
                }

                // Reference
                if (!processed) {
                    for (; p < srcIdx + afterAxisSize; p++) {
                        const size_t scale_offset = p / scale_group_size;
                        auto cur_zp = have_zp ? zp[p / zp_group_size] : 0;
                        pdst[dst_idx] =
                            static_cast<OUT_TYPE>((static_cast<float>(srcData[p]) - cur_zp) * scale[scale_offset]);
                        dst_idx++;
                    }
                }
            }
        } else {
            for (size_t i = 0; i < betweenBatchAndAxisSize; i++) {
                size_t dstIdx = c2 + specIdxAndAfterAxSize * i;
                for (size_t p = 0; p < afterAxisSize; p++)
                    dstData[dstIdx] = 0;
            }
        }
    });
}

int8_t Gather::get_i4(const uint8_t& val, bool high) {
    if (high) {
        if (val & 0x80) {
            return static_cast<int8_t>((val >> 4) | 0xf8);
        } else {
            return static_cast<int8_t>(val >> 4);
        }
    }
    if (val & 0x8) {
        // Just fill in the high 4 bits with 1
        return static_cast<int8_t>(val | 0xf8);
    } else {
        return static_cast<int8_t>(val & 0xF);
    }
}

int8_t Gather::get_u4(const uint8_t& val, bool high) {
    if (high) {
        return (val >> 4) & 0xF;
    }
    return val & 0xF;
}

struct ExecCompressedContext {
    Gather* node;
    element::Type inType;
};

template <typename OUT_PRECISION>
struct ExecCompressedDispatcher {
    void operator()(ExecCompressedContext& ctx) {
        if (ctx.inType.bitwidth() == 8) {
            ExecCompressed8Bit_dispatch(ctx);
        } else {
            ExecCompressed4Bit_dispatch(ctx);
        }
    }

    template <typename IN_PRECISION>
    struct ExecCompressed8BitDispatcher {
        void operator()(ExecCompressedContext& ctx) {
            ctx.node->execCompressed8Bit<OUT_PRECISION, IN_PRECISION>();
        }
    };

private:
    void ExecCompressed8Bit_dispatch(ExecCompressedContext& ctx) {
        OV_SWITCH(intel_cpu,
                  ExecCompressed8BitDispatcher,
                  ctx,
                  ctx.inType,
                  OV_CASE(element::u8, uint8_t),
                  OV_CASE(element::i8, int8_t));
    }
    void ExecCompressed4Bit_dispatch(ExecCompressedContext& ctx) {
        switch (ctx.inType) {
        case element::u4:
            return ctx.node->execCompressed4Bit<OUT_PRECISION, Gather::get_u4>();
        case element::i4:
            return ctx.node->execCompressed4Bit<OUT_PRECISION, Gather::get_i4>();
        default:
            break;
        }
    }
};

void Gather::execCompressed() {
    auto in_precison = getParentEdgeAt(GATHER_DATA)->getMemoryPtr()->getPrecision();
    auto out_precision = getChildEdgeAt(0)->getMemoryPtr()->getPrecision();
    ExecCompressedContext ctx{this, in_precison};

    OV_SWITCH(intel_cpu,
              ExecCompressedDispatcher,
              ctx,
              out_precision,
              OV_CASE(element::f32, float),
              OV_CASE(element::bf16, bfloat16),
              OV_CASE(element::f16, float16));
}

void Gather::execReference() {
    const int32_t* srcIndices = getSrcDataAtPortAs<const int32_t>(GATHER_INDICES);
    const uint8_t* srcData = getSrcDataAtPortAs<const uint8_t>(GATHER_DATA);
    uint8_t* dstData = getDstDataAtPortAs<uint8_t>(0);

    const size_t dstAfterBatchSize = betweenBatchAndAxisSize * specIdxAndAfterAxSizeB;
    parallel_for2d(beforeBatchSize, specIndicesSize, [&](const size_t b, const size_t j) {
        int ii = srcIndices[b * specIndicesSize + j];
        if (ii < 0) {
            if (m_reverse_indexing)
                ii += axisDim;
            else
                ii = axisDim;
        }
        const size_t idx = ii;
        const size_t c2 = dstAfterBatchSize * b + afterAxisSizeInBytes * j;
        if (idx < static_cast<size_t>(axisDim)) {
            size_t c1 = srcAfterBatchSizeInBytes * b + afterAxisSizeInBytes * idx;
            for (size_t i = 0; i < betweenBatchAndAxisSize; i++) {
                size_t srcIdx = c1 + axisAndAfterAxisSizeInBytes * i;
                size_t dstIdx = c2 + specIdxAndAfterAxSizeB * i;

                cpu_memcpy(&dstData[dstIdx], &srcData[srcIdx], afterAxisSizeInBytes);
            }
        } else {
            for (size_t i = 0; i < betweenBatchAndAxisSize; i++) {
                memset(&dstData[c2 + specIdxAndAfterAxSizeB * i], 0, afterAxisSizeInBytes);
            }
        }
    });
}

void Gather::exec1DCase() {
    DEBUG_LOG(getName(), " exec1DCase");
    auto* pdst = getDstDataAtPortAs<uint32_t>(0);
    auto srcMemPtr = getSrcMemoryAtPort(GATHER_DATA);
    auto idxMemPtr = getSrcMemoryAtPort(GATHER_INDICES);
    const auto* psrc = srcMemPtr->getDataAs<const uint32_t>();
    const auto* pidx = idxMemPtr->getDataAs<int32_t>();

    const auto& idxDims = idxMemPtr->getStaticDims();
    const auto idxCnt = (idxDims.size() == 0) ? 1 : idxDims[0];
    auto axisDim = srcMemPtr->getStaticDims()[0];
    for (size_t i = 0; i < idxCnt; i++) {
        auto ii = pidx[i];
        if (ii < 0) {
            if (m_reverse_indexing)
                ii += axisDim;
            else
                ii = axisDim;
        }
        pdst[i] = psrc[ii];
    }
}

bool Gather::created() const {
    return getType() == Type::Gather;
}

bool Gather::isExecutable() const {
    return !isInPlace() && Node::isExecutable();
}

void Gather::resolveInPlaceEdges(Edge::LOOK look) {
    if (!(look & Edge::LOOK_UP) || !isInPlace()) {
        Node::resolveInPlaceEdges(look);
        return;
    }

    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        THROW_CPU_NODE_ERR("does not have preferable primitive descriptor.");
    constexpr size_t outputPort = 0;

    auto& config = selected_pd->getConfig();
    size_t inplaceInpIndx = selected_pd->getConfig().outConfs[outputPort].inPlace();
    const auto baseDim = inputShapes.front().getDims()[m_axis];
    OPENVINO_ASSERT(baseDim != Shape::UNDEFINED_DIM,
                    "Gather node: ",
                    getName(),
                    " can not use inPlace memory with splitting on dynamic dimention");
    auto baseMemMngr = getParentEdgeAt(inplaceInpIndx)->getMemory().getMemoryMngr();
    const auto index = constIndices.front();
    const ptrdiff_t offset = index < 0 ? baseDim + index : index;
    const auto& childEdges = getChildEdgesAtPort(outputPort);
    for (auto& childEdge : childEdges) {
        OPENVINO_ASSERT(childEdge->getStatus() == Edge::Status::NotAllocated,
                        " Unexpected edge status in node: ",
                        getName(),
                        " with type ",
                        getTypeStr());

        auto memMngr = std::make_shared<PartitionedMemoryMngr>(baseMemMngr, baseDim, offset);
        auto newMem = std::make_shared<Memory>(getEngine(), config.outConfs[outputPort].getMemDesc(), memMngr);

        childEdge->reuse(newMem);
    }
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
