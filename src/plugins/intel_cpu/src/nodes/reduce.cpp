// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce.h"

#include "eltwise.h"
#include "fake_quantize.h"
#include "ie_parallel.hpp"
#include "utils/bfloat16.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <common/primitive_hashing_utils.hpp>

using namespace ov::intel_cpu::node;
using namespace ov::intel_cpu::kernel;
using namespace InferenceEngine;
using namespace dnnl::impl::cpu;

#define SET_SRC_DIM_VALUE(batch, channel, depth, height, width) IB = batch;   \
                                                                IC = channel; \
                                                                ID = depth;   \
                                                                IH = height;  \
                                                                IW = width;
#define SET_DST_DIM_VALUE(batch, channel, depth, height, width) OB = batch;   \
                                                                OC = channel; \
                                                                OD = depth;   \
                                                                OH = height;  \
                                                                OW = width;

#define GET_OFF(field) offsetof(JitReduceCallArgs, field)
#define GET_OFF_POST(field) offsetof(JitReducePostCallArgs, field)

#define GET_PTR_N_PLN              const uint8_t    *in_ptr_n      = in_ptr       + src_data_size * ib * IC * ID * IH * IW;               \
                                         uint8_t    *out_ptr_n     = out_ptr      + dst_data_size * ob * OC * OD * OH * OW;
#define GET_PTR_NC_PLN             const uint8_t    *in_ptr_nc     = in_ptr_n     + src_data_size * ic * ID * IH * IW;                    \
                                         uint8_t    *out_ptr_nc    = out_ptr_n    + dst_data_size * oc * OD * OH * OW;
#define GET_PTR_NCD_PLN            const uint8_t    *in_ptr_ncd    = in_ptr_nc    + src_data_size * id * IH * IW;                         \
                                         uint8_t    *out_ptr_ncd   = out_ptr_nc   + dst_data_size * od * OH * OW;
#define GET_PTR_NCDH_PLN           const uint8_t    *in_ptr_ncdh   = in_ptr_ncd   + src_data_size * ih * IW;                              \
                                         uint8_t    *out_ptr_ncdh  = out_ptr_ncd  + dst_data_size * oh * OW;
#define GET_PTR_NCD_BASE_PTR_N_PLN const uint8_t    *in_ptr_ncd    = in_ptr_n     + src_data_size * (ic * ID + id) * IH * IW;             \
                                         uint8_t    *out_ptr_ncd   = out_ptr_n    + dst_data_size * (oc * OD + od) * OH * OW;
#define GET_PTR_N_BLK              const uint8_t    *in_ptr_n      = in_ptr       + src_data_size * ib * ICB * ID * IH * IW * blk_size;   \
                                         uint8_t    *out_ptr_n     = out_ptr      + dst_data_size * ob * OCB * OD * OH * OW * blk_size;
#define GET_PTR_NC_BLK             const uint8_t    *in_ptr_nc     = in_ptr_n     + src_data_size * icb * ID * IH * IW * blk_size;        \
                                         uint8_t    *out_ptr_nc    = out_ptr_n    + dst_data_size * ocb * OD * OH * OW * blk_size;
#define GET_PTR_NCD_BLK            const uint8_t    *in_ptr_ncd    = in_ptr_nc    + src_data_size * id * IH * IW * blk_size;              \
                                         uint8_t    *out_ptr_ncd   = out_ptr_nc   + dst_data_size * od * OH * OW * blk_size;
#define GET_PTR_NCDH_BLK           const uint8_t    *in_ptr_ncdh   = in_ptr_ncd   + src_data_size * ih * IW * blk_size;                   \
                                         uint8_t    *out_ptr_ncdh  = out_ptr_ncd  + dst_data_size * oh * OW * blk_size;
#define GET_PTR_NCDHW_BLK          const uint8_t    *in_ptr_ncdhw  = in_ptr_ncdh  + src_data_size * iw * blk_size;                        \
                                         uint8_t    *out_ptr_ncdhw = out_ptr_ncdh + dst_data_size * ow * blk_size;
#define GET_PTR_NCD_BASE_PTR_N_BLK const uint8_t    *in_ptr_ncd    = in_ptr_n     + src_data_size * (icb * ID + id) * IH * IW * blk_size; \
                                         uint8_t    *out_ptr_ncd   = out_ptr_n    + dst_data_size * (ocb * OD + od) * OH * OW * blk_size;

namespace {

struct ReduceKey {
    JitReduceConfigParams jcp;
    dnnl::post_ops postOps;

    size_t hash() const;
    bool operator==(const ReduceKey& rhs) const;
};

size_t ReduceKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;
    seed = hash_combine(seed, jcp.layout);
    seed = hash_combine(seed, jcp.reduce_mode);
    seed = hash_combine(seed, jcp.src_dt);
    seed = hash_combine(seed, jcp.dst_dt);
    seed = get_post_op_hash(seed, *postOps.get());

    return seed;
}

bool ReduceKey::operator==(const ReduceKey &rhs) const {
    return jcp.layout == rhs.jcp.layout && jcp.reduce_mode == rhs.jcp.reduce_mode &&
           jcp.src_dt == rhs.jcp.src_dt && jcp.dst_dt == rhs.jcp.dst_dt && *postOps.get() == *rhs.postOps.get();
}

} // namespace

const std::map<const ov::DiscreteTypeInfo, std::function<void(const std::shared_ptr<ov::Node>&, Reduce&)>> Reduce::initializers = {
    {ov::op::v4::ReduceL1::get_type_info_static(), [](const std::shared_ptr<ov::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceL1;
    }},
    {ov::op::v4::ReduceL2::get_type_info_static(), [](const std::shared_ptr<ov::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceL2;
    }},
    {ov::op::v1::ReduceLogicalAnd::get_type_info_static(), [](const std::shared_ptr<ov::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceAnd;
    }},
    {ov::op::v1::ReduceLogicalOr::get_type_info_static(), [](const std::shared_ptr<ov::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceOr;
    }},
    {ov::op::v1::ReduceMax::get_type_info_static(), [](const std::shared_ptr<ov::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceMax;
    }},
    {ov::op::v1::ReduceMean::get_type_info_static(), [](const std::shared_ptr<ov::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceMean;
    }},
    {ov::op::v1::ReduceMin::get_type_info_static(), [](const std::shared_ptr<ov::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceMin;
    }},
    {ov::op::v1::ReduceProd::get_type_info_static(), [](const std::shared_ptr<ov::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceProd;
    }},
    {ov::op::v1::ReduceSum::get_type_info_static(), [](const std::shared_ptr<ov::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceSum;
    }}
};

bool Reduce::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!op->get_type_info().is_castable(ov::op::util::ArithmeticReductionKeepDims::get_type_info_static()) &&
                !op->get_type_info().is_castable(ov::op::util::LogicalReductionKeepDims::get_type_info_static())) {
            errorMessage = "Reduce node with name " + op->get_friendly_name() + " is not derived from ArithmeticReductionKeepDims or LogicalReductionKeepDims";
            return false;
        }
        const auto idxIn = op->get_input_node_shared_ptr(REDUCE_INDEXES);
        if (idxIn->get_type_info() != ov::op::v0::Constant::get_type_info_static()) {
            errorMessage = "Only const 'reduce_indexes' input is supported";
            return false;
        }
        if (idxIn->get_element_type() != ov::element::i32 && idxIn->get_element_type() != ov::element::i64) {
            errorMessage = "Only i32 and i64 'reduce_indexes' input is supported";
            return false;
        }
        if (initializers.find(op->get_type_info()) == initializers.end()) {
            errorMessage = "Doesn't support Reduce algorithm: " +  std::string(op->get_type_info().name);
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Reduce::Reduce(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
        : Node(op, context, NgraphShapeInferFactory(op, PortMask(REDUCE_INDEXES))) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }
std::cout << "REDUCE CTR" << std::endl;

    errorPrefix = "Reduce node with name '" + getName() + "'";
    initializers.at(op->get_type_info())(op, *this);

    if (const auto reduction = std::dynamic_pointer_cast<ov::op::util::ReductionBase>(op)) {
        keepDims = reduction->get_keep_dims();
    }
    const auto idxIn = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(REDUCE_INDEXES));
    if (idxIn->get_element_type() == ov::element::i32) {
        const auto tmpData = idxIn->get_vector<int32_t>();
        rawAxes.assign(tmpData.begin(), tmpData.end());
    } else if (idxIn->get_element_type() == ov::element::i64) {
        rawAxes = idxIn->get_vector<int64_t>();
    }

    vec_reduceDH_prc.clear();
    setJITBeyond5D();
}

void Reduce::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    if (getParentEdges().size() != 2) {
        IE_THROW() << errorPrefix << " gets incorrect number of input edges!";
    }
    if (getChildEdges().empty()) {
        IE_THROW() << errorPrefix << " gets incorrect number of output edges!";
    }

    if (getInputShapeAtPort(REDUCE_INDEXES).getRank() != 1) {
        IE_THROW() << errorPrefix << " gets incorrect index vector dimension! Index vector should be 1 dimension.";
    }

    if (keepDims) {
        if (getInputShapeAtPort(REDUCE_DATA).getRank() != getOutputShapeAtPort(0).getRank())
            IE_THROW() << errorPrefix << " gets incorrect number of input/output dimensions!";
    } else {
        // In fact, after the Reduce operation, the shape must be a scalar if the previous one was 1d.
        // But for now, 0d tensor (scalar) is emulated as 1d tensor. Skip checking in such cases.
        bool is_emulated_0d_as_1d = getInputShapeAtPort(REDUCE_DATA).getRank() == 1 && getOutputShapeAtPort(0).getRank() == 1;
        if (getInputShapeAtPort(REDUCE_DATA).getRank() <= getOutputShapeAtPort(0).getRank() && !is_emulated_0d_as_1d)
            IE_THROW() << errorPrefix << "gets incorrect number of input/output dimensions!";
    }
}

void Reduce::initSupportedPrimitiveDescriptors() {
    const auto inputPrc0 = getOriginalInputPrecisionAtPort(REDUCE_DATA);
    const auto inputPrc1 = getOriginalInputPrecisionAtPort(REDUCE_INDEXES);
    outputPrc = getOriginalOutputPrecisionAtPort(0);

    if (!fusedWith.empty()) {
        outputPrc = fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0);
    }

    jit_mode = canApplyJIT(inputPrc0, outputPrc);

    if (jit_mode) {
        // Since in jit mode we use the output memory as an intermediate accumulator for certain reduce modes, we can't use BF16 output precision due to
        // the possible accuracy loss. Therefore, for such mods, we will change the output precision to FP32.
        if (Precision::BF16 == outputPrc) {
            if (!x64::mayiuse(x64::avx512_core)) {
                    outputPrc = Precision::FP32;
            } else if (algorithm != Algorithm::ReduceAnd && algorithm != Algorithm::ReduceOr &&
                       algorithm != Algorithm::ReduceMin && algorithm != Algorithm::ReduceMax) {
                            outputPrc = Precision::FP32;
            }
        }
    }

    support_split = algorithm != Algorithm::ReduceL2 && algorithm != Algorithm::ReduceLogSumExp &&
                    algorithm != Algorithm::ReduceSumSquare && inputPrc0 == outputPrc;

    src_data_size = inputPrc0.size();
    dst_data_size = outputPrc.size();

    NodeConfig config;
    config.dynBatchSupport = false;
    config.inConfs.resize(2);
    config.outConfs.resize(1);
    config.inConfs[REDUCE_DATA].constant(false);
    config.inConfs[REDUCE_INDEXES].constant(false);
    config.outConfs[0].constant(false);
    config.inConfs[REDUCE_DATA].inPlace(-1);
    config.inConfs[REDUCE_INDEXES].inPlace(-1);
    config.outConfs[0].inPlace(-1);

    auto& creatorsMap = BlockedDescCreator::getCommonCreators();

    auto pushDesc = [&](const LayoutType &inFormat, const LayoutType &outFormat, const Precision &inPrecision0, const Precision &inPrecision1,
            const Precision &outPrecision, const impl_desc_type &impl_type) {
        config.inConfs[REDUCE_DATA].setMemDesc(creatorsMap.at(inFormat)->createSharedDesc(inPrecision0, getInputShapeAtPort(REDUCE_DATA)));
        config.inConfs[REDUCE_INDEXES].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(inPrecision1,
                                                                                                 getInputShapeAtPort(REDUCE_INDEXES)));
        config.outConfs[0].setMemDesc(creatorsMap.at(outFormat)->createSharedDesc(outPrecision, getOutputShapeAtPort(0)));
        supportedPrimitiveDescriptors.push_back({config, impl_type});
    };

    if (jit_mode) {
        impl_desc_type impl_type = impl_desc_type::jit_sse42;
        if (x64::mayiuse(x64::avx512_core)) {
            impl_type = impl_desc_type::jit_avx512;
        } else if (x64::mayiuse(x64::avx2)) {
            impl_type = impl_desc_type::jit_avx2;
        }

        pushDesc(LayoutType::ncsp, LayoutType::ncsp, inputPrc0, inputPrc1, outputPrc, impl_type);
        if ((getInputShapeAtPort(REDUCE_DATA).getRank() == 4 || getInputShapeAtPort(REDUCE_DATA).getRank() == 5) &&
                getInputShapeAtPort(REDUCE_DATA).getMinDims()[1] > 1) {
            if (keepDims) {
                if (x64::mayiuse(x64::avx512_core)) {
                    pushDesc(LayoutType::nspc, LayoutType::nspc, inputPrc0, inputPrc1, outputPrc, impl_type);
                    pushDesc(LayoutType::nCsp16c, LayoutType::nCsp16c, inputPrc0, inputPrc1, outputPrc, impl_type);
                } else if (x64::mayiuse(x64::avx2) || x64::mayiuse(x64::sse41)) {
                    pushDesc(LayoutType::nspc, LayoutType::nspc, inputPrc0, inputPrc1, outputPrc, impl_type);
                    pushDesc(LayoutType::nCsp8c, LayoutType::nCsp8c, inputPrc0, inputPrc1, outputPrc, impl_type);
                }
            } else {
                if (x64::mayiuse(x64::avx512_core)) {
                    pushDesc(LayoutType::nspc, LayoutType::ncsp, inputPrc0, inputPrc1, outputPrc, impl_type);
                    pushDesc(LayoutType::nCsp16c, LayoutType::ncsp, inputPrc0, inputPrc1, outputPrc, impl_type);
                } else if (x64::mayiuse(x64::avx2) || x64::mayiuse(x64::sse41)) {
                    pushDesc(LayoutType::nspc, LayoutType::ncsp, inputPrc0, inputPrc1, outputPrc, impl_type);
                    pushDesc(LayoutType::nCsp8c, LayoutType::ncsp, inputPrc0, inputPrc1, outputPrc, impl_type);
                }
            }
        }
    } else {
        pushDesc(LayoutType::ncsp, LayoutType::ncsp, Precision::FP32, Precision::FP32, Precision::FP32, impl_desc_type::ref);
    }
}

bool Reduce::isExecutable() const {
    return !isInputTensorAtPortEmpty(REDUCE_DATA);
}

void Reduce::prepareParams() {
    srcDims = getParentEdgesAtPort(REDUCE_DATA)[0]->getMemory().getDesc().getShape().getDims();
    std::vector<int64_t> reduceAxes;
    if (jit_mode && jit_beyond_5D) {
        reduceAxes = update_src_dims();
    } else {
        reduceAxes = rawAxes;
    }

    auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    const SizeVector &dst_dims = dstMemPtr->getDesc().getShape().getDims();
    dst_size = dstMemPtr->GetSize();
    calcProcessDstDims(reduceAxes, dst_dims);
    if (jit_mode) {
        set_reduce_dim_flags();
    }

    auto builder = [&](const ReduceKey &key) -> std::shared_ptr<JitReducePostKernelBase> {
        std::shared_ptr<JitReducePostKernelBase> postKernel;

        if (x64::mayiuse(x64::avx512_core)) {
            postKernel.reset(new JitReducePostKernel<x64::avx512_core>(key.jcp, *attr.get()));
        } else if (x64::mayiuse(x64::avx2)) {
            postKernel.reset(new JitReducePostKernel<x64::avx2>(key.jcp, *attr.get()));
        } else if (x64::mayiuse(x64::sse41)) {
            postKernel.reset(new JitReducePostKernel<x64::sse41>(key.jcp, *attr.get()));
        }
        if (postKernel) {
            postKernel->create_ker();
        }

        return postKernel;
    };

    if (compile_post_kernel) {
        setPostOps(attr, dst_dims, true);

        ReduceKey key = {jcp, attr.get_post_ops()};
        auto cache = context->getParamsCache();
        auto result = cache->getOrCreate(key, builder);
        if (!result.first) {
            IE_THROW() << errorPrefix << " has not found JitReducePostKernel.";
        }

        reducePostKernel = result.first;
        jit_mode = jit_mode && reducePostKernel;

        if (!isDynamicNode()) {
            compile_post_kernel = false;
        }
    }
}

void Reduce::createPrimitive() {
    if (!isExecutable()) {
        return;
    }
    auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto &srcMemPtr = getParentEdgeAt(REDUCE_DATA)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->isAllocated())
        IE_THROW() << errorPrefix << " has not allocated destination memory.";
    if (!srcMemPtr || !srcMemPtr->isAllocated())
        IE_THROW() << errorPrefix << " has not allocate input memory.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW() << errorPrefix << " has nullable preferable primitive descriptor";

    if (srcMemPtr->getDesc().hasLayoutType(LayoutType::ncsp)) {
        layout = ReduceLayoutType::reduce_ncsp;
    } else if (srcMemPtr->getDesc().hasLayoutType(LayoutType::nspc)) {
        layout = ReduceLayoutType::reduce_nspc;
    } else {
        layout = ReduceLayoutType::reduce_blocked;
    }

    // hybrid layout: nspc/blocked layout for input and ncsp for output
    // !keepDims is needed to avoid hybrid layout for cases eg. (A, B, C, D) reduce to (A, 1, 1, 1)
    if (!keepDims && (layout == ReduceLayoutType::reduce_nspc || layout == ReduceLayoutType::reduce_blocked)) {
        is_hybrid_layout = dstMemPtr->getDesc().hasLayoutType(LayoutType::ncsp);
    }

    auto selectedPD = getSelectedPrimitiveDescriptor();
    jcp = JitReduceConfigParams();
    jcp.src_dt = DnnlExtensionUtils::IEPrecisionToDataType(selectedPD->getConfig().inConfs[REDUCE_DATA].getMemDesc()->getPrecision());
    jcp.dst_dt = DnnlExtensionUtils::IEPrecisionToDataType(selectedPD->getConfig().outConfs[0].getMemDesc()->getPrecision());
    jcp.src_data_size = DnnlExtensionUtils::sizeOfDataType(jcp.src_dt);
    jcp.dst_data_size = DnnlExtensionUtils::sizeOfDataType(jcp.dst_dt);
    jcp.layout = layout;
    jcp.reduce_mode = getAlgorithm();

    compile_post_kernel = true;

    if (x64::mayiuse(x64::avx512_core)) {
        blk_size = 16;
    } else {
        blk_size = 8;
    }

    if (inputShapesDefined()) {
        if (needPrepareParams())
            prepareParams();
        updateLastInputDims();
    }

    if (x64::mayiuse(x64::avx512_core)) {
        reduceKernel.reset(new JitReduceKernel<x64::avx512_core>(jcp));
    } else if (x64::mayiuse(x64::avx2)) {
        reduceKernel.reset(new JitReduceKernel<x64::avx2>(jcp));
    } else if (x64::mayiuse(x64::sse41)) {
        reduceKernel.reset(new JitReduceKernel<x64::sse41>(jcp));
    }
    if (reduceKernel) {
        reduceKernel->create_ker();
    }
    jit_mode = jit_mode && reduceKernel;
}

void Reduce::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

void Reduce::execute(dnnl::stream strm) {
    auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto &srcMemPtr = getParentEdgeAt(REDUCE_DATA)->getMemoryPtr();

    const uint8_t *src_data = reinterpret_cast<const uint8_t *>(srcMemPtr->GetPtr());
    uint8_t *dst_data = reinterpret_cast<uint8_t *>(dstMemPtr->GetPtr());

    if (jit_mode) {
        if (is_hybrid_layout) {
            dst_data = reinterpret_cast<uint8_t *>(prc_mem.get_data_handle());
        }
        reduce_type(src_data, dst_data, dst_size);
    } else {
        if (layout == ReduceLayoutType::reduce_ncsp) {
            auto in_ptr = reinterpret_cast<const float *>(src_data);
            auto out_ptr = reinterpret_cast<float *>(dst_data);
            reduce_ref(in_ptr, out_ptr);
        } else {
            IE_THROW() << errorPrefix << " supports only plain layout on machine w/o sse42.";
        }
    }
}

void Reduce::reduce_type(const uint8_t *in_ptr, uint8_t *out_ptr, size_t dst_size) {
    initDstData(out_ptr, dst_size);
    reduce_stride = IW;

    if (layout == ReduceLayoutType::reduce_ncsp || layout == ReduceLayoutType::reduce_nspc) {
        reduce_PLN(in_ptr, out_ptr);
    } else {
        if (ReduceC && (IC % blk_size)) {
            reduce_BLK_concern_padding(in_ptr, out_ptr);
        } else {
            reduce_BLK(in_ptr, out_ptr);
        }
    }

    if (is_hybrid_layout) {
        uint8_t *proc_ptr = out_ptr;
        auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
        out_ptr = reinterpret_cast<uint8_t *>(dstMemPtr->GetPtr());
        if (layout == ReduceLayoutType::reduce_nspc) {
            nspc2ncsp(proc_ptr, out_ptr);
        } else {
            blocked2ncsp(proc_ptr, out_ptr);
        }
    }
}

void Reduce::reduce_PLN(const uint8_t *in_ptr, uint8_t *out_ptr) {
    if (ReduceN && !ReduceC && !ReduceD && !ReduceH && !ReduceW) {
        size_t IA = IC * ID * IH * IW;
        reduce_stride = IA;
        parallel_for(IA / blk_size, [&](size_t iba){
            size_t oba = iba;
            reduce_kernel_process(in_ptr + iba * blk_size * src_data_size, out_ptr + oba * blk_size * dst_data_size,
                                  blk_size, 0, IB);
        });

        size_t tail_start = IA / blk_size * blk_size;
        reduce_kernel_process(in_ptr + tail_start * src_data_size, out_ptr + tail_start * dst_data_size,
                              IA - tail_start, 0, IB);
    } else {
        for (size_t ib = 0; ib < IB; ib++) {
            size_t ob = ReduceN ? 0 : ib; GET_PTR_N_PLN;
            if (!ReduceC && !ReduceD && ReduceW) {
                size_t work_amount = ReduceH ? IH * IW : IW;
                if (work_amount < blk_size && x64::mayiuse(x64::avx2)) {
                    size_t outer_size = ReduceH ? IC * ID : IC * ID * IH;
                    size_t inner_size = ReduceH ? IH * IW : IW;
                    size_t output_inner_size = ReduceH ? OH * OW : OW;
                    size_t IK = outer_size / blk_size;
                    std::vector<int> index_buf(blk_size);
                    for (size_t i = 0; i < blk_size; i++) {
                        index_buf[i] = i * work_amount * src_data_size;
                    }
                    parallel_for(IK, [&](size_t ik) {
                        size_t ok = ik;
                        reduce_kernel_process(in_ptr_n + ik * blk_size * inner_size * src_data_size,
                                              out_ptr_n + ok * blk_size * output_inner_size * dst_data_size,
                                              work_amount, 1, 0, static_cast<int *>(&index_buf[0]));
                    });
                    size_t tail_start = IK * blk_size;
                    size_t IT = outer_size - tail_start;
                    parallel_for(IT, [&](size_t it) {
                        size_t ot = it;
                        reduce_kernel_process(in_ptr_n + (tail_start + it) * inner_size * src_data_size,
                                              out_ptr_n + (tail_start + ot) * output_inner_size * dst_data_size, work_amount, 1);
                    });
                } else {
                    if (ReduceH) {
                        parallel_for2d(IC, ID, [&](size_t ic, size_t id) {
                            size_t oc = ic, od = id; GET_PTR_NCD_BASE_PTR_N_PLN;
                            reduce_kernel_process(in_ptr_ncd, out_ptr_ncd, work_amount, 1);
                        });
                    } else {
                        parallel_for3d(IC, ID, IH, [&](size_t ic, size_t id, size_t ih) {
                            size_t oc = ic, od = id; GET_PTR_NCD_BASE_PTR_N_PLN;
                            size_t oh = ih; GET_PTR_NCDH_PLN;
                            reduce_kernel_process(in_ptr_ncdh, out_ptr_ncdh, work_amount, 1);
                        });
                    }
                }
            } else if (ReduceH && ReduceW) {
                for (size_t ic = 0; ic < IC; ic++) {
                    size_t oc = ReduceC ? 0 : ic; GET_PTR_NC_PLN;
                    for (size_t id = 0; id < ID; id++) {
                        size_t od = ReduceD ? 0 : id; GET_PTR_NCD_PLN;
                        reduce_kernel_process(in_ptr_ncd, out_ptr_ncd, IH * IW, 1);
                    }
                }
            } else if (!ReduceH && ReduceW) {
                for (size_t ic = 0; ic < IC; ic++) {
                    size_t oc = ReduceC ? 0 : ic; GET_PTR_NC_PLN;
                    for (size_t id = 0; id < ID; id++) {
                        size_t od = ReduceD ? 0 : id; GET_PTR_NCD_PLN;
                        parallel_for(IH, [&](size_t ih){
                            size_t oh = ih; GET_PTR_NCDH_PLN;
                            reduce_kernel_process(in_ptr_ncdh, out_ptr_ncdh, IW, 1);
                        });
                    }
                }
            } else if (ReduceW) {
                for (size_t ic = 0; ic < IC; ic++) {
                    size_t oc = ReduceC ? 0 : ic; GET_PTR_NC_PLN;
                    for (size_t id = 0; id < ID; id++) {
                        size_t od = ReduceD ? 0 : id; GET_PTR_NCD_PLN;
                        for (size_t ih = 0; ih < IH; ih++) {
                            size_t oh = ReduceH ? 0 : ih; GET_PTR_NCDH_PLN;
                            reduce_kernel_process(in_ptr_ncdh, out_ptr_ncdh, IW, 1);
                        }
                    }
                }
            } else if (!ReduceC && !ReduceD && ReduceH && !ReduceW) {
                parallel_for2d(IC, ID, [&](size_t ic, size_t id) {
                    size_t oc = ic, od = id; GET_PTR_NCD_BASE_PTR_N_PLN;
                    parallel_for(IW / blk_size, [&](size_t ibw){
                        size_t obw = ibw;
                        reduce_kernel_process(in_ptr_ncd + ibw * blk_size * src_data_size, out_ptr_ncd + obw * blk_size * dst_data_size,
                                              blk_size, 0, IH);
                    });
                    size_t tail_start = IW / blk_size * blk_size;
                    reduce_kernel_process(in_ptr_ncd + tail_start * src_data_size, out_ptr_ncd + tail_start * dst_data_size,
                                          IW - tail_start, 0, IH);
                });
            } else if (!ReduceC && ReduceD && ReduceH && !ReduceW) {
                size_t IWB = IW / blk_size;
                if (ReduceDH_opt) {
                    if (IWB > 0) {
                        // reduce parallelly in D dimension
                        // step1: !ReduceD && ReduceH && !ReduceW
                        uint8_t *prc_ptr_n = &vec_reduceDH_prc[0];
                        initDstData(prc_ptr_n, prc_size);
                        parallel_for2d(ID, IWB, [&](size_t id, size_t iwb){
                            size_t pd = id, pwb = iwb;
                            reduce_kernel_process(in_ptr_n + (id * IH * IW + iwb * blk_size) * src_data_size,
                                                prc_ptr_n + (pd * PW + pwb * blk_size) * prc_data_size, blk_size, 0, IH);
                        });
                        // step2: ReduceD
                        reduce_stride = PW;
                        parallel_for(IWB, [&](size_t iwb){
                            size_t pwb = iwb, owb = iwb;
                            reduce_kernel_process(prc_ptr_n + pwb * blk_size * prc_data_size,
                                                out_ptr_n + owb * blk_size * dst_data_size, blk_size, 0, ID);
                        });
                    }
                    // reduce tail
                    reduce_stride = IW;
                    size_t tail_start = IWB * blk_size;
                    parallel_for(IW - tail_start, [&](size_t i_tail) {
                        reduce_kernel_process(in_ptr_n + (tail_start + i_tail) * src_data_size, out_ptr_n + (tail_start + i_tail) * dst_data_size,
                                            1, 0, ID * IH);
                    });
                } else {
                    parallel_for(IC, [&](size_t ic) {
                        size_t oc = ic; GET_PTR_NC_PLN;
                        parallel_for(IWB, [&](size_t iwb){
                            size_t owb = iwb;
                            reduce_kernel_process(in_ptr_nc + iwb * blk_size * src_data_size, out_ptr_nc + owb * blk_size * dst_data_size,
                                                blk_size, 0, ID * IH);
                        });
                        size_t tail_start = IWB * blk_size;
                        parallel_for(IW - tail_start, [&](size_t i_tail) {
                            reduce_kernel_process(in_ptr_nc + (tail_start + i_tail) * src_data_size, out_ptr_nc + (tail_start + i_tail) * dst_data_size,
                                                1, 0, ID * IH);
                        });
                    });
                }
            } else if (ReduceC && ReduceD && ReduceH && !ReduceW) {
                parallel_for(IW / blk_size, [&](size_t ibw){
                    size_t obw = ibw;
                    reduce_kernel_process(in_ptr_n + ibw * blk_size * src_data_size, out_ptr_n + obw * blk_size * dst_data_size,
                                          blk_size, 0, IC * ID * IH);
                });

                size_t tail_start = IW / blk_size * blk_size;
                reduce_kernel_process(in_ptr_n + tail_start * src_data_size, out_ptr_n + tail_start * dst_data_size,
                                      IW - tail_start, 0, IC * ID * IH);
            } else if (ReduceC && !ReduceD && !ReduceH && !ReduceW) {
                size_t IS = ID * IH * IW;
                reduce_stride = IS;
                parallel_for(IS / blk_size, [&](size_t ibs){
                    size_t obs = ibs;
                    reduce_kernel_process(in_ptr_n + ibs * blk_size * src_data_size, out_ptr_n + obs * blk_size * dst_data_size,
                                          blk_size, 0, IC);
                });

                size_t tail_start = IS / blk_size * blk_size;
                reduce_kernel_process(in_ptr_n + tail_start * src_data_size, out_ptr_n + tail_start * dst_data_size,
                                      IS - tail_start, 0, IC);
            } else {
                for (size_t ic = 0; ic < IC; ic++) {
                    size_t oc = ReduceC ? 0 : ic; GET_PTR_NC_PLN;
                    for (size_t id = 0; id < ID; id++) {
                        size_t od = ReduceD ? 0 : id; GET_PTR_NCD_PLN;
                        for (size_t ih = 0; ih < IH; ih++) {
                            size_t oh = ReduceH ? 0 : ih; GET_PTR_NCDH_PLN;
                            for (size_t ibw = 0; ibw < IW / blk_size; ibw++) {
                                size_t obw = ibw;
                                reduce_kernel_process(in_ptr_ncdh + ibw * blk_size * src_data_size,
                                                      out_ptr_ncdh + obw * blk_size * dst_data_size, blk_size, 0);
                            }
                            size_t tail_start = IW / blk_size * blk_size;
                            reduce_kernel_process(in_ptr_ncdh + tail_start * src_data_size, out_ptr_ncdh + tail_start * dst_data_size, IW - tail_start, 0);
                        }
                    }
                }
            }
        }
    }

    reduce_kernel_post_process(out_ptr);
}

void Reduce::reduce_BLK(const uint8_t *in_ptr, uint8_t *out_ptr) {
    size_t ICB = div_up(IC, blk_size);
    size_t OCB = div_up(OC, blk_size);

    for (size_t ib = 0; ib < IB; ib++) {
        size_t ob = ReduceN ? 0 : ib; GET_PTR_N_BLK;
        if (!ReduceC && !ReduceD && ReduceH && ReduceW) {
            parallel_for2d(ICB, ID, [&](size_t icb, size_t id) {
                size_t ocb = icb, od = id; GET_PTR_NCD_BASE_PTR_N_BLK;
                reduce_kernel_process(in_ptr_ncd, out_ptr_ncd, IH * IW * blk_size);
            });
        } else if (ReduceC && ReduceD && ReduceH && ReduceW) {
            if (!support_split) {
                reduce_kernel_process(in_ptr_n, out_ptr_n, ICB * ID * IH * IW * blk_size);
            } else {
                // reduce parallelly
                // step1: !ReduceC && ReduceD && ReduceH && ReduceW
                size_t prc_size = ICB * blk_size * dst_data_size;
                std::vector<uint8_t> vec_prc(prc_size);
                initDstData(vec_prc.data(), prc_size);
                uint8_t *out_ptr_n_cp = out_ptr_n;
                out_ptr_n = vec_prc.data();
                parallel_for(ICB, [&](size_t icb) {
                    size_t ocb = icb; GET_PTR_NC_BLK;
                    reduce_kernel_process(in_ptr_nc, out_ptr_nc, ID * IH * IW * blk_size);
                });
                // step2: ReduceC
                reduce_kernel_process(out_ptr_n, out_ptr_n_cp, ICB * blk_size);
            }
        } else if (ReduceW) {
            for (size_t icb = 0; icb < ICB; icb++) {
                size_t ocb = ReduceC ? 0 : icb; GET_PTR_NC_BLK;
                for (size_t id = 0; id < ID; id++) {
                    size_t od = ReduceD ? 0 : id; GET_PTR_NCD_BLK;
                    for (size_t ih = 0; ih < IH; ih++) {
                        size_t oh = ReduceH ? 0 : ih; GET_PTR_NCDH_BLK;
                        reduce_kernel_process(in_ptr_ncdh, out_ptr_ncdh, IW * blk_size);
                    }
                }
            }
        } else if (ReduceC && !ReduceD && !ReduceH && !ReduceW) {
            reduce_stride = ID * IH * IW * blk_size;
            parallel_for3d(ID, IH, IW, [&](size_t id, size_t ih, size_t iw) {
                size_t icb = 0, ocb = 0; GET_PTR_NC_BLK;
                size_t od = id; GET_PTR_NCD_BLK;
                size_t oh = ih; GET_PTR_NCDH_BLK;
                size_t ow = iw; GET_PTR_NCDHW_BLK;
                reduce_kernel_process(in_ptr_ncdhw, out_ptr_ncdhw, blk_size, 0, ICB);
            });
        } else {
            for (size_t icb = 0; icb < ICB; icb++) {
                size_t ocb = ReduceC ? 0 : icb; GET_PTR_NC_BLK;
                for (size_t id = 0; id < ID; id++) {
                    size_t od = ReduceD ? 0 : id; GET_PTR_NCD_BLK;
                    for (size_t ih = 0; ih < IH; ih++) {
                        size_t oh = ReduceH ? 0 : ih; GET_PTR_NCDH_BLK;
                        parallel_for(IW, [&](size_t iw) {
                            size_t ow = iw; GET_PTR_NCDHW_BLK;
                            reduce_kernel_process(in_ptr_ncdhw, out_ptr_ncdhw, blk_size);
                        });
                    }
                }
            }
        }
    }

    reduce_kernel_post_process(out_ptr);
}

void Reduce::reduce_BLK_concern_padding(const uint8_t *in_ptr, uint8_t *out_ptr) {
    size_t ICB = div_up(IC, blk_size);
    size_t OCB = div_up(OC, blk_size);

    auto reduceSkipPadding = [&](const uint8_t *in_ptr_ncd, uint8_t *out_ptr_ncd, size_t ic) {
        size_t blk_valid_size = IC - ic;
        for (size_t ih = 0; ih < IH; ih++) {
            size_t oh = ReduceH ? 0 : ih; GET_PTR_NCDH_BLK;
            for (size_t iw = 0; iw < IW; iw++) {
                size_t ow = ReduceW ? 0 : iw; GET_PTR_NCDHW_BLK;
                reduce_kernel_process(in_ptr_ncdhw, out_ptr_ncdhw, blk_valid_size);
            }
        }
    };

    for (size_t ib = 0; ib < IB; ib++) {
        size_t ob = ReduceN ? 0 : ib; GET_PTR_N_BLK;
        if (!ReduceD && ReduceH && ReduceW) {
            for (size_t icb = 0; icb < ICB; icb++) {
                size_t ocb = 0;;
                size_t ic = icb * blk_size;
                parallel_for(ID, [&](size_t id) {
                    size_t od = id; GET_PTR_NCD_BASE_PTR_N_BLK;
                    if (ic + blk_size <= IC) {
                        reduce_kernel_process(in_ptr_ncd, out_ptr_ncd, IH * IW * blk_size);
                    } else {
                        reduceSkipPadding(in_ptr_ncd, out_ptr_ncd, ic);
                    }
                });
            }
        } else if (ReduceD && ReduceH && ReduceW) {
            for (size_t icb = 0; icb < ICB; icb++) {
                size_t ocb = 0; GET_PTR_NC_BLK;
                size_t ic = icb * blk_size;
                if (ic + blk_size <= IC) {
                    reduce_kernel_process(in_ptr_nc, out_ptr_nc, ID * IH * IW * blk_size);
                } else {
                    for (size_t id = 0; id < ID; id++) {
                        size_t od = 0; GET_PTR_NCD_BLK;
                        reduceSkipPadding(in_ptr_ncd, out_ptr_ncd, ic);
                    }
                }
            }
        } else if (ReduceW) {
            for (size_t icb = 0; icb < ICB; icb++) {
                size_t ocb = 0; GET_PTR_NC_BLK;
                size_t ic = icb * blk_size;
                for (size_t id = 0; id < ID; id++) {
                    size_t od = ReduceD ? 0 : id; GET_PTR_NCD_BLK;
                    if (ic + blk_size <= IC) {
                        for (size_t ih = 0; ih < IH; ih++) {
                            size_t oh = ReduceH ? 0 : ih; GET_PTR_NCDH_BLK;
                            reduce_kernel_process(in_ptr_ncdh, out_ptr_ncdh, IW * blk_size);
                        }
                    } else {
                        reduceSkipPadding(in_ptr_ncd, out_ptr_ncd, ic);
                    }
                }
            }
        } else {
            for (size_t icb = 0; icb < ICB; icb++) {
                size_t ocb = 0; GET_PTR_NC_BLK;
                size_t ic = icb * blk_size;
                for (size_t id = 0; id < ID; id++) {
                    size_t od = ReduceD ? 0 : id; GET_PTR_NCD_BLK;
                    if (ic + blk_size <= IC) {
                        for (size_t ih = 0; ih < IH; ih++) {
                            size_t oh = ReduceH ? 0 : ih; GET_PTR_NCDH_BLK;
                            parallel_for(IW, [&](size_t iw) {
                                size_t ow = iw; GET_PTR_NCDHW_BLK;
                                reduce_kernel_process(in_ptr_ncdhw, out_ptr_ncdhw, blk_size);
                            });
                        }
                    } else {
                        reduceSkipPadding(in_ptr_ncd, out_ptr_ncd, ic);
                    }
                }
            }
        }
    }

    reduce_kernel_post_process(out_ptr);
}

inline void Reduce::reduce_kernel_process(const uint8_t *in_p, uint8_t *out_p, size_t work_amount,
                                                    size_t reduce_w, size_t work_batch, const int *tab_idx) {
    auto arg = JitReduceCallArgs();
    arg.src = static_cast<const void *>(in_p);
    arg.idx = tab_idx;
    arg.dst = static_cast<void *>(out_p);
    arg.work_amount = work_amount;
    arg.work_batch = work_batch;
    arg.reduce_w = reduce_w;
    arg.reduce_stride = reduce_stride;

    (*reduceKernel)(&arg);
}

inline void Reduce::reduce_kernel_post_process(uint8_t *out_ptr) {
    const size_t integerDivisor = IB * IC * ID * IH * IW / (OB * OC * OD * OH * OW);
    const float divisor = static_cast<float>(integerDivisor);
    if (layout == ReduceLayoutType::reduce_ncsp || layout == ReduceLayoutType::reduce_nspc) {
        parallel_for2d(OB, OC, [&](size_t ob, size_t oc) {
            uint8_t *out_p = out_ptr + (ob * OC + oc) * OD * OH * OW * dst_data_size;
            auto arg = JitReducePostCallArgs();
            arg.dst = static_cast<void *>(out_p);
            arg.oc_off = layout == ReduceLayoutType::reduce_nspc ? 0 : oc * sizeof(float);
            arg.channel_size = layout == ReduceLayoutType::reduce_nspc ? OW : OC; // OW is related to nspc-ncsp dimension reinterpret
            arg.work_amount = OD * OH * OW;
            arg.divisor = &divisor;
            arg.post_op_data = static_cast<const void **>(postOpsDataPtrs.data());
            (*reducePostKernel)(&arg);
        });
    } else {
        size_t OCB = div_up(OC, blk_size);
        parallel_for2d(OB, OCB, [&](size_t ob, size_t ocb) {
            uint8_t *out_p = out_ptr + (ob * OCB + ocb) * OD * OH * OW * blk_size * dst_data_size;
            auto arg = JitReducePostCallArgs();
            arg.dst = static_cast<void *>(out_p);
            arg.reduce_c = ReduceC ? 1 : 0;
            arg.oc_off = ocb * blk_size * sizeof(float);
            arg.work_amount = OD * OH * OW * blk_size;
            arg.divisor = &divisor;
            arg.post_op_data = static_cast<const void **>(postOpsDataPtrs.data());
            (*reducePostKernel)(&arg);
        });
    }
}

void Reduce::nspc2ncsp(uint8_t *proc_ptr, uint8_t *out_ptr) {
    // dimension reinterpret after nspc reusing routine reduce_PLN
    // demote -- nspc -- ncsp
    //  DIM0  --   B  --  B
    //  DIM1  --   C  --  W
    //  DIM2  --   D  --  C
    //  DIM3  --   H  --  D
    //  DIM4  --   W  --  H
    const size_t DIM0 = OB;
    const size_t DIM1 = OW;
    const size_t DIM2 = OC;
    const size_t DIM3 = OD;
    const size_t DIM4 = OH;
    const size_t stride1 = DIM2 * DIM3 * DIM4;
    const size_t stride0 = stride1 * DIM1;

    if (dst_data_size == 4) {
        auto src_data = reinterpret_cast<const float *>(proc_ptr);
        auto dst_data = reinterpret_cast<float *>(out_ptr);
        parallel_for2d(DIM0, stride1, [&](size_t b, size_t j) {
            auto src_off = b * stride0 + j * DIM1;
            auto dst_off = b * stride0 + j;
            for (size_t dim1 = 0; dim1 < DIM1; dim1++) {
                dst_data[dst_off] = src_data[src_off];
                src_off++;
                dst_off += stride1;
            }
        });
    } else if (dst_data_size == 2) {
        auto src_data = reinterpret_cast<const uint16_t *>(proc_ptr);
        auto dst_data = reinterpret_cast<uint16_t *>(out_ptr);
        parallel_for2d(DIM0, stride1, [&](size_t b, size_t j) {
            auto src_off = b * stride0 + j * DIM1;
            auto dst_off = b * stride0 + j;
            for (size_t dim1 = 0; dim1 < DIM1; dim1++) {
                dst_data[dst_off] = src_data[src_off];
                src_off++;
                dst_off += stride1;
            }
        });
    } else {
        auto src_data = reinterpret_cast<const uint8_t *>(proc_ptr);
        auto dst_data = reinterpret_cast<uint8_t *>(out_ptr);
        parallel_for2d(DIM0, stride1, [&](size_t b, size_t j) {
            auto src_off = b * stride0 + j * DIM1;
            auto dst_off = b * stride0 + j;
            for (size_t dim1 = 0; dim1 < DIM1; dim1++) {
                dst_data[dst_off] = src_data[src_off];
                src_off++;
                dst_off += stride1;
            }
        });
    }
}

void Reduce::blocked2ncsp(uint8_t *proc_ptr, uint8_t *out_ptr) {
    const size_t DIM0 = OB;
    const size_t DIM1 = OC;
    const size_t DIM2 = OD;
    const size_t DIM3 = OH;
    const size_t DIM4 = OW;
    const size_t stride1 = DIM2 * DIM3 * DIM4;
    const size_t src_stride0 = stride1 * div_up(OC, blk_size) * blk_size;
    const size_t dst_stride0 = stride1 * DIM1;

    if (dst_data_size == 4) {
        auto src_data = reinterpret_cast<const float *>(proc_ptr);
        auto dst_data = reinterpret_cast<float *>(out_ptr);
        parallel_for2d(DIM0, stride1, [&](size_t b, size_t j) {
            auto src_off = b * src_stride0 + j * blk_size;
            auto dst_off = b * dst_stride0 + j;
            for (size_t dim1 = 0; dim1 + blk_size <= DIM1; dim1 += blk_size) {
                for (size_t k = 0; k < blk_size; k++) {
                    dst_data[dst_off] = src_data[src_off];
                    src_off++;
                    dst_off += stride1;
                }
                src_off += (stride1 - 1) * blk_size;
            }
            size_t tail = DIM1 % blk_size;
            for (size_t k = 0; k < tail; k++) {
                dst_data[dst_off] = src_data[src_off];
                src_off++;
                dst_off += stride1;
            }
        });
    } else if (dst_data_size == 2) {
        auto src_data = reinterpret_cast<const uint16_t *>(proc_ptr);
        auto dst_data = reinterpret_cast<uint16_t *>(out_ptr);
        parallel_for2d(DIM0, stride1, [&](size_t b, size_t j) {
            auto src_off = b * src_stride0 + j * blk_size;
            auto dst_off = b * dst_stride0 + j;
            for (size_t dim1 = 0; dim1 + blk_size <= DIM1; dim1 += blk_size) {
                for (size_t k = 0; k < blk_size; k++) {
                    dst_data[dst_off] = src_data[src_off];
                    src_off++;
                    dst_off += stride1;
                }
                src_off += (stride1 - 1) * blk_size;
            }
            size_t tail = DIM1 % blk_size;
            for (size_t k = 0; k < tail; k++) {
                dst_data[dst_off] = src_data[src_off];
                src_off++;
                dst_off += stride1;
            }
        });
    } else {
        auto src_data = reinterpret_cast<const uint8_t *>(proc_ptr);
        auto dst_data = reinterpret_cast<uint8_t *>(out_ptr);
        parallel_for2d(DIM0, stride1, [&](size_t b, size_t j) {
            auto src_off = b * src_stride0 + j * blk_size;
            auto dst_off = b * dst_stride0 + j;
            for (size_t dim1 = 0; dim1 + blk_size <= DIM1; dim1 += blk_size) {
                for (size_t k = 0; k < blk_size; k++) {
                    dst_data[dst_off] = src_data[src_off];
                    src_off++;
                    dst_off += stride1;
                }
                src_off += (stride1 - 1) * blk_size;
            }
            size_t tail = DIM1 % blk_size;
            for (size_t k = 0; k < tail; k++) {
                dst_data[dst_off] = src_data[src_off];
                src_off++;
                dst_off += stride1;
            }
        });
    }
}

inline void Reduce::initDstData(uint8_t *out_ptr, size_t dst_size) {
    switch (algorithm) {
        case Algorithm::ReduceL1:
        case Algorithm::ReduceL2:
        case Algorithm::ReduceLogSum:
        case Algorithm::ReduceLogSumExp:
        case Algorithm::ReduceMean:
        case Algorithm::ReduceOr:
        case Algorithm::ReduceSum:
        case Algorithm::ReduceSumSquare:
            memset(out_ptr, 0, dst_size);
            break;
        case Algorithm::ReduceAnd:
        case Algorithm::ReduceProd:
            if (outputPrc == Precision::FP32) {
                auto out_p = reinterpret_cast<float *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = static_cast<float>(1); });
            } else if (outputPrc == Precision::I32) {
                auto out_p = reinterpret_cast<int32_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = static_cast<int32_t>(1); });
            } else if (outputPrc == Precision::I64) {
                auto out_p = reinterpret_cast<int64_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = static_cast<int64_t>(1); });
            } else if (outputPrc == Precision::BF16) {
                auto out_p = reinterpret_cast<bfloat16_t*>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = static_cast<bfloat16_t>(1); });
            } else if (outputPrc == Precision::U8) {
                auto out_p = reinterpret_cast<uint8_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = static_cast<uint8_t>(1); });
            } else if (outputPrc == Precision::I8) {
                auto out_p = reinterpret_cast<int8_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = static_cast<int8_t>(1); });
            }
            break;
        case Algorithm::ReduceMax:
            if (outputPrc == Precision::FP32) {
                auto out_p = reinterpret_cast<float *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<float>::lowest(); });
            } else if (outputPrc == Precision::I32) {
                auto out_p = reinterpret_cast<int32_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<int32_t>::min(); });
            } else if (outputPrc == Precision::I64) {
                auto out_p = reinterpret_cast<int64_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<int64_t>::min(); });
            } else if (outputPrc == Precision::BF16) {
                auto out_p = reinterpret_cast<bfloat16_t*>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<bfloat16_t>::lowest(); });
            } else if (outputPrc == Precision::U8) {
                auto out_p = reinterpret_cast<uint8_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<uint8_t>::min(); });
            } else if (outputPrc == Precision::I8) {
                auto out_p = reinterpret_cast<int8_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<int8_t>::min(); });
            }
            break;
        case Algorithm::ReduceMin:
            if (outputPrc == Precision::FP32) {
                auto out_p = reinterpret_cast<float *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<float>::max(); });
            } else if (outputPrc == Precision::I32) {
                auto out_p = reinterpret_cast<int32_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<int32_t>::max(); });
            } else if (outputPrc == Precision::I64) {
                auto out_p = reinterpret_cast<int64_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<int64_t>::max(); });
            } else if (outputPrc == Precision::BF16) {
                auto out_p = reinterpret_cast<bfloat16_t*>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<bfloat16_t>::max(); });
            } else if (outputPrc == Precision::U8) {
                auto out_p = reinterpret_cast<uint8_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<uint8_t>::max(); });
            } else if (outputPrc == Precision::I8) {
                auto out_p = reinterpret_cast<int8_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<int8_t>::max(); });
            }
            break;
        default:
            IE_THROW() << errorPrefix << " gets unsupported reduce mode.";
    }
}

inline void Reduce::create_working_memory() {
    auto rank = getInputShapeAtPort(REDUCE_DATA).getRank();
    memory::format_tag format = (layout == ReduceLayoutType::reduce_nspc) ? (rank == 4 ? dnnl::memory::format_tag::nhwc : dnnl::memory::format_tag::ndhwc)
                                        : (rank == 4 ? (x64::mayiuse(x64::avx512_core) ? dnnl::memory::format_tag::nChw16c : dnnl::memory::format_tag::nChw8c)
                                                     : (x64::mayiuse(x64::avx512_core) ? dnnl::memory::format_tag::nCdhw16c : dnnl::memory::format_tag::nCdhw8c));
    auto prc_dims = rank == 4 ? std::vector<size_t>{OB, OC, OH, OW} : std::vector<size_t>{OB, OC, OD, OH, OW};
    auto desc = memory::desc(DnnlExtensionUtils::convertToDnnlDims(prc_dims), DnnlExtensionUtils::IEPrecisionToDataType(outputPrc), format);
    prc_mem = memory(desc, getEngine());
    dst_size = desc.get_size();
}

inline void Reduce::create_DH_working_memory() {
    ReduceDH_opt = layout == ReduceLayoutType::reduce_nspc && !isDynamicNode() && support_split &&
                   !ReduceC && ReduceD && ReduceH && !ReduceW && IC == 1 && ID > 1;
    if (ReduceDH_opt) {
        PD = ID;
        PW = IW / blk_size * blk_size;
        prc_data_size = src_data_size;
        prc_size = PD * PW * src_data_size;
        if (prc_size > vec_reduceDH_prc.size()) {
            vec_reduceDH_prc.resize(prc_size);
        }
    }
}

inline void Reduce::calcProcessDstDims(std::vector<int64_t> &reduce_axes, const SizeVector &dst_dims) {
    std::set<size_t> axes;
    SizeVector out_dims;
    process_dst_dims.clear();
    axes_for_reduction.clear();
    for (auto &axis : reduce_axes) {
        if (axis < 0)
            axis += srcDims.size();
        if (static_cast<size_t>(axis) > srcDims.size())
            IE_THROW() << errorPrefix << " exceeds data tensor dimension on index to reduce";
        axes.insert(static_cast<size_t>(axis));
    }
    for (size_t i = 0; i < srcDims.size(); i++) {
        bool found = false;
        for (auto axis : axes) {
            if (i == axis) {
                found = true;
                break;
            }
        }
        if (found) {
            if (keepDims) out_dims.push_back(1);
            process_dst_dims.push_back(1);
            axes_for_reduction.push_back(i);
        } else {
            out_dims.push_back(srcDims[i]);
            process_dst_dims.push_back(srcDims[i]);
        }
    }
    if (jit_mode && jit_beyond_5D) {
        if (std::accumulate(out_dims.begin(), out_dims.end(), size_t(1), std::multiplies<size_t>()) !=
            std::accumulate(dst_dims.begin(), dst_dims.end(), size_t(1), std::multiplies<size_t>()))
            IE_THROW() << errorPrefix << "gets incorrect number of output dimensions!";
    } else {
        for (size_t i = 0; i < std::min(out_dims.size(), dst_dims.size()); i++) {
            if (out_dims[i] != dst_dims[i])
                IE_THROW() << errorPrefix << "gets incorrect number of output dimensions!";
        }
    }
}

inline void Reduce::set_reduce_dim_flags() {
    size_t dims_size = srcDims.size();
    if (dims_size == 5) {
        SET_SRC_DIM_VALUE(srcDims[0], srcDims[1], srcDims[2], srcDims[3], srcDims[4]);
        SET_DST_DIM_VALUE(process_dst_dims[0], process_dst_dims[1], process_dst_dims[2], process_dst_dims[3], process_dst_dims[4]);
    } else if (dims_size == 4) {
        SET_SRC_DIM_VALUE(srcDims[0], srcDims[1], 1, srcDims[2], srcDims[3]);
        SET_DST_DIM_VALUE(process_dst_dims[0], process_dst_dims[1], 1, process_dst_dims[2], process_dst_dims[3]);
    } else if (dims_size == 3) {
        SET_SRC_DIM_VALUE(1, srcDims[0], 1, srcDims[1], srcDims[2]);
        SET_DST_DIM_VALUE(1, process_dst_dims[0], 1, process_dst_dims[1], process_dst_dims[2]);
    } else if (dims_size == 2) {
        SET_SRC_DIM_VALUE(1, 1, 1, srcDims[0], srcDims[1]);
        SET_DST_DIM_VALUE(1, 1, 1, process_dst_dims[0], process_dst_dims[1]);
    } else {
        SET_SRC_DIM_VALUE(1, srcDims[0], 1, 1, 1);
        SET_DST_DIM_VALUE(1, process_dst_dims[0], 1, 1, 1);
    }

    // must be done before the following dimension change
    if (is_hybrid_layout) {
        create_working_memory();
    }

    // Reducing a dimesion in nspc layout can be treated as reducing another dimension in ncsp layout,
    // eg. reducing C in nspc can be treated as reducing W in ncsp layout, so that the routine reduce_PLN can be reused.
    // nspc -- ncsp
    //    D -- C
    //    H -- D
    //    W -- H
    //    C -- W
    if (layout == ReduceLayoutType::reduce_nspc) {
        size_t ITmp = IC; IC = ID; ID = IH; IH = IW; IW = ITmp;
        size_t OTmp = OC; OC = OD; OD = OH; OH = OW; OW = OTmp;
    }

    ReduceN = IB != OB && OB == 1;
    ReduceC = IC != OC && OC == 1;
    ReduceD = ID != OD && OD == 1;
    ReduceH = IH != OH && OH == 1;
    ReduceW = IW != OW && OW == 1;

    // must be done after the above dimension change
    create_DH_working_memory();

    // suit for parallel
    if (ReduceH && IW == 1) {
        ReduceW = true;
    }
    if (ReduceC && ReduceH && ID == 1) {
        ReduceD = true;
    }
}

inline void Reduce::reduce_ref(const float *in_ptr, float *out_ptr) {
    switch (algorithm) {
        case Algorithm::ReduceAnd:
            reduce_ref_process(in_ptr, out_ptr, 1, [](float x, float y)->float { return x && y; });
            break;
        case Algorithm::ReduceL1:
            reduce_ref_process(in_ptr, out_ptr, 0, [](float old, float y)->float { return old + (y >= 0 ? y : -y); });
            break;
        case Algorithm::ReduceL2:
            reduce_ref_process(in_ptr, out_ptr, 0, [](float old, float y)->float { return old + y * y; });
            break;
        case Algorithm::ReduceLogSum:
            reduce_ref_process(in_ptr, out_ptr, 0, [](float x, float y)->float { return x + y; });
            break;
        case Algorithm::ReduceLogSumExp:
            reduce_ref_process(in_ptr, out_ptr, 0, [](float old, float y)->float { return old + expf(y); });
            break;
        case Algorithm::ReduceMax:
            reduce_ref_process(in_ptr, out_ptr, std::numeric_limits<float>::lowest(),
                                                    [](float x, float y)->float { return x > y ? x : y; });
            break;
        case Algorithm::ReduceMean:
            reduce_ref_process(in_ptr, out_ptr, 0, [](float x, float y)->float { return x + y; });
            break;
        case Algorithm::ReduceMin:
            reduce_ref_process(in_ptr, out_ptr, std::numeric_limits<float>::max(),
                                                    [](float x, float y)->float { return x < y ? x : y; });
            break;
        case Algorithm::ReduceOr:
            reduce_ref_process(in_ptr, out_ptr, 0, [](float x, float y)->float { return x || y; });
            break;
        case Algorithm::ReduceProd:
            reduce_ref_process(in_ptr, out_ptr, 1, [](float x, float y)->float { return x * y; });
            break;
        case Algorithm::ReduceSum:
            reduce_ref_process(in_ptr, out_ptr, 0, [](float x, float y)->float { return x + y; });
            break;
        case Algorithm::ReduceSumSquare:
            reduce_ref_process(in_ptr, out_ptr, 0, [](float old, float y)->float { return old + y * y; });
            break;
    default:
        IE_THROW() << errorPrefix << "gets unsupported reduce mode.";
    }
}

void Reduce::reduce_ref_process(const float *in_ptr, float *out_ptr, float init_value, std::function<float(float, float)> func) {
    size_t work_amount_dst = 1, reduced_dims_work_amount = 1;
    for (size_t i = 0; i < process_dst_dims.size(); i++)
        work_amount_dst *= process_dst_dims[i];
    for (size_t i = 0; i < srcDims.size(); i++)
        reduced_dims_work_amount *= srcDims[i];
    reduced_dims_work_amount /= work_amount_dst;

    SizeVector src_strides = getParentEdgeAt(REDUCE_DATA)->getMemory().GetDescWithType<BlockedMemoryDesc>()->getStrides();
    parallel_nt(0, [&](const int ithr, const int nthr) {
        int j;
        size_t i, start = 0, end = 0;
        SizeVector dst_counters(process_dst_dims.size(), 0);
        splitter(work_amount_dst, nthr, ithr, start, end);
        for (j = process_dst_dims.size() - 1, i = start; j >= 0; j--) {
            dst_counters[j] = i % process_dst_dims[j];
            i /= process_dst_dims[j];
        }
        for (size_t src_idx = 0, dst_idx = start; dst_idx < end; ++dst_idx) {
            float reduce_prod = init_value;
            bool update_idx = true;
            SizeVector src_counters = dst_counters;
            for (i = 0; i < reduced_dims_work_amount; ++i) {
                if (update_idx) {
                    src_idx = 0;
                    for (j = 0; j < static_cast<int>(srcDims.size()); ++j)
                        src_idx += (src_counters[j] % srcDims[j]) * src_strides[j];
                    update_idx = false;
                }
                reduce_prod = func(reduce_prod, in_ptr[src_idx]);
                for (j = axes_for_reduction.size() - 1; j >= 0; j--) {
                    src_counters[axes_for_reduction[j]]++;
                    if (src_counters[axes_for_reduction[j]] < srcDims[axes_for_reduction[j]]) {
                        src_idx += src_strides[axes_for_reduction[j]];
                        break;
                    } else {
                        src_counters[axes_for_reduction[j]] = 0;
                        update_idx = true;
                    }
                }
            }
            out_ptr[dst_idx] = reduce_prod;
            for (j = process_dst_dims.size() - 1; j >= 0; j--) {
                dst_counters[j]++;
                if (dst_counters[j] < process_dst_dims[j])
                    break;
                else
                    dst_counters[j] = 0;
            }
        }
    });

    reduce_ref_map(out_ptr, work_amount_dst, reduced_dims_work_amount);
}

inline void Reduce::reduce_ref_map(float *out_ptr, size_t work_amount_dst, size_t reduced_dims_work_amount) {
    switch (algorithm) {
        case Algorithm::ReduceAnd:
        case Algorithm::ReduceL1:
        case Algorithm::ReduceMax:
        case Algorithm::ReduceMin:
        case Algorithm::ReduceOr:
        case Algorithm::ReduceProd:
        case Algorithm::ReduceSum:
        case Algorithm::ReduceSumSquare:
            break;
        case Algorithm::ReduceL2:
            parallel_for(work_amount_dst, [&](size_t i) {
                out_ptr[i] = std::sqrt(out_ptr[i]);
            });
            break;
        case Algorithm::ReduceLogSum:
        case Algorithm::ReduceLogSumExp:
            parallel_for(work_amount_dst, [&](size_t i) {
                out_ptr[i] = logf(out_ptr[i]);
            });
            break;
        case Algorithm::ReduceMean:
            parallel_for(work_amount_dst, [&](size_t i) {
                out_ptr[i] /= reduced_dims_work_amount;
            });
            break;
        default:
            IE_THROW() << errorPrefix << "gets unsupported reduce mode.";
    }
}

void Reduce::setPostOps(dnnl::primitive_attr &attr, const VectorDims &postOpDims, bool initWeights) {
    dnnl::post_ops ops;
    postOpsDataPtrs.clear();
    for (auto &node : fusedWith) {
        auto* fakeQuantizeNode = dynamic_cast<FakeQuantize *>(node.get());
        if (fakeQuantizeNode) {
            fakeQuantizeNode->appendPostOps(ops, {}, postOpsDataPtrs);
            continue;
        }

        auto* eltwiseNode = dynamic_cast<Eltwise *>(node.get());
        if (eltwiseNode) {
            eltwiseNode->appendPostOps(ops, postOpDims, postOpsDataPtrs, getFusingAxis());
            continue;
        }
        IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType()) << " node is not implemented";
    }

    attr.set_post_ops(ops);
}

void Reduce::setJITBeyond5D() {
    jit_beyond_5D = false;
    if (getInputShapeAtPort(REDUCE_DATA).getRank() > 5) {
        for (auto &axis : rawAxes) {
            if (axis < 0)
                axis += static_cast<int>(getInputShapeAtPort(REDUCE_DATA).getRank());
        }

        if (rawAxes.size() <= 1) {
            jit_beyond_5D = true;
        } else {
            for (size_t i = 1; i < rawAxes.size(); i++) {
                if (rawAxes[i] != rawAxes[i - 1] + 1) {
                    jit_beyond_5D = false;
                    break;
                }
                jit_beyond_5D = true;
            }
        }
    }
}

std::vector<int64_t> Reduce::update_src_dims() {
    std::vector<int64_t> reduce_axes = rawAxes;

    if (reduce_axes.size() < 1)
        return reduce_axes;

    size_t axis_dim = 1;
    size_t outer_dim = 1;
    size_t inner_dim = 1;
    int outer_end = reduce_axes[0];
    int inner_start = reduce_axes[reduce_axes.size() - 1];
    for (size_t i = 0; i < srcDims.size(); i++) {
        if (static_cast<int>(i) < outer_end) {
            outer_dim *= srcDims[i];
        } else if (static_cast<int>(i) > inner_start) {
            inner_dim *= srcDims[i];
        } else {
            axis_dim *= srcDims[i];
        }
    }

    reduce_axes.clear();
    reduce_axes.push_back(1);

    srcDims.clear();
    srcDims.push_back(outer_dim);
    srcDims.push_back(axis_dim);
    srcDims.push_back(inner_dim);

    return reduce_axes;
}

bool Reduce::canApplyJIT(const Precision &inputPrc, const Precision &outputPrc) const {
    static const Precision supportedPrecisions[] = {
            Precision::FP32,
            Precision::BF16,
            Precision::I32,
            Precision::I64,
            Precision::I8,
            Precision::U8
    };

    return (x64::mayiuse(x64::sse41)) && (getInputShapeAtPort(REDUCE_DATA).getRank() <= 5 || jit_beyond_5D) &&
           std::find(std::begin(supportedPrecisions), std::end(supportedPrecisions), inputPrc) != std::end(supportedPrecisions) &&
           std::find(std::begin(supportedPrecisions), std::end(supportedPrecisions), outputPrc) != std::end(supportedPrecisions);
}

int Reduce::getFusingAxis() const {
    int channelAxis = 1;
    if (!keepDims) {
        for (auto &raw_axis : rawAxes) {
            int axis = raw_axis >= 0 ? raw_axis : raw_axis + static_cast<int>(getInputShapeAtPort(REDUCE_DATA).getRank());
            if (axis == 1) {
                // channel axis has been reduced and doesn't exist any more
                channelAxis = -1;
                break;
            } else if (axis == 0) {
                channelAxis = 0;
            }
        }
    }
    return channelAxis;
}

bool Reduce::canFuse(const NodePtr& node) const {
    const auto inputPrc = getOriginalInputPrecisionAtPort(REDUCE_DATA);
    const auto outputPrc = getOriginalOutputPrecisionAtPort(0);
    if (!canApplyJIT(inputPrc, outputPrc) || jit_beyond_5D || algorithm == Algorithm::ReduceAnd || algorithm == Algorithm::ReduceOr) {
        return false;
    }

    // In jit mode we use the output memory as an intermediate accumulator for certain reduce modes.
    // If the post ops node has a lower precision for such modes, post ops fusing won't be supposted, in order to avoid accuracy loss.
    if (outputPrc == Precision::FP32 &&
            !node->getOriginalOutputPrecisions().empty() && node->getOriginalOutputPrecisionAtPort(0) != Precision::FP32) {
        if (!one_of(algorithm, Algorithm::ReduceAnd, Algorithm::ReduceOr, Algorithm::ReduceMin, Algorithm::ReduceMax)) {
            return false;
        }
    }

    return canFuseSimpleOperation(node);
}

bool Reduce::created() const {
    return getType() == Type::Reduce;
}
