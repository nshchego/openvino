// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <vector>
#include <utility>
#include <map>
#include <memory>

namespace MKLDNNPlugin {

struct jit_transpose_conf_t {
    uint32_t ndims;
    InferenceEngine::SizeVector dst_block_dims;
    InferenceEngine::SizeVector src_strides;
    InferenceEngine::SizeVector dst_strides;
    int n;
    int data_size;

    bool supported_dynamic_batch = false;
};

struct jit_args_transpose {
    const void* src;
    const void* dst;
};

struct jit_uni_transpose_kernel {
    void (*ker_)(const jit_args_transpose *);

    void operator()(const jit_args_transpose *args) { assert(ker_); ker_(args); }

    jit_transpose_conf_t jpp;

    virtual void create_ker() = 0;

    explicit jit_uni_transpose_kernel(jit_transpose_conf_t jpp) : ker_(nullptr), jpp(jpp) {}
    virtual ~jit_uni_transpose_kernel() {}
};

class MKLDNNTransposeNode : public MKLDNNNode {
public:
    MKLDNNTransposeNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNTransposeNode() override = default;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }

    const InferenceEngine::SizeVector& getOrder() const {
        return order;
    }

private:
    InferenceEngine::SizeVector order;
    InferenceEngine::Precision prec;

    typedef std::function<void(int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr)> transposeImpl;
    typedef std::function<bool(int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr)> isApplicable;
    struct TransposeImpl {
        TransposeImpl(transposeImpl f0, isApplicable f1): execute(std::move(f0)), isValidParams(std::move(f1)) {}

        transposeImpl execute;
        isApplicable isValidParams;
    };

    static const std::multimap<InferenceEngine::SizeVector, TransposeImpl> OptimizedCases;
    std::shared_ptr<jit_uni_transpose_kernel> transpose_kernel;
};

}  // namespace MKLDNNPlugin

