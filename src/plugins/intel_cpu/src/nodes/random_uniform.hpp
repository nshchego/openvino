// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include <random>

namespace ov {
namespace intel_cpu {
namespace node {

class RandomUniform : public Node {
public:
    union OutputType {
        int32_t i32;
        int64_t i64;
        float f32;
        double f64;
    };

    RandomUniform(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;

    void initSupportedPrimitiveDescriptors() override;

    bool created() const override;

    // bool canBeInPlace() const override {
    //     return false;
    // }

    bool needPrepareParams() const override;

    void execute(dnnl::stream strm) override;

    void executeDynamicImpl(dnnl::stream strm) override;

    bool isExecutable() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    std::string getPrimitiveDescriptorType() const override;

private:
    void computeOnnx(void* out, size_t work_amount);
    std::pair<uint64_t, uint64_t> computeTf(void* out, size_t work_amount, const std::pair<uint64_t, uint64_t>& prev_state);
    std::pair<uint64_t, uint64_t> computeTfParallel(void* out, size_t work_amount, const std::pair<uint64_t, uint64_t>& prev_state);

    template <typename T, typename DISTR_TYPE>
    void generateData(DISTR_TYPE distribution, void* out, size_t work_amount);

    void initOutShape(VectorDims& dst, const void* src, const element::Type& shape_type, size_t len);

    void initEdgeValues(OutputType& dst, const void* src, const element::Type& output_type);

    enum { SHAPE = 0, MIN_VAL, MAX_VAL };
    enum AlgoType { ONNX, TF };

    bool m_const_inputs[3] = {false, false, false};

    ov::element::Type m_shape_prc;
    ov::element::Type m_output_prc;
    uint64_t m_global_seed = 0;
    uint64_t m_op_seed = 0;
    std::pair<uint64_t, uint64_t> m_state {0llu, 0llu};

    VectorDims m_out_shape;
    OutputType m_min_val;
    OutputType m_max_val;
    AlgoType algo = TF;

    std::default_random_engine m_generator;

    // TF PHILOX constants
    // Determines how many sequence elements of RNG sequence are skipped between runs.
    // Can be any positive value, 256 is chosen for parity with Tensorflow.
    static constexpr uint64_t SKIP_CONST = 256;

    // Philox algorithm returns 4 elements of RNG sequence per each invocation
    static constexpr size_t PHILOX_GROUP_SIZE = 4;
    static constexpr size_t ROUNDS_NUMBER = 10;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
