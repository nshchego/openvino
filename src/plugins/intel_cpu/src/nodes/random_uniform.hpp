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
    union edge {
        int32_t i32;
        int64_t i64;
        float f32;
        double f64;
    };

public:
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

private:
    void compute(void* out, size_t work_amount);

    template <typename T, typename DISTR_TYPE>
    void generateData(DISTR_TYPE distribution, void* out, size_t work_amount);

    void initOutShape(VectorDims& dst, const void* src, const element::Type& shape_type, size_t len);

    void initEdge(edge& dst, const void* src, const element::Type& output_type);

    enum { SHAPE = 0, MIN_VAL, MAX_VAL };

    bool m_const_inputs[3] = {false, false, false};

    ov::element::Type m_shape_prc;
    ov::element::Type m_output_prc;
    int m_global_seed = 0;
    int m_op_seed = 0;

    VectorDims m_out_shape;
    edge m_min_val;
    edge m_max_val;
    // void* m_min_val;
    // void* m_max_max;

    std::default_random_engine m_generator;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
