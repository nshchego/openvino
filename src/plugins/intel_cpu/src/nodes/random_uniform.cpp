// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "random_uniform.hpp"

#include "ie_parallel.hpp"
#include "ie_ngraph_utils.hpp"
#include <openvino/op/constant.hpp>
#include <openvino/op/random_uniform.hpp>

using namespace dnnl::impl::cpu;

namespace ov {
namespace intel_cpu {
namespace node {

bool RandomUniform::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
std::cout << "RandomUniform::isSupportedOperation" << std::endl;
// return false;
    try {
        if (op->get_type_info() != op::v8::RandomUniform::get_type_info_static()) {
            errorMessage = "Only RandomUniform operation from the opset8 is supported by the CPU plugin.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

RandomUniform::RandomUniform(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
        : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    // RandomUniform should generate new sequence each run even if all inputs are constants. So that method Node::IsConstant()
    // doesn't return 'True' for RandomUniform with all constant inputs and the node generates new values for each inference,
    // we set 'NoConst' value for 'ConstantType' in ctor.
    constant = ConstantType::NoConst;

    m_shape_prc = op->get_input_element_type(SHAPE);
    if (!one_of(m_shape_prc, element::i32, element::i64)) {
        m_shape_prc = element::i32;
    }

    auto rnd_op = as_type_ptr<op::v8::RandomUniform>(op);
    // m_output_prc = rnd_op->get_out_type();
    m_global_seed = rnd_op->get_global_seed();
std::cout << "CPU m_global_seed: " << m_global_seed << std::endl;
    m_op_seed = rnd_op->get_op_seed();
    // m_op_seed = 284;
std::cout << "CPU m_op_seed: " << m_op_seed << std::endl;

    m_output_prc = op->get_output_element_type(0);
    if (m_output_prc.is_real() && !one_of(m_output_prc, element::f32, element::f16, element::bf16)) {
        m_output_prc = element::f32;
    }
    if (m_output_prc.is_integral() && !one_of(m_output_prc, element::i32, element::i64)) {
        m_output_prc = element::i32;
    }
std::cout << "CPU m_output_prc: " << m_output_prc << std::endl;

    for (int i = 0; i < op->get_input_size(); i++) {
        if (is_type<op::v0::Constant>(op->get_input_node_ptr(i))) {
            m_const_inputs[i] = true;
        }
    }

    if (m_const_inputs[SHAPE]) {
std::cout << "RandomUniform::RandomUniform m_const_inputs[SHAPE]; dyn: " << isDynamicNgraphNode(op) << std::endl;
        initOutShape(m_out_shape, as_type<op::v0::Constant>(op->get_input_node_ptr(SHAPE))->get_data_ptr(), m_shape_prc,
                op->get_input_shape(SHAPE)[0]);
std::cout << "Out shape {";
for (auto val : m_out_shape) {
    std::cout << val << "; ";
}
std::cout << "}" << std::endl;
    }
    if (m_const_inputs[MIN_VAL]) {
std::cout << "RandomUniform::RandomUniform m_const_inputs[MIN_VAL]" << std::endl;
        initEdgeValues(m_min_val, as_type<op::v0::Constant>(op->get_input_node_ptr(MIN_VAL))->get_data_ptr(), m_output_prc);
std::cout << "CPU m_min_val: " << m_min_val.f32 << std::endl;
    }
    if (m_const_inputs[MAX_VAL]) {
std::cout << "RandomUniform::RandomUniform m_const_inputs[MAX_VAL]" << std::endl;
        initEdgeValues(m_max_val, as_type<op::v0::Constant>(op->get_input_node_ptr(MAX_VAL))->get_data_ptr(), m_output_prc);
std::cout << "CPU m_max_val: " << m_max_val.f32 << std::endl;
    }

    m_generator = std::default_random_engine{m_op_seed};
}

void RandomUniform::getSupportedDescriptors() {
    if (getParentEdges().size() != 3) {
        THROW_CPU_NODE_ERR << "has incorrect number of input edges.";
    }
    if (getChildEdges().empty()) {
        THROW_CPU_NODE_ERR << "has incorrect number of output edges.";
    }
}

void RandomUniform::initSupportedPrimitiveDescriptors() {
    auto shape_prc = InferenceEngine::details::convertPrecision(m_shape_prc);
    auto out_prc = InferenceEngine::details::convertPrecision(m_output_prc);

    addSupportedPrimDesc({{LayoutType::ncsp, shape_prc, m_const_inputs[SHAPE]},
                          {LayoutType::ncsp, out_prc, m_const_inputs[MIN_VAL]},
                          {LayoutType::ncsp, out_prc, m_const_inputs[MAX_VAL]}},
                         {{LayoutType::ncsp, out_prc}},
                         ref_any); // TODO: expand
}

void RandomUniform::createPrimitive() {
#if defined(OPENVINO_ARCH_X86_64)
    if (m_algo == TF) {
        kernel::RandomUniform::CompileParams jcp;

        // jcp.inDataPrc     = dataPrecision;
        // jcp.gridPrc       = gridPrecision;
        // jcp.dynamicShapes = isDynamicNode();
        // jcp.alignCorners  = alignCorners;
        // jcp.interpolationMode = interpolationMode;
        // jcp.paddingMode   = paddingMode;

        // const auto& srcDataDims = getInputShapeAtPort(IN_DATA).getDims();
        // if (!jcp.dynamicShapes) {
        //     jcp.batchNum       = srcDataDims[0];
        //     jcp.cannelNum      = srcDataDims[1];
        //     jcp.dynamicBatch   = false;
        //     jcp.dynamicChannel = false;
        //     jcp.srcBatchStepB  = std::accumulate(srcDataDims.begin() + 1, srcDataDims.end(), dataTypeSize, std::multiplies<Dim>());
        // } else {
        //     jcp.dynamicBatch   = srcDataDims[0] == Shape::UNDEFINED_DIM;
        //     jcp.batchNum       = jcp.dynamicBatch ? 1lu : srcDataDims[0];
        //     jcp.dynamicChannel = srcDataDims[1] == Shape::UNDEFINED_DIM;
        //     jcp.cannelNum      = jcp.dynamicChannel ? 1lu : srcDataDims[1];
        // }

        m_jit_kernel = kernel::createInstance<RandomUniform>(jcp);
        // if (x64::mayiuse(x64::avx512_core)) {
        //     m_jit_kernel.reset(new kernel::RandomUniform<x64::avx512_core>(jcp));
        // } else if (x64::mayiuse(x64::avx2)) {
        //     m_jit_kernel.reset(new kernel::RandomUniform<x64::avx2>(jcp));
        // } else if (x64::mayiuse(x64::sse41)) {
        //     m_jit_kernel.reset(new kernel::RandomUniform<x64::sse41>(jcp));
        // }
        // if (!m_jit_kernel) {
        //     THROW_CPU_NODE_ERR << "could not create JIT kernel.";
        // }
        // m_jit_kernel->create_kernel();
    }
#endif // OPENVINO_ARCH_X86_64

    // nthr = parallel_get_max_threads();
    // execParamsPerThread.resize(nthr);
    // if (!x64::mayiuse(x64::avx512_core)) {
    //     const auto dataElPerVec = m_jit_kernel->getDataElPerVec();
    //     parallel_nt(nthr, [&](const int ithr, const int nthr) {
    //         auto& p = execParamsPerThread[ithr];

    //         p.srcHeightF.resize(dataElPerVec);
    //         p.srcWidthF.resize(dataElPerVec);
    //         p.srcWidthB.resize(dataElPerVec);
    //         p.dataTypeSize.resize(dataElPerVec);
    //         p.srcHeightSub1F.resize(dataElPerVec);
    //         p.srcWidthSub1F.resize(dataElPerVec);
    //         p.srcHeightMul2F.resize(dataElPerVec);
    //         p.srcWidthMul2F.resize(dataElPerVec);
    //         p.srcHeightMul2Sub1F.resize(dataElPerVec);
    //         p.srcWidthMul2Sub1F.resize(dataElPerVec);
    //         if (alignCorners) {
    //             p.wDenormCoefF.resize(dataElPerVec);
    //             p.hDenormCoefF.resize(dataElPerVec);
    //         }
    //         if (interpolationMode == GridSampleInterpolationMode::BICUBIC) {
    //             const size_t vecNum = paddingMode == GridSamplePaddingMode::ZEROS ? 32 : 16;
    //             p.buffer.resize(dataElPerVec * dataTypeSize * vecNum);
    //         }
    //     });
    // }

    Node::createPrimitive();
}

void RandomUniform::execute(dnnl::stream strm) {
// std::cout << "[CPU] RandomUniform::execute" << std::endl;
    if (!m_const_inputs[SHAPE]) {
        auto memPtr = getParentEdgeAt(SHAPE)->getMemoryPtr();
        initOutShape(m_out_shape, memPtr->getData(), m_shape_prc, memPtr->getShape().getElementsCount());

        redefineOutputMemory({m_out_shape});
    }
    if (!m_const_inputs[MIN_VAL]) {
        initEdgeValues(m_min_val, getParentEdgeAt(MIN_VAL)->getMemoryPtr()->getData(), m_output_prc);
    }
    if (!m_const_inputs[MAX_VAL]) {
        initEdgeValues(m_max_val, getParentEdgeAt(MAX_VAL)->getMemoryPtr()->getData(), m_output_prc);
    }

    auto dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    const auto out_el_num = dstMemPtr->getShape().getElementsCount();
// std::cout << "[CPU] RandomUniform::execute out_el_num: " << out_el_num << std::endl;

    if (m_algo == TF) {
// std::cout << "[CPU] RandomUniform::execute m_state={" << m_state.first << ";" << m_state.second << "}" << std::endl;
        m_state = computeTf(dstMemPtr->getData(), out_el_num, m_state);
    } else if (m_algo == ONNX) {
        computeOnnx(dstMemPtr->getData(), out_el_num);
    } else {
        THROW_CPU_NODE_ERR << "unsupported algorithm.";
    }
}

void RandomUniform::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

///////// ONNX algo //////////
void RandomUniform::computeOnnx(void* out, size_t work_amount) {
    switch (m_output_prc) {
        case element::f32: {
            generateData<float, std::uniform_real_distribution<float>>(
                    std::uniform_real_distribution<float>{m_min_val.f32, m_max_val.f32}, out, work_amount);
        } break;
        case element::i32: {
            generateData<int32_t, std::uniform_int_distribution<int32_t>>(
                    std::uniform_int_distribution<int32_t>{m_min_val.i32, m_max_val.i32}, out, work_amount);
        } break;
        case element::i64: {
            generateData<int64_t, std::uniform_int_distribution<int64_t>>(
                    std::uniform_int_distribution<int64_t>{m_min_val.i64, m_max_val.i64}, out, work_amount);
        } break;
        case element::f64: {
            generateData<double, std::uniform_real_distribution<double>>(
                    std::uniform_real_distribution<double>{m_min_val.f64, m_max_val.f64}, out, work_amount);
        } break;
        default:
            THROW_CPU_NODE_ERR << "has unsupported output type: " << m_output_prc;
    }
}

template <typename T, typename DISTR_TYPE>
void RandomUniform::generateData(DISTR_TYPE distribution, void* out, size_t work_amount) {
    auto dst = reinterpret_cast<T*>(out);
    for (size_t i = 0; i < work_amount; i++) {
        *dst = distribution(m_generator);
        dst++;
    }
}

///////// TF algo //////////

namespace {
// Following const values are taken from the original paper:
// https://www.thesalmons.org/john/random123/papers/random123sc11.pdf
constexpr uint32_t CRUSH_RESISTANCE_CONST_LOWER_VALUE = 0x9E3779B9;
constexpr uint32_t CRUSH_RESISTANCE_CONST_UPPER_VALUE = 0xBB67AE85;
constexpr uint64_t STATISTIC_MAXIMIZING_MULTIPLIER_N = 0xD2511F53;
constexpr uint64_t STATISTIC_MAXIMIZING_MULTIPLIER_COUNTER = 0xCD9E8D57;
constexpr uint64_t ROUNDS_NUMBER = 10llu;

// Concatenates two uint32 values into single uint64 values.
uint64_t unite_high_low(uint32_t high, uint32_t low) {
    return (static_cast<uint64_t>(high) << 32) + low;
}

void calculateRound(const uint32_t* key, uint32_t* counter, uint32_t* n) {
    // Each round performs following updating for n and counter:
    // left uint32 part = mullo(R, M)
    // right uint32 part  = mulhi(R, M) xor k xor L
    // mulhi(a, b) = floor((a * b) / 2^32)
    // mullo(a, b) = (a * b) mod 2^32,
    // where M - statistic_maximizing_multiplier const
    uint64_t prod_0 = STATISTIC_MAXIMIZING_MULTIPLIER_N * n[0];
    uint64_t prod_1 = STATISTIC_MAXIMIZING_MULTIPLIER_COUNTER * counter[0];
    n[0] = static_cast<uint32_t>(prod_1 >> 32) ^ n[1] ^ key[0];
    n[1] = static_cast<uint32_t>(prod_1);
    counter[0] = static_cast<uint32_t>(prod_0 >> 32) ^ counter[1] ^ key[1];
    counter[1] = static_cast<uint32_t>(prod_0);
}

void raiseKey(uint32_t* key) {
    key[0] += CRUSH_RESISTANCE_CONST_LOWER_VALUE;
    key[1] += CRUSH_RESISTANCE_CONST_UPPER_VALUE;
}

// Helper function for converting uint32 values to float32. Sets fractional part of
// floating value with bits from uint32 value. Resulting value is in interval [0,1).
float uint32ToFloat(uint32_t x) {
    // float32 is formatted as follows: sign(1 bit) exponent(8 bits) mantissa(23 bits).
    // The value is interpreted using following formula:
    // (-1)^sign * 1, mantissa * 2 ^ (exponent - 127)
    // Here we set the following values:
    // sign = 0
    // exponent = 127, for obtaining a zero exponent.
    // mantissa = 23 right bits from generated uint32 random value.

    RandomUniform::OutputType out_val = {(static_cast<uint32_t>(127) << 23) | (x & 0x7fffffu)};
    return out_val.f32 - 1.0f;
}

// Helper function for converting uint32 values to float16.Sets fractional part of
// floating value with bits from uint32 value. Resulting value is in interval [0,1).
float16 uint32ToFloat16(uint32_t x) {
    // float16 is formatted as follows: sign(1 bit) exponent(5 bits) mantissa(10 bits). The value is interpreted
    // The value is interpreted using following formula:
    // (-1)^sign * 1, mantissa * 2 ^ (exponent - 15)
    // Here we set the following values:
    // sign = 0
    // exponent = 15, for obtaining a zero exponent.
    // mantissa = 10 right bits from generated uint32 random value.

    uint16_t x_uint16 = static_cast<uint16_t>(x);
    RandomUniform::OutputType out_val = {(static_cast<uint16_t>(15) << 10) | (x_uint16 & 0x3ffu)};
    return out_val.f16 - static_cast<float16>(1);
}

// Helper function for converting uint32 values to bfloat16. Sets fractional part of
// floating value with bits from uint32 value. Resulting value is in interval [0,1).
bfloat16 uint32ToBfloat16(uint32_t x) {
    // bfloat16 is formatted as follows: sign(1 bit) exponent(8 bits) mantissa(7 bits). The value is interpreted
    // The value is interpreted using following formula:
    // (-1)^sign * 1, mantissa * 2 ^ (exponent - 127)
    // Here we set the following values:
    // sign = 0
    // exponent = 127, for obtaining a zero exponent.
    // mantissa = 7 right bits from generated uint32 random value.

    uint16_t x_uint16 = static_cast<uint16_t>(x);
    RandomUniform::OutputType out_val = {(static_cast<uint16_t>(127) << 7) | (x_uint16 & 0x7fu)};
    return out_val.bf16 - static_cast<bfloat16>(1);
}

void runPhilox(uint64_t key, uint64_t counter, uint64_t n, uint32_t* res) {
    uint32_t* key_32 = reinterpret_cast<uint32_t*>(&key);
    uint32_t* counter_32 = reinterpret_cast<uint32_t*>(&counter);
    uint32_t* n_32 = reinterpret_cast<uint32_t*>(&n);

    for (size_t i = 0; i < ROUNDS_NUMBER; i++) {
        calculateRound(key_32, counter_32, n_32);
        if (i < ROUNDS_NUMBER - 1)
            raiseKey(key_32);
    }

    res[0] = n_32[0];
    res[1] = n_32[1];
    res[2] = counter_32[0];
    res[3] = counter_32[1];
}

// Converts uint32 values to destination type and normalizes to required range.
template <typename T>
void convertToOutputType(const uint32_t* res,
                            size_t step,
                            const element::Type& elem_type,
                            T min_val,
                            T max_val,
                            uint8_t* out,
                            size_t k,
                            size_t elem_count,
                            T (*convert_single_input)(uint32_t) = nullptr,
                            T (*convert_two_inputs)(uint32_t, uint32_t, T, T) = nullptr,
                            T (*mod_func)(uint32_t, T, T) = nullptr) {
    std::vector<T> res_out_type(step);
    if (elem_type.size() > 4) {
        // Each element of resulting sequence is formed using two uint32 values
        res_out_type[0] = convert_two_inputs(res[0], res[1], min_val, max_val);
        res_out_type[1] = convert_two_inputs(res[2], res[3], min_val, max_val);
    } else {
        // Each element of resulting sequence is formed using single uint32 value
        std::transform(res,
                       res + step,
                       res_out_type.data(),
                       [&min_val, &max_val, &convert_single_input, &mod_func](uint32_t elem) {
                           if (convert_single_input != nullptr) {
                               return convert_single_input(elem) * (max_val - min_val) + min_val;
                           } else {
                               return mod_func(elem, min_val, max_val);
                           }
                       });
    }

    memcpy(out + k * elem_type.size(), res_out_type.data(), std::min(step, elem_count - k) * elem_type.size());
}

}  // namespace

std::pair<uint64_t, uint64_t> RandomUniform::computeTf(void* out, size_t out_el_num, const std::pair<uint64_t, uint64_t>& prev_state) {
    // When both seed values are equal to zero RandomUniform should generate non-deterministic sequence.
    if (m_global_seed == 0 && m_op_seed == 0) {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
        m_global_seed = std::rand();
    }

    // Get previous counter state
    uint64_t n_state = prev_state.first;
    uint64_t counter_state = prev_state.second;

    // Initialize Philox key and counters
    const uint64_t key = m_global_seed;
    uint64_t counter = counter_state > 0 ? counter_state : m_op_seed;

    // Each run of Philox algorithm generates 4 uint32 values.
    // If output_type is int32, f32, bf16, or f16 each value is converted to
    // corresponding type so we have 4 result values. For f64 and i64 we use
    // a pair of values for conversion, so we have 2 result values.
    // Step indicates how many values we generate in one iteration.

    auto out_u8 = reinterpret_cast<uint8_t*>(out);
    const auto groups_num = (out_el_num + PHILOX_GROUP_SIZE - 1) / PHILOX_GROUP_SIZE;

    if (m_jit_kernel) {
        const auto el_per_vec = m_jit_kernel->getVectorLen() / m_output_prc.size();
        const size_t step = m_output_prc.size() > 4 ? el_per_vec : (el_per_vec * 2);

        auto threadBody = [&](const int ithr, const int nthr) {
            const auto groups_per_thr = (groups_num + nthr - 1) / nthr;
            auto start = ithr * groups_per_thr * PHILOX_GROUP_SIZE;
            auto end = (ithr + 1) * groups_per_thr * PHILOX_GROUP_SIZE;
            if (end > out_el_num) {
                end = out_el_num;
            }
            if (start >= out_el_num || end - start <= 0) {
                return;
            }
            uint64_t n = n_state + start / PHILOX_GROUP_SIZE;
// printf("[CPU][%d] exec out_el_num: %ld; step: %ld; start: %ld; end: %ld\n", ithr, out_el_num, step, start, end);

            uint32_t res[4];
            for (size_t k = start; k < end; k += step) {
                // generate 4 random uint32 values using Philox algorithm
// printf("[CPU][%d] key: %ld; counter: %ld; n: %ld\n", ithr, key, counter, n);
                // runPhilox(key, counter, n, res);
// printf("[CPU][%d] key: %u; counter: %ld; n: %ld; res={%u; %u; %u; %u}\n", ithr, key[0], counter, n, res[0], res[1], res[2], res[3]);

                kernel::RandomUniformExecArgs args;
                (*m_jit_kernel)(&args);

                // convert values to corresponding output_type
                switch (m_output_prc) {
                    case element::Type_t::f32: {
                        convertToOutputType<float>(res, step, m_output_prc, m_min_val.f32, m_max_val.f32, out_u8, k, out_el_num, uint32ToFloat);
                    } break;
                    case element::Type_t::f16: {
                        convertToOutputType<float16>(res,
                                                    step,
                                                    m_output_prc,
                                                    m_min_val.f16,
                                                    m_max_val.f16,
                                                    out_u8,
                                                    k,
                                                    out_el_num,
                                                    uint32ToFloat16);
                    } break;
                    case element::Type_t::bf16: {
                        convertToOutputType<bfloat16>(res,
                                                    step,
                                                    m_output_prc,
                                                    m_min_val.bf16,
                                                    m_max_val.bf16,
                                                    out_u8,
                                                    k,
                                                    out_el_num,
                                                    uint32ToBfloat16);
                    } break;
                    case element::Type_t::i32: {
                        convertToOutputType<int>(res,
                                                    step,
                                                    m_output_prc,
                                                    m_min_val.i32,
                                                    m_max_val.i32,
                                                    out_u8,
                                                    k,
                                                    out_el_num,
                                                    nullptr,
                                                    nullptr,
                                                    [](uint32_t x, int mn, int mx) {
                                                        return static_cast<int>(x % (mx - mn) + mn);
                                                    });
                    } break;
                    case element::Type_t::i64: {
                        convertToOutputType<int64_t>(res,
                                                    step,
                                                    m_output_prc,
                                                    m_min_val.i64,
                                                    m_max_val.i64,
                                                    out_u8,
                                                    k,
                                                    out_el_num,
                                                    nullptr,
                                                    [](uint32_t a, uint32_t b, int64_t mn, int64_t mx) {
                                                        return static_cast<int64_t>(unite_high_low(b, a) % (mx - mn) + mn);
                                                    });
                    } break;
                    default: OPENVINO_THROW("Unsupported type of RandomUniform: ", m_output_prc.to_string());
                }

                if (++n == 0) {
std::cout << "[CPU] RandomUniform::computeTf (++n == 0)" << std::endl;
                    ++counter;
                }
            }
        };

        // if (out_el_num < PHILOX_PARALLEL_EXECUTION_THRESHOLD) {
        //     parallel_nt(1, threadBody);
        // } else {
            parallel_nt(0, threadBody);
        // }
    } else {
        const size_t step = m_output_prc.size() > 4 ? 2 : 4;

        auto threadBody = [&](const int ithr, const int nthr) {
            const auto groups_per_thr = (groups_num + nthr - 1) / nthr;
            auto start = ithr * groups_per_thr * PHILOX_GROUP_SIZE;
            auto end = (ithr + 1) * groups_per_thr * PHILOX_GROUP_SIZE;
            if (end > out_el_num) {
                end = out_el_num;
            }
            if (start >= out_el_num || end - start <= 0) {
                return;
            }
            uint64_t n = n_state + start / PHILOX_GROUP_SIZE;
// printf("[CPU][%d] exec out_el_num: %ld; step: %ld; start: %ld; end: %ld\n", ithr, out_el_num, step, start, end);

            uint32_t res[4];
            for (size_t k = start; k < end; k += step) {
                // generate 4 random uint32 values using Philox algorithm
// printf("[CPU][%d] key: %ld; counter: %ld; n: %ld\n", ithr, key, counter, n);
                runPhilox(key, counter, n, res);
// printf("[CPU][%d] key: %u; counter: %ld; n: %ld; res={%u; %u; %u; %u}\n", ithr, key[0], counter, n, res[0], res[1], res[2], res[3]);

                // convert values to corresponding output_type
                switch (m_output_prc) {
                    case element::Type_t::f32: {
                        convertToOutputType<float>(res, step, m_output_prc, m_min_val.f32, m_max_val.f32, out_u8, k, out_el_num, uint32ToFloat);
                    } break;
                    case element::Type_t::f16: {
                        convertToOutputType<float16>(res,
                                                    step,
                                                    m_output_prc,
                                                    m_min_val.f16,
                                                    m_max_val.f16,
                                                    out_u8,
                                                    k,
                                                    out_el_num,
                                                    uint32ToFloat16);
                    } break;
                    case element::Type_t::bf16: {
                        convertToOutputType<bfloat16>(res,
                                                    step,
                                                    m_output_prc,
                                                    m_min_val.bf16,
                                                    m_max_val.bf16,
                                                    out_u8,
                                                    k,
                                                    out_el_num,
                                                    uint32ToBfloat16);
                    } break;
                    case element::Type_t::i32: {
                        convertToOutputType<int>(res,
                                                    step,
                                                    m_output_prc,
                                                    m_min_val.i32,
                                                    m_max_val.i32,
                                                    out_u8,
                                                    k,
                                                    out_el_num,
                                                    nullptr,
                                                    nullptr,
                                                    [](uint32_t x, int mn, int mx) {
                                                        return static_cast<int>(x % (mx - mn) + mn);
                                                    });
                    } break;
                    case element::Type_t::i64: {
                        convertToOutputType<int64_t>(res,
                                                    step,
                                                    m_output_prc,
                                                    m_min_val.i64,
                                                    m_max_val.i64,
                                                    out_u8,
                                                    k,
                                                    out_el_num,
                                                    nullptr,
                                                    [](uint32_t a, uint32_t b, int64_t mn, int64_t mx) {
                                                        return static_cast<int64_t>(unite_high_low(b, a) % (mx - mn) + mn);
                                                    });
                    } break;
                    default: OPENVINO_THROW("Unsupported type of RandomUniform: ", m_output_prc.to_string());
                }

                if (++n == 0) {
std::cout << "[CPU] RandomUniform::computeTf (++n == 0)" << std::endl;
                    ++counter;
                }
            }
        };

        if (out_el_num < PHILOX_PARALLEL_EXECUTION_THRESHOLD) {
            parallel_nt(1, threadBody);
        } else {
            parallel_nt(0, threadBody);
        }
    }

    // Calculate counter values for next RandomUniform run
    const uint64_t skip_count = out_el_num * SKIP_CONST;
    n_state += skip_count;
    if (n_state < skip_count)
        counter_state++;

    return { n_state, counter_state };
}
//////////////////////////////

void RandomUniform::initOutShape(VectorDims& dst, const void* src, const element::Type& shape_type, size_t len) {
    switch (shape_type) {
        case element::i32: {
            auto data = reinterpret_cast<const int32_t*>(src);
            dst.assign(data, data + len);
        } break;
        case element::i64: {
            auto data = reinterpret_cast<const int64_t*>(src);
            dst.assign(data, data + len);
        } break;
        default:
            THROW_CPU_NODE_ERR << "has unsupported shape precision: " << m_output_prc;
    }
}

void RandomUniform::initEdgeValues(OutputType& dst, const void* src, const element::Type& output_type) {
    switch (output_type) {
        case element::f32:
            dst.f32 = *reinterpret_cast<const float*>(src);
            break;
        case element::i32:
            dst.i32 = *reinterpret_cast<const int32_t*>(src);
            break;
        case element::i64:
            dst.i64 = *reinterpret_cast<const int64_t*>(src);
            break;
        case element::f64:
            dst.f64 = *reinterpret_cast<const double*>(src);
            break;
        default:
            THROW_CPU_NODE_ERR << "has unsupported output precision: " << output_type;
    }
}

std::string RandomUniform::getPrimitiveDescriptorType() const {
    std::string str_type = "ref";
    auto selectedPrimitiveDesc = getSelectedPrimitiveDescriptor();
    if (selectedPrimitiveDesc->getConfig().outConfs[0].getMemDesc()->getPrecision() != InferenceEngine::Precision::U8) {
        str_type += "_" + std::string(selectedPrimitiveDesc->getConfig().outConfs[0].getMemDesc()->getPrecision().name());
    } else {
        str_type += "_I8";
    }
    return str_type;
}

bool RandomUniform::needPrepareParams() const {
    return false;
}

bool RandomUniform::isExecutable() const {
    return !isInputTensorAtPortEmpty(SHAPE);
}

bool RandomUniform::created() const {
    return getType() == Type::RandomUniform;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
