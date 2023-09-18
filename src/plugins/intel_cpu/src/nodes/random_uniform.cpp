// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "random_uniform.hpp"

#include "ie_parallel.hpp"
#include "ie_ngraph_utils.hpp"
#include <openvino/op/constant.hpp>
#include <openvino/op/random_uniform.hpp>

namespace ov {
namespace intel_cpu {
namespace node {

bool RandomUniform::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
std::cout << "RandomUniform::isSupportedOperation" << std::endl;
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
    // we set 'NoConst' value for 'ConstantType' in ctor
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
    if (m_output_prc.is_real() && !one_of(m_output_prc, element::f32, element::f64)) {
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
                         ref_any);
}

void RandomUniform::execute(dnnl::stream strm) {
// std::cout << "RandomUniform::execute" << std::endl;
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

    if (algo == TF) {
        m_state = computeTf(dstMemPtr->getData(), dstMemPtr->getShape().getElementsCount(), m_state);
    } else if (algo == ONNX) {
        computeOnnx(dstMemPtr->getData(), dstMemPtr->getShape().getElementsCount());
    } else {
        THROW_CPU_NODE_ERR << "unsupported random algorithm.";
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

// Following const values are taken from the original paper:
// https://www.thesalmons.org/john/random123/papers/random123sc11.pdf
constexpr uint32_t CRUSH_RESISTANCE_CONST_LOWER_VALUE = 0x9E3779B9;
constexpr uint32_t crush_resistance_const_upper_value = 0xBB67AE85;
constexpr uint64_t statistic_maximizing_multiplier_n = 0xD2511F53;
constexpr uint64_t statistic_maximizing_multiplier_counter = 0xCD9E8D57;

namespace {

// Splits uint64 value into two uint32 values with right and left part of original value.
std::pair<uint32_t, uint32_t> split_high_low(uint64_t value) {
    uint32_t low = static_cast<uint32_t>(value);
    uint32_t high = static_cast<uint32_t>(value >> 32);
    return {low, high};
}

// Concatenates two uint32 values into single uint64 values.
uint64_t unite_high_low(uint32_t high, uint32_t low) {
    return (static_cast<uint64_t>(high) << 32) + low;
}

// Runs single "round" of Philox algorithm.
void calculate_round(uint64_t key, uint64_t& counter, uint64_t& n) {
    // Split key, counter and n into two uint32 values.
    auto counter_lr = split_high_low(counter);
    auto key_lr = split_high_low(key);
    auto n_lr = split_high_low(n);

    // Each round performs following updating for n and counter:
    // left uint32 part = mullo(R, M)
    // right uint32 part  = mulhi(R, M) xor k xor L
    // mulhi(a, b) = floor((a * b) / 2^32)
    // mullo(a, b) = (a * b) mod 2^32,
    // where M - statistic_maximizing_multiplier const
    auto prod0 = split_high_low(statistic_maximizing_multiplier_n * n_lr.first);
    auto prod1 = split_high_low(statistic_maximizing_multiplier_counter * counter_lr.first);
    n_lr.first = prod1.second ^ n_lr.second ^ key_lr.first;
    n_lr.second = prod1.first;
    counter_lr.first = prod0.second ^ counter_lr.second ^ key_lr.second;
    counter_lr.second = prod0.first;

    // Unite counter and n into uint64 values.
    counter = unite_high_low(counter_lr.second, counter_lr.first);
    n = unite_high_low(n_lr.second, n_lr.first);
}

// Increases key value.
void raise_key(uint64_t& key) {
    auto key_lr = split_high_low(key);
    key_lr.first += CRUSH_RESISTANCE_CONST_LOWER_VALUE;
    key_lr.second += crush_resistance_const_upper_value;
    key = unite_high_low(key_lr.second, key_lr.first);
}

// Helper function for converting uint32 values to float32. Sets fractional part of
// floating value with bits from uint32 value. Resulting value is in interval [0,1).
float uint32_to_float(uint32_t x) {
    // float32 is formatted as follows: sign(1 bit) exponent(8 bits) mantissa(23 bits). The value is interpreted
    // The value is interpreted using following formula:
    // (-1)^sign * 1, mantissa * 2 ^ (exponent - 127)
    // Here we set the following values:
    // sign = 0
    // exponent = 127, for obtaining a zero exponent.
    // mantissa = 23 right bits from generated uint32 random value.

    RandomUniform::OutputType out_val = {(static_cast<uint32_t>(127) << 23) | (x & 0x7fffffu)};
    // out_val.f32 = ((static_cast<uint32_t>(127) << 23) | (x & 0x7fffffu));
    return out_val.f32 - 1.0f;
}

// Helper function for converting uint32 values to float16.Sets fractional part of
// floating value with bits from uint32 value. Resulting value is in interval [0,1).
// float16 uint32_to_float16(uint32_t x) {
//     // float16 is formatted as follows: sign(1 bit) exponent(5 bits) mantissa(10 bits). The value is interpreted
//     // The value is interpreted using following formula:
//     // (-1)^sign * 1, mantissa * 2 ^ (exponent - 15)
//     // Here we set the following values:
//     // sign = 0
//     // exponent = 15, for obtaining a zero exponent.
//     // mantissa = 10 right bits from generated uint32 random value.

//     uint16_t x_uint16 = static_cast<uint16_t>(x);
//     RandomUniform::OutputType out_val = {(static_cast<uint16_t>(15) << 10) | (x_uint16 & 0x3ffu)};
//     return out_val.f16 - static_cast<float16>(1);
// }

// Helper function for converting uint32 values to double. Sets fractional part of
// floating double with bits from uint32 values. Resulting value is in interval [0,1).
double uint32_to_double(uint32_t x1, uint32_t x2) {
    // float64 is formatted as follows: sign(1 bit) exponent(11 bits) mantissa(52 bits). The value is interpreted
    // The value is interpreted using following formula:
    // (-1)^sign * 1, mantissa * 2 ^ (exponent - 1023)
    // Here we set the following values:
    // sign = 0
    // exponent = 1023, for obtaining a zero exponent.
    // mantissa = 52 right bits from two concatenated uint32 values from random integer generator.

    uint64_t significant = ((static_cast<uint64_t>(x1) & 0xfffffu) << 32) | static_cast<uint64_t>(x2);
    RandomUniform::OutputType out_val = {((static_cast<uint64_t>(1023) << 52) | significant)};
    return out_val.f64 - 1.0;
}

// Helper function for converting uint32 values to bfloat16. Sets fractional part of
// floating value with bits from uint32 value. Resulting value is in interval [0,1).
// bfloat16 uint32_to_bfloat16(uint32_t x) {
//     // bfloat16 is formatted as follows: sign(1 bit) exponent(8 bits) mantissa(7 bits). The value is interpreted
//     // The value is interpreted using following formula:
//     // (-1)^sign * 1, mantissa * 2 ^ (exponent - 127)
//     // Here we set the following values:
//     // sign = 0
//     // exponent = 127, for obtaining a zero exponent.
//     // mantissa = 7 right bits from generated uint32 random value.

//     uint16_t x_uint16 = static_cast<uint16_t>(x);
//     RandomUniform::OutputType out_val = {(static_cast<uint16_t>(127) << 7) | (x_uint16 & 0x7fu)};
//     return out_val.bf16 - static_cast<bfloat16>(1);
// }

// Runs Philox algorithm.
void run_philox(uint64_t key, uint64_t counter, uint64_t n, size_t n_rounds, std::vector<uint32_t>& res) {
    for (size_t i = 0; i < n_rounds; i++) {
        calculate_round(key, counter, n);
        if (i < n_rounds - 1)
            raise_key(key);
    }
    auto res1 = split_high_low(n);
    auto res2 = split_high_low(counter);
    res[0] = res1.first;
    res[1] = res1.second;
    res[2] = res2.first;
    res[3] = res2.second;
}

// Converts uint32 values to destination type and normalizes to required range.
template <typename T>
void convert_to_output_type(const std::vector<uint32_t>& res,
                            size_t step,
                            const element::Type& elem_type,
                            T min_val,
                            T max_val,
                            void* out,
                            size_t k,
                            size_t elem_count,
                            T (*convert_single_input)(uint32_t) = nullptr,
                            T (*convert_two_inputs)(uint32_t, uint32_t, T, T) = nullptr,
                            T (*mod_func)(uint32_t, T, T) = nullptr) {
    // Get min and max values
    // T mn[1];
    // T mx[1];
    // memcpy(mn, min_val, elem_type.size());
    // memcpy(mx, max_val, elem_type.size());

    std::vector<T> res_out_type(step);
    if (elem_type.size() > 4) {
        // Each element of resulting sequence is formed using two uint32 values
        res_out_type[0] = convert_two_inputs(res[0], res[1], min_val, max_val);
        res_out_type[1] = convert_two_inputs(res[2], res[3], min_val, max_val);
    } else {
        // Each element of resulting sequence is formed using single uint32 value
        std::transform(res.data(),
                       res.data() + step,
                       res_out_type.data(),
                       [&min_val, &max_val, &convert_single_input, &mod_func](uint32_t elem) {
                           if (convert_single_input != nullptr) {
                               return convert_single_input(elem) * (max_val - min_val) + min_val;
                           } else {
                               return mod_func(elem, min_val, max_val);
                           }
                       });
    }

    auto out_char = static_cast<char*>(out);
    memcpy(out_char + k * elem_type.size(), res_out_type.data(), std::min(step, elem_count - k) * elem_type.size());
}

}  // namespace

// Implementation of RandomUniform that uses Philox algorithm as inner random unsigned integer generator.
std::pair<uint64_t, uint64_t> RandomUniform::computeTf(void* out, size_t work_amount, const std::pair<uint64_t, uint64_t>& prev_state) {
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
    uint64_t n = n_state;

    // Philox algorithm returns 4 elements of RNG sequence per each invocation
    const size_t philox_output_size = 4;

    // Each run of Philox algorithm generates 4 uint32 values.
    // If output_type is int32, f32, bf16, or f16 each value is converted to
    // corresponding type so we have 4 result values. For f64 and i64 we use
    // a pair of values for conversion, so we have 2 result values.
    // Step indicates how many values we generate in one iteration.
    const size_t step = m_output_prc.size() > 4 ? 2 : 4;
    const size_t rounds_number = 10;

    for (size_t k = 0; k < work_amount; k += step) {
        // generate 4 random uint32 values using Philox algorithm
        std::vector<uint32_t> res(philox_output_size);
        run_philox(key, counter, n, rounds_number, res);

        // convert values to corresponding output_type
        switch (m_output_prc) {
            case element::Type_t::f32: {
                convert_to_output_type<float>(res, step, m_output_prc, m_min_val.f32, m_max_val.f32, out, k, work_amount, uint32_to_float);
            } break;
            // case element::Type_t::f16: {
            //     convert_to_output_type<float16>(res,
            //                                     step,
            //                                     m_output_prc,
            //                                     min_val,
            //                                     max_val,
            //                                     out,
            //                                     k,
            //                                     work_amount,
            //                                     uint32_to_float16);
            // } break;
            // case element::Type_t::bf16: {
            //     convert_to_output_type<bfloat16>(res,
            //                                     step,
            //                                     m_output_prc,
            //                                     min_val,
            //                                     max_val,
            //                                     out,
            //                                     k,
            //                                     work_amount,
            //                                     uint32_to_bfloat16);
            // } break;
            case element::Type_t::i32: {
                convert_to_output_type<int>(res,
                                            step,
                                            m_output_prc,
                                            m_min_val.i32,
                                            m_max_val.i32,
                                            out,
                                            k,
                                            work_amount,
                                            nullptr,
                                            nullptr,
                                            [](uint32_t x, int mn, int mx) {
                                                return static_cast<int>(x % (mx - mn) + mn);
                                            });
            } break;
            case element::Type_t::f64: {
                convert_to_output_type<double>(res,
                                            step,
                                            m_output_prc,
                                            m_min_val.f64,
                                            m_max_val.f64,
                                            out,
                                            k,
                                            work_amount,
                                            nullptr,
                                            [](uint32_t a, uint32_t b, double mn, double mx) {
                                                return uint32_to_double(a, b) * (mx - mn) + mn;
                                            });
            } break;
            case element::Type_t::i64: {
                convert_to_output_type<int64_t>(res,
                                            step,
                                            m_output_prc,
                                            m_min_val.i64,
                                            m_max_val.i64,
                                            out,
                                            k,
                                            work_amount,
                                            nullptr,
                                            [](uint32_t a, uint32_t b, int64_t mn, int64_t mx) {
                                                return static_cast<int64_t>(unite_high_low(b, a) % (mx - mn) + mn);
                                            });
            } break;
            default: OPENVINO_THROW("Unsupported type of RandomUniform: ", m_output_prc.to_string());
        }
        if (++n == 0)
            ++counter;
    }

    // Calculate counter values for next RandomUniform run
    const uint64_t skip_count = work_amount * SKIP_CONST;
    n_state += skip_count;
    if (n_state < skip_count)
        counter_state++;

    return {n_state, counter_state};
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
