// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/op_types.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/logical_xor.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/squared_difference.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/op/util/binary_elementwise_comparison.hpp"
#include "openvino/op/util/binary_elementwise_logical.hpp"
#include "openvino/op/util/unary_elementwise_arithmetic.hpp"
#include "openvino/op/xor.hpp"

bool ov::op::util::is_unary_elementwise_arithmetic(const ov::Node* node) {
    return ov::is_type<ov::op::util::UnaryElementwiseArithmetic>(node);
}

bool ov::op::util::is_binary_elementwise_arithmetic(const ov::Node* node) {
    return ov::is_type<ov::op::util::BinaryElementwiseArithmetic>(node);
}

bool ov::op::util::is_binary_elementwise_comparison(const ov::Node* node) {
    return ov::is_type<ov::op::util::BinaryElementwiseComparison>(node);
}

bool ov::op::util::is_binary_elementwise_logical(const ov::Node* node) {
    return ov::is_type<ov::op::util::BinaryElementwiseLogical>(node);
}

bool ov::op::util::supports_auto_broadcast(const ov::Node* node) {
    return ov::is_type<ov::op::v1::Select>(node) ||
           ov::is_type<ov::op::v0::SquaredDifference>(node) ||
           ov::is_type<ov::op::util::BinaryElementwiseComparison>(node) ||
           ov::is_type<ov::op::util::BinaryElementwiseLogical>(node) ||
           ov::is_type<ov::op::util::BinaryElementwiseArithmetic>(node);
}

bool ov::op::util::is_op(const ov::Node* node) {
    return ov::is_type<ov::op::Op>(node);
}

bool ov::op::util::is_parameter(const ov::Node* node) {
    return ov::is_type<ov::op::v0::Parameter>(node);
}

bool ov::op::util::is_output(const ov::Node* node) {
    return ov::is_type<ov::op::v0::Result>(node);
}

bool ov::op::util::is_sink(const ov::Node* node) {
    return ov::is_type<ov::op::Sink>(node);
}

bool ov::op::util::is_constant(const ov::Node* node) {
    return ov::is_type<ov::op::v0::Constant>(node);
}

bool ov::op::util::is_commutative(const ov::Node* node) {
    return ov::is_type<ov::op::v1::Add>(node) ||
           ov::is_type<ov::op::v1::Equal>(node) ||
           ov::is_type<ov::op::v1::LogicalAnd>(node) ||
           ov::is_type<ov::op::v1::LogicalOr>(node) ||
           ov::is_type<ov::op::v1::LogicalXor>(node) ||
           ov::is_type<ov::op::v1::Maximum>(node) ||
           ov::is_type<ov::op::v1::Minimum>(node) ||
           ov::is_type<ov::op::v1::Multiply>(node) ||
           ov::is_type<ov::op::v1::NotEqual>(node) ||
           ov::is_type<ov::op::v0::Xor>(node);
}

bool ov::op::util::is_unary_elementwise_arithmetic(const std::shared_ptr<ov::Node>& node) {
    return is_unary_elementwise_arithmetic(node.get());
}
bool ov::op::util::is_binary_elementwise_arithmetic(const std::shared_ptr<ov::Node>& node) {
    return is_binary_elementwise_arithmetic(node.get());
}
bool ov::op::util::is_binary_elementwise_comparison(const std::shared_ptr<ov::Node>& node) {
    return is_binary_elementwise_comparison(node.get());
}
bool ov::op::util::is_binary_elementwise_logical(const std::shared_ptr<ov::Node>& node) {
    return is_binary_elementwise_logical(node.get());
}

bool ov::op::util::supports_auto_broadcast(const std::shared_ptr<ov::Node>& node) {
    return supports_auto_broadcast(node.get());
}

bool ov::op::util::is_op(const std::shared_ptr<ov::Node>& node) {
    return is_op(node.get());
}
bool ov::op::util::is_parameter(const std::shared_ptr<ov::Node>& node) {
    return is_parameter(node.get());
}
bool ov::op::util::is_output(const std::shared_ptr<ov::Node>& node) {
    return is_output(node.get());
}
bool ov::op::util::is_sink(const std::shared_ptr<ov::Node>& node) {
    return is_sink(node.get());
}
bool ov::op::util::is_constant(const std::shared_ptr<ov::Node>& node) {
    return is_constant(node.get());
}
bool ov::op::util::is_commutative(const std::shared_ptr<ov::Node>& node) {
    return is_commutative(node.get());
}
