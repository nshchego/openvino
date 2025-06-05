// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes_factory.hpp"

#include "nodes/reference.h"
#include "utils/serialization/internal_types.hpp"

namespace ov::intel_cpu {

template <>
Node* NodesFactory<const std::shared_ptr<ov::Node>&>::create(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context) {
printf("--CPU-- NodesFactory::create '%s':'%s'\n", op->get_type_name(), op->get_friendly_name().data());
    Node* new_node = nullptr;
    std::string error_message;

    try {
printf("--CPU-- FACTORY ov::Node. Try to create CPU node %s\n", op->get_type_name());
        std::unique_ptr<Node> ol(createNodeIfRegistered(intel_cpu, TypeFromName(op->get_type_name()), op, context));
        if (ol != nullptr && ol->created()) {
            new_node = ol.release();
        }
    } catch (const ov::Exception& ex) {
        if (dynamic_cast<const ov::NotImplemented*>(&ex) != nullptr) {
            error_message += ex.what();
        } else {
            throw;
        }
    }

    if (new_node == nullptr) {
        try {
printf("    Try to create Reference node\n");
            std::unique_ptr<Node> ol(new node::Reference(op, context, error_message));
            if (ol != nullptr && ol->created()) {
                new_node = ol.release();
            }
        } catch (const ov::Exception& ex) {
            if (dynamic_cast<const ov::NotImplemented*>(&ex) != nullptr) {
                const std::string curr_error_mess = ex.what();
                if (!curr_error_mess.empty()) {
                    error_message += error_message.empty() ? curr_error_mess : "\n" + curr_error_mess;
                }
            } else {
                throw;
            }
        }
    }

    if (!new_node) {
        std::string error_details;
        if (!error_message.empty()) {
            error_details = "\nDetails:\n" + error_message;
        }
        OPENVINO_THROW("Unsupported operation of type: ",
                       op->get_type_name(),
                       " name: ",
                       op->get_friendly_name(),
                       error_details);
    }

    return new_node;
}

template <>
Node* NodesFactory<BinaryInputBuffer&>::create(BinaryInputBuffer& ib, const GraphContext::CPtr& context) {
    Node* new_node = nullptr;
    std::string error_message;
    intel_cpu::Type node_type;
    ib >> node_type;

    try {
printf("--CPU-- FACTORY Stream. Try to deserialize CPU node %d\n", int(node_type));
        std::unique_ptr<Node> ol(createNodeIfRegistered(intel_cpu, node_type, ib, context));
        if (ol != nullptr && ol->created()) {
            new_node = ol.release();
        }
    } catch (const ov::Exception& ex) {
        if (dynamic_cast<const ov::NotImplemented*>(&ex) != nullptr) {
            error_message += ex.what();
        } else {
            throw;
        }
    }

    if (new_node == nullptr && node_type == Type::Reference) {
        try {
            std::unique_ptr<Node> ol(new node::Reference(ib, context, error_message));
            if (ol != nullptr && ol->created()) {
                new_node = ol.release();
            }
        } catch (const ov::Exception& ex) {
            if (dynamic_cast<const ov::NotImplemented*>(&ex) != nullptr) {
                const std::string curr_error_mess = ex.what();
                if (!curr_error_mess.empty()) {
                    error_message += error_message.empty() ? curr_error_mess : "\n" + curr_error_mess;
                }
            } else {
                throw;
            }
        }
    }

    if (!new_node) {
        std::string error_details;
        if (!error_message.empty()) {
            error_details = "\nDetails:\n" + error_message;
        }
        OPENVINO_THROW("Unsupported operation of type: ",
                       NameFromType(node_type),
                       error_details);
    }

    return new_node;
}

}  // namespace ov::intel_cpu
