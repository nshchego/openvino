// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes_factory.hpp"
#include "nodes/reference.h"

namespace ov::intel_cpu {

template <>
Node* NodesFactory<const std::shared_ptr<ov::Node>&>::create(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context) {
printf("--CPU-- NodesFactory::create '%s':'%s'\n", op->get_type_name(), op->get_friendly_name().data());
    Node* newNode = nullptr;
    std::string errorMessage;
    if (newNode == nullptr) {
        try {
printf("    Try to create CPU node\n");
            std::unique_ptr<Node> ol(createNodeIfRegistered(intel_cpu, TypeFromName(op->get_type_name()), op, context));
            if (ol != nullptr && ol->created()) {
                newNode = ol.release();
            }
        } catch (const ov::Exception& ex) {
            if (dynamic_cast<const ov::NotImplemented*>(&ex) != nullptr) {
                errorMessage += ex.what();
            } else {
                throw;
            }
        }
    }

    if (newNode == nullptr) {
        try {
printf("    Try to create Reference node\n");
            std::unique_ptr<Node> ol(new node::Reference(op, context, errorMessage));
            if (ol != nullptr && ol->created()) {
                newNode = ol.release();
            }
        } catch (const ov::Exception& ex) {
            if (dynamic_cast<const ov::NotImplemented*>(&ex) != nullptr) {
                const std::string currErrorMess = ex.what();
                if (!currErrorMess.empty()) {
                    errorMessage += errorMessage.empty() ? currErrorMess : "\n" + currErrorMess;
                }
            } else {
                throw;
            }
        }
    }

    if (!newNode) {
        std::string errorDetails;
        if (!errorMessage.empty()) {
            errorDetails = "\nDetails:\n" + errorMessage;
        }
        OPENVINO_THROW("Unsupported operation of type: ",
                       op->get_type_name(),
                       " name: ",
                       op->get_friendly_name(),
                       errorDetails);
    }

    return newNode;
}

template <>
Node* NodesFactory<BinaryInputBuffer&>::create(BinaryInputBuffer& ib, const GraphContext::CPtr& context) {
    Node* newNode = nullptr;
    std::string errorMessage;
    Type node_type;
    ib >> make_data(&node_type, sizeof(Type));

    if (newNode == nullptr) {
        try {
            std::unique_ptr<Node> ol(createNodeIfRegistered(intel_cpu, node_type, ib, context));
            if (ol != nullptr && ol->created()) {
                newNode = ol.release();
            }
        } catch (const ov::Exception& ex) {
            if (dynamic_cast<const ov::NotImplemented*>(&ex) != nullptr) {
                errorMessage += ex.what();
            } else {
                throw;
            }
        }
    }

    // if (newNode == nullptr) {
    //     try {
    //         std::unique_ptr<Node> ol(new Reference(op, context, errorMessage));
    //         if (ol != nullptr && ol->created()) {
    //             newNode = ol.release();
    //         }
    //     } catch (const ov::Exception& ex) {
    //         if (dynamic_cast<const ov::NotImplemented*>(&ex) != nullptr) {
    //             const std::string currErrorMess = ex.what();
    //             if (!currErrorMess.empty()) {
    //                 errorMessage += errorMessage.empty() ? currErrorMess : "\n" + currErrorMess;
    //             }
    //         } else {
    //             throw;
    //         }
    //     }
    // }

    if (!newNode) {
        std::string errorDetails;
        if (!errorMessage.empty()) {
            errorDetails = "\nDetails:\n" + errorMessage;
        }
        OPENVINO_THROW("Unsupported operation of type: ",
                       NameFromType(node_type),
                       " name: ",
                    //    op->get_friendly_name(),
                       errorDetails);
    }

    return newNode;
}

}  // namespace ov::intel_cpu
