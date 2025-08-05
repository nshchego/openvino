// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/core/extension.hpp"
#include "openvino/core/visibility.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

namespace ov {

class OPENVINO_API IOpPostProcessExtension {
public:
    virtual void update_node(ov::Node& node, ov::AttributeVisitor& fallback_visitor) const = 0;
};
}  // namespace ov
