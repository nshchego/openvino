// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "node.h"

namespace ov::intel_cpu {

template <typename SrcType>
class NodesFactory : public openvino::cc::Factory<Type, Node*(SrcType src, const GraphContext::CPtr&)> {
public:
    NodesFactory();

    Node* create(SrcType ib, const GraphContext::CPtr& context);

    static NodesFactory& factory() {
        static NodesFactory factory_instance;
        return factory_instance;
    }
};

}  // namespace ov::intel_cpu
