// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cache/multi_cache.h"
#include "cpu_memory.h"
#include "dnnl_scratch_pad.h"
#include "graph_context.h"
#include "memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/visibility.hpp"
#include "weights_cache.hpp"
#include "utils/serialization/internal_types.hpp"
#include "utils/serialization/vector_serializer.hpp"

namespace ov::intel_cpu {

// @todo another option is to determine shape relation by executor type
enum class ShapeTolerance : uint8_t { Agnostic, Dependant };

enum class ExecutorType : uint8_t {
    Undefined,
    Reference,
    Graph,
    Common,
    Jit,
    Dnnl,
    Acl,
    Mlas,
    Shl,
    Kleidiai,
};

enum class OperationType : uint8_t { FullyConnected, MatMul, Convolution, Eltwise };

std::string ExecutorTypeToString(ExecutorType type);
ExecutorType ExecutorTypeFromString(const std::string& typeStr);

class ExecutorContext {
public:
    using Ptr = std::shared_ptr<ExecutorContext>;
    using CPtr = std::shared_ptr<const ExecutorContext>;

    ExecutorContext(const GraphContext::CPtr& graphContext,
                    std::vector<impl_desc_type> impl_priorities,
                    std::shared_ptr<std::unordered_map<std::string, MemoryPtr>> private_weight_cache = nullptr)
        : m_runtime_cache(graphContext->getParamsCache()),
          m_scratch_pads(graphContext->getScratchPads()),
          m_weights_cache(graphContext->getWeightsCache()),
          m_engine(graphContext->getEngine()),
          m_impl_priorities(std::move(impl_priorities)),
          m_private_weight_cache(std::move(private_weight_cache)),
          m_num_numa_nodes(graphContext->getNumNumaNodes()) {
        auto cpuStreamsExecutor = graphContext->getCPUStreamExecutor();
        m_cur_numa_node_id = std::max(0, cpuStreamsExecutor ? cpuStreamsExecutor->get_numa_node_id() : m_cur_numa_node_id);
    }

    ExecutorContext(BinaryInputBuffer& in_buf, const GraphContext::CPtr& context)
        : m_runtime_cache(context->getParamsCache()),
          m_scratch_pads(context->getScratchPads()),
          m_weights_cache(context->getWeightsCache()),
          m_engine(context->getEngine()) {
        load(in_buf);
    }

    [[nodiscard]] MultiCachePtr getRuntimeCache() const {
        auto runtimeCachePtr = m_runtime_cache.lock();
        assert(runtimeCachePtr);
        return runtimeCachePtr;
    }

    [[nodiscard]] DnnlScratchPadPtr getScratchPad() const {
        return m_scratch_pads[m_cur_numa_node_id];
    }

    [[nodiscard]] std::shared_ptr<std::unordered_map<std::string, MemoryPtr>> getPrivateWeightCache() const {
        return m_private_weight_cache;
    }

    [[nodiscard]] const dnnl::engine& getEngine() const {
        return m_engine;
    }

    [[nodiscard]] const std::vector<impl_desc_type>& getImplPriorities() const {
        return m_impl_priorities;
    }

    [[nodiscard]] WeightsSharing::Ptr getWeightsCache() const {
        return m_weights_cache;
    }

    DECLARE_SERIALIZATION_OBJECT_MEMBERS(ov::intel_cpu::ExecutorContext)

    void save(BinaryOutputBuffer& out_buf) const;

    void load(BinaryInputBuffer& in_buf);

private:
    // weak_ptr is required to avoid cycle dependencies with MultiCache
    // since ExecutorContext is stored in Executor itself
    MultiCacheWeakPtr m_runtime_cache;
    std::vector<DnnlScratchPadPtr> m_scratch_pads;
    WeightsSharing::Ptr m_weights_cache;
    const dnnl::engine& m_engine;
    std::vector<impl_desc_type> m_impl_priorities;
    // @todo remove after global cache is used exclusevly
    std::shared_ptr<std::unordered_map<std::string, MemoryPtr>> m_private_weight_cache;
    int m_num_numa_nodes;
    int m_cur_numa_node_id = -1;
};

class ExecutorFactoryLegacy {
public:
    explicit ExecutorFactoryLegacy(ExecutorContext::CPtr context) : context(std::move(context)) {}
    virtual ~ExecutorFactoryLegacy() = default;

    const ExecutorContext::CPtr context;
};

using ExecutorFactoryLegacyPtr = std::shared_ptr<ExecutorFactoryLegacy>;
using ExecutorFactoryLegacyCPtr = std::shared_ptr<const ExecutorFactoryLegacy>;

class Executor {
public:
    // returns false if the stage has failed and the executor must be rejected
    virtual bool update([[maybe_unused]] const MemoryArgs& memory) {
        OPENVINO_THROW_NOT_IMPLEMENTED("This version of the 'update' method is not implemented by executor");
        return false;
    }

    virtual void execute() const {
        OPENVINO_THROW_NOT_IMPLEMENTED("This version of the 'execute' method is not implemented by executor");
    }

    virtual void execute() {
        OPENVINO_THROW_NOT_IMPLEMENTED("This version of the 'execute' method is not implemented by executor");
    }
    // dnnl_fullyconnected 3D workaround version
    virtual void execute([[maybe_unused]] const MemoryArgs& memory) {
        OPENVINO_THROW_NOT_IMPLEMENTED("This version of the 'execute' method is not implemented by executor");
    }
    // legacy version
    virtual void exec([[maybe_unused]] const std::vector<MemoryCPtr>& src,
                      [[maybe_unused]] const std::vector<MemoryPtr>& dst) {
        OPENVINO_THROW_NOT_IMPLEMENTED("This version of the 'execute' method is not implemented by executor");
    }
    [[nodiscard]] virtual impl_desc_type implType() const = 0;
    virtual void moveMemToNumaNode([[maybe_unused]] int numaID) {
        OPENVINO_THROW_NOT_IMPLEMENTED("This version of the 'moveMemToNumaNode' method is not implemented by executor");
    }
    virtual ~Executor() = default;
};

using ExecutorPtr = std::shared_ptr<Executor>;

}  // namespace ov::intel_cpu
