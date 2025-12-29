// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "executor.hpp"
#include "memory_format_filter.hpp"
#include "nodes/executors/executor_config.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "nodes/executors/implementations.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/printers.hpp"
#include "nodes/executors/variable_executor.hpp"
#include "openvino/core/except.hpp"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu {

template <typename Attrs>
class ExecutorFactory {
public:
    using ExecutorImplementationRef = std::reference_wrapper<const ExecutorImplementation<Attrs>>;

    ExecutorFactory(Attrs attrs,
                    ExecutorContext::CPtr context,
                    const MemoryDescArgs& descriptors,
                    const MemoryFormatFilter& memoryFormatFilter = {},
                    const std::string& implementationPriority = {})
        : m_attrs(std::move(attrs)),
          m_context(std::move(context)),
          m_suitable_implementations(filter(m_attrs, descriptors, memoryFormatFilter, implementationPriority)) {
        // printf("--CPU-- ExecutorFactory ctr 1 type name: %s\n", typeid(*this).name());
        OPENVINO_ASSERT(!m_suitable_implementations.empty(), "No suitable implementations found");
    }

    ExecutorFactory(BinaryInputBuffer& in_buf, const GraphContext::CPtr& graph_context) {
        // printf("--CPU-- ExecutorFactory ctr 2 type name: %s\n", typeid(*this).name());
        in_buf.check_position();  // TODO: Remove
        in_buf(m_context, graph_context);
        // m_context = std::make_shared<ExecutorContext>(in_buf, context);
        load(in_buf);
        OPENVINO_ASSERT(!m_suitable_implementations.empty(), "[CPU] No suitable executor implementation found.");
    }

    // ~ExecutorFactory() = default;

    /**
     * @brief Retrieves the proper memory descriptors based on the provided memory descriptors.
     *
     * Examines the given executor configuration and determines the appropriate
     * memory descriptors to be used.
     *
     * @param descriptors memory descriptors.
     * @return MemoryDescArgs The list of proper memory descriptors based on the configuration.
     * @todo Create proper memory descriptors for all the implementations
     *       to fully enable graph's layout propagation functionality
     */
    [[nodiscard]] std::vector<MemoryDescArgs> getProperMemoryDescriptors(const MemoryDescArgs& descriptors) const {
        DEBUG_LOG("Preconfiguring memory descriptors");

        executor::Config<Attrs> config{descriptors, m_attrs};

        auto getProperMemoryDescArgs = [](const ExecutorImplementationRef& impl,
                                          const executor::Config<Attrs>& config) {
            if (auto optimalConfig = impl.get().createOptimalConfig(config)) {
                return optimalConfig->descs;
            }

            return config.descs;
        };

        std::vector<MemoryDescArgs> memoryDescArgs;
        memoryDescArgs.reserve(m_suitable_implementations.size());
        for (const auto& impl : m_suitable_implementations) {
            memoryDescArgs.emplace_back(getProperMemoryDescArgs(impl, config));
        }

        return memoryDescArgs;
    }

    /**
     * @brief Creates an Executor instance based on the provided memory arguments.
     *
     * Depending on the number of available implementations, returns:
     * - VariableExecutor, if the number of implementations is two or more
     * - Simple Executor, if there is only one available implementation
     *
     * @param memory memory arguments.
     * @param initVariableExecutor whether to init first available implementation of variable executor or not.
     *        This option is mostly a workaround at the moment.
     *        In general it might be beneficial to initialize all the shape dependent implementations
     *        of the variable executor in advance to avoid first-time call delays.
     *
     * @return A shared pointer to the created Executor.
     */
    ExecutorPtr make(const MemoryArgs& memory, bool initVariableExecutor = true) {
        std::vector<ExecutorImplementationRef> implementations;

        auto acceptsConfig = [](const ExecutorImplementationRef& impl, const executor::Config<Attrs>& config) {
            // current config is already considered as the optimal one
            return !impl.get().createOptimalConfig(config).has_value();
        };

        // Filter out implementations that still require changes in configuration
        for (const auto& impl : m_suitable_implementations) {
            auto config = createConfig(memory, m_attrs);

            if (!acceptsConfig(impl, config)) {
                continue;
            }

            implementations.push_back(impl);

            if (impl.get().shapeAgnostic() &&
                impl.get().type() != ExecutorType::Acl) {  // @todo fix acl_eltwise precision mapping)
                break;  // there is no way an implementation with a lower priority will be chosen
            }
        }

        OPENVINO_ASSERT(
            !implementations.empty(),
            "No suitable implementations."
            "This may indicate that the provided memory descriptors are not compatible with any implementation.");

        // only single executor is available
        if (implementations.size() == 1) {
            const auto& theOnlyImplementation = implementations.front().get();
            return theOnlyImplementation.create(m_attrs, memory, m_context);
        }

        return std::make_shared<VariableExecutor<Attrs>>(memory,
                                                         m_attrs,
                                                         m_context,
                                                         implementations,
                                                         initVariableExecutor);
    }

    // DECLARE_SERIALIZATION_OBJECT_MEMBERS(ov::intel_cpu::ExecutorFactory<Attrs>)
    static const std::string& get_type_info_s();

    virtual const std::string& get_type_info() const { return get_type_info_s(); }

    void save(BinaryOutputBuffer& out_buf) const {
        out_buf.dump_position();  // TODO: remove
        out_buf << m_context;
        out_buf.dump_position();  // TODO: remove
        out_buf << m_attrs;
        out_buf << m_suitable_implementations.size();
        out_buf.dump_position();  // TODO: remove
        for (const auto& impl : m_suitable_implementations) {
           out_buf << std::string(impl.get().name());  // TODO: replace to char*
        }
        out_buf.dump_position();  // TODO: remove
    }

    void load(BinaryInputBuffer& in_buf) {
        size_t impl_size = 0UL;
        //const char* impl_name;
        std::string impl_name;

        in_buf.check_position();  // TODO: Remove
        in_buf >> m_attrs;
        in_buf >> impl_size;
        in_buf.check_position();  // TODO: Remove

        const auto& implementations = getImplementations<Attrs>();
        for (size_t i = 0UL; i < impl_size; i++) {
            in_buf >> impl_name;
            for (const auto& impl : implementations) {
                // if (strcmp(impl_name, impl.name())) {
                if (impl_name.compare(impl.name()) == 0) {
                    m_suitable_implementations.push_back(std::ref(impl));
                    break;
                }
            }
        }
        in_buf.check_position();  // TODO: Remove
    }

    void set_attr(Attrs attrs) {
        m_attrs = attrs;
    }

private:
    /**
     * @brief Filters and retrieves suitable implementations based on the provided executor configuration.
     *
     * @param attrs The attributes used for filtering implementations.
     * @param descs The memory descriptor arguments.
     * @param implementationPriority Optional. The name of the implementation to prioritize.
     *        If specified, only the implementation with this name will be considered.
     *
     * @note If an implementation is shape agnostic, no further implementations with lower
     *       priority are considered.
     */
    static std::vector<ExecutorImplementationRef> filter(const Attrs& attrs,
                                                         const MemoryDescArgs& descs,
                                                         const MemoryFormatFilter& memoryFormatFilter = {},
                                                         const std::string& implementationPriority = {}) {
        const auto& implementations = getImplementations<Attrs>();
        std::vector<ExecutorImplementationRef> suitableImplementations;
        const executor::Config<Attrs> config{descs, attrs};

        for (const auto& implementation : implementations) {
            DEBUG_LOG("Processing implementation: ", implementation.name());
            if (!implementationPriority.empty() && implementation.name() != implementationPriority) {
                DEBUG_LOG("Implementation: ",
                          implementation.name(),
                          " does not match priority: ",
                          implementationPriority);
                continue;
            }

            if (!implementation.supports(config, memoryFormatFilter)) {
                DEBUG_LOG("Implementation is NOT supported: ", implementation.name());
                continue;
            }

            DEBUG_LOG("Implementation is supported: ", implementation.name());
            suitableImplementations.push_back(std::ref(implementation));
        }

        const bool hasShapeAgnosticNonReferenceImplementation =
            std::any_of(suitableImplementations.begin(),
                        suitableImplementations.end(),
                        [](const ExecutorImplementationRef& impl) {
                            return impl.get().type() != ExecutorType::Reference &&
                                   impl.get().type() != ExecutorType::Acl &&  // @todo fix acl_eltwise precision mapping
                                   impl.get().shapeAgnostic();
                        });

        // Consider reference implementation only if there is nothing else available
        if (hasShapeAgnosticNonReferenceImplementation) {
            suitableImplementations.erase(std::remove_if(suitableImplementations.begin(),
                                                         suitableImplementations.end(),
                                                         [](const ExecutorImplementationRef& impl) {
                                                             return impl.get().type() == ExecutorType::Reference;
                                                         }),
                                          suitableImplementations.end());
        }

        return suitableImplementations;
    }

    Attrs m_attrs;
    ExecutorContext::CPtr m_context;
    std::vector<ExecutorImplementationRef> m_suitable_implementations;
};

template <typename Attrs>
using ExecutorFactoryPtr = std::shared_ptr<ExecutorFactory<Attrs>>;

template <typename Attrs>
using ExecutorFactoryCPtr = std::shared_ptr<const ExecutorFactory<Attrs>>;

}  // namespace ov::intel_cpu
