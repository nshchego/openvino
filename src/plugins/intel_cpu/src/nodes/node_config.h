// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "memory_desc/blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "openvino/core/except.hpp"
#include "utils/serialization/bind.hpp"

namespace ov::intel_cpu {

class PortDescBase {
public:
    virtual ~PortDescBase() = default;

    /**
     * @brief Check if the port desc can be accepted.
     *
     * @warning This operation is not commutative desc1.isCompatible(desc2) != desc2.isCompatible(desc1) in general case
     *
     * @return True if the port desc may be accepted false otherwise
     */
    [[nodiscard]] bool isCompatible(const PortDescBase& rhs) const {
        return typeid(*this) == typeid(rhs) && this->compareImpl(rhs);
    }
    [[nodiscard]] virtual MemoryDescPtr getMemDesc() const = 0;

    virtual const std::string& get_type_info() const = 0;

    virtual void save(BinaryOutputBuffer& ob) const = 0;

    virtual void load(BinaryInputBuffer& in_buf) = 0;

protected:
    [[nodiscard]] virtual bool compareImpl(const PortDescBase& rhs) const = 0;
};

using PortDescBasePtr = std::shared_ptr<PortDescBase>;
using PortDescBaseCPtr = std::shared_ptr<const PortDescBase>;

template <class T>
class PortDescBase_ : public PortDescBase {
protected:
    PortDescBase_() = default;
    [[nodiscard]] bool compareImpl(const PortDescBase& rhs) const override /*final*/ {
        return static_cast<const T&>(*this).isCompatible(static_cast<const T&>(rhs));
    }
};

class PortDescGeneric : public PortDescBase_<PortDescGeneric> {
public:
    PortDescGeneric() = default;

    explicit PortDescGeneric(MemoryDescPtr mem_desc) : m_mem_desc(std::move(mem_desc)) {
        OPENVINO_ASSERT(m_mem_desc, "ParameterMismatch: PortDescGeneric constructor got nullptr");
        //if (auto dn = std::dynamic_pointer_cast<DnnlMemoryDesc>(m_mem_desc)) {
        //    printf("DnnlMemoryDesc passed\n");
        //}
        //if (auto dn = std::dynamic_pointer_cast<DnnlBlockedMemoryDesc>(m_mem_desc)) {
        //    printf("DnnlBlockedMemoryDesc passed\n");
        //}
    }
    [[nodiscard]] bool isCompatible(const PortDescGeneric& rhs) const {
        return m_mem_desc->isCompatible(*rhs.m_mem_desc);
    }
    [[nodiscard]] MemoryDescPtr getMemDesc() const override {
        return m_mem_desc;
    }

    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_cpu::PortDescGeneric)

    void save(BinaryOutputBuffer& ob) const override;

    void load(BinaryInputBuffer& in_buf) override;

private:
    MemoryDescPtr m_mem_desc;
};

class PortDescBlocked : public PortDescBase_<PortDescBlocked> {
public:
    using CmpMask = BlockedMemoryDesc::CmpMask;

    PortDescBlocked() = default;

    PortDescBlocked(BlockedMemoryDescPtr mem_desc, CmpMask cmpMask) : m_mem_desc(std::move(mem_desc)), m_cmp_mask(cmpMask) {
        OPENVINO_ASSERT(m_mem_desc, "ParameterMismatch: PortDescBlocked constructor got nullptr");
        //if (auto dn = std::dynamic_pointer_cast<DnnlMemoryDesc>(m_mem_desc)) {
        //    printf("DnnlMemoryDesc passed\n");
        //}
        //if (auto dn = std::dynamic_pointer_cast<DnnlBlockedMemoryDesc>(m_mem_desc)) {
        //    printf("DnnlBlockedMemoryDesc passed\n");
        //}
    }
    [[nodiscard]] bool isCompatible(const PortDescBlocked& rhs) const {
        return m_mem_desc->isCompatible(*rhs.m_mem_desc, m_cmp_mask) && (((~m_cmp_mask) | rhs.m_cmp_mask).all());
    }
    [[nodiscard]] MemoryDescPtr getMemDesc() const override {
        return m_mem_desc;
    }

    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_cpu::PortDescBlocked)

    void save(BinaryOutputBuffer& ob) const override;

    void load(BinaryInputBuffer& in_buf) override;

private:
    BlockedMemoryDescPtr m_mem_desc;
    CmpMask m_cmp_mask = BlockedMemoryDesc::FULL_MASK;
};

class PortConfig {
public:
    PortConfig() = default;

    explicit PortConfig(const MemoryDescPtr& desc,
                        BlockedMemoryDesc::CmpMask cmpMask = BlockedMemoryDesc::FULL_MASK,
                        int inPlacePort = -1,
                        bool isConstant = false)
        : m_port_desc(createPortDesc(desc, cmpMask)),
          m_in_place_port(inPlacePort),
          m_constant(isConstant) {}

    // prevent implicit convertion of cmpMask
    PortConfig(MemoryDescPtr desc, int cmpMask, int inPlacePort = -1, bool isConstant = false) = delete;

    PortConfig(const PortConfig& rhs) = default;

    PortConfig& operator=(const PortConfig& rhs) = default;

    PortConfig(PortConfig&& rhs) = default;
    PortConfig& operator=(PortConfig&& rhs) = default;

    [[nodiscard]] int inPlace() const {
        return m_in_place_port;
    }

    void inPlace(int port) {
        m_in_place_port = port;
    }

    [[nodiscard]] bool constant() const {
        return m_constant;
    }

    void constant(bool constant) {
        m_constant = constant;
    }

    [[nodiscard]] MemoryDescPtr getMemDesc() const {
        return m_port_desc->getMemDesc();
    }

    [[nodiscard]] PortDescBasePtr getPortDesc() const {
        return m_port_desc;
    }

    void setMemDesc(const MemoryDescPtr& desc) {
        m_port_desc = createPortDesc(desc, BlockedMemoryDesc::FULL_MASK);
    }

    void setMemDesc(const BlockedMemoryDescPtr& desc, BlockedMemoryDesc::CmpMask cmpMask) {
        m_port_desc = createPortDesc(desc, cmpMask);
    }

    [[nodiscard]] bool hasZeroDims() const {
        const auto desc = getMemDesc();
        return desc->getShape().hasZeroDims() && !desc->empty();
    }

    void save(BinaryOutputBuffer& ob) const;

    void load(BinaryInputBuffer& in_buf);

private:
    static PortDescBasePtr createPortDesc(const MemoryDescPtr& desc, BlockedMemoryDesc::CmpMask cmpMask) {
        if (desc->getType() & MemoryDescType::Blocked) {
            return createPortDesc(std::dynamic_pointer_cast<BlockedMemoryDesc>(desc), cmpMask);
        }

        return std::make_shared<PortDescGeneric>(desc);
    }

    static PortDescBasePtr createPortDesc(const BlockedMemoryDescPtr& desc, BlockedMemoryDesc::CmpMask cmpMask) {
        return std::make_shared<PortDescBlocked>(desc, cmpMask);
    }

    PortDescBasePtr m_port_desc;
    int m_in_place_port = -1;
    bool m_constant = false;
};

struct NodeConfig {
    NodeConfig() = default;

    NodeConfig(std::vector<PortConfig> inConfs, std::vector<PortConfig> outConfs)
        : inConfs(std::move(inConfs)),
          outConfs(std::move(outConfs)) {}

    void save(BinaryOutputBuffer& ob) const;

    void load(BinaryInputBuffer& in_buf);

    std::vector<PortConfig> inConfs;
    std::vector<PortConfig> outConfs;
};

}  // namespace ov::intel_cpu
