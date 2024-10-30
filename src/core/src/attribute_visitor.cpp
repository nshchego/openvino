// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/attribute_visitor.hpp"

#include "openvino/core/attribute_adapter.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"


void ov::AttributeVisitor::start_structure(const std::string& name) {
    m_context.push_back(name);
}

void ov::AttributeVisitor::start_structure(const char* name) {
    m_context.push_back(name);
}

std::string ov::AttributeVisitor::finish_structure() {
    std::string result = m_context.back();
    m_context.pop_back();
    return result;
}

std::string ov::AttributeVisitor::get_name_with_context() {
    //std::stringstream result;
    //result.iword(512);
    //static const char sep = '.';
    //for (const auto& c : m_context) {
    //    result << c << sep;
    //}
    //auto strt = result.str();
    //strt.pop_back();
    //return strt;


    //std::cout << "get_name_with_context: \"" << strt << "\" tellp: " << result.tellp() <<
    //    std::endl;
    //const std::streamsize size = static_cast<std::streamsize>(result.tellp());
    //std::string res;
    //res.reserve(size + 5);
    //char tmp[5];
    //char* tmp = &(res[0]);
    //result.get(tmp, size);
    //result.read(tmp, size);
    //result.get(&(res[0]), size - 1);
    //result.get(const_cast<char*>(res.data()), size - 1l);
    //std::cout << "get_name_with_context: \"" << strt << "\" size: " << strt.size() << std::endl;
    //printf("get_name_with_context: \"%s\"; tellp: \"%lu\"\n", result.str().data(), result.tellp());
    //return strt;


    //std::ostringstream result;
    //std::string sep = "";
    //for (const auto& c : m_context) {
    //    result << sep << c;
    //    sep = ".";
    //}
    //return result.str();


    std::string result;
    //std::cout << "result capacity: " << result.capacity() << std::endl;
    //result.reserve(64);
    static const char sep = '.';
    for (const auto& c : m_context) {
        result.append(c).push_back(sep);
    }
    result.pop_back();
    //if (result.size() >= 128) {
        //std::cout << "result: \"" << result << "\"; size: " << result.size() << std::endl;
    //}

    return result;
}

void ov::AttributeVisitor::on_adapter(const std::string& name, VisitorAdapter& adapter) {
    adapter.visit_attributes(*this);
}

void ov::AttributeVisitor::on_adapter(const std::string& name, ValueAccessor<void*>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const std::string& name, ValueAccessor<std::string>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
};

void ov::AttributeVisitor::on_adapter(const std::string& name, ValueAccessor<bool>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
};

void ov::AttributeVisitor::on_adapter(const std::string& name, ValueAccessor<int8_t>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const std::string& name, ValueAccessor<int16_t>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const std::string& name, ValueAccessor<int32_t>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const std::string& name, ValueAccessor<int64_t>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const std::string& name, ValueAccessor<uint8_t>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const std::string& name, ValueAccessor<uint16_t>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const std::string& name, ValueAccessor<uint32_t>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const std::string& name, ValueAccessor<uint64_t>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const std::string& name, ValueAccessor<float>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const std::string& name, ValueAccessor<double>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const std::string& name, ValueAccessor<std::vector<int8_t>>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const std::string& name, ValueAccessor<std::vector<int16_t>>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const std::string& name, ValueAccessor<std::vector<int32_t>>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const std::string& name, ValueAccessor<std::vector<int64_t>>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const std::string& name, ValueAccessor<std::vector<uint8_t>>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const std::string& name, ValueAccessor<std::vector<uint16_t>>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const std::string& name, ValueAccessor<std::vector<uint32_t>>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const std::string& name, ValueAccessor<std::vector<uint64_t>>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const std::string& name, ValueAccessor<std::vector<float>>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const std::string& name, ValueAccessor<std::vector<double>>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const std::string& name, ValueAccessor<std::vector<std::string>>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

void ov::AttributeVisitor::on_adapter(const std::string& name, ValueAccessor<std::shared_ptr<ov::Model>>& adapter) {
    on_adapter(name, static_cast<ValueAccessor<void>&>(adapter));
}

constexpr const char* ov::AttributeVisitor::invalid_node_id;

void ov::AttributeVisitor::register_node(const std::shared_ptr<ov::Node>& node, node_id_t id) {
    if (id == invalid_node_id) {
        id = node->get_friendly_name();
    }
    m_id_node_map[id] = node;
    m_node_id_map[node] = std::move(id);
}

std::shared_ptr<ov::Node> ov::AttributeVisitor::get_registered_node(node_id_t id) {
    auto it = m_id_node_map.find(id);
    return it == m_id_node_map.end() ? std::shared_ptr<ov::Node>() : it->second;
}

ov::AttributeVisitor::node_id_t ov::AttributeVisitor::get_registered_node_id(const std::shared_ptr<ov::Node>& node) {
    auto it = m_node_id_map.find(node);
    return it == m_node_id_map.end() ? invalid_node_id : it->second;
}
