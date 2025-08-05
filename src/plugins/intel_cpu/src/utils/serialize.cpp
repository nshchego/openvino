// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/serialize.hpp"

#include <utility>

#include "openvino/core/constant_writer.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/core/xml_serialize_util.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/mmap_object.hpp"
#include "serialize.hpp"

namespace ov::intel_cpu {
class WeightlessWriter : public ov::util::ConstantWriter {
public:
    explicit WeightlessWriter(ov::util::ConstantWriter& other) : ov::util::ConstantWriter(other), m_offset{} {}
    WeightlessWriter(std::ostream& bin_file) : ov::util::ConstantWriter(bin_file), m_offset{} {}

    WeightlessWriter::FilePosition write([[maybe_unused]] const char* ptr,
                                         size_t size,
                                         size_t& new_size,
                                         [[maybe_unused]] bool compress_to_fp16,
                                         [[maybe_unused]] ov::element::Type src_type,
                                         [[maybe_unused]] bool ptr_is_temporary) override {
        // new_size = size; // TODO will set size to 0, no data in weights (CPU specific)
        auto offset = m_offset;
        m_offset += size;
        return offset;
    }

private:
    WeightlessWriter::FilePosition m_offset;
};

class XmlSerializer : public ov::util::XmlSerializer {
public:
    XmlSerializer(pugi::xml_node& data,
                  const std::string& node_type_name,
                  ov::util::ConstantWriter& constant_write_handler,
                  int64_t version,
                  bool deterministic = false,
                  bool compress_to_fp16 = false,
                  ov::element::Type output_element_type = ov::element::dynamic,
                  bool data_is_temporary = false,
                  bool wl_mode = false)
        : ov::util::XmlSerializer(data,
                                  node_type_name,
                                  constant_write_handler,
                                  version,
                                  deterministic,
                                  compress_to_fp16,
                                  output_element_type,
                                  data_is_temporary),
          m_wl_const_writer(constant_write_handler),
          m_use_weightless_writer(false),
          m_wl_mode(wl_mode) {}

private:
    bool append_rt_attribute(pugi::xml_node& node, const ov::RuntimeAttribute& attribute) override {
        if (auto wl_attr = ov::as_type<const ov::WeightlessCacheAttribute>(&attribute)) {
            const auto& type_info = attribute.get_type_info();
            node.append_attribute("name").set_value(type_info.name);
            node.append_attribute("version").set_value(type_info.get_version().c_str());
            node.append_attribute("type").set_value(ov::util::get_ir_precision_name(wl_attr->original_dtype));
            node.append_attribute("offset").set_value(wl_attr->bin_offset);
            node.append_attribute("size").set_value(wl_attr->original_size);
            return true;
        } else {
            return ov::util::XmlSerializer::append_rt_attribute(node, attribute);
        }
    }

    bool append_node_attributes(ov::Node& node) override {
        m_use_weightless_writer =
            m_wl_mode && node.get_rt_info().count(ov::WeightlessCacheAttribute::get_type_info_static()) != 0;
        auto result = ov::util::XmlSerializer::append_node_attributes(node);
        m_use_weightless_writer = false;
        return result;
    }

    ov::util::ConstantWriter& get_constant_write_handler() override {
        return m_wl_mode && m_use_weightless_writer ? m_wl_const_writer
                                                    : ov::util::XmlSerializer::get_constant_write_handler();
    }

    std::unique_ptr<ov::util::XmlSerializer> make_visitor(pugi::xml_node& data,
                                                          const std::string& node_type_name,
                                                          ov::util::ConstantWriter& constant_write_handler,
                                                          int64_t version,
                                                          bool deterministic,
                                                          bool compress_to_fp16,
                                                          ov::element::Type output_element_type,
                                                          bool data_is_temporary) const override {
        return std::make_unique<XmlSerializer>(data,
                                               node_type_name,
                                               constant_write_handler,
                                               version,
                                               deterministic,
                                               compress_to_fp16,
                                               output_element_type,
                                               data_is_temporary);
    }

    WeightlessWriter m_wl_const_writer;
    bool m_use_weightless_writer;  // Flag to indicate if we are using a weightless writer
    bool m_wl_mode;
};

class StreamSerialize : public ov::pass::StreamSerialize {
public:
    StreamSerialize(std::ostream& stream,
                    std::function<void(std::ostream&)> custom_data_serializer,
                    ModelSerializer::CacheEncrypt cache_encrypt,
                    ov::pass::Serialize::Version version,
                    bool wl_mode)
        : ov::pass::StreamSerialize(stream, std::move(custom_data_serializer), std::move(cache_encrypt), version),
          m_weightless_mode(wl_mode) {}

private:
    std::unique_ptr<util::XmlSerializer> make_serializer(pugi::xml_node& data,
                                                         const std::string& node_type_name,
                                                         util::ConstantWriter& constant_write_handler,
                                                         int64_t version,
                                                         bool deterministic,
                                                         bool compress_to_fp16,
                                                         ov::element::Type output_element_type,
                                                         bool data_is_temporary) const override {
        return std::make_unique<XmlSerializer>(data,
                                               node_type_name,
                                               constant_write_handler,
                                               version,
                                               deterministic,
                                               compress_to_fp16,
                                               output_element_type,
                                               data_is_temporary,
                                               m_weightless_mode);
    }

    bool m_weightless_mode;
};

////////// ModelSerializer //////////

ModelSerializer::ModelSerializer(std::ostream& ostream, CacheEncrypt encrypt_fn, bool wl_mode)
    : m_ostream(ostream),
      m_cache_encrypt(std::move(encrypt_fn)),
      m_wl_mode(wl_mode) {}

void ModelSerializer::operator<<(const std::shared_ptr<ov::Model>& model) {
    auto serialize_info = [&](std::ostream& stream) {
        pugi::xml_document xml_doc;
        pugi::xml_node root = xml_doc.append_child("cnndata");
        root.append_child("outputs");
        xml_doc.save(stream);
    };
    StreamSerialize serializer(m_ostream,
                               serialize_info,
                               m_cache_encrypt,
                               pass::Serialize::Version::UNSPECIFIED,
                               m_wl_mode);
    serializer.run_on_model(model);
}

////////// ModelDeserializer //////////

ModelDeserializer::ModelDeserializer(std::istream& model_stream,
                                     std::shared_ptr<ov::AlignedBuffer> model_buffer,
                                     ModelBuilder fn,
                                     const CacheDecrypt& decrypt_fn,
                                     bool decript_from_string,
                                     std::string origin_weights_path)
    : m_istream(model_stream),
      m_model_builder(std::move(fn)),
      m_decript_from_string(decript_from_string),
      m_model_buffer(std::move(model_buffer)),
      m_origin_weights_path(std::move(origin_weights_path)) {
    if (m_decript_from_string) {
        m_cache_decrypt.m_decrypt_str = decrypt_fn.m_decrypt_str;
    } else {
        m_cache_decrypt.m_decrypt_char = decrypt_fn.m_decrypt_char;
    }
}

void ModelDeserializer::set_info(pugi::xml_node& root, std::shared_ptr<ov::Model>& model) {}

void ModelDeserializer::operator>>(std::shared_ptr<ov::Model>& model) {
    if (m_model_buffer) {
        process_mmap(model, m_model_buffer);
    } else {
        process_stream(model);
    }
}

void ModelDeserializer::process_mmap(std::shared_ptr<ov::Model>& model,
                                     const std::shared_ptr<ov::AlignedBuffer>& mmemory) {
    // Note: Don't use seekg with mmaped stream. This may affect the performance of some models.
    // Get file size before seek content.
    // Blob from cache may have other header, so need to skip this.
    auto buffer_base = reinterpret_cast<char*>(mmemory->get_ptr());
    const auto file_size = mmemory->size();
    const size_t hdr_pos = m_istream.tellg();

    pass::StreamSerialize::DataHeader hdr = {};
    std::memcpy(reinterpret_cast<char*>(&hdr), buffer_base + hdr_pos, sizeof hdr);

    // Check if model header contains valid data.
    bool is_valid_model = (hdr.custom_data_offset == sizeof(hdr) + hdr_pos) &&
                          (hdr.custom_data_size == hdr.consts_offset - hdr.custom_data_offset) &&
                          (hdr.consts_size == hdr.model_offset - hdr.consts_offset) &&
                          ((hdr.model_size = file_size - hdr.model_offset) != 0u);
    if (!is_valid_model) {
        OPENVINO_THROW("[CPU] Could not deserialize by device xml header.");
    }

    // Read model input/output precisions.
    pugi::xml_document xml_in_out_doc;
    if (hdr.custom_data_size > 0lu) {
        auto res = xml_in_out_doc.load_buffer(buffer_base + hdr.custom_data_offset,
                                              hdr.custom_data_size,
                                              pugi::parse_default,
                                              pugi::encoding_utf8);
        if (res.status != pugi::status_ok) {
            OPENVINO_THROW("[CPU] Could to deserialize custom data.");
        }
    }

    // Map blob content
    std::shared_ptr<ov::AlignedBuffer> weights_buf;
    if (hdr.consts_size) {
        weights_buf =
            std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(buffer_base + hdr.consts_offset,
                                                                                   hdr.consts_size,
                                                                                   mmemory);
    }
    std::shared_ptr<ov::AlignedBuffer> origin_weights_buf;
    if (!m_origin_weights_path.empty()) {
        auto mmap = ov::load_mmap_object(m_origin_weights_path);
        origin_weights_buf =
            std::make_shared<ov::SharedBuffer<std::shared_ptr<MappedMemory>>>(mmap->data(), mmap->size(), mmap);
    }

    // XML content
    auto xml_buff = std::make_shared<std::string>();
    if (m_cache_decrypt) {
        if (m_decript_from_string) {
            xml_buff->assign(buffer_base + hdr.model_offset, hdr.model_size);
            *xml_buff = m_cache_decrypt.m_decrypt_str(*xml_buff);
        } else {
            xml_buff->reserve(hdr.model_size + 1);
            m_cache_decrypt.m_decrypt_char(&((*xml_buff)[0]), buffer_base + hdr.model_offset, hdr.model_size);
        }
    } else {
        xml_buff->assign(buffer_base + hdr.model_offset, hdr.model_size);
    }
    std::shared_ptr<ov::AlignedBuffer> model_buf =
        std::make_shared<ov::SharedBuffer<std::shared_ptr<std::string>>>(&((*xml_buff)[0]), hdr.model_size, xml_buff);

    model = m_model_builder(model_buf, weights_buf, origin_weights_buf);

    // Set Info
    pugi::xml_node root = xml_in_out_doc.child("cnndata");
    set_info(root, model);
}

void ModelDeserializer::process_stream(std::shared_ptr<ov::Model>& model) {
    const size_t hdr_pos = m_istream.tellg();
    m_istream.seekg(0, m_istream.end);
    const size_t file_size = m_istream.tellg();
    m_istream.seekg(hdr_pos, m_istream.beg);

    pass::StreamSerialize::DataHeader hdr = {};
    m_istream.read(reinterpret_cast<char*>(&hdr), sizeof hdr);

    // Check if model header contains valid data.
    bool is_valid_model = (hdr.custom_data_offset == sizeof(hdr) + hdr_pos) &&
                          (hdr.custom_data_size == hdr.consts_offset - hdr.custom_data_offset) &&
                          (hdr.consts_size == hdr.model_offset - hdr.consts_offset) &&
                          ((hdr.model_size = file_size - hdr.model_offset) != 0u);
    if (!is_valid_model) {
        OPENVINO_THROW("[CPU] Could not deserialize by device xml header.");
    }

    // read model input/output precisions
    m_istream.seekg(hdr.custom_data_offset);

    pugi::xml_document xmlInOutDoc;
    if (hdr.custom_data_size > 0) {
        std::string xmlInOutString;
        xmlInOutString.resize(hdr.custom_data_size);
        m_istream.read(const_cast<char*>(xmlInOutString.c_str()), hdr.custom_data_size);
        auto res = xmlInOutDoc.load_string(xmlInOutString.c_str());
        if (res.status != pugi::status_ok) {
            OPENVINO_THROW("NetworkNotRead: The inputs and outputs information is invalid.");
        }
    }

    // read blob content
    auto data_blob = std::make_shared<ov::Tensor>(ov::element::u8, ov::Shape({hdr.consts_size}));
    m_istream.seekg(hdr.consts_offset);
    if (hdr.consts_size) {
        m_istream.read(static_cast<char*>(data_blob->data(ov::element::u8)), hdr.consts_size);
    }
    std::shared_ptr<ov::AlignedBuffer> origin_weights_buf;
    if (!m_origin_weights_path.empty()) {
        auto mmap = ov::load_mmap_object(m_origin_weights_path);
        origin_weights_buf =
            std::make_shared<ov::SharedBuffer<std::shared_ptr<MappedMemory>>>(mmap->data(), mmap->size(), mmap);
    }

    // read XML content
    auto xml_string = std::make_shared<std::string>();
    m_istream.seekg(hdr.model_offset);
    xml_string->resize(hdr.model_size);
    m_istream.read(const_cast<char*>(xml_string->data()), hdr.model_size);
    if (m_cache_decrypt) {
        if (m_decript_from_string) {
            *xml_string = m_cache_decrypt.m_decrypt_str(*xml_string);
        } else {
            m_cache_decrypt.m_decrypt_char(const_cast<char*>(xml_string->data()),
                                           xml_string->data(),
                                           xml_string->size());
        }
    }

    auto model_buf =
        std::make_shared<ov::SharedBuffer<std::shared_ptr<std::string>>>(const_cast<char*>(xml_string->data()),
                                                                         xml_string->size(),
                                                                         xml_string);
    auto weights_buf = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::Tensor>>>(
        reinterpret_cast<char*>(data_blob->data(ov::element::u8)),
        hdr.consts_size,
        data_blob);

    model = m_model_builder(model_buf, weights_buf, origin_weights_buf);

    // Set Info
    pugi::xml_node root = xmlInOutDoc.child("cnndata");
    set_info(root, model);
}

}  // namespace ov::intel_cpu
