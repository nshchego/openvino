// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "openvino/core/core_visibility.hpp"

namespace ov {

namespace detail {
constexpr size_t fnv1a_basis = static_cast<size_t>(14695981039346656037ull);
constexpr size_t fnv1a_prime = static_cast<size_t>(1099511628211ull);

constexpr size_t fnv1a_hash_impl(const char* str, size_t idx, size_t hash_val) {
    return str[idx] == '\0'
               ? hash_val
               : fnv1a_hash_impl(str,
                                 idx + 1,
                                 (hash_val ^ static_cast<unsigned char>(str[idx])) * fnv1a_prime);
}

constexpr size_t fnv1a_hash(const char* str) {
    return str ? fnv1a_hash_impl(str, 0, fnv1a_basis) : 0;
}

constexpr size_t hash_combine_constexpr(size_t val, size_t seed) {
    return seed ^ (val + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

constexpr size_t compute_type_hash(const char* name, const char* version_id) {
    return hash_combine_constexpr(fnv1a_hash(name), hash_combine_constexpr(fnv1a_hash(version_id), 0));
}
}  // namespace detail

/**
 * @brief Type information for a type system without inheritance; instances have exactly one type not
 * related to any other type.
 *
 * Supports three functions, ov::is_type<Type>, ov::as_type<Type>, and ov::as_type_ptr<Type> for type-safe
 * dynamic conversions via static_cast/static_ptr_cast without using C++ RTTI.
 * Type must have a static type_info member and a virtual get_type_info() member that
 * returns a reference to its type_info member.
 * @ingroup ov_model_cpp_api
 */
struct OPENVINO_API DiscreteTypeInfo {
    const char* name;
    const char* version_id;
    // A pointer to a parent type info; used for casting and inheritance traversal, not for
    // exact type identification
    const DiscreteTypeInfo* parent;

        constexpr DiscreteTypeInfo()
                : name(nullptr),
                    version_id(nullptr),
                    parent(nullptr),
                    hash_value(0) {}
    DiscreteTypeInfo(const DiscreteTypeInfo&) = default;
    DiscreteTypeInfo(DiscreteTypeInfo&&) = default;
    DiscreteTypeInfo& operator=(const DiscreteTypeInfo&) = default;

    explicit constexpr DiscreteTypeInfo(const char* _name,
                                        const char* _version_id,
                                        const DiscreteTypeInfo* _parent = nullptr)
        : name(_name),
          version_id(_version_id),
          parent(_parent),
                    hash_value(detail::compute_type_hash(_name, _version_id)) {}

    constexpr DiscreteTypeInfo(const char* _name, const DiscreteTypeInfo* _parent = nullptr)
        : name(_name),
          version_id(nullptr),
          parent(_parent),
                    hash_value(detail::compute_type_hash(_name, nullptr)) {}

    bool is_castable(const DiscreteTypeInfo& target_type) const;

    std::string get_version() const;

    // For use as a key
    bool operator<(const DiscreteTypeInfo& b) const;
    bool operator<=(const DiscreteTypeInfo& b) const;
    bool operator>(const DiscreteTypeInfo& b) const;
    bool operator>=(const DiscreteTypeInfo& b) const;
    bool operator==(const DiscreteTypeInfo& b) const;
    bool operator!=(const DiscreteTypeInfo& b) const;

    operator std::string() const;

    constexpr size_t hash() const;

private:
    size_t hash_value;
};

constexpr inline size_t DiscreteTypeInfo::hash() const {
    return hash_value;
}

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const DiscreteTypeInfo& info);

namespace frontend {
class ConversionExtensionBase;
}  // namespace frontend

template <typename T>
constexpr bool use_ov_dynamic_cast() {
#if defined(__ANDROID__)
    return true;
#else
    return std::is_base_of_v<ov::frontend::ConversionExtensionBase, T>;
#endif
}

/// \brief Tests if value is a pointer/shared_ptr that can be statically cast to a
/// Type*/shared_ptr<Type>
template <typename Type, typename Value>
std::enable_if_t<
    std::is_convertible_v<decltype(std::declval<Value>()->get_type_info().is_castable(Type::get_type_info_static())),
                          bool>,
    bool>
is_type(const Value& value) {
    return value && value->get_type_info().is_castable(Type::get_type_info_static());
}

/// \brief Tests if value is a pointer/shared_ptr that can be statically cast to any of the specified types
template <typename... Types, typename Value>
bool is_type_any_of(const Value& value) {
    return (is_type<Types>(value) || ...);
}

/// Casts a Value* to a Type* if it is of type Type, nullptr otherwise
template <typename Type, typename Value>
std::enable_if_t<std::is_convertible_v<decltype(static_cast<Type*>(std::declval<Value>())), Type*>, Type*> as_type(
    Value value) {
    if constexpr (use_ov_dynamic_cast<Type>())
        return is_type<Type>(value) ? static_cast<Type*>(value) : nullptr;
    else
        return dynamic_cast<Type*>(value);
}

namespace util {
template <typename T>
struct AsTypePtr;
/// Casts a std::shared_ptr<Value> to a std::shared_ptr<Type> if it is of type
/// Type, nullptr otherwise
template <typename In>
struct AsTypePtr<std::shared_ptr<In>> {
    template <typename Type>
    static std::shared_ptr<Type> call(const std::shared_ptr<In>& value) {
        return ov::is_type<Type>(value) ? std::static_pointer_cast<Type>(value) : std::shared_ptr<Type>();
    }
};
}  // namespace util

/// Casts a std::shared_ptr<Value> to a std::shared_ptr<Type> if it is of type
/// Type, nullptr otherwise
template <typename Type, typename Value>
auto as_type_ptr(const Value& value) -> decltype(::ov::util::AsTypePtr<Value>::template call<Type>(value)) {
    if constexpr (use_ov_dynamic_cast<Type>())
        return ::ov::util::AsTypePtr<Value>::template call<Type>(value);
    else
        return std::dynamic_pointer_cast<Type>(value);
}
}  // namespace ov

namespace std {
template <>
struct OPENVINO_API hash<ov::DiscreteTypeInfo> {
    size_t operator()(const ov::DiscreteTypeInfo& k) const;
};
}  // namespace std
