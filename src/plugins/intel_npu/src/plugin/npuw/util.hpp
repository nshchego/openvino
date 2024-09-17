// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "logging.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
namespace npuw {
namespace util {

bool is_set(const std::size_t sub_idx, const std::string& opt);

// Every great project has its own string class...
// NB: Newer C++ standards would allow to use string views or smt
ov::Tensor tensor_from_const(const std::shared_ptr<ov::Node>& node);

bool starts_with(const std::string& str, const std::string& prefix);

std::string fmt(std::size_t number, std::size_t total);

struct UnpackOptions {
    bool bUseOvParallelFor;
    size_t nPartitions;  // if 0 we use 64 elements step in parallel for, otherwise  target workload is dynamically
                         // calculated
    bool bStrictPartitioning;  // cannot reduce partitions in favor of speed
    explicit UnpackOptions(bool useParallelFor, size_t nPartitions, bool bStrictPartitioning)
        : bUseOvParallelFor(useParallelFor),
          nPartitions(nPartitions),
          bStrictPartitioning(bStrictPartitioning) {}
};

void unpack(const ov::SoPtr<ov::ITensor>& from,
            const ov::SoPtr<ov::ITensor>& to,
            const UnpackOptions& unpack_options = UnpackOptions{true, 16, false});

void unpack(const ov::SoPtr<ov::ITensor>& from,
            const ov::SoPtr<ov::ITensor>& scale,
            const ov::SoPtr<ov::ITensor>& to,
            const UnpackOptions& unpack_options = UnpackOptions{true, 16, false});

void unpack(const ov::SoPtr<ov::ITensor>& from,
            const ov::SoPtr<ov::ITensor>& zerop,
            const ov::SoPtr<ov::ITensor>& scale,
            const ov::SoPtr<ov::ITensor>& to,
            const UnpackOptions& unpack_options = UnpackOptions{true, 16, false});

void to_f32(const ov::Tensor& in, ov::Tensor& out);
void to_f16(ov::Tensor& t);
void transpose(ov::Tensor& t);
void permute(ov::Tensor& t, const std::vector<std::size_t>& axes);

namespace at {
template <class M>
struct Impl {
    using V = typename M::mapped_type;

    M* m = nullptr;
    explicit Impl(M* pM) : m(pM) {}

    template <typename K>
    V& at(const K& k) {
        const auto iter = m->find(k);
        if (iter == m->end()) {
            std::stringstream ss;
            ss << "Key " << k << " is not found in a map of type " << typeid(m).name();
            const auto msg = ss.str();
            LOG_ERROR(msg);
            throw std::out_of_range(msg);
        }
        return iter->second;
    }

    template <typename K>
    const V& at(const K& k) const {
        return const_cast<Impl*>(this)->at(k);
    }
};

template <typename M>
Impl<M> _(M* pM) {
    return Impl<M>(pM);
}

template <typename M>
Impl<M> _(std::shared_ptr<M> pM) {
    return Impl<M>(pM.get());
}

}  // namespace at

}  // namespace util
}  // namespace npuw
}  // namespace ov
