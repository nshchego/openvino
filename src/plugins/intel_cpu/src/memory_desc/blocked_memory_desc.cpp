// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "blocked_memory_desc.h"

#include <cctype>
#include <cstddef>
#include <sstream>
#include <string>
#include <unordered_set>

#include "utils/general_utils.h"
#include "utils/serialization/vector_serializer.hpp"

namespace ov::intel_cpu {

/* c++11 requires to have a definition in cpp file */

bool BlockedMemoryDesc::isCompatibleInternal(const BlockedMemoryDesc& rhs, CmpMask cmpMask) const {
    if (this->getShape() != rhs.getShape() || this->getPrecision() != rhs.getPrecision()) {
        return false;
    }

    if (!dimsEqualWeak(this->getBlockDims(), rhs.getBlockDims())) {
        return false;
    }

    if (!dimsEqualWeak(this->getOffsetPaddingToData(), rhs.getOffsetPaddingToData())) {
        return false;
    }

    const auto& thisStrides = this->getStrides();
    const auto& rhsStrides = rhs.getStrides();

    if (thisStrides.size() != rhsStrides.size()) {
        return false;
    }

    for (size_t i = 0; i < thisStrides.size(); i++) {
        if (cmpMask.test(i) && !dimsEqualWeak(thisStrides[i], rhsStrides[i])) {
            return false;
        }
    }

    if (!dimsEqualWeak(this->getOrder(), rhs.getOrder())) {
        return false;
    }

    if (cmpMask.test(OFFSET_MASK_POS)) {
        return dimsEqualWeak(this->getOffsetPadding(), rhs.getOffsetPadding());
    }

    return true;
}

std::string BlockedMemoryDesc::serializeFormat() const {
    std::stringstream result;
    char startLetter = 'a';
    std::unordered_set<size_t> blockedAxis;
    const auto& order = getOrder();
    const auto& shape = getShape();
    for (size_t i = shape.getRank(); i < order.size(); ++i) {
        blockedAxis.insert(order[i]);
    }

    for (size_t i = 0; i < shape.getRank(); ++i) {
        auto nextLetter = static_cast<char>(startLetter + order[i]);
        if (blockedAxis.count(i)) {
            nextLetter = static_cast<char>(toupper(nextLetter));
        }
        result << nextLetter;
    }

    const auto& blkDims = getBlockDims();
    for (size_t i = shape.getRank(); i < order.size(); ++i) {
        result << blkDims[i] << static_cast<char>(startLetter + order[i]);
    }

    return result.str();
}

void BlockedMemoryDesc::save(BinaryOutputBuffer& ob) const {
    MemoryDesc::save(ob);
    ob << ob.get_pos();  // TODO: remove

    ob << blockedDims;
    ob << strides;
    ob << order;
    ob << m_offset_padding_to_data;

    ob << ob.get_pos();  // TODO: remove
}

void BlockedMemoryDesc::load(BinaryInputBuffer& in_buf) {
    MemoryDesc::load(in_buf);
    validate_stream_offset(in_buf);  // TODO: remove

    in_buf >> blockedDims;
    in_buf >> strides;
    in_buf >> order;
    in_buf >> m_offset_padding_to_data;

    validate_stream_offset(in_buf);  // TODO: remove
}

}  // namespace ov::intel_cpu
