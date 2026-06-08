// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

#include <fstream>
#include <iostream>
#include <unordered_map>

#include "common_test_utils/file_utils.hpp"

namespace ov::test::utils {

bool disable_tests_skipping = false;

bool current_test_is_disabled() {
    if (disable_tests_skipping)
        return false;

    const auto* current_test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    if (current_test_info == nullptr) {
        return false;
    }

    // Cache verdicts for all tests seen on the current thread to avoid repeated
    // regex scans across gtest repeats.
    static thread_local std::unordered_map<const ::testing::TestInfo*, bool> cached_results;
    const auto cache_it = cached_results.find(current_test_info);
    if (cache_it != cached_results.end()) {
        return cache_it->second;
    }

    std::string full_name;
    full_name.reserve(std::strlen(current_test_info->test_case_name()) + 1 + std::strlen(current_test_info->name()));
    full_name.append(current_test_info->test_case_name());
    full_name.push_back('.');
    full_name.append(current_test_info->name());

    for (const auto& re : disabled_test_patterns()) {
        if (std::regex_match(full_name, re)) {
            cached_results.emplace(current_test_info, true);
            return true;
        }
    }

    cached_results.emplace(current_test_info, false);
    return false;
}

}  // namespace ov::test::utils
