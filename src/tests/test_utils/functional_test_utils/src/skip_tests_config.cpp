// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

#include <fstream>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include "common_test_utils/file_utils.hpp"

namespace ov::test::utils {

bool disable_tests_skipping = false;
namespace {

bool precomputed_disabled_tests_enabled = false;
std::unordered_set<std::string> precomputed_disabled_tests;

std::string basename_from_path(const char* path) {
    if (path == nullptr) {
        return {};
    }

    const std::string full_path(path);
    const auto pos = full_path.find_last_of("/\\");
    return pos == std::string::npos ? full_path : full_path.substr(pos + 1);
}

bool is_cpu_func_tests_binary(const char* argv0) {
    const std::string bin_name = basename_from_path(argv0);
    return bin_name == "ov_cpu_func_tests" || bin_name == "ov_cpu_func_tests_subset";
}

}  // namespace

void set_disabled_tests_filter_from_patterns(const char* argv0) {
    if (disable_tests_skipping || !is_cpu_func_tests_binary(argv0)) {
        return;
    }

    const auto& patterns = disabled_test_patterns();
    if (patterns.empty()) {
        return;
    }

    auto* unit_test = ::testing::UnitTest::GetInstance();
    std::unordered_set<std::string> disabled_tests;

    for (int suite_idx = 0; suite_idx < unit_test->total_test_suite_count(); ++suite_idx) {
        const auto* test_suite = unit_test->GetTestSuite(suite_idx);
        if (test_suite == nullptr) {
            continue;
        }

        for (int test_idx = 0; test_idx < test_suite->total_test_count(); ++test_idx) {
            const auto* test_info = test_suite->GetTestInfo(test_idx);
            if (test_info == nullptr) {
                continue;
            }

            std::string full_name;
            full_name.reserve(std::strlen(test_info->test_case_name()) + 1 + std::strlen(test_info->name()));
            full_name.append(test_info->test_case_name());
            full_name.push_back('.');
            full_name.append(test_info->name());

            for (const auto& re : patterns) {
                if (std::regex_match(full_name, re)) {
                    disabled_tests.emplace(std::move(full_name));
                    break;
                }
            }
        }
    }

    precomputed_disabled_tests = std::move(disabled_tests);
    precomputed_disabled_tests_enabled = true;
}

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

    if (precomputed_disabled_tests_enabled) {
        return precomputed_disabled_tests.find(full_name) != precomputed_disabled_tests.end();
    }

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
