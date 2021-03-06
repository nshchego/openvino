// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpp/ie_cnn_network.h"
#include "mkldnn_graph.h"

#include <memory>

namespace MKLDNNPlugin {

InferenceEngine::CNNNetwork dump_graph_as_ie_ngraph_net(const MKLDNNGraph &graph);

}  // namespace MKLDNNPlugin
