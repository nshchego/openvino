// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>

#include "common/tile_broadcast_utils.h"
#include <ngraph/op/tile.hpp>

namespace MKLDNNPlugin {

class MKLDNNTileNode : public MKLDNNNode, public TileBroadcastCommon {
public:
    MKLDNNTileNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNTileNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    void notOptimizedExecute(mkldnn::stream strm);
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

protected:
    bool needPrepareParams() const override;
    void prepareParams() override;
    bool needShapeInfer() const override;
    std::vector<VectorDims> shapeInfer() const override;

private:
    static const size_t TILE_INPUT = 0lu;
    static const size_t TILE_REPEATS = 1lu;

    int axis = -1;
    int tiles = 0;
    bool noTiling = false;
    mutable InferenceEngine::SizeVector originRepeats;

    std::string errorPrefix;
};

}  // namespace MKLDNNPlugin

