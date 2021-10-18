// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

//#include <mkldnn_node.h>
#include "common/tile_broadcast_utils.h"

#include <memory>
#include <string>
#include <vector>


namespace MKLDNNPlugin {

class MKLDNNBroadcastNode : public MKLDNNNode, public TileBroadcastCommon {
public:
    MKLDNNBroadcastNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNBroadcastNode() override = default;

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

private:
    enum AutoBroadcastType {
        NUMPY,
        EXPLICIT
    };
    AutoBroadcastType broadcastType;

    static const size_t INPUT_DATA_IDX = 0;
    static const size_t TARGET_SHAPE_IDX = 1;
    static const size_t AXES_MAPPING_IDX = 2;

    std::vector<int32_t> axesMapping;
    VectorDims targetDims;

    std::string errorPrefix;
};

}  // namespace MKLDNNPlugin
