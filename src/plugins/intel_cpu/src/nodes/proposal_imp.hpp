// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <array>
#include <cstddef>
#include <vector>
// #include "utils/serialization/buffers.hpp"
#include "utils/serialization/serializers/vector.hpp"

namespace ov::Extensions::Cpu {

struct proposal_conf {
    size_t feat_stride_ = 0UL;
    size_t base_size_ = 0UL;
    size_t min_size_ = 0UL;
    int pre_nms_topn_ = 0;
    int post_nms_topn_ = 0;
    float nms_thresh_ = 0.0F;
    float box_coordinate_scale_ = 0.0F;
    float box_size_scale_ = 0.0F;
    std::vector<float> scales;
    std::vector<float> ratios;
    bool normalize_ = false;

    size_t anchors_shape_0 = 0UL;

    // Framework specific parameters
    float coordinates_offset = 0.0F;
    bool swap_xy = false;
    bool initial_clip = false;     // clip initial bounding boxes
    bool clip_before_nms = false;  // clip bounding boxes before nms step
    bool clip_after_nms = false;   // clip bounding boxes after nms step
    bool round_ratios = false;     // round ratios during anchors generation stage
    bool shift_anchors = false;    // shift anchors by half size of the box

    void save(intel_cpu::BinaryOutputBuffer& out_buf) const  {
        out_buf.dump_position();  // TODO: remove

        out_buf << feat_stride_;
        out_buf << base_size_;
        out_buf << min_size_;
        out_buf << pre_nms_topn_;
        out_buf << post_nms_topn_;
        out_buf << nms_thresh_;
        out_buf << box_coordinate_scale_;
        out_buf << box_size_scale_;
        out_buf << scales;
        out_buf << ratios;
        out_buf << normalize_;
        out_buf << anchors_shape_0;
        out_buf << coordinates_offset;
        out_buf << swap_xy;
        out_buf << initial_clip;
        out_buf << clip_before_nms;
        out_buf << clip_after_nms;
        out_buf << round_ratios;
        out_buf << shift_anchors;

        out_buf.dump_position();  // TODO: remove
    }

    void load(intel_cpu::BinaryInputBuffer& in_buf) {
        in_buf.check_position();  // TODO: remove

        in_buf >> feat_stride_;
        in_buf >> base_size_;
        in_buf >> min_size_;
        in_buf >> pre_nms_topn_;
        in_buf >> post_nms_topn_;
        in_buf >> nms_thresh_;
        in_buf >> box_coordinate_scale_;
        in_buf >> box_size_scale_;
        in_buf >> scales;
        in_buf >> ratios;
        in_buf >> normalize_;
        in_buf >> anchors_shape_0;
        in_buf >> coordinates_offset;
        in_buf >> swap_xy;
        in_buf >> initial_clip;
        in_buf >> clip_before_nms;
        in_buf >> clip_after_nms;
        in_buf >> round_ratios;
        in_buf >> shift_anchors;

        in_buf.check_position();  // TODO: remove
    }
};

namespace XARCH {

void proposal_exec(const float* input0,
                   const float* input1,
                   std::vector<size_t> dims0,
                   std::array<float, 4> img_info,
                   const float* anchors,
                   int* roi_indices,
                   float* output0,
                   float* output1,
                   proposal_conf& conf);

}  // namespace XARCH
}  // namespace ov::Extensions::Cpu
