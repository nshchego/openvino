// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// #include <cfenv>
// #include <cmath>
// #include <functional>
// #include <numeric>

#include "ngraph/util.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
namespace {
constexpr size_t in_batch_axis = 0;
constexpr size_t in_channel_axis = 1;
constexpr size_t filter_out_ch_axis = 0;
constexpr size_t filter_in_ch_axis = 1;
constexpr size_t out_batch_axis = 0;
constexpr size_t out_channel_axis = 1;

struct ConvolutionParams {
    std::vector<int64_t> strides;
    std::vector<int64_t> dilation;
    std::vector<int64_t> pads_begin;
    std::vector<int64_t> pads_end;
    std::vector<int64_t> output_padding;

    ConvolutionParams(const Strides& strides_,
                      const Strides& dilation_,
                      const CoordinateDiff& pads_begin_,
                      const CoordinateDiff& pads_end_,
                      const CoordinateDiff& output_padding_ = {0, 0, 0})
        : strides{strides_.begin(), strides_.end()},
          dilation{dilation_.begin(), dilation_.end()},
          pads_begin{pads_begin_.begin(), pads_begin_.end()},
          pads_end{pads_end_.begin(), pads_end_.end()},
          output_padding{output_padding_.begin(), output_padding_.end()} {};
};

template <typename Int>
constexpr inline bool in_range(Int val, std::pair<Int, Int> range) noexcept {
    return val >= range.first && val < range.second;
}

template <typename T>
void convolve_3D_channels(const ConvolutionParams& p,
                          const T* batch,
                          const Shape& batch_shape,
                          const T* filter,
                          const Shape& filter_shape,
                          T*& out) {
// static int counter = 0;
// static double total_time = 0.;
// counter++;
// auto t1 = std::chrono::high_resolution_clock::now();

    const int dilation_z = static_cast<int>(p.dilation[0]);
    const int dilation_y = static_cast<int>(p.dilation[1]);
    const int dilation_x = static_cast<int>(p.dilation[2]);

    const int pad_begin_z = static_cast<int>(p.pads_begin[0]);
    const int pad_begin_y = static_cast<int>(p.pads_begin[1]);
    const int pad_begin_x = static_cast<int>(p.pads_begin[2]);

    const int stride_z = static_cast<int>(p.strides[0]);
    const int stride_y = static_cast<int>(p.strides[1]);
    const int stride_x = static_cast<int>(p.strides[2]);

    const int input_size_z = static_cast<int>(batch_shape[1]);
    const int input_size_y = static_cast<int>(batch_shape[2]);
    const int input_size_x = static_cast<int>(batch_shape[3]);

    const int input_size_yx  = input_size_y * input_size_x;
    const int input_size_zyx = input_size_z * input_size_yx;

    const size_t f_channels = static_cast<int>(filter_shape[0]);
    const int filter_size_z = static_cast<int>(filter_shape[1]);
    const int filter_size_y = static_cast<int>(filter_shape[2]);
    const int filter_size_x = static_cast<int>(filter_shape[3]);
    const int filter_size_yx = filter_size_y * filter_size_x;

    const int dilated_filter_size_z = static_cast<int>(filter_size_z + (filter_size_z - 1) * (dilation_z - 1));
    const int dilated_filter_size_y = static_cast<int>(filter_size_y + (filter_size_y - 1) * (dilation_y - 1));
    const int dilated_filter_size_x = static_cast<int>(filter_size_x + (filter_size_x - 1) * (dilation_x - 1));

    const int i_z_lim = static_cast<int>(p.pads_end[0] + input_size_z - dilated_filter_size_z + p.output_padding[0]);
    const int i_y_lim = static_cast<int>(p.pads_end[1] + input_size_y - dilated_filter_size_y + p.output_padding[1]);
    const int i_x_lim = static_cast<int>(p.pads_end[2] + input_size_x - dilated_filter_size_x + p.output_padding[2]);

    const int f_z_increment = dilation_z * input_size_yx;
    const int f_y_increment = dilation_y * input_size_x;
    const int f_x_increment = dilation_x;

    const int f_z_block = filter_size_z * f_z_increment;
    const int f_y_block = filter_size_y * f_y_increment;
    const int f_x_block = filter_size_x * f_x_increment;

    for (int i_z = -pad_begin_z; i_z <= i_z_lim; i_z += stride_z) {
        const int s_z_shift = i_z * input_size_yx;
        const int f_z_up_bound = f_z_block + s_z_shift;

        for (int i_y = -pad_begin_y; i_y <= i_y_lim; i_y += stride_y) {
            const int i_y_m = i_y * input_size_x;

            for (int i_x = -pad_begin_x; i_x <= i_x_lim; i_x += stride_x) {
                auto input_channel = batch;
                auto filter_channel = filter;
                T sum = 0;
                size_t filter_channels_count = f_channels;

                while (filter_channels_count--) {
                    for (int f_z_i = s_z_shift; f_z_i < f_z_up_bound; f_z_i += f_z_increment) {
                        if (f_z_i < 0 || f_z_i >= input_size_zyx) {
                            filter_channel += filter_size_yx;
                            continue;
                        }
                        const int y_up_bound = f_z_i + input_size_yx;
                        const int y_shift = f_z_i + i_y_m;
                        for (int f_y_i = y_shift; f_y_i < f_y_block + y_shift; f_y_i += f_y_increment) {
                            if (f_y_i < f_z_i || f_y_i >= y_up_bound) {
                                filter_channel += filter_size_x;
                                continue;
                            }
                            const int x_up_bound = input_size_x + f_y_i;
                            for (int f_x_i = f_y_i + i_x; f_x_i < f_x_block + f_y_i + i_x; f_x_i += f_x_increment, filter_channel++) {
                                if (f_x_i < f_y_i || f_x_i >= x_up_bound) {
                                    continue;
                                }

                                sum += input_channel[f_x_i] * filter_channel[0];
                            }
                        }
                    }
                    input_channel += input_size_zyx;
                }
                *out = sum;
                ++out;
            }
        }
    }
    
// auto t2 = std::chrono::high_resolution_clock::now();
// std::chrono::duration<double, std::milli> ms_double = t2 - t1;
// total_time += ms_double.count();
// if (counter == 745490) {
//     std::cout << "AVG TIME: " << total_time / double(counter) << std::endl;
// }
}

inline void extend_to_3D(ConvolutionParams& p, Shape& in_shape, Shape& filter_shape) {
    int spatial_rank = static_cast<int>(in_shape.size() - 2);
    if (spatial_rank < 3) {
        int missing_dims = 3 - spatial_rank;
        p.dilation.insert(std::prev(p.dilation.end(), spatial_rank), missing_dims, 1);
        p.strides.insert(std::prev(p.strides.end(), spatial_rank), missing_dims, 1);
        p.pads_begin.insert(std::prev(p.pads_begin.end(), spatial_rank), missing_dims, 0);
        p.pads_end.insert(std::prev(p.pads_end.end(), spatial_rank), missing_dims, 0);
        p.output_padding.insert(std::prev(p.output_padding.end(), spatial_rank), missing_dims, 0);
        in_shape.insert(std::next(in_shape.end(), -spatial_rank), missing_dims, 1);
        filter_shape.insert(std::prev(filter_shape.end(), spatial_rank), missing_dims, 1);
    }
}

void infer_forward_conv_output_shape(const Shape& in_spatial_shape,
                                     const Shape& f_spatial_shape,
                                     Shape& out_spatial_shape,
                                     const Strides& strides,
                                     const Strides& dilations,
                                     const CoordinateDiff& pads_begin,
                                     const CoordinateDiff& pads_end) {
    for (size_t idx = 0; idx < in_spatial_shape.size(); idx++) {
        size_t in_padded_dim = in_spatial_shape[idx] + pads_begin[idx] + pads_end[idx];
        size_t filter_dilated_dim = dilations[idx] * (f_spatial_shape[idx] - 1) + 1;
        size_t out_spatial_dim = (in_padded_dim - filter_dilated_dim) / strides[idx] + 1;
        out_spatial_shape.push_back(out_spatial_dim);
    }
}

inline void validate_convolution_parameters(const Shape& in_shape,
                                            const Shape& f_shape,
                                            const Shape& out_shape,
                                            const Strides& strides,
                                            const Strides& dilations,
                                            const CoordinateDiff& pads_begin,
                                            const CoordinateDiff& pads_end) {
    // this implementation supports 1D, 2D and 3D convolutions
    NGRAPH_CHECK(in_shape.size() >= 3 && in_shape.size() <= 5, "Unsupported input rank: ", in_shape);

    NGRAPH_CHECK(in_shape.size() == f_shape.size(),
                 "Incompatible input ranks: ",
                 in_shape.size(),
                 " and ",
                 f_shape.size());

    NGRAPH_CHECK(in_shape[in_channel_axis] == f_shape[filter_in_ch_axis],
                 "Incompatible input channels in data batch and filters shapes: ",
                 in_shape[in_channel_axis],
                 " and ",
                 f_shape[filter_in_ch_axis]);

    NGRAPH_CHECK(in_shape.size() == out_shape.size(),
                 "Incompatible input and output ranks: ",
                 in_shape.size(),
                 " and ",
                 out_shape.size());

    const auto spatial_dims = in_shape.size() - 2;
    NGRAPH_CHECK(strides.size() == spatial_dims, "Strides not definied for all and only spatial dimensions");

    NGRAPH_CHECK(dilations.size() == spatial_dims, "Dilations not defined for all and only spatial dimensions");

    NGRAPH_CHECK((pads_begin.size() == pads_end.size()) && (pads_begin.size() == spatial_dims),
                 "Pads not defined for all and only spatial dimensions");

    Shape out_spatial_shape{std::next(out_shape.begin(), 2), std::end(out_shape)};
    Shape infered_out_spatial_shape{};
    infer_forward_conv_output_shape(Shape{std::next(in_shape.begin(), 2), std::end(in_shape)},
                                    Shape{std::next(f_shape.begin(), 2), std::end(f_shape)},
                                    infered_out_spatial_shape,
                                    strides,
                                    dilations,
                                    pads_begin,
                                    pads_end);
    NGRAPH_CHECK(out_spatial_shape == infered_out_spatial_shape, "Incorrect output shape provided");
}
}  // namespace

template <typename T>
void convolution(const T* in,
                 const T* f,
                 T* out,
                 const Shape& in_shape,
                 const Shape& f_shape,
                 const Shape& out_shape,
                 const Strides& strides,
                 const Strides& dilations,
                 const CoordinateDiff& pads_begin,
                 const CoordinateDiff& pads_end)

{
    validate_convolution_parameters(in_shape, f_shape, out_shape, strides, dilations, pads_begin, pads_end);

    // here we are converting all param types to int's to avoid arithmetic issues
    // (e.g signed + unsigned) in indexes calculation later
    ConvolutionParams params{strides, dilations, pads_begin, pads_end};

    // here we are extending spatial dimensions to 3D, because we are going to use 3D
    // convolution implementation to convolve also in 1D & 2D case
    Shape input_shape{in_shape};
    Shape filters_shape{f_shape};
    if (in_shape.size() < 5) {
        extend_to_3D(params, input_shape, filters_shape);
    }

    const size_t batches_count = input_shape[in_batch_axis];
    const Shape batch_shape(++input_shape.begin(), input_shape.end());
    const size_t batch_size = shape_size(batch_shape);

    const size_t filters_count = filters_shape[filter_out_ch_axis];
    const Shape filter_shape(++filters_shape.begin(), filters_shape.end());
    const size_t filter_size = shape_size(filter_shape);

    auto batch = in;
    for (size_t batch_idx = 0; batch_idx < batches_count; ++batch_idx) {
        auto filter = f;
        for (size_t f_idx = 0; f_idx < filters_count; ++f_idx) {
            convolve_3D_channels(params, batch, batch_shape, filter, filter_shape, out);
            filter += filter_size;
        }
        batch += batch_size;
    }
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph

// can't be removed currently due to arm-plugin dependency
#include "ngraph/runtime/reference/convolution_backprop_data.hpp"
