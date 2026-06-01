// Copyright (c) 2024 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <dlfcn.h>
#include <glog/logging.h>
#include <glog/raw_logging.h>

#include<hip/hip_runtime.h>

#include "paddle/cinn/runtime/cinn_runtime.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/cinn/runtime/sycl/sycl_backend_api.h"
#include "paddle/cinn/runtime/sycl/sycl_util.h"
#include "paddle/cinn/utils/profiler.h"
#include "paddle/common/enforce.h"

#include <iostream>

namespace cinn {
namespace runtime {
namespace sycl {

void cinn_call_sycl_kernel(void *kernel_fn,
                           void *v_args,
                           int num_args,
                           int grid_x,
                           int grid_y,
                           int grid_z,
                           int block_x,
                           int block_y,
                           int block_z,
                           int shared_memory_bytes,
                           void *stream) {
  VLOG(3) << "cinn_call_sycl_kernel, grid_dim={" << grid_x << ", " << grid_y
          << ", " << grid_z << "}, block_dim={" << block_x << ", " << block_y
          << ", " << block_z << "}, num_args=" << num_args;
  
  hipStream_t hs = reinterpret_cast<hipStream_t>(stream);

  error_t err = hipStreamSynchronize(hs);

  if (err != hipSuccess) {
    std::cerr << "[HIP ERROR] hipStreamSynchronize failed: " << std::endl;
  }
  /*
  std::cout << "cinn_call_sycl_kernel, grid_dim={" << grid_x << ", " << grid_y
          << ", " << grid_z << "}, block_dim={" << block_x << ", " << block_y
          << ", " << block_z << "}, num_args=" << num_args << std::endl;
  */
  std::vector<void *> kernel_args;
  {
    cinn::utils::RecordEvent record_run("prepare_args",
                                        cinn::utils::EventType::kInstruction);
    kernel_args.reserve(num_args);
    cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
    for (int idx = 0; idx < num_args; ++idx) {
      if (args[idx].type_code() == ::cinn_type_code<cinn_buffer_t *>()) {
        void *addr = static_cast<cinn_buffer_t *>(args[idx])->memory;
        std::stringstream ss;
        ss << std::hex << addr;
        VLOG(4) << "sycl kernel arg[" << idx
                << "] is a buffer, addr=" << ss.str();
        kernel_args.emplace_back(addr);
      } else {
        kernel_args.emplace_back((args[idx].data_addr()));
      }
    }
  }

  {
    cinn::utils::RecordEvent record_run("syclLaunchKernel",
                                        cinn::utils::EventType::kInstruction);
    void (*kernel_func)(::sycl::queue & Q,
                        ::sycl::range<3> k0_dimGrid,
                        ::sycl::range<3> k0_dimBlock,
                        void **void_args) =
        (void (*)(::sycl::queue & Q,
                  ::sycl::range<3> k0_dimGrid,
                  ::sycl::range<3> k0_dimBlock,
                  void **void_args))(kernel_fn);
    //std::cout<<"Before get now queue"<<std::endl;
    ::sycl::queue *Queue = SYCLBackendAPI::Global()->get_now_queue(stream);
    ::sycl::range<3> Grid(grid_z, grid_y, grid_x);
    ::sycl::range<3> Block(block_z, block_y, block_x);
    //std::cout<<"Run sycl kernel"<<std::endl;
    SYCL_CALL(kernel_func(*Queue, Grid, Block, kernel_args.data()));
  }
}

void infer_shape_set_value(int row, int col, int64_t value, int64_t **v) {
  v[row][col] = value;
}

void cinn_call_sycl_memset(
    void *v_args, int num_args, int value, size_t count, void *stream) {
  PADDLE_ENFORCE_EQ(num_args,
                    1,
                    ::common::errors::PreconditionNotMet(
                        "The cinn_call_sycl_memset only accept a output."));
  VLOG(4) << "call cinn_call_sycl_memset with value=" << value
          << ", count=" << count;

  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  void *output = args[0].operator cinn_buffer_t *()->memory;

  std::stringstream ss;
  ss << std::hex << output;
  VLOG(4) << "cinn_call_sycl_memset: " << ss.str();

  ::sycl::queue *Queue = SYCLBackendAPI::Global()->get_now_queue(stream);

  SYCL_CALL(Queue->memset(output, value, count));
}

void cinn_call_sycl_memcpy(void *v_args,
                           int num_args,
                           size_t count,
                           void *stream) {
  PADDLE_ENFORCE_EQ(
      num_args,
      2,
      ::common::errors::PreconditionNotMet(
          "The cinn_call_sycl_memcpy only accept a input and a output."));
  VLOG(4) << "call cinn_call_sycl_memcpy with count=" << count;

  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  void *input = args[0].operator cinn_buffer_t *()->memory;
  void *output = args[1].operator cinn_buffer_t *()->memory;

  if (input == output) {
    VLOG(3) << "cinn_call_sycl_memcpy: skip copy as input addr is the same as "
               "output.";
    return;
  }

  std::stringstream ss;
  ss << std::hex << input << " -> " << output;
  VLOG(4) << "cinn_call_sycl_memcpy: " << ss.str();

  ::sycl::queue *Queue = SYCLBackendAPI::Global()->get_now_queue(stream);

  SYCL_CALL(Queue->memcpy(output, input, count));
}

#ifdef CINN_WITH_CNNL

namespace {
[[noreturn]] void ThrowPureSycl2020CnnlUnsupported(const char* fn_name) {
  PADDLE_THROW(::common::errors::Unimplemented(
      "%s is not implemented in the pure SYCL 2020 runtime path. "
      "The previous implementation depended on CNRT/CNNL native queues and "
      "sycl::queue::get_native<::sycl::backend::cnrt>(), which are not part "
      "of the SYCL 2020 standard. Build without CINN_WITH_CNNL for the pure "
      "SYCL 2020 path, or keep a separate vendor-native backend implementation.",
      fn_name));
}
}  // namespace

void cinn_call_cnnl_gaussian_random(
    void *v_args, int num_args, float mean, float std, int seed) {
  ThrowPureSycl2020CnnlUnsupported("cinn_call_cnnl_gaussian_random");
}

void cinn_call_cnnl_uniform_random(
    void *v_args, int num_args, float min, float max, int seed) {
  ThrowPureSycl2020CnnlUnsupported("cinn_call_cnnl_uniform_random");
}

void cinn_call_cnnl_randint(void *v_args, int num_args, int seed) {
  ThrowPureSycl2020CnnlUnsupported("cinn_call_cnnl_randint");
}

void cinn_call_cnnl_matmul(void *v_args,
                           int num_args,
                           bool trans_a,
                           bool trans_b,
                           bool trans_o,
                           float alpha,
                           float beta,
                           int a1,
                           int a2,
                           int a3,
                           int a4,
                           int b1,
                           int b2,
                           int b3,
                           int b4) {
  ThrowPureSycl2020CnnlUnsupported("cinn_call_cnnl_matmul");
}

void cinn_call_cnnl_conv2d_forward(void *v_args,
                                   int num_args,
                                   int format,
                                   float alpha,
                                   float beta,
                                   int input_n,
                                   int input_c,
                                   int input_h,
                                   int input_w,
                                   int filter_n,
                                   int filter_c,
                                   int filter_h,
                                   int filter_w,
                                   int pad_h,
                                   int pad_w,
                                   int stride_h,
                                   int stride_w,
                                   int dilation_h,
                                   int dilation_w,
                                   int groups,
                                   int output_n,
                                   int output_c,
                                   int output_h,
                                   int output_w) {
  ThrowPureSycl2020CnnlUnsupported("cinn_call_cnnl_conv2d_forward");
}

void cinn_call_cnnl_conv2d_backward_data(void *v_args,
                                         int num_args,
                                         int format,
                                         float alpha,
                                         float beta,
                                         int input_n,
                                         int input_c,
                                         int input_h,
                                         int input_w,
                                         int filter_n,
                                         int filter_c,
                                         int filter_h,
                                         int filter_w,
                                         int pad_h,
                                         int pad_w,
                                         int stride_h,
                                         int stride_w,
                                         int dilation_h,
                                         int dilation_w,
                                         int groups,
                                         int output_n,
                                         int output_c,
                                         int output_h,
                                         int output_w) {
  ThrowPureSycl2020CnnlUnsupported("cinn_call_cnnl_conv2d_backward_data");
}

void cinn_call_cnnl_conv2d_backward_filter(void *v_args,
                                           int num_args,
                                           int format,
                                           float alpha,
                                           float beta,
                                           int input_n,
                                           int input_c,
                                           int input_h,
                                           int input_w,
                                           int filter_n,
                                           int filter_c,
                                           int filter_h,
                                           int filter_w,
                                           int pad_h,
                                           int pad_w,
                                           int stride_h,
                                           int stride_w,
                                           int dilation_h,
                                           int dilation_w,
                                           int groups,
                                           int output_n,
                                           int output_c,
                                           int output_h,
                                           int output_w) {
  ThrowPureSycl2020CnnlUnsupported("cinn_call_cnnl_conv2d_backward_filter");
}

void cinn_call_cnnl_pool2d_forward(void *v_args,
                                   int num_args,
                                   int mode,
                                   int format,
                                   float alpha,
                                   float beta,
                                   int input_n,
                                   int input_c,
                                   int input_h,
                                   int input_w,
                                   int kernel_h,
                                   int kernel_w,
                                   int pad_h,
                                   int pad_w,
                                   int stride_h,
                                   int stride_w,
                                   int output_n,
                                   int output_c,
                                   int output_h,
                                   int output_w) {
  ThrowPureSycl2020CnnlUnsupported("cinn_call_cnnl_pool2d_forward");
}

void cinn_call_cnnl_pool2d_backward(void *v_args,
                                    int num_args,
                                    int mode,
                                    int format,
                                    float alpha,
                                    float beta,
                                    int input_n,
                                    int input_c,
                                    int input_h,
                                    int input_w,
                                    int kernel_h,
                                    int kernel_w,
                                    int pad_h,
                                    int pad_w,
                                    int stride_h,
                                    int stride_w,
                                    int output_n,
                                    int output_c,
                                    int output_h,
                                    int output_w) {
  ThrowPureSycl2020CnnlUnsupported("cinn_call_cnnl_pool2d_backward");
}

#endif  // CINN_WITH_CNNL

}  // namespace sycl
}  // namespace runtime
}  // namespace cinn
