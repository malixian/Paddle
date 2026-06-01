// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#pragma once

#include <sycl/sycl.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/runtime/backend_api.h"
#include "paddle/common/enforce.h"

using cinn::common::Arch;

namespace cinn {
namespace runtime {
namespace sycl {

inline const char* SYCLGetErrorString(const std::error_code& error_code) {
  if (error_code.category() != ::sycl::sycl_category()) {
    return "NON-SYCL ERROR";
  }

  switch (static_cast<::sycl::errc>(error_code.value())) {
    case ::sycl::errc::runtime:
      return "RUNTIME ERROR";
    case ::sycl::errc::kernel:
      return "KERNEL ERROR";
    case ::sycl::errc::accessor:
      return "ACCESSOR ERROR";
    case ::sycl::errc::nd_range:
      return "ND_RANGE ERROR";
    case ::sycl::errc::event:
      return "EVENT ERROR";
    case ::sycl::errc::kernel_argument:
      return "KERNEL ARGUMENT ERROR";
    case ::sycl::errc::build:
      return "BUILD ERROR";
    case ::sycl::errc::invalid:
      return "INVALID ERROR";
    case ::sycl::errc::memory_allocation:
      return "MEMORY ALLOCATION ERROR";
    case ::sycl::errc::platform:
      return "PLATFORM ERROR";
    case ::sycl::errc::profiling:
      return "PROFILING ERROR";
    case ::sycl::errc::feature_not_supported:
      return "FEATURE NOT SUPPORTED";
    case ::sycl::errc::kernel_not_supported:
      return "KERNEL NOT SUPPORTED";
    case ::sycl::errc::backend_mismatch:
      return "BACKEND MISMATCH";
    default:
      return "UNKNOWN SYCL ERROR";
  }
}

/*!
 * \brief Protected SYCL call.
 *
 * SYCL 2020 exposes exception details through sycl::exception::code();
 * get_cl_code() was a SYCL 1.2.1 / DPC++ compatibility API and is not used
 * here so this header can be compiled as a SYCL 2020 implementation.
 */
#define SYCL_CALL(func)                                                   \
  do {                                                                    \
    try {                                                                 \
      func;                                                               \
    } catch (const ::sycl::exception& e) {                                \
      PADDLE_THROW(::common::errors::Fatal(                               \
          "SYCL Driver Error in Paddle CINN: failed with error code %d "  \
          "(%s): %s",                                                    \
          e.code().value(),                                               \
          ::cinn::runtime::sycl::SYCLGetErrorString(e.code()),            \
          e.what()));                                                     \
    }                                                                     \
  } while (0)

class SYCLBackendAPI final : public BackendAPI {
 public:
  SYCLBackendAPI() = default;
  ~SYCLBackendAPI() override = default;

  static SYCLBackendAPI* Global();

  /*!
   * \brief Initialize SYCL devices, contexts and queues.
   * \param arch CINN target architecture. The current implementation keeps the
   *             BackendAPI signature and selects SYCL GPU devices first.
   */
  void Init(Arch arch);
  void set_device(int device_id) final;
  int get_device() final;
  int get_device_property(DeviceProperty device_property,
                          std::optional<int> device_id = std::nullopt) final;
  void* malloc(size_t numBytes) final;
  void free(void* data) final;
  void memset(void* data, int value, size_t numBytes) final;
  void memcpy(void* dest,
              const void* src,
              size_t numBytes,
              MemcpyType type) final;
  void device_sync() final;
  void stream_sync(void* stream) final;
  ::sycl::queue* get_now_queue(void* stream = nullptr);
  std::string GetGpuVersion();
  std::array<int, 3> get_max_grid_dims(
      std::optional<int> device_id = std::nullopt) final;
  std::array<int, 3> get_max_block_dims(
      std::optional<int> device_id = std::nullopt) final;

 private:
  void CheckInitialized() const;
  int NormalizeDeviceId(std::optional<int> device_id) const;

  // SYCL 2020 objects are value types. No backend-native HIP handles are stored
  // here, so the API is portable across SYCL backends.
  std::vector<::sycl::device> devices_;
  std::vector<::sycl::context> contexts_;
  std::vector<std::vector<::sycl::queue>> queues_;

  int now_device_id_{0};
  bool initialized_{false};
};

}  // namespace sycl
}  // namespace runtime
}  // namespace cinn

