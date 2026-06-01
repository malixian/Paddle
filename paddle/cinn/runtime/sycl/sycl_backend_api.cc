#include "paddle/cinn/runtime/sycl/sycl_backend_api.h"

#include <glog/logging.h>

#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif

#include <hip/hip_runtime.h>

#include <sycl/backend.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/detail/backend_traits_hip.hpp>

#include <algorithm>
#include <array>
#include <iostream>
#include <optional>
#include <vector>

namespace cinn {
namespace runtime {
namespace sycl {

namespace {

#define HIP_CHECK(cmd)                                                       \
  do {                                                                       \
    hipError_t error = (cmd);                                                \
    if (error != hipSuccess) {                                               \
      std::cerr << "HIP error: " << hipGetErrorString(error)                 \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;      \
      std::abort();                                                          \
    }                                                                        \
  } while (0)

::sycl::async_handler MakeAsyncHandler() {
  return [](::sycl::exception_list exceptions) {
    for (const std::exception_ptr& e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (const ::sycl::exception& e) {
        PADDLE_THROW(::common::errors::Fatal(
            "Caught asynchronous SYCL exception:\n %s ", e.what()));
      }
    }
  };
}

}  // namespace

SYCLBackendAPI* SYCLBackendAPI::Global() {
  static auto* inst = new SYCLBackendAPI();
  return inst;
}

void SYCLBackendAPI::Init(Arch arch) {
  if (initialized_) return;

  auto devices = ::sycl::device::get_devices(::sycl::info::device_type::gpu);
  PADDLE_ENFORCE_GT(
      devices.size(),
      static_cast<size_t>(0),
      ::common::errors::InvalidArgument("No valid SYCL gpu device found."));

  ::sycl::backend backend;
  arch.Match(
      [&](common::UnknownArch) {
        SYCL_CALL(backend = ::sycl::device::get_devices(
                                ::sycl::info::device_type::gpu)[0]
                                .get_backend());
      },
      [&](common::X86Arch) { CINN_NOT_IMPLEMENTED },
      [&](common::ARMArch) { CINN_NOT_IMPLEMENTED },
      [&](common::CustomDeviceArch) { CINN_NOT_IMPLEMENTED },
      [&](common::NVGPUArch) { backend = ::sycl::backend::ext_oneapi_cuda; },
      [&](common::HygonDCUArchHIP) { CINN_NOT_IMPLEMENTED },
      [&](common::HygonDCUArchSYCL) {
        backend = ::sycl::backend::ext_oneapi_hip;
      });

  if (this->devices_.size() < 8) {
    for (auto device : devices) {
      if (device.get_backend() == backend) {
        this->devices_.push_back(device);
      }
    }
  }

  if (this->devices_.empty()) {
    std::cerr << "No valid gpu device matched given arch \n";
  }

  this->contexts_.resize(this->devices_.size());
  this->queues_.resize(this->devices_.size());
  initialized_ = true;
}

void SYCLBackendAPI::set_device(int device_id) {
  if (!initialized_) Init(common::UnknownArch{});

  PADDLE_ENFORCE_GE(device_id,
                    0,
                    ::common::errors::InvalidArgument(
                        "please set valid device id! device id", device_id));

  PADDLE_ENFORCE_LE(
      device_id,
      static_cast<int>(this->devices_.size()) - 1,
      ::common::errors::InvalidArgument("set valid device id! device id: ",
                                        device_id,
                                        " > max device id:",
                                        this->devices_.size() - 1));

  this->now_device_id_ = device_id;

  if (this->queues_[device_id].empty()) {
    this->contexts_[device_id] =
        ::sycl::context(this->devices_[device_id], MakeAsyncHandler());

    ::sycl::property_list q_prop{::sycl::property::queue::in_order()};
    this->queues_[device_id].emplace_back(
        this->contexts_[device_id], this->devices_[device_id], q_prop);
  }
}

int SYCLBackendAPI::get_device() { return this->now_device_id_; }

int SYCLBackendAPI::get_device_property(DeviceProperty device_property,
                                        std::optional<int> device_id) {
  int index = device_id.value_or(this->now_device_id_);
  int rv = -1;

  switch (device_property) {
    case DeviceProperty::MaxBlockDimX: {
      //::sycl::id<3> max_work_item_sizes =
      //    this->devices_[index]
      //        .get_info<::sycl::info::device::max_work_item_sizes<3>>();
      //rv = static_cast<int>(max_work_item_sizes[0]);
      rv = 1024;
      break;
    }
    case DeviceProperty::MaxBlockDimY: {
      //::sycl::id<3> max_work_item_sizes =
      //    this->devices_[index]
      //        .get_info<::sycl::info::device::max_work_item_sizes<3>>();
      //rv = static_cast<int>(max_work_item_sizes[1]);
      rv = 1024;
      break;
    }
    case DeviceProperty::MaxBlockDimZ: {
      //::sycl::id<3> max_work_item_sizes =
      //    this->devices_[index]
      //        .get_info<::sycl::info::device::max_work_item_sizes<3>>();
      //rv = static_cast<int>(max_work_item_sizes[2]);
      rv = 1024;
      break;
    }
    case DeviceProperty::MaxGridDimX: {
      rv = 2147483647;
      break;
    }
    case DeviceProperty::MaxGridDimY: {
      rv = 2147483647;
      break;
    }
    case DeviceProperty::MaxGridDimZ: {
      rv = 2147483647;
      break;
    }
    case DeviceProperty::MaxSharedMemoryPerBlock: {
      rv = static_cast<int>(
          this->devices_[index].get_info<::sycl::info::device::local_mem_size>());
      break;
    }
    case DeviceProperty::MaxThreadsPerBlock: {
      rv = static_cast<int>(
          this->devices_[index]
              .get_info<::sycl::info::device::max_work_group_size>());
      break;
    }
    case DeviceProperty::MaxThreadsPerSM: {
      rv = static_cast<int>(
          this->devices_[index]
              .get_info<::sycl::info::device::max_work_group_size>());
      break;
    }
    case DeviceProperty::MultiProcessorCount: {
      rv = static_cast<int>(
          this->devices_[index]
              .get_info<::sycl::info::device::max_compute_units>());
      break;
    }
    case DeviceProperty::MaxBlocksPerSM: {
      PADDLE_THROW(::common::errors::InvalidArgument(
          "SYCL Not supported device property : MaxBlocksPerSM !"));
      break;
    }
    case DeviceProperty::WarpSize: {
      std::vector<size_t> sub_group_sizes =
          this->devices_[index]
              .get_info<::sycl::info::device::sub_group_sizes>();
      size_t max_sub_group_size =
          *std::max_element(std::begin(sub_group_sizes),
                            std::end(sub_group_sizes));
      rv = static_cast<int>(max_sub_group_size);
      break;
    }
    default:
      PADDLE_THROW(::common::errors::InvalidArgument(
          "SYCL Not supported device property !"));
  }
  return rv;
}

void* SYCLBackendAPI::malloc(size_t numBytes) {
  VLOG(3) << "sycl malloc";

  auto* Q = get_now_queue(nullptr);
  void* dev_mem = nullptr;

  SYCL_CALL(dev_mem = ::sycl::malloc_device(numBytes, *Q));

  PADDLE_ENFORCE_NE(dev_mem,
                    nullptr,
                    ::common::errors::InvalidArgument(
                        "allocate sycl device memory failure!"));
  return dev_mem;
}

void SYCLBackendAPI::free(void* data) {
  VLOG(3) << "sycl free";
  auto* Q = get_now_queue(nullptr);
  SYCL_CALL(::sycl::free(data, *Q));
}

void SYCLBackendAPI::memset(void* data, int value, size_t numBytes) {
  VLOG(3) << "sycl memset";
  auto* Q = get_now_queue(nullptr);
  SYCL_CALL(Q->memset(data, value, numBytes).wait());
}

void SYCLBackendAPI::memcpy(void* dest,
                            const void* src,
                            size_t numBytes,
                            MemcpyType type) {
  VLOG(3) << "sycl memcpy";
  (void)type;

  auto* Q = get_now_queue(nullptr);
  SYCL_CALL(Q->memcpy(dest, src, numBytes).wait());
}

void SYCLBackendAPI::device_sync() {
  VLOG(3) << "sycl device sync";
  for (auto& queues_in_one_device : this->queues_) {
    for (auto& queue : queues_in_one_device) {
      SYCL_CALL(queue.wait_and_throw());
    }
  }
}

void SYCLBackendAPI::stream_sync(void* stream) {
  VLOG(3) << "sycl stream sync";

  if (stream == nullptr) {
    SYCL_CALL(get_now_queue(nullptr)->wait_and_throw());
    return;
  }

  SYCL_CALL(static_cast<::sycl::queue*>(stream)->wait_and_throw());
}

::sycl::queue* SYCLBackendAPI::get_now_queue(void* raw_stream) {
  if (!initialized_) {
    Init(common::UnknownArch{});
  }

  // In the SYCL backend, the stream object is represented by sycl::queue*.
  // Do not construct a SYCL queue from hipStream_t here.  The HIP backend
  // native stream should be obtained inside a host_task through
  // interop_handle::get_native_queue<backend::ext_oneapi_hip>(), as in the
  // verified HIP interop test.
  //if (raw_stream != nullptr) {
  //  return static_cast<::sycl::queue*>(raw_stream);
  //}

  PADDLE_ENFORCE_GT(
      this->devices_.size(),
      static_cast<size_t>(0),
      ::common::errors::InvalidArgument("No valid SYCL gpu device found."));

  if (this->now_device_id_ < 0 ||
      this->now_device_id_ >= static_cast<int>(this->devices_.size())) {
    this->now_device_id_ = 0;
  }

  if (this->queues_[now_device_id_].empty()) {
    this->contexts_[now_device_id_] =
        ::sycl::context(this->devices_[now_device_id_], MakeAsyncHandler());

    ::sycl::property_list q_prop{::sycl::property::queue::in_order()};
    this->queues_[now_device_id_].emplace_back(
        this->contexts_[now_device_id_],
        this->devices_[now_device_id_],
        q_prop);
  }

  return &this->queues_[now_device_id_][0];
}

std::string SYCLBackendAPI::GetGpuVersion() {
  ::sycl::device device = this->devices_[now_device_id_];
  ::sycl::backend backend = device.get_backend();

  switch (backend) {
    case ::sycl::backend::ext_oneapi_cuda: {
      std::string gpu_version = "sm_";
      std::string version_with_point =
          device.get_info<::sycl::info::device::driver_version>();
      size_t pos = version_with_point.find(".");
      if (pos != std::string::npos) {
        gpu_version += version_with_point.substr(0, pos) +
                       version_with_point.substr(pos + 1,
                                                 version_with_point.size());
      }
      return gpu_version;
    }
    case ::sycl::backend::ext_oneapi_hip: {
      std::string gpu_version =
          device.get_info<::sycl::info::device::version>();
      size_t pos = gpu_version.find(":");
      if (pos != std::string::npos) gpu_version = gpu_version.substr(0, pos);
      return gpu_version;
    }
    default:
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Error! use unknown sycl backend!"));
  }
}

std::array<int, 3> SYCLBackendAPI::get_max_block_dims(
    std::optional<int> device_id) {
  int index = device_id.value_or(this->now_device_id_);

  //::sycl::id<3> max_work_item_sizes =
  //    this->devices_[index]
  //        .get_info<::sycl::info::device::max_work_item_sizes<3>>();

  return std::array<int, 3>{
      static_cast<int>(2048),
      static_cast<int>(2048),
      static_cast<int>(2048)};
}

std::array<int, 3> SYCLBackendAPI::get_max_grid_dims(
    std::optional<int> device_id) {
  (void)device_id;
  return std::array<int, 3>{2147483647, 2147483647, 2147483647};
}

}  // namespace sycl
}  // namespace runtime
}  // namespace cinn
