/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once

#include <roccache/conv_kernels.h>

#include <mutex>  // NOLINT

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/common/port.h"

namespace phi {
namespace dynload {
extern std::once_flag roccache_dso_flag;
extern void *roccache_dso_handle;

#define DECLARE_DYNAMIC_LOAD_ROCCACHE_WRAP(__name)                    \
  struct DynLoad__##__name {                                        \
    template <typename... Args>                                     \
    void operator()(Args... args) {                      \
      using roccacheFunc = decltype(&::__name);                      \
      std::call_once(roccache_dso_flag, []() {                       \
        roccache_dso_handle = phi::dynload::GetROCCACHEDsoHandle();    \
      });                                                           \
      static void *p_##__name = dlsym(roccache_dso_handle, #__name); \
      return reinterpret_cast<roccacheFunc>(p_##__name)(args...);    \
    }                                                               \
  };                                                                \
  extern DynLoad__##__name __name

#define ROCCACHE_CACHE_ROUTINE_EACH(__macro)      \
  __macro(rocCacheConv2dFwd);

ROCCACHE_CACHE_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_ROCCACHE_WRAP);

}  // namespace dynload
}  // namespace phi
