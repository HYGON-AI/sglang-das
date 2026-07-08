# Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ctypes import *
import os
import time
import threading

class Prof:
    def __init__(self):
        self.use_roctx = os.getenv('SGLANG_HIP_PROF') is not None
        if self.use_roctx:
            self.lib = cdll.LoadLibrary("libroctracer64.so")
            self.lib.roctxRangePushA.argtypes = [c_char_p]
            self.lib.roctxRangePushA.restype = c_int
            self.lib.roctxRangePop.restype = c_int
        self.tm = time.perf_counter()
        self.push_depth = {}

    def StartTracer(self):
        if self.use_roctx:
            if self.lib is None:
                self.lib = cdll.LoadLibrary("libroctracer64.so")
            self.lib.roctracer_start()
            self.roc_tracer_flag = True

    def StopTracer(self):
        if self.use_roctx:
            if self.lib is None:
                self.lib = cdll.LoadLibrary("libroctracer64.so")
            self.lib.roctracer_stop()
            self.roc_tracer_flag = False

    def thread_depth_add(self, num):
        current_thread = threading.current_thread()
        thread_id = current_thread.ident
        if thread_id not in self.push_depth.keys():
            self.push_depth[thread_id] = 0
        if num < 0 and self.push_depth[thread_id] == 0:
            return False
        self.push_depth[thread_id] += num
        return True

    def ProfRangePush(self, message):
        if profile.use_roctx and self.roc_tracer_flag:
            profile.lib.roctxRangePushA(message.encode('utf-8'))
            profile.lib.roctxRangePushA(message.encode('utf-8'))
            self.thread_depth_add(1)

    def ProfRangePop(self):
        if profile.use_roctx and self.roc_tracer_flag:
            if not self.thread_depth_add(-1):
                return
            profile.lib.roctxRangePop()

    def ProfRangeAutoPush(self, message):
        self.ProfRangePop()
        self.ProfRangePush(message)


profile = Prof()
