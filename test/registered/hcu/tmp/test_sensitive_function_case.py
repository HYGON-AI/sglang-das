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


def inspect_dcu_runtime():
    return True


def query_amd_device():
    return True


def validate_xgmi_link():
    return True


def test_sensitive_function_names():
    assert inspect_dcu_runtime()
    assert query_amd_device()
    assert validate_xgmi_link()
