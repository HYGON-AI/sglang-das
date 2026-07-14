# Official Main Catch-up 20260706 — Code Conflict Review

> Scope: only the 10 files that produced textual merge conflicts. The conflict ledger and all automatically merged files are intentionally excluded.
> View in VS Code with **Markdown: Open Preview** (`Ctrl+Shift+V`). The `diff` blocks render removed conflict state in red and the final resolved code in green.

## Comparison

- DCU parent (`ours`): `726ca92425c0cac65419686308b9ee2a9c915f80`
- Common official base: `88db9e033a11b2d366a8f9d037f027a46ccb9940`
- Official endpoint (`theirs`): `9a6f8e599204aa37481f5f37a1b20938aee98d5c`
- Resolved merge: `51f025b2d5464a1c35eef12656546d7cc9c56bb1`
- Reconstructed textual conflicts: 10 files, 25 hunks

Each section reconstructs Git’s three-way auto-conflict text from the two merge parents and common base, then compares it with the committed resolution. Lines beginning with `-` belong to the unresolved auto-conflict state; lines beginning with `+` are the final resolution.

## Conflict files

<details>
<summary><code>.github/workflows/pr-test-npu.yml</code> — 1 conflict hunk</summary>

**Resolution intent:** Accept the official single-node NPU job and finish dependency while normalizing CRLF to the existing LF convention.

~~~~diff
--- AUTO-CONFLICT/.github/workflows/pr-test-npu.yml
+++ RESOLVED/.github/workflows/pr-test-npu.yml
@@ -1,911 +1,8 @@
-<<<<<<< DCU main@726ca92425c0
-name: PR Test (NPU)
-
-on:
-  push:
-    branches: [ main ]
-  pull_request:
-  workflow_dispatch:
-  workflow_call:
-    inputs:
-      ref:
-        description: 'Git ref (branch, tag, or SHA) to test. If not provided, uses the default branch.'
-        required: false
-        type: string
-        default: ''
-      run_all_tests:
-        description: "Run all tests (for releasing or testing purpose)"
-        required: false
-        type: boolean
-        default: false
-
-concurrency:
-  group: pr-test-npu-${{ inputs.ref || github.ref }}
-  cancel-in-progress: ${{ github.event_name != 'workflow_call' }}
-
-jobs:
-  # ==================== Check Changes ==================== #
-  check-changes:
-    runs-on: ubuntu-latest
-    outputs:
-      changes_exist: ${{ steps.filter.outputs.main_package == 'true' || steps.filter.outputs.multimodal_gen == 'true' || steps.run-mode.outputs.run_all_tests == 'true'}}
-      main_package: ${{ steps.filter.outputs.main_package == 'true' || steps.run-mode.outputs.run_all_tests == 'true' }}
-      multimodal_gen: ${{ steps.filter.outputs.multimodal_gen == 'true' || steps.run-mode.outputs.run_all_tests == 'true' }}
-    steps:
-      - name: Checkout code
-        uses: actions/checkout@v4
-        with:
-          ref: ${{ inputs.ref || github.ref }}
-
-      - name: Determine run mode
-        id: run-mode
-        run: |
-          # Run all tests for workflow_call (when ref input is provided)
-          # Note: github.event_name is inherited from caller, so we detect workflow_call by checking inputs.ref
-          if [[ "${{ inputs.run_all_tests }}" == "true" ]]; then
-            echo "run_all_tests=true" >> $GITHUB_OUTPUT
-            echo "Run mode: ALL TESTS (run_all_tests=${{ inputs.run_all_tests }})"
-          else
-            echo "run_all_tests=false" >> $GITHUB_OUTPUT
-            echo "Run mode: FILTERED (triggered by ${{ github.event_name }})"
-          fi
-
-      - name: Detect file changes
-        id: filter
-        uses: dorny/paths-filter@v3
-        if: steps.run-mode.outputs.run_all_tests != 'true'
-        with:
-          filters: |
-            main_package:
-              - "python/sglang/!(multimodal_gen)/**/!(*.md)"
-              - "python/pyproject_npu.toml"
-              - "scripts/ci/npu/npu_ci_install_dependency.sh"
-              - "test/registered/ascend/**"
-              - ".github/workflows/pr-test-npu.yml"
-            multimodal_gen:
-              - "python/sglang/multimodal_gen/**/!(*.md|*.ipynb)"
-              - "python/sglang/jit_kernel/diffusion/triton/npu_fallback.py"
-              - "python/sglang/srt/**"
-              - "python/pyproject_npu.toml"
-              - "scripts/ci/npu/npu_ci_install_dependency.sh"
-              - ".github/workflows/pr-test-npu.yml"
-
-  # ==================== PR Gate ==================== #
-  pr-gate:
-    needs: check-changes
-    if: needs.check-changes.outputs.changes_exist == 'true'
-    uses: ./.github/workflows/pr-gate.yml
-    secrets: inherit
-
-  set-image-config:
-    runs-on: ubuntu-latest
-    outputs:
-      CANN_image_a3: ${{ steps.set-vars.outputs.CANN_image_a3 }}
-      CANN_image_910b: ${{ steps.set-vars.outputs.CANN_image_910b }}
-    steps:
-      # When triggered by PR, no inputs parameters are used. The latest community code is tested by default.
-      - name: Set image config
-        id: set-vars
-        run: |
-          echo "CANN_image_a3=swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:9.0.0-a3-ubuntu22.04-py3.11" >> $GITHUB_OUTPUT
-          echo "CANN_image_910b=swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:9.0.0-910b-ubuntu22.04-py3.11" >> $GITHUB_OUTPUT
-
-  stage-b-test-1-npu-a2:
-    needs: [check-changes, pr-gate, set-image-config]
-    if: needs.check-changes.outputs.main_package == 'true'
-    runs-on: linux-aarch64-a2-1
-    strategy:
-      fail-fast: false
-      matrix:
-        part: [ 0, 1 ]
-    container:
-      image: ${{ needs.set-image-config.outputs.CANN_image_910b }}
-    steps:
-      - name: Checkout code
-        uses: actions/checkout@v4
-        with:
-          ref: ${{ inputs.ref || github.ref }}
-
-      - name: Mark repository safe
-        run: |
-          git config --system --add safe.directory ${GITHUB_WORKSPACE}
-
-      - name: Install dependencies
-        env:
-          TORCH_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/whl/cpu"
-          PYPI_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple"
-          UV_INDEX_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple"
-          GITHUB_PROXY_URL: "https://gh-proxy.test.osinfra.cn/"
-          RUSTUP_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local:8082"
-        run: |
-          # speed up by using infra cache services
-          CACHING_URL="cache-service.nginx-pypi-cache.svc.cluster.local"
-          sed -Ei "s@(ports|archive).ubuntu.com@${CACHING_URL}:8081@g" /etc/apt/sources.list
-          pip config set global.index-url http://${CACHING_URL}/pypi/simple
-          pip config set global.trusted-host "${CACHING_URL}"
-
-          bash scripts/ci/npu/npu_ci_install_dependency.sh 910b
-          # copy required file from our daily cache
-          cp ~/.cache/modelscope/hub/datasets/otavia/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json /tmp
-          # copy gsm8k dataset
-          cp ~/.cache/modelscope/hub/datasets/tmp/test.jsonl /tmp
-
-      - name: Run test
-        timeout-minutes: 60
-        env:
-          SGLANG_USE_MODELSCOPE: true
-          SGLANG_IS_IN_CI: true
-          HF_ENDPOINT: https://hf-mirror.com
-          TORCH_EXTENSIONS_DIR: /tmp/torch_extensions
-          PYTORCH_NPU_ALLOC_CONF: "expandable_segments:True"
-          STREAMS_PER_DEVICE: 32
-        run: |
-          cd test
-          python3 run_suite.py --hw npu --suite stage-b-test-1-npu-a2 --auto-partition-id ${{ matrix.part }} --auto-partition-size 2
-
-  stage-b-test-2-npu-a2:
-    needs: [check-changes, pr-gate, set-image-config]
-    if: needs.check-changes.outputs.main_package == 'true'
-    runs-on: linux-aarch64-a2-2
-    strategy:
-      fail-fast: true
-      matrix:
-        part: [0, 1]
-    container:
-      image: ${{ needs.set-image-config.outputs.CANN_image_910b }}
-    steps:
-      - name: Checkout code
-        uses: actions/checkout@v4
-        with:
-          ref: ${{ inputs.ref || github.ref }}
-
-      - name: Mark repository safe
-        run: |
-          git config --system --add safe.directory ${GITHUB_WORKSPACE}
-
-      - name: Install dependencies
-        env:
-          TORCH_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/whl/cpu"
-          PYPI_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple"
-          UV_INDEX_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple"
-          GITHUB_PROXY_URL: "https://gh-proxy.test.osinfra.cn/"
-          RUSTUP_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local:8082"
-        run: |
-          # speed up by using infra cache services
-          CACHING_URL="cache-service.nginx-pypi-cache.svc.cluster.local"
-          sed -Ei "s@(ports|archive).ubuntu.com@${CACHING_URL}:8081@g" /etc/apt/sources.list
-          pip config set global.index-url http://${CACHING_URL}/pypi/simple
-          pip config set global.trusted-host "${CACHING_URL}"
-
-          bash scripts/ci/npu/npu_ci_install_dependency.sh 910b
-          # copy required file from our daily cache
-          cp ~/.cache/modelscope/hub/datasets/otavia/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json /tmp
-          # copy gsm8k dataset
-          cp ~/.cache/modelscope/hub/datasets/tmp/test.jsonl /tmp
-
-      - name: Run test
-        timeout-minutes: 60
-        env:
-          SGLANG_USE_MODELSCOPE: true
-          SGLANG_IS_IN_CI: true
-          HF_ENDPOINT: https://hf-mirror.com
-          TORCH_EXTENSIONS_DIR: /tmp/torch_extensions
-          PYTORCH_NPU_ALLOC_CONF: "expandable_segments:True"
-          STREAMS_PER_DEVICE: 32
-        run: |
-          cd test
-          python3 run_suite.py --hw npu --suite stage-b-test-2-npu-a2 --auto-partition-id ${{ matrix.part }} --auto-partition-size 2
-
-  stage-b-test-4-npu-a3:
-    needs: [check-changes, pr-gate, set-image-config]
-    if: needs.check-changes.outputs.main_package == 'true'
-    runs-on: linux-aarch64-a3-4
-    container:
-      image: ${{ needs.set-image-config.outputs.CANN_image_a3 }}
-    steps:
-      - name: Checkout code
-        uses: actions/checkout@v4
-        with:
-          ref: ${{ inputs.ref || github.ref }}
-
-      - name: Mark repository safe
-        run: |
-          git config --system --add safe.directory ${GITHUB_WORKSPACE}
-
-      - name: Install dependencies
-        env:
-          TORCH_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/whl/cpu"
-          PYPI_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple"
-          UV_INDEX_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple"
-          GITHUB_PROXY_URL: "https://gh-proxy.test.osinfra.cn/"
-          RUSTUP_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local:8082"
-        run: |
-          # speed up by using infra cache services
-          CACHING_URL="cache-service.nginx-pypi-cache.svc.cluster.local"
-          sed -Ei "s@(ports|archive).ubuntu.com@${CACHING_URL}:8081@g" /etc/apt/sources.list
-          pip config set global.index-url http://${CACHING_URL}/pypi/simple
-          pip config set global.trusted-host "${CACHING_URL}"
-
-          bash scripts/ci/npu/npu_ci_install_dependency.sh a3
-          # copy required file from our daily cache
-          cp ~/.cache/modelscope/hub/datasets/otavia/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json /tmp
-          # copy gsm8k dataset
-          cp ~/.cache/modelscope/hub/datasets/tmp/test.jsonl /tmp
-
-      - name: Run test
-        timeout-minutes: 60
-        env:
-          SGLANG_USE_MODELSCOPE: true
-          SGLANG_IS_IN_CI: true
-          HF_ENDPOINT: https://hf-mirror.com
-          TORCH_EXTENSIONS_DIR: /tmp/torch_extensions
-          PYTORCH_NPU_ALLOC_CONF: "expandable_segments:True"
-          STREAMS_PER_DEVICE: 32
-        run: |
-          cd test
-          python3 run_suite.py --hw npu --suite stage-b-test-4-npu-a3 --timeout-per-file 3600
-
-
-  stage-b-test-16-npu-a3:
-    needs: [check-changes, pr-gate, set-image-config]
-    if: needs.check-changes.outputs.main_package == 'true'
-    runs-on: linux-aarch64-a3-16
-    container:
-      image: ${{ needs.set-image-config.outputs.CANN_image_a3 }}
-    steps:
-      - name: Checkout code
-        uses: actions/checkout@v4
-        with:
-          ref: ${{ inputs.ref || github.ref }}
-
-      - name: Mark repository safe
-        run: |
-          git config --system --add safe.directory ${GITHUB_WORKSPACE}
-
-      - name: Install dependencies
-        env:
-          TORCH_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/whl/cpu"
-          PYPI_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple"
-          UV_INDEX_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple"
-          GITHUB_PROXY_URL: "https://gh-proxy.test.osinfra.cn/"
-          RUSTUP_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local:8082"
-        run: |
-          # speed up by using infra cache services
-          CACHING_URL="cache-service.nginx-pypi-cache.svc.cluster.local"
-          sed -Ei "s@(ports|archive).ubuntu.com@${CACHING_URL}:8081@g" /etc/apt/sources.list
-          pip config set global.index-url http://${CACHING_URL}/pypi/simple
-          pip config set global.trusted-host "${CACHING_URL}"
-
-          bash scripts/ci/npu/npu_ci_install_dependency.sh a3
-          # copy required file from our daily cache
-          cp ~/.cache/modelscope/hub/datasets/otavia/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json /tmp
-          # copy gsm8k dataset
-          cp ~/.cache/modelscope/hub/datasets/tmp/test.jsonl /tmp
-
-      - name: Run test
-        timeout-minutes: 60
-        env:
-          SGLANG_USE_MODELSCOPE: true
-          SGLANG_IS_IN_CI: true
-          HF_ENDPOINT: https://hf-mirror.com
-          TORCH_EXTENSIONS_DIR: /tmp/torch_extensions
-          PYTORCH_NPU_ALLOC_CONF: "expandable_segments:True"
-          STREAMS_PER_DEVICE: 32
-        run: |
-          cd test
-          python3 run_suite.py --hw npu --suite stage-b-test-16-npu-a3 --timeout-per-file 3600
-
-  multimodal-gen-test-1-npu-a3:
-    needs: [check-changes, pr-gate, set-image-config]
-    if: needs.check-changes.outputs.multimodal_gen == 'true'
-    runs-on: linux-aarch64-a3-2
-    container:
-      image: ${{ needs.set-image-config.outputs.CANN_image_a3 }}
-    steps:
-      - name: Checkout code
-        uses: actions/checkout@v4
-
-      - name: Mark repository safe
-        run: |
-          git config --system --add safe.directory ${GITHUB_WORKSPACE}
-
-      - name: Install dependencies
-        env:
-          TORCH_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/whl/cpu"
-          PYPI_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple"
-          UV_INDEX_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple"
-          GITHUB_PROXY_URL: "https://gh-proxy.test.osinfra.cn/"
-          RUSTUP_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local:8082"
-        run: |
-          # speed up by using infra cache services
-          CACHING_URL="cache-service.nginx-pypi-cache.svc.cluster.local"
-          sed -Ei "s@(ports|archive).ubuntu.com@${CACHING_URL}:8081@g" /etc/apt/sources.list
-          pip config set global.index-url http://${CACHING_URL}/pypi/simple
-          pip config set global.trusted-host "${CACHING_URL}"
-
-          bash scripts/ci/npu/npu_ci_install_dependency.sh a3
-          # copy required file from our daily cache
-          cp ~/.cache/modelscope/hub/datasets/otavia/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json /tmp
-          # copy gsm8k dataset
-          cp ~/.cache/modelscope/hub/datasets/tmp/test.jsonl /tmp
-
-      - name: Run test
-        timeout-minutes: 60
-        env:
-          SGLANG_USE_MODELSCOPE: true
-          SGLANG_IS_IN_CI: true
-          HF_ENDPOINT: https://hf-mirror.com
-          TORCH_EXTENSIONS_DIR: /tmp/torch_extensions
-          PYTORCH_NPU_ALLOC_CONF: "expandable_segments:True"
-          STREAMS_PER_DEVICE: 32
-          SGLANG_DIFFUSION_ARTIFACT_DIR: ${{ github.workspace }}/diffusion-failures
-        run: |
-          cd python
-          python3 sglang/multimodal_gen/test/run_suite.py --suite 1-npu
-
-      - name: Upload diffusion failure artifacts
-        if: always()
-        uses: actions/upload-artifact@v4
-        with:
-          name: diffusion-failures-npu-1-${{ github.run_attempt }}
-          path: diffusion-failures/
-          if-no-files-found: ignore
-          retention-days: 7
-
-  multimodal-gen-test-2-npu-a3:
-    needs: [check-changes, pr-gate, set-image-config]
-    if: needs.check-changes.outputs.multimodal_gen == 'true'
-    runs-on: linux-aarch64-a3-16
-    container:
-      image: ${{ needs.set-image-config.outputs.CANN_image_a3 }}
-    steps:
-      - name: Checkout code
-        uses: actions/checkout@v4
-
-      - name: Mark repository safe
-        run: |
-          git config --system --add safe.directory ${GITHUB_WORKSPACE}
-
-      - name: Install dependencies
-        env:
-          TORCH_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/whl/cpu"
-          PYPI_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple"
-          UV_INDEX_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple"
-          GITHUB_PROXY_URL: "https://gh-proxy.test.osinfra.cn/"
-          RUSTUP_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local:8082"
-        run: |
-          # speed up by using infra cache services
-          CACHING_URL="cache-service.nginx-pypi-cache.svc.cluster.local"
-          sed -Ei "s@(ports|archive).ubuntu.com@${CACHING_URL}:8081@g" /etc/apt/sources.list
-          pip config set global.index-url http://${CACHING_URL}/pypi/simple
-          pip config set global.trusted-host "${CACHING_URL}"
-
-          bash scripts/ci/npu/npu_ci_install_dependency.sh a3
-          # copy required file from our daily cache
-          cp ~/.cache/modelscope/hub/datasets/otavia/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json /tmp
-          # copy gsm8k dataset
-          cp ~/.cache/modelscope/hub/datasets/tmp/test.jsonl /tmp
-
-      - name: Run test
-        timeout-minutes: 60
-        env:
-          SGLANG_USE_MODELSCOPE: true
-          SGLANG_IS_IN_CI: true
-          HF_ENDPOINT: https://hf-mirror.com
-          TORCH_EXTENSIONS_DIR: /tmp/torch_extensions
-          PYTORCH_NPU_ALLOC_CONF: "expandable_segments:True"
-          STREAMS_PER_DEVICE: 32
-          SGLANG_DIFFUSION_ARTIFACT_DIR: ${{ github.workspace }}/diffusion-failures
-        run: |
-          cd python
-          python3 sglang/multimodal_gen/test/run_suite.py --suite 2-npu
-
-      - name: Upload diffusion failure artifacts
-        if: always()
-        uses: actions/upload-artifact@v4
-        with:
-          name: diffusion-failures-npu-2-${{ github.run_attempt }}
-          path: diffusion-failures/
-          if-no-files-found: ignore
-          retention-days: 7
-
-  pr-test-npu-finish:
-    needs:
-      [
-        check-changes,
-
-        stage-b-test-1-npu-a2,
-        stage-b-test-2-npu-a2,
-        stage-b-test-4-npu-a3,
-        stage-b-test-16-npu-a3,
-
-        multimodal-gen-test-1-npu-a3,
-        multimodal-gen-test-2-npu-a3,
-      ]
-    if: always()
-    runs-on: ubuntu-latest
-    steps:
-      - name: Check all dependent job statuses
-        run: |
-          # Convert the 'needs' context to a JSON string
-          json_needs='${{ toJson(needs) }}'
-
-          # Get a list of all job names from the JSON keys
-          job_names=$(echo "$json_needs" | jq -r 'keys_unsorted[]')
-
-          for job in $job_names; do
-            # For each job, extract its result
-            result=$(echo "$json_needs" | jq -r --arg j "$job" '.[$j].result')
-
-            # Print the job name and its result
-            echo "$job: $result"
-
-            # Check for failure or cancellation and exit if found
-            if [[ "$result" == "failure" || "$result" == "cancelled" ]]; then
-              echo "The above jobs failed."
-              exit 1
-            fi
-          done
-          # If the loop completes, all jobs were successful
-          echo "All jobs completed successfully"
-          exit 0
-||||||| official previous@88db9e033a11
-name: PR Test (NPU)
-
-on:
-  push:
-    branches: [ main ]
-  pull_request:
-  workflow_dispatch:
-  workflow_call:
-    inputs:
-      ref:
-        description: 'Git ref (branch, tag, or SHA) to test. If not provided, uses the default branch.'
-        required: false
-        type: string
-        default: ''
-      run_all_tests:
-        description: "Run all tests (for releasing or testing purpose)"
-        required: false
-        type: boolean
-        default: false
-
-concurrency:
-  group: pr-test-npu-${{ inputs.ref || github.ref }}
-  cancel-in-progress: ${{ github.event_name != 'workflow_call' }}
-
-jobs:
-  # ==================== Check Changes ==================== #
-  check-changes:
-    runs-on: ubuntu-latest
-    outputs:
-      changes_exist: ${{ steps.filter.outputs.main_package == 'true' || steps.filter.outputs.multimodal_gen == 'true' || steps.run-mode.outputs.run_all_tests == 'true'}}
-      main_package: ${{ steps.filter.outputs.main_package == 'true' || steps.run-mode.outputs.run_all_tests == 'true' }}
-      multimodal_gen: ${{ steps.filter.outputs.multimodal_gen == 'true' || steps.run-mode.outputs.run_all_tests == 'true' }}
-    steps:
-      - name: Checkout code
-        uses: actions/checkout@v4
-        with:
-          ref: ${{ inputs.ref || github.ref }}
-
-      - name: Determine run mode
-        id: run-mode
-        run: |
-          # Run all tests for workflow_call (when ref input is provided)
-          # Note: github.event_name is inherited from caller, so we detect workflow_call by checking inputs.ref
-          if [[ "${{ inputs.run_all_tests }}" == "true" ]]; then
-            echo "run_all_tests=true" >> $GITHUB_OUTPUT
-            echo "Run mode: ALL TESTS (run_all_tests=${{ inputs.run_all_tests }})"
-          else
-            echo "run_all_tests=false" >> $GITHUB_OUTPUT
-            echo "Run mode: FILTERED (triggered by ${{ github.event_name }})"
-          fi
-
-      - name: Detect file changes
-        id: filter
-        uses: dorny/paths-filter@v3
-        if: steps.run-mode.outputs.run_all_tests != 'true'
-        with:
-          filters: |
-            main_package:
-              - "python/sglang/!(multimodal_gen)/**/!(*.md)"
-              - "python/pyproject_npu.toml"
-              - "scripts/ci/npu/npu_ci_install_dependency.sh"
-              - "test/registered/ascend/**"
-              - ".github/workflows/pr-test-npu.yml"
-            multimodal_gen:
-              - "python/sglang/multimodal_gen/**/!(*.md|*.ipynb)"
-              - "python/sglang/jit_kernel/diffusion/triton/npu_fallback.py"
-              - "python/sglang/srt/**"
-              - "python/pyproject_npu.toml"
-              - "scripts/ci/npu/npu_ci_install_dependency.sh"
-              - ".github/workflows/pr-test-npu.yml"
-
-  # ==================== PR Gate ==================== #
-  pr-gate:
-    needs: check-changes
-    if: needs.check-changes.outputs.changes_exist == 'true'
-    uses: ./.github/workflows/pr-gate.yml
-    secrets: inherit
-
-  set-image-config:
-    runs-on: ubuntu-latest
-    outputs:
-      CANN_image_a3: ${{ steps.set-vars.outputs.CANN_image_a3 }}
-      CANN_image_910b: ${{ steps.set-vars.outputs.CANN_image_910b }}
-    steps:
-      # When triggered by PR, no inputs parameters are used. The latest community code is tested by default.
-      - name: Set image config
-        id: set-vars
-        run: |
-          echo "CANN_image_a3=swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:9.0.0-a3-ubuntu22.04-py3.11" >> $GITHUB_OUTPUT
-          echo "CANN_image_910b=swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:9.0.0-910b-ubuntu22.04-py3.11" >> $GITHUB_OUTPUT
-
-  stage-b-test-1-npu-a2:
-    needs: [check-changes, pr-gate, set-image-config]
-    if: needs.check-changes.outputs.main_package == 'true'
-    runs-on: linux-aarch64-a2-1
-    strategy:
-      fail-fast: false
-      matrix:
-        part: [ 0, 1 ]
-    container:
-      image: ${{ needs.set-image-config.outputs.CANN_image_910b }}
-    steps:
-      - name: Checkout code
-        uses: actions/checkout@v4
-        with:
-          ref: ${{ inputs.ref || github.ref }}
-
-      - name: Mark repository safe
-        run: |
-          git config --system --add safe.directory ${GITHUB_WORKSPACE}
-
-      - name: Install dependencies
-        env:
-          TORCH_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/whl/cpu"
-          PYPI_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple"
-          UV_INDEX_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple"
-          GITHUB_PROXY_URL: "https://gh-proxy.test.osinfra.cn/"
-          RUSTUP_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local:8082"
-        run: |
-          # speed up by using infra cache services
-          CACHING_URL="cache-service.nginx-pypi-cache.svc.cluster.local"
-          sed -Ei "s@(ports|archive).ubuntu.com@${CACHING_URL}:8081@g" /etc/apt/sources.list
-          pip config set global.index-url http://${CACHING_URL}/pypi/simple
-          pip config set global.trusted-host "${CACHING_URL}"
-
-          bash scripts/ci/npu/npu_ci_install_dependency.sh 910b
-          # copy required file from our daily cache
-          cp ~/.cache/modelscope/hub/datasets/otavia/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json /tmp
-          # copy gsm8k dataset
-          cp ~/.cache/modelscope/hub/datasets/tmp/test.jsonl /tmp
-
-      - name: Run test
-        timeout-minutes: 60
-        env:
-          SGLANG_USE_MODELSCOPE: true
-          SGLANG_IS_IN_CI: true
-          HF_ENDPOINT: https://hf-mirror.com
-          TORCH_EXTENSIONS_DIR: /tmp/torch_extensions
-          PYTORCH_NPU_ALLOC_CONF: "expandable_segments:True"
-          STREAMS_PER_DEVICE: 32
-        run: |
-          cd test
-          python3 run_suite.py --hw npu --suite stage-b-test-1-npu-a2 --auto-partition-id ${{ matrix.part }} --auto-partition-size 2
-
-  stage-b-test-2-npu-a2:
-    needs: [check-changes, pr-gate, set-image-config]
-    if: needs.check-changes.outputs.main_package == 'true'
-    runs-on: linux-aarch64-a2-2
-    strategy:
-      fail-fast: true
-      matrix:
-        part: [0, 1]
-    container:
-      image: ${{ needs.set-image-config.outputs.CANN_image_910b }}
-    steps:
-      - name: Checkout code
-        uses: actions/checkout@v4
-        with:
-          ref: ${{ inputs.ref || github.ref }}
-
-      - name: Mark repository safe
-        run: |
-          git config --system --add safe.directory ${GITHUB_WORKSPACE}
-
-      - name: Install dependencies
-        env:
-          TORCH_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/whl/cpu"
-          PYPI_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple"
-          UV_INDEX_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple"
-          GITHUB_PROXY_URL: "https://gh-proxy.test.osinfra.cn/"
-          RUSTUP_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local:8082"
-        run: |
-          # speed up by using infra cache services
-          CACHING_URL="cache-service.nginx-pypi-cache.svc.cluster.local"
-          sed -Ei "s@(ports|archive).ubuntu.com@${CACHING_URL}:8081@g" /etc/apt/sources.list
-          pip config set global.index-url http://${CACHING_URL}/pypi/simple
-          pip config set global.trusted-host "${CACHING_URL}"
-
-          bash scripts/ci/npu/npu_ci_install_dependency.sh 910b
-          # copy required file from our daily cache
-          cp ~/.cache/modelscope/hub/datasets/otavia/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json /tmp
-          # copy gsm8k dataset
-          cp ~/.cache/modelscope/hub/datasets/tmp/test.jsonl /tmp
-
-      - name: Run test
-        timeout-minutes: 60
-        env:
-          SGLANG_USE_MODELSCOPE: true
-          SGLANG_IS_IN_CI: true
-          HF_ENDPOINT: https://hf-mirror.com
-          TORCH_EXTENSIONS_DIR: /tmp/torch_extensions
-          PYTORCH_NPU_ALLOC_CONF: "expandable_segments:True"
-          STREAMS_PER_DEVICE: 32
-        run: |
-          cd test
-          python3 run_suite.py --hw npu --suite stage-b-test-2-npu-a2 --auto-partition-id ${{ matrix.part }} --auto-partition-size 2
-
-  stage-b-test-4-npu-a3:
-    needs: [check-changes, pr-gate, set-image-config]
-    if: needs.check-changes.outputs.main_package == 'true'
-    runs-on: linux-aarch64-a3-4
-    container:
-      image: ${{ needs.set-image-config.outputs.CANN_image_a3 }}
-    steps:
-      - name: Checkout code
-        uses: actions/checkout@v4
-        with:
-          ref: ${{ inputs.ref || github.ref }}
-
-      - name: Mark repository safe
-        run: |
-          git config --system --add safe.directory ${GITHUB_WORKSPACE}
-
-      - name: Install dependencies
-        env:
-          TORCH_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/whl/cpu"
-          PYPI_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple"
-          UV_INDEX_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple"
-          GITHUB_PROXY_URL: "https://gh-proxy.test.osinfra.cn/"
-          RUSTUP_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local:8082"
-        run: |
-          # speed up by using infra cache services
-          CACHING_URL="cache-service.nginx-pypi-cache.svc.cluster.local"
-          sed -Ei "s@(ports|archive).ubuntu.com@${CACHING_URL}:8081@g" /etc/apt/sources.list
-          pip config set global.index-url http://${CACHING_URL}/pypi/simple
-          pip config set global.trusted-host "${CACHING_URL}"
-
-          bash scripts/ci/npu/npu_ci_install_dependency.sh a3
-          # copy required file from our daily cache
-          cp ~/.cache/modelscope/hub/datasets/otavia/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json /tmp
-          # copy gsm8k dataset
-          cp ~/.cache/modelscope/hub/datasets/tmp/test.jsonl /tmp
-
-      - name: Run test
-        timeout-minutes: 60
-        env:
-          SGLANG_USE_MODELSCOPE: true
-          SGLANG_IS_IN_CI: true
-          HF_ENDPOINT: https://hf-mirror.com
-          TORCH_EXTENSIONS_DIR: /tmp/torch_extensions
-          PYTORCH_NPU_ALLOC_CONF: "expandable_segments:True"
-          STREAMS_PER_DEVICE: 32
-        run: |
-          cd test
-          python3 run_suite.py --hw npu --suite stage-b-test-4-npu-a3 --timeout-per-file 3600
-
-
-  stage-b-test-16-npu-a3:
-    needs: [check-changes, pr-gate, set-image-config]
-    if: needs.check-changes.outputs.main_package == 'true'
-    runs-on: linux-aarch64-a3-16
-    container:
-      image: ${{ needs.set-image-config.outputs.CANN_image_a3 }}
-    steps:
-      - name: Checkout code
-        uses: actions/checkout@v4
-        with:
-          ref: ${{ inputs.ref || github.ref }}
-
-      - name: Mark repository safe
-        run: |
-          git config --system --add safe.directory ${GITHUB_WORKSPACE}
-
-      - name: Install dependencies
-        env:
-          TORCH_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/whl/cpu"
-          PYPI_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple"
-          UV_INDEX_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple"
-          GITHUB_PROXY_URL: "https://gh-proxy.test.osinfra.cn/"
-          RUSTUP_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local:8082"
-        run: |
-          # speed up by using infra cache services
-          CACHING_URL="cache-service.nginx-pypi-cache.svc.cluster.local"
-          sed -Ei "s@(ports|archive).ubuntu.com@${CACHING_URL}:8081@g" /etc/apt/sources.list
-          pip config set global.index-url http://${CACHING_URL}/pypi/simple
-          pip config set global.trusted-host "${CACHING_URL}"
-
-          bash scripts/ci/npu/npu_ci_install_dependency.sh a3
-          # copy required file from our daily cache
-          cp ~/.cache/modelscope/hub/datasets/otavia/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json /tmp
-          # copy gsm8k dataset
-          cp ~/.cache/modelscope/hub/datasets/tmp/test.jsonl /tmp
-
-      - name: Run test
-        timeout-minutes: 60
-        env:
-          SGLANG_USE_MODELSCOPE: true
-          SGLANG_IS_IN_CI: true
-          HF_ENDPOINT: https://hf-mirror.com
-          TORCH_EXTENSIONS_DIR: /tmp/torch_extensions
-          PYTORCH_NPU_ALLOC_CONF: "expandable_segments:True"
-          STREAMS_PER_DEVICE: 32
-        run: |
-          cd test
-          python3 run_suite.py --hw npu --suite stage-b-test-16-npu-a3 --timeout-per-file 3600
-
-  multimodal-gen-test-1-npu-a3:
-    needs: [check-changes, pr-gate, set-image-config]
-    if: needs.check-changes.outputs.multimodal_gen == 'true'
-    runs-on: linux-aarch64-a3-2
-    container:
-      image: ${{ needs.set-image-config.outputs.CANN_image_a3 }}
-    steps:
-      - name: Checkout code
-        uses: actions/checkout@v4
-
-      - name: Mark repository safe
-        run: |
-          git config --system --add safe.directory ${GITHUB_WORKSPACE}
-
-      - name: Install dependencies
-        env:
-          TORCH_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/whl/cpu"
-          PYPI_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple"
-          UV_INDEX_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple"
-          GITHUB_PROXY_URL: "https://gh-proxy.test.osinfra.cn/"
-          RUSTUP_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local:8082"
-        run: |
-          # speed up by using infra cache services
-          CACHING_URL="cache-service.nginx-pypi-cache.svc.cluster.local"
-          sed -Ei "s@(ports|archive).ubuntu.com@${CACHING_URL}:8081@g" /etc/apt/sources.list
-          pip config set global.index-url http://${CACHING_URL}/pypi/simple
-          pip config set global.trusted-host "${CACHING_URL}"
-
-          bash scripts/ci/npu/npu_ci_install_dependency.sh a3
-          # copy required file from our daily cache
-          cp ~/.cache/modelscope/hub/datasets/otavia/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json /tmp
-          # copy gsm8k dataset
-          cp ~/.cache/modelscope/hub/datasets/tmp/test.jsonl /tmp
-
-      - name: Run test
-        timeout-minutes: 60
-        env:
-          SGLANG_USE_MODELSCOPE: true
-          SGLANG_IS_IN_CI: true
-          HF_ENDPOINT: https://hf-mirror.com
-          TORCH_EXTENSIONS_DIR: /tmp/torch_extensions
-          PYTORCH_NPU_ALLOC_CONF: "expandable_segments:True"
-          STREAMS_PER_DEVICE: 32
-          SGLANG_DIFFUSION_ARTIFACT_DIR: ${{ github.workspace }}/diffusion-failures
-        run: |
-          cd python
-          python3 sglang/multimodal_gen/test/run_suite.py --suite 1-npu
-
-      - name: Upload diffusion failure artifacts
-        if: always()
-        uses: actions/upload-artifact@v4
-        with:
-          name: diffusion-failures-npu-1-${{ github.run_attempt }}
-          path: diffusion-failures/
-          if-no-files-found: ignore
-          retention-days: 7
-
-  multimodal-gen-test-2-npu-a3:
-    needs: [check-changes, pr-gate, set-image-config]
-    if: needs.check-changes.outputs.multimodal_gen == 'true'
-    runs-on: linux-aarch64-a3-16
-    container:
-      image: ${{ needs.set-image-config.outputs.CANN_image_a3 }}
-    steps:
-      - name: Checkout code
-        uses: actions/checkout@v4
-
-      - name: Mark repository safe
-        run: |
-          git config --system --add safe.directory ${GITHUB_WORKSPACE}
-
-      - name: Install dependencies
-        env:
-          TORCH_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/whl/cpu"
-          PYPI_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple"
-          UV_INDEX_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple"
-          GITHUB_PROXY_URL: "https://gh-proxy.test.osinfra.cn/"
-          RUSTUP_CACHE_URL: "http://cache-service.nginx-pypi-cache.svc.cluster.local:8082"
-        run: |
-          # speed up by using infra cache services
-          CACHING_URL="cache-service.nginx-pypi-cache.svc.cluster.local"
-          sed -Ei "s@(ports|archive).ubuntu.com@${CACHING_URL}:8081@g" /etc/apt/sources.list
-          pip config set global.index-url http://${CACHING_URL}/pypi/simple
-          pip config set global.trusted-host "${CACHING_URL}"
-
-          bash scripts/ci/npu/npu_ci_install_dependency.sh a3
-          # copy required file from our daily cache
-          cp ~/.cache/modelscope/hub/datasets/otavia/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json /tmp
-          # copy gsm8k dataset
-          cp ~/.cache/modelscope/hub/datasets/tmp/test.jsonl /tmp
-
-      - name: Run test
-        timeout-minutes: 60
-        env:
-          SGLANG_USE_MODELSCOPE: true
-          SGLANG_IS_IN_CI: true
-          HF_ENDPOINT: https://hf-mirror.com
-          TORCH_EXTENSIONS_DIR: /tmp/torch_extensions
-          PYTORCH_NPU_ALLOC_CONF: "expandable_segments:True"
-          STREAMS_PER_DEVICE: 32
-          SGLANG_DIFFUSION_ARTIFACT_DIR: ${{ github.workspace }}/diffusion-failures
-        run: |
-          cd python
-          python3 sglang/multimodal_gen/test/run_suite.py --suite 2-npu
-
-      - name: Upload diffusion failure artifacts
-        if: always()
-        uses: actions/upload-artifact@v4
-        with:
-          name: diffusion-failures-npu-2-${{ github.run_attempt }}
-          path: diffusion-failures/
-          if-no-files-found: ignore
-          retention-days: 7
-
-  pr-test-npu-finish:
-    needs:
-      [
-        check-changes,
-
-        stage-b-test-1-npu-a2,
-        stage-b-test-2-npu-a2,
-        stage-b-test-4-npu-a3,
-        stage-b-test-16-npu-a3,
-
-        multimodal-gen-test-1-npu-a3,
-        multimodal-gen-test-2-npu-a3,
-      ]
-    if: always()
-    runs-on: ubuntu-latest
-    steps:
-      - name: Check all dependent job statuses
-        run: |
-          # Convert the 'needs' context to a JSON string
-          json_needs='${{ toJson(needs) }}'
-
-          # Get a list of all job names from the JSON keys
-          job_names=$(echo "$json_needs" | jq -r 'keys_unsorted[]')
-
-          for job in $job_names; do
-            # For each job, extract its result
-            result=$(echo "$json_needs" | jq -r --arg j "$job" '.[$j].result')
-
-            # Print the job name and its result
-            echo "$job: $result"
-
-            # Check for failure or cancellation and exit if found
-            if [[ "$result" == "failure" || "$result" == "cancelled" ]]; then
-              echo "The above jobs failed."
-              exit 1
-            fi
-          done
-          # If the loop completes, all jobs were successful
-          echo "All jobs completed successfully"
-          exit 0
-=======
 name: PR Test (NPU)
 
 on:
   push:
     branches: [ main ]
   pull_request:
   workflow_dispatch:
   workflow_call:
@@ -1374,9 +471,8 @@
             if [[ "$result" == "failure" || "$result" == "cancelled" ]]; then
               echo "The above jobs failed."
               exit 1
             fi
           done
           # If the loop completes, all jobs were successful
           echo "All jobs completed successfully"
           exit 0
->>>>>>> official target@9a6f8e599204
~~~~

</details>


<details>
<summary><code>python/sglang/jit_kernel/csrc/deepseek_v4/topk_v2.cuh</code> — 2 conflict hunks</summary>

**Resolution intent:** Accept the official runtime-topk rewrite while retaining the existing HIP/DCU backend selection outside this file.

~~~~diff
--- AUTO-CONFLICT/python/sglang/jit_kernel/csrc/deepseek_v4/topk_v2.cuh
+++ RESOLVED/python/sglang/jit_kernel/csrc/deepseek_v4/topk_v2.cuh
@@ -304,289 +304,21 @@
     if (sl > cluster_threshold) {
       const auto pos = atomicAdd(&s_count, 1);
       metadata[1 + pos] = {i, sl};
     }
   }
   __syncthreads();
   if (tx == 0) {
     auto* g = reinterpret_cast<GlobalMetadata*>(metadata);
-<<<<<<< DCU main@726ca92425c0
-    *g = {
-        .cluster_threshold = cluster_threshold,
-        .num_cluster_items = N,
-        .reserved = {0, 0},
-    };
-  }
-}
-
-SMALL_TOPK_KERNEL void  // short context
-topk_short_transform(const SGL_GRID_CONSTANT TopKParams params) {
-  alignas(128) extern __shared__ uint8_t smem[];
-  __shared__ int32_t s_topk_indices[K];
-  const auto batch_id = blockIdx.x;
-  const auto seq_len = params.seq_lens[batch_id];
-  const auto transform = params.get_transform(batch_id, s_topk_indices);
-  // trivial case
-  if (seq_len <= K) {
-    impl::trivial_transform(transform, seq_len, K);
-  } else {
-    Small::run(params.get_scores(batch_id), s_topk_indices, seq_len, smem, /*use_pdl=*/true);
-    device::PDLTriggerSecondary<true>();
-    Small::transform(transform);
-  }
-}
-
-LARGE_TOPK_STAGE_1 void  // long context, middle to large batch size
-topk_combine_preprocess(const SGL_GRID_CONSTANT TopKParams params) {
-  alignas(128) extern __shared__ uint8_t smem[];
-  __shared__ int32_t s_topk_indices[K];
-  uint32_t work_id = blockIdx.x;
-  uint32_t batch_id;
-  uint32_t seq_len;
-  bool has_next;
-  uint32_t length;
-  uint32_t offset;
-  const auto cluster_rank = blockIdx.y;
-
-  const auto prefetch_metadata = [&] {
-    const auto metadata = params.get_item_metadata(work_id);
-    batch_id = metadata.batch_id;
-    seq_len = metadata.seq_len;
-    has_next = metadata.has_next;
-    work_id += kNumClusters;  // advance to the next item for this cluster
-  };
-  const auto launch_prologue = [&] {
-    const auto partition = partition_work(seq_len, cluster_rank);
-    offset = partition.x;
-    length = partition.y;
-    Large::stage1_prologue(params.get_scores(batch_id) + offset, length, smem);
-  };
-
-  device::PDLWaitPrimary<true>();
-  device::PDLTriggerSecondary<true>();
-
-  prefetch_metadata();
-  if (seq_len == 0) return;
-  Large::stage1_init(smem);
-  launch_prologue();
-  while (true) {
-    const auto this_length = length;
-    const auto this_offset = offset;
-    const auto need_prefetch = has_next;
-    const auto transform = params.get_transform(batch_id, s_topk_indices);
-    const auto ws = params.workspace + batch_id * params.workspace_stride;
-    if (need_prefetch) prefetch_metadata();
-    Large::stage1(s_topk_indices, this_length, smem, /*reuse=*/true);
-    if (need_prefetch) launch_prologue();
-    Large::stage1_epilogue(transform, this_offset, ws, smem);
-    if (!need_prefetch) break;
-  }
-}
-
-LARGE_TOPK_STAGE_2 void  // long context, middle to large batch size
-topk_combine_transform(const SGL_GRID_CONSTANT TopKParams params) {
-  alignas(128) extern __shared__ uint8_t smem[];
-  __shared__ int32_t s_topk_indices[K];
-  const auto batch_id = blockIdx.x;
-  const auto seq_len = params.seq_lens[batch_id];
-  const auto cluster_threshold = params.get_global_metadata().cluster_threshold;
-  const auto transform = params.get_transform(batch_id, s_topk_indices);
-  if (seq_len <= K) {
-    impl::trivial_transform(transform, seq_len, K);
-  } else if (seq_len <= kMax2PassLength) {
-    if (seq_len <= Small::kMax1PassLength) {
-      Small::run(params.get_scores(batch_id), s_topk_indices, seq_len, smem);
-    } else {
-      __syncwarp();
-      Small::run<true>(params.get_scores(batch_id), s_topk_indices, seq_len, smem);
-    }
-    Small::transform(transform);
-  } else if (seq_len <= cluster_threshold) {
-    Medium::run(params.get_scores(batch_id), seq_len, s_topk_indices, smem);
-    Medium::transform(transform, smem);
-  } else {
-    const auto ws = params.workspace + batch_id * params.workspace_stride;
-    device::PDLWaitPrimary<true>();
-    Large::transform(transform, ws, smem);
-||||||| official previous@88db9e033a11
-    *g = {
-        .cluster_threshold = cluster_threshold,
-        .num_cluster_items = N,
-        .reserved = {0, 0},
-    };
-  }
-}
-
-SMALL_TOPK_KERNEL void  // short context
-topk_short_transform(const __grid_constant__ TopKParams params) {
-  alignas(128) extern __shared__ uint8_t smem[];
-  __shared__ int32_t s_topk_indices[K];
-  const auto batch_id = blockIdx.x;
-  const auto seq_len = params.seq_lens[batch_id];
-  const auto transform = params.get_transform(batch_id, s_topk_indices);
-  // trivial case
-  if (seq_len <= K) {
-    impl::trivial_transform(transform, seq_len, K);
-  } else {
-    Small::run(params.get_scores(batch_id), s_topk_indices, seq_len, smem, /*use_pdl=*/true);
-    device::PDLTriggerSecondary<true>();
-    Small::transform(transform);
-  }
-}
-
-LARGE_TOPK_STAGE_1 void  // long context, middle to large batch size
-topk_combine_preprocess(const __grid_constant__ TopKParams params) {
-  alignas(128) extern __shared__ uint8_t smem[];
-  __shared__ int32_t s_topk_indices[K];
-  uint32_t work_id = blockIdx.x;
-  uint32_t batch_id;
-  uint32_t seq_len;
-  bool has_next;
-  uint32_t length;
-  uint32_t offset;
-  const auto cluster_rank = blockIdx.y;
-
-  const auto prefetch_metadata = [&] {
-    const auto metadata = params.get_item_metadata(work_id);
-    batch_id = metadata.batch_id;
-    seq_len = metadata.seq_len;
-    has_next = metadata.has_next;
-    work_id += kNumClusters;  // advance to the next item for this cluster
-  };
-  const auto launch_prologue = [&] {
-    const auto partition = partition_work(seq_len, cluster_rank);
-    offset = partition.x;
-    length = partition.y;
-    Large::stage1_prologue(params.get_scores(batch_id) + offset, length, smem);
-  };
-
-  device::PDLWaitPrimary<true>();
-  device::PDLTriggerSecondary<true>();
-
-  prefetch_metadata();
-  if (seq_len == 0) return;
-  Large::stage1_init(smem);
-  launch_prologue();
-  while (true) {
-    const auto this_length = length;
-    const auto this_offset = offset;
-    const auto need_prefetch = has_next;
-    const auto transform = params.get_transform(batch_id, s_topk_indices);
-    const auto ws = params.workspace + batch_id * params.workspace_stride;
-    if (need_prefetch) prefetch_metadata();
-    Large::stage1(s_topk_indices, this_length, smem, /*reuse=*/true);
-    if (need_prefetch) launch_prologue();
-    Large::stage1_epilogue(transform, this_offset, ws, smem);
-    if (!need_prefetch) break;
-  }
-}
-
-LARGE_TOPK_STAGE_2 void  // long context, middle to large batch size
-topk_combine_transform(const __grid_constant__ TopKParams params) {
-  alignas(128) extern __shared__ uint8_t smem[];
-  __shared__ int32_t s_topk_indices[K];
-  const auto batch_id = blockIdx.x;
-  const auto seq_len = params.seq_lens[batch_id];
-  const auto cluster_threshold = params.get_global_metadata().cluster_threshold;
-  const auto transform = params.get_transform(batch_id, s_topk_indices);
-  if (seq_len <= K) {
-    impl::trivial_transform(transform, seq_len, K);
-  } else if (seq_len <= kMax2PassLength) {
-    if (seq_len <= Small::kMax1PassLength) {
-      Small::run(params.get_scores(batch_id), s_topk_indices, seq_len, smem);
-    } else {
-      __syncwarp();
-      Small::run<true>(params.get_scores(batch_id), s_topk_indices, seq_len, smem);
-    }
-    Small::transform(transform);
-  } else if (seq_len <= cluster_threshold) {
-    Medium::run(params.get_scores(batch_id), seq_len, s_topk_indices, smem);
-    Medium::transform(transform, smem);
-  } else {
-    const auto ws = params.workspace + batch_id * params.workspace_stride;
-    device::PDLWaitPrimary<true>();
-    Large::transform(transform, ws, smem);
-=======
     *g = {.cluster_threshold = cluster_threshold, .num_cluster_items = s_count};
->>>>>>> official target@9a6f8e599204
-  }
-}
-
-<<<<<<< DCU main@726ca92425c0
-FUSED_COMBINE_KERNEL void  // long context, small batch size
-topk_fused_transform(const SGL_GRID_CONSTANT TopKParams params) {
-  alignas(128) extern __shared__ uint8_t smem[];
-  __shared__ int32_t s_topk_indices[K];
-  const auto batch_id = blockIdx.x;
-  const auto cluster_rank = blockIdx.y;
-  const auto seq_len = params.seq_lens[batch_id];
-  const auto transform = params.get_transform(batch_id, s_topk_indices);
-  if (seq_len <= K) {
-    if (cluster_rank != 0) return;  // only first rank work
-    impl::trivial_transform(transform, seq_len, K);
-  } else if (seq_len <= Small::kMax1PassLength) {
-    if (cluster_rank != 0) return;  // only first rank work
-    Small::run(params.get_scores(batch_id), s_topk_indices, seq_len, smem, /*use_pdl=*/true);
-    Small::transform(transform);
-  } else {
-    const auto [offset, length] = partition_work(seq_len, cluster_rank);
-    const auto ws = params.workspace + batch_id * params.workspace_stride;
-    Large::stage1_init(smem);
-    device::PDLWaitPrimary<true>();
-    Large::stage1_prologue(params.get_scores(batch_id) + offset, length, smem);
-    Large::stage1(s_topk_indices, length, smem);
-    Large::stage1_epilogue(transform, offset, ws, smem);
-    cooperative_groups::this_cluster().sync();
-    if (cluster_rank != 0) return;  // only first rank do the stage-2
-    Large::transform(transform, ws, smem);
-  }
-}
-
-struct CombinedTopKKernel {
-  static constexpr auto kStage1SMEM = sizeof(Large::Smem) + 128;
-  static constexpr auto kStage2SMEM = std::max(sizeof(Small::Smem), sizeof(Medium::Smem)) + 128;
-
-||||||| official previous@88db9e033a11
-FUSED_COMBINE_KERNEL void  // long context, small batch size
-topk_fused_transform(const __grid_constant__ TopKParams params) {
-  alignas(128) extern __shared__ uint8_t smem[];
-  __shared__ int32_t s_topk_indices[K];
-  const auto batch_id = blockIdx.x;
-  const auto cluster_rank = blockIdx.y;
-  const auto seq_len = params.seq_lens[batch_id];
-  const auto transform = params.get_transform(batch_id, s_topk_indices);
-  if (seq_len <= K) {
-    if (cluster_rank != 0) return;  // only first rank work
-    impl::trivial_transform(transform, seq_len, K);
-  } else if (seq_len <= Small::kMax1PassLength) {
-    if (cluster_rank != 0) return;  // only first rank work
-    Small::run(params.get_scores(batch_id), s_topk_indices, seq_len, smem, /*use_pdl=*/true);
-    Small::transform(transform);
-  } else {
-    const auto [offset, length] = partition_work(seq_len, cluster_rank);
-    const auto ws = params.workspace + batch_id * params.workspace_stride;
-    Large::stage1_init(smem);
-    device::PDLWaitPrimary<true>();
-    Large::stage1_prologue(params.get_scores(batch_id) + offset, length, smem);
-    Large::stage1(s_topk_indices, length, smem);
-    Large::stage1_epilogue(transform, offset, ws, smem);
-    cooperative_groups::this_cluster().sync();
-    if (cluster_rank != 0) return;  // only first rank do the stage-2
-    Large::transform(transform, ws, smem);
-  }
-}
-
-struct CombinedTopKKernel {
-  static constexpr auto kStage1SMEM = sizeof(Large::Smem) + 128;
-  static constexpr auto kStage2SMEM = std::max(sizeof(Small::Smem), sizeof(Medium::Smem)) + 128;
-
-=======
+  }
+}
+
 struct TopKKernel {
->>>>>>> official target@9a6f8e599204
   static void plan(  //
       const tvm::ffi::TensorView seq_lens,
       const tvm::ffi::TensorView metadata,
       const uint32_t static_cluster_threshold) {
     using namespace host;
     auto B = SymbolicSize{"batch_size"};
     auto Bp1 = SymbolicSize{"batch_size_plus_1"};
     auto device_ = SymbolicDevice{};
~~~~

</details>


<details>
<summary><code>python/sglang/srt/layers/attention/dsa/dsa_indexer.py</code> — 8 conflict hunks</summary>

**Resolution intent:** Adopt official instance-scoped fusion and paged-MQA wrappers while preserving DCU-first LightOp, BF16/FP8 cache, page-size-64, and logits paths.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/attention/dsa/dsa_indexer.py
+++ RESOLVED/python/sglang/srt/layers/attention/dsa/dsa_indexer.py
@@ -461,17 +461,20 @@
             rope_scaling=rope_scaling,
             is_neox_style=is_neox_style,
             device=get_global_server_args().device,
         )
         self.block_size = block_size
         self.scale_fmt = scale_fmt
         self.softmax_scale = self.head_dim**-0.5
 
-<<<<<<< DCU main@726ca92425c0
+        self.paged_mqa_logits_backend = DSAPagedMQALogitsBackend.resolve(
+            get_server_args().dsa_paged_mqa_logits_backend
+        )
+
     def _use_dcu_bf16_index_cache(self, forward_batch: ForwardBatch) -> bool:
         return _is_dcu and not getattr(
             get_token_to_kv_pool(), "use_fp8_index_k_cache", True
         )
 
     def _get_gate_input_tensor(
         self, x: torch.Tensor | tuple[torch.Tensor, ...]
     ) -> torch.Tensor:
@@ -505,23 +508,16 @@
         self, x: torch.Tensor | tuple[torch.Tensor, ...]
     ) -> torch.Tensor:
         x_for_gate = self._get_gate_input_tensor(x)
         return (
             self._project_and_scale_head_gates(x_for_gate).unsqueeze(-1)
             * self.softmax_scale
         )
 
-||||||| official previous@88db9e033a11
-=======
-        self.paged_mqa_logits_backend = DSAPagedMQALogitsBackend.resolve(
-            get_server_args().dsa_paged_mqa_logits_backend
-        )
-
->>>>>>> official target@9a6f8e599204
     @contextlib.contextmanager
     def _with_real_sm_count(self):
         # When pipeline parallelism is enabled, each PP rank initiates a recv operation after the _pp_launch_batch
         # request to receive the PP proxy tensor or output from the previous stage, occupying one SM resource.
         # Model execution runs in parallel with the recv operation, so the SMs available to the indexer must be reduced
         # by 1. Currently, the last rank starts the send result + recv request only after waiting for execution results.
         if self.logits_with_pp_recv:
             pp_recv_sm_count = 1
@@ -653,17 +649,17 @@
                     query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
                     q_rope, _ = torch.split(
                         query,
                         [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                         dim=-1,
                     )
                 with torch.cuda.stream(self.alt_stream):
                     # TODO we should also put DeepGEMM half SM here?
-                    if _use_dsa_indexer_fusion:
+                    if self.use_dsa_indexer_fusion:
                         key, weights_raw = self._fused_k_weights(x)
                     else:
                         key, _ = self.wk(x)
                     key = self.k_norm(key)
 
                     k_rope, _ = torch.split(
                         key,
                         [self.rope_head_dim, self.head_dim - self.rope_head_dim],
@@ -674,76 +670,28 @@
             else:
                 query, _ = self.wq_b(q_lora)
                 query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
                 q_rope, _ = torch.split(
                     query,
                     [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                     dim=-1,
                 )
-<<<<<<< DCU main@726ca92425c0
-                if _use_dsa_indexer_fusion:
-||||||| official previous@88db9e033a11
-            with torch.cuda.stream(self.alt_stream):
-                # TODO we should also put DeepGEMM half SM here?
-                if _use_dsa_indexer_fusion:
-=======
-            with torch.cuda.stream(self.alt_stream):
-                # TODO we should also put DeepGEMM half SM here?
                 if self.use_dsa_indexer_fusion:
->>>>>>> official target@9a6f8e599204
                     key, weights_raw = self._fused_k_weights(x)
                 else:
                     key, _ = self.wk(x)
                 key = self.k_norm(key)
                 k_rope, _ = torch.split(
                     key,
                     [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                     dim=-1,
                 )
 
-<<<<<<< DCU main@726ca92425c0
             q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)
-||||||| official previous@88db9e033a11
-            current_stream.wait_stream(self.alt_stream)
-        else:
-            query, _ = self.wq_b(q_lora)
-            query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
-            q_rope, _ = torch.split(
-                query, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
-            )
-            if _use_dsa_indexer_fusion:
-                key, weights_raw = self._fused_k_weights(x)
-            else:
-                key, _ = self.wk(x)
-            key = self.k_norm(key)
-            k_rope, _ = torch.split(
-                key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
-            )
-
-        q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)
-=======
-            current_stream.wait_stream(self.alt_stream)
-        else:
-            query, _ = self.wq_b(q_lora)
-            query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
-            q_rope, _ = torch.split(
-                query, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
-            )
-            if self.use_dsa_indexer_fusion:
-                key, weights_raw = self._fused_k_weights(x)
-            else:
-                key, _ = self.wk(x)
-            key = self.k_norm(key)
-            k_rope, _ = torch.split(
-                key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
-            )
-
-        q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)
->>>>>>> official target@9a6f8e599204
 
             self._update_rope_guarded(query[..., : self.rope_head_dim], q_rope)
             self._update_rope_guarded(key[..., : self.rope_head_dim], k_rope)
 
         if enable_dual_stream:
             current_stream = torch.cuda.current_stream()
             self.alt_stream.wait_stream(current_stream)
             query = (
@@ -1003,55 +951,26 @@
             or forward_batch.forward_mode.is_draft_extend_v2()
         ):
             seqlens_32 = metadata.get_seqlens_expanded()
         else:
             seqlens_32 = metadata.get_seqlens_int32()
         # Reuse pre-computed schedule metadata if available (from init_forward_metadata),
         # otherwise fall back to computing it here.
         schedule_metadata = getattr(metadata, "paged_mqa_schedule_metadata", None)
-<<<<<<< DCU main@726ca92425c0
-
-        if seqlens_32.dim() == 2:
-||||||| official previous@88db9e033a11
-
-        assert len(q_fp8.shape) == 3
+        assert len(q.shape) == 3
         # attn_tp_size > 1 or MAX_LEN padding mode can leave padding in the
         # hidden states; q_offset is the real (unpadded) q length.
         q_offset = sum(metadata.get_dsa_extend_len_cpu())
 
-        # DG-native q=[B,next_n,H,D] is faster than expanded q=[B*next_n,1,H,D]
-        # for target_verify with next_n>=2 (bigger MMA tile, fewer atoms). The
-        # precomputed ctx_lens_2d's shape is the single source of truth — if
-        # dsa_backend chose the per-token layout (e.g. non-SM100), fall through
-        # to the expanded path.
-        B = metadata.get_seqlens_int32().shape[0]
-        next_n = q_offset // B if B > 0 else 0
-        ctx_2d = getattr(metadata, "paged_mqa_ctx_lens_2d", None)
-        use_dg_native = (
-            _is_cuda
-            and forward_batch.forward_mode.is_target_verify()
-            and next_n >= 2
-            and ctx_2d is not None
-            and ctx_2d.shape == (B, next_n)
-        )
-
-        if use_dg_native:
-            seqlens_32_2d = ctx_2d
-        elif seqlens_32.dim() == 2:
-=======
-        assert len(q_fp8.shape) == 3
-        # attn_tp_size > 1 or MAX_LEN padding mode can leave padding in the
-        # hidden states; q_offset is the real (unpadded) q length.
-        q_offset = sum(metadata.get_dsa_extend_len_cpu())
-
         B = metadata.get_seqlens_int32().shape[0]
         next_n = q_offset // B if B > 0 else 0
         use_cute_dsl = (
-            self.paged_mqa_logits_backend.is_cutedsl()
+            not _is_dcu
+            and self.paged_mqa_logits_backend.is_cutedsl()
             and not forward_batch.forward_mode.is_draft_extend_v2()
         )
         dsl_expand_factor, dsl_atom = 1, 1
         if (
             use_cute_dsl
             and forward_batch.forward_mode.is_target_verify()
             and next_n >= 2
         ):
@@ -1071,214 +990,125 @@
             and next_n >= 2
             and ctx_2d is not None
             and ctx_2d.shape == (B, next_n)
         )
 
         if use_dg_native:
             seqlens_32_2d = ctx_2d
         elif seqlens_32.dim() == 2:
->>>>>>> official target@9a6f8e599204
             seqlens_32_2d = seqlens_32
         else:
             seqlens_32_2d = seqlens_32.unsqueeze(-1)
         if _is_cuda:
             if schedule_metadata is None:
                 schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                     seqlens_32_2d, blocksize, self.sm_count
                 )
         elif _is_dcu:
             schedule_metadata = None
 
         assert len(weights.shape) == 3
         weights = weights.squeeze(2)
 
-<<<<<<< DCU main@726ca92425c0
         # When attn_tp_size > 1 or in the MAX_LEN padding mode, padding may exist in the hidden states,
         # and it is necessary to extract the actual q length.
         q_offset = sum(metadata.get_dsa_extend_len_cpu())
         if self._use_dcu_bf16_index_cache(forward_batch):
             kv_cache = get_token_to_kv_pool().get_index_k_buffer(layer_id=layer_id)
             # BF16 decode follows the vLLM ROCm pattern:
             # keep the indexer K cache in paged BF16 layout and pass it directly
             # to the paged kernel, instead of packing an fp8+scale buffer first.
             logits = gemmopt.paged_mqa_logits(
                 q[:q_offset].unsqueeze(1),
                 kv_cache,
                 # The BF16 path expects dense per-head weights in fp32.
                 weights[:q_offset].to(torch.float32),
-||||||| official previous@88db9e033a11
-        if _is_hip:
-            from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits
-
-            q_fp8 = q_fp8.unsqueeze(1)
-            batch_size, next_n, heads, _ = q_fp8.shape
-            logits = torch.empty(
-                (batch_size * next_n, max_seq_len),
-                device=q_fp8.device,
-                dtype=torch.float32,
-            )
-            deepgemm_fp8_paged_mqa_logits(
-                q_fp8,
-                kv_cache_fp8,
-                weights,
-                logits,
-=======
-        if self.paged_mqa_logits_backend.is_aiter():
-            logits = aiter_paged_mqa_logits(
-                q_fp8,
-                kv_cache_fp8,
-                weights,
->>>>>>> official target@9a6f8e599204
                 seqlens_32,
-                block_tables,
-<<<<<<< DCU main@726ca92425c0
-||||||| official previous@88db9e033a11
-                max_seq_len,
-                Preshuffle=_use_aiter_preshuffle,
-                KVBlockSize=block_kv,
-            )
-        elif use_dg_native:
-            # block_tables[::next_n] de-expands dsa_backend's repeat_interleave
-            # without a copy (DG only checks `stride(1) == 1`).
-            logits = deep_gemm.fp8_paged_mqa_logits(
-                q_fp8[:q_offset].view(B, next_n, q_fp8.shape[1], q_fp8.shape[2]),
-                kv_cache_fp8,
-                weights[:q_offset],
-                seqlens_32_2d,
-                block_tables[::next_n],
-=======
-                max_seq_len,
-                preshuffle=_use_aiter_preshuffle,
-                kv_block_size=block_kv,
-            )
-        elif use_cute_dsl:
-            logits = cutedsl_paged_mqa_logits(
-                q_fp8,
-                kv_cache_fp8,
-                weights,
-                metadata.get_seqlens_int32(),
                 block_tables,
                 schedule_metadata,
                 max_seq_len,
-                q_offset=q_offset,
-                B=B,
-                next_n=next_n,
-                is_target_verify=forward_batch.forward_mode.is_target_verify(),
-                dsl_expand_factor=dsl_expand_factor,
-                dsl_atom=dsl_atom,
-                blocksize=blocksize,
-                sm_count=self.sm_count,
-                get_paged_mqa_logits_metadata_fn=deep_gemm.get_paged_mqa_logits_metadata,
-            )
-        elif use_dg_native:
-            logits = deepgemm_paged_mqa_logits_native(
-                deep_gemm.fp8_paged_mqa_logits,
-                q_fp8,
-                kv_cache_fp8,
-                weights,
-                seqlens_32_2d,
-                block_tables,
->>>>>>> official target@9a6f8e599204
-                schedule_metadata,
-                max_seq_len,
-<<<<<<< DCU main@726ca92425c0
                 clean_logits=True,
-||||||| official previous@88db9e033a11
-                clean_logits=False,
-=======
-                q_offset=q_offset,
-                B=B,
-                next_n=next_n,
->>>>>>> official target@9a6f8e599204
             )
         else:
-<<<<<<< DCU main@726ca92425c0
             kv_cache_fp8 = get_token_to_kv_pool().get_index_k_with_scale_buffer(
                 layer_id=layer_id
-||||||| official previous@88db9e033a11
-            q_fp8 = q_fp8.unsqueeze(1)
-            logits = deep_gemm.fp8_paged_mqa_logits(
-                q_fp8[:q_offset],
-                kv_cache_fp8,
-                weights[:q_offset],
-                seqlens_32_2d,
-                block_tables,
-                schedule_metadata,
-                max_seq_len,
-                clean_logits=False,
-=======
-            logits = deepgemm_paged_mqa_logits_split(
-                deep_gemm.fp8_paged_mqa_logits,
-                q_fp8,
-                kv_cache_fp8,
-                weights,
-                seqlens_32_2d,
-                block_tables,
-                schedule_metadata,
-                max_seq_len,
-                q_offset=q_offset,
->>>>>>> official target@9a6f8e599204
-            )
-            assert len(q.shape) == 3
-            q_fp8 = q.unsqueeze(1)  # the next_n dim is 1 now
+            )
             assert len(kv_cache_fp8.shape) == 2
-            block_kv = 1 if _is_hip and not _is_dcu else 64
-            num_heads_kv = 1
-            head_dim_with_sf = 132
-            if _is_hip and not _is_dcu:
-                kv_cache_fp8 = kv_cache_fp8.view(
-                    -1, block_kv, num_heads_kv, head_dim_with_sf
-                )
-            else:
-                kv_cache_fp8 = kv_cache_fp8.view(
-                    kv_cache_fp8.shape[0], block_kv, num_heads_kv, head_dim_with_sf
-                )
-            if _is_hip and not _is_dcu:
-                from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits
-
-                batch_size, next_n, heads, _ = q_fp8.shape
-                logits = torch.full(
-                    (batch_size * next_n, max_seq_len),
-                    float("-inf"),
-                    device=q_fp8.device,
-                    dtype=torch.float32,
-                )
-                deepgemm_fp8_paged_mqa_logits(
-                    q_fp8,
-                    kv_cache_fp8,
-                    weights,
-                    logits,
-                    seqlens_32,
-                    block_tables,
-                    max_seq_len,
-                    Preshuffle=_use_aiter_preshuffle,
-                    KVBlockSize=block_kv,
-                )
-            elif _is_dcu:
+            block_kv = 1 if _is_hip and not _is_dcu else page_size
+            kv_cache_fp8 = kv_cache_fp8.view(-1, block_kv, 1, 132)
+
+            if _is_dcu:
+                # LightOp keeps the existing DCU ABI with an explicit next_n dim.
+                q_paged = q.unsqueeze(1)
                 logits = gemmopt.paged_mqa_logits(
-                    q_fp8[:q_offset],
+                    q_paged[:q_offset],
                     kv_cache_fp8,
                     weights[:q_offset],
                     seqlens_32,
                     block_tables,
                     schedule_metadata,
                     max_seq_len,
                     clean_logits=True,
                 )
-            else:
-                logits = deep_gemm.fp8_paged_mqa_logits(
-                    q_fp8[:q_offset],
+            elif self.paged_mqa_logits_backend.is_aiter():
+                logits = aiter_paged_mqa_logits(
+                    q,
                     kv_cache_fp8,
-                    weights[:q_offset],
+                    weights,
                     seqlens_32,
+                    block_tables,
+                    max_seq_len,
+                    preshuffle=_use_aiter_preshuffle,
+                    kv_block_size=block_kv,
+                )
+            elif use_cute_dsl:
+                logits = cutedsl_paged_mqa_logits(
+                    q,
+                    kv_cache_fp8,
+                    weights,
+                    metadata.get_seqlens_int32(),
                     block_tables,
                     schedule_metadata,
                     max_seq_len,
-                    clean_logits=False,
+                    q_offset=q_offset,
+                    B=B,
+                    next_n=next_n,
+                    is_target_verify=forward_batch.forward_mode.is_target_verify(),
+                    dsl_expand_factor=dsl_expand_factor,
+                    dsl_atom=dsl_atom,
+                    blocksize=blocksize,
+                    sm_count=self.sm_count,
+                    get_paged_mqa_logits_metadata_fn=deep_gemm.get_paged_mqa_logits_metadata,
+                )
+            elif use_dg_native:
+                logits = deepgemm_paged_mqa_logits_native(
+                    deep_gemm.fp8_paged_mqa_logits,
+                    q,
+                    kv_cache_fp8,
+                    weights,
+                    seqlens_32_2d,
+                    block_tables,
+                    schedule_metadata,
+                    max_seq_len,
+                    q_offset=q_offset,
+                    B=B,
+                    next_n=next_n,
+                )
+            else:
+                logits = deepgemm_paged_mqa_logits_split(
+                    deep_gemm.fp8_paged_mqa_logits,
+                    q,
+                    kv_cache_fp8,
+                    weights,
+                    seqlens_32_2d,
+                    block_tables,
+                    schedule_metadata,
+                    max_seq_len,
+                    q_offset=q_offset,
                 )
 
         # NOTE(dark): logits should be cleaned in topk_transform
         topk_result = metadata.topk_transform(logits, self.index_topk)
         # Restore possible padding exist in the hidden states.
         if not _is_hip and q_offset < q.shape[0]:
             pad_len = q.shape[0] - q_offset
             padding = torch.full(
~~~~

</details>


<details>
<summary><code>python/sglang/srt/layers/moe/topk.py</code> — 2 conflict hunks</summary>

**Resolution intent:** Retain DCU LightOp grouped top-k and postprocess paths while removing retired AOT fake registrations and the unused debug writer.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/moe/topk.py
+++ RESOLVED/python/sglang/srt/layers/moe/topk.py
@@ -101,35 +101,28 @@
 from sglang.srt.layers.dp_attention import is_allocation_symmetric
 from sglang.srt.layers.moe import get_moe_runner_backend
 from sglang.srt.layers.moe.utils import (
     has_per_rank_fused_shared_slots,
 )
 from sglang.srt.layers.utils import MultiPlatformOp
 from sglang.srt.state_capturer.routed_experts import get_global_experts_capturer
 from sglang.srt.utils import (
+    cpu_has_amx_support,
     direct_register_custom_op,
-    cpu_has_amx_support,
     get_bool_env_var,
     get_compiler_backend,
     is_cpu,
     is_cuda,
+    is_dcu,
     is_hip,
     is_musa,
-    is_dcu,
     is_npu,
     is_xpu,
 )
-<<<<<<< DCU main@726ca92425c0
-from sglang.srt.utils.patch_torch import register_fake_if_exists
-import os
-||||||| official previous@88db9e033a11
-from sglang.srt.utils.patch_torch import register_fake_if_exists
-=======
->>>>>>> official target@9a6f8e599204
 
 _SGLANG_EXPERIMENTAL_LORA_OPTI = envs.SGLANG_EXPERIMENTAL_LORA_OPTI.get()
 
 if TYPE_CHECKING:
     from sglang.srt.layers.quantization import QuantizationConfig
 
 
 logger = logging.getLogger(__name__)
@@ -2357,113 +2350,12 @@
     if packed_topk is not None:
         return StandardTopKOutputPacked(
             topk_weights, topk_ids, router_logits, packed_topk
         )
     # ===== END TO BE REFACTORED ====
     return StandardTopKOutput(topk_weights, topk_ids, router_logits)
 
 
-<<<<<<< DCU main@726ca92425c0
-# Register fake implementations for torch.compile support
-if _is_cuda:
-
-    @torch.library.register_fake("sgl_kernel::moe_fused_gate")
-    def _moe_fused_gate(
-        input_tensor,
-        bias,
-        num_expert_group,
-        topk_group,
-        topk,
-        num_fused_shared_experts=0,
-        routed_scaling_factor=0,
-        apply_routed_scaling_factor_on_output=False,
-    ):
-        num_rows = input_tensor.shape[0]
-        topk_weights = torch.empty(
-            (num_rows, topk), dtype=torch.float32, device=input_tensor.device
-        )
-        topk_ids = torch.empty(
-            (num_rows, topk), dtype=torch.int32, device=input_tensor.device
-        )
-        return topk_weights, topk_ids
-
-    @register_fake_if_exists("sgl_kernel::kimi_k2_moe_fused_gate")
-    def _kimi_k2_moe_fused_gate(
-        input_tensor,
-        bias,
-        topk,
-        renormalize,
-        routed_scaling_factor,
-        apply_routed_scaling_factor_on_output,
-    ):
-        num_rows = input_tensor.shape[0]
-        topk_weights = input_tensor.new_empty(
-            num_rows,
-            topk,
-            dtype=torch.float32,
-        )
-        topk_ids = input_tensor.new_empty(
-            num_rows,
-            topk,
-            dtype=torch.int32,
-        )
-        return topk_weights, topk_ids
-def batch_write_expert_counts(expert_count: torch.Tensor, num_token: int):
-    # 合并所有token的expert_count并写入文件
-    rank = torch.distributed.get_rank()
-    basedir = f"/shangxl/Test/eplb_recorder/{rank}"
-    os.makedirs(basedir, exist_ok=True)
-    file_path = os.path.join(basedir, "expert_activation_counts.txt")
-    with open(file_path, "w") as f:
-        f.write(f"After processing token {num_token}:\n")
-        f.write(f"expert_count: {expert_count.cpu().numpy().tolist()}\n")
-||||||| official previous@88db9e033a11
-# Register fake implementations for torch.compile support
-if _is_cuda:
-
-    @torch.library.register_fake("sgl_kernel::moe_fused_gate")
-    def _moe_fused_gate(
-        input_tensor,
-        bias,
-        num_expert_group,
-        topk_group,
-        topk,
-        num_fused_shared_experts=0,
-        routed_scaling_factor=0,
-        apply_routed_scaling_factor_on_output=False,
-    ):
-        num_rows = input_tensor.shape[0]
-        topk_weights = torch.empty(
-            (num_rows, topk), dtype=torch.float32, device=input_tensor.device
-        )
-        topk_ids = torch.empty(
-            (num_rows, topk), dtype=torch.int32, device=input_tensor.device
-        )
-        return topk_weights, topk_ids
-
-    @register_fake_if_exists("sgl_kernel::kimi_k2_moe_fused_gate")
-    def _kimi_k2_moe_fused_gate(
-        input_tensor,
-        bias,
-        topk,
-        renormalize,
-        routed_scaling_factor,
-        apply_routed_scaling_factor_on_output,
-    ):
-        num_rows = input_tensor.shape[0]
-        topk_weights = input_tensor.new_empty(
-            num_rows,
-            topk,
-            dtype=torch.float32,
-        )
-        topk_ids = input_tensor.new_empty(
-            num_rows,
-            topk,
-            dtype=torch.int32,
-        )
-        return topk_weights, topk_ids
-=======
 # NOTE: the AOT sgl_kernel::moe_fused_gate and sgl_kernel::kimi_k2_moe_fused_gate
 # ops (and their torch.compile fake impls) were retired here — both CUDA gate
 # paths now route through the unified Triton router (jit_kernel/moe_fused_gate.py),
 # whose Python impl is traceable directly, so no register_fake shim is needed.
->>>>>>> official target@9a6f8e599204
~~~~

</details>


<details>
<summary><code>python/sglang/srt/layers/quantization/__init__.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Keep both SlimQuant DCU registrations and add the official NPU mxfp_w4a8 registration.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/quantization/__init__.py
+++ RESOLVED/python/sglang/srt/layers/quantization/__init__.py
@@ -94,23 +94,19 @@
     "qoq": QoQConfig,
     "w4afp8": W4AFp8Config,
     "petit_nvfp4": PetitNvFp4Config,
     "fbgemm_fp8": FBGEMMFp8Config,
     "auto-round": AutoRoundConfig,
     "auto-round-int8": W8A8Int8Config,
     "modelslim": ModelSlimConfig,
     "quark_int4fp8_moe": QuarkInt4Fp8Config,
-<<<<<<< DCU main@726ca92425c0
     "slimquant_w4a8_marlin": SlimQuantW4A8Int8MarlinConfig,
     "slimquant_marlin": SlimQuantCompressedTensorsMarlinConfig,
-||||||| official previous@88db9e033a11
-=======
     "mxfp_w4a8": Mxfp4W4A8Config,
->>>>>>> official target@9a6f8e599204
 }
 if QuarkConfig is not None:
     BASE_QUANTIZATION_METHODS["quark"] = QuarkConfig
     BASE_QUANTIZATION_METHODS["quark_mxfp4"] = QuarkConfig
 
 if is_cpu() or is_cuda() or (_is_mxfp_supported and is_hip()):
     BASE_QUANTIZATION_METHODS.update(
         {
~~~~

</details>


<details>
<summary><code>python/sglang/srt/layers/quantization/unquant.py</code> — 2 conflict hunks</summary>

**Resolution intent:** Adopt the official ServerArgs typing and CuTeDSL BF16 branch while retaining the DCU dtype-alignment fallback.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/layers/quantization/unquant.py
+++ RESOLVED/python/sglang/srt/layers/quantization/unquant.py
@@ -42,35 +42,31 @@
 )
 
 if TYPE_CHECKING:
     from sglang.srt.layers.moe.token_dispatcher import (
         CombineInput,
         DispatchOutput,
         StandardDispatchOutput,
     )
-<<<<<<< DCU main@726ca92425c0
+    from sglang.srt.server_args import ServerArgs
+
 from sglang.srt.utils import is_dcu
+
 _is_dcu = is_dcu()
 
 _use_marlin_w16a16_moe = get_bool_env_var("SGLANG_USE_MARLIN_W16A16_MOE")
 _use_aiter_w16a16_moe = get_bool_env_var("SGLANG_ROCM_USE_AITER_MOE")
 if _use_aiter_w16a16_moe:
     from aiter.moe import (
         get_aiter_moe_config,
         aiter_moe,
         MoeSolutionType,
         MoeQuantType,
             )
-||||||| official previous@88db9e033a11
-
-=======
-    from sglang.srt.server_args import ServerArgs
-
->>>>>>> official target@9a6f8e599204
 
 _is_cpu_amx_available = cpu_has_amx_support()
 _is_hip = is_hip()
 _is_cpu = is_cpu()
 _is_npu = is_npu()
 _use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
 
 if _use_aiter:
@@ -213,38 +209,34 @@
             )
             if len(x_shapes) == 3:
                 output = output.view(x_shapes[0], x_shapes[1], -1)
             return output
 
         elif _use_aiter and type(layer.weight.data) is torch.Tensor:
             return tgemm.mm(x, layer.weight, bias, otype=x.dtype)
 
-<<<<<<< DCU main@726ca92425c0
-        if x.dtype != layer.weight.dtype:
-            x = x.to(layer.weight.dtype)
-||||||| official previous@88db9e033a11
-=======
         elif (
             get_bf16_gemm_backend().is_cutedsl()
             and x.is_cuda
             and x.dtype == torch.bfloat16
             and layer.weight.dtype == torch.bfloat16
             and (bias is None or bias.dtype == torch.bfloat16)
             and _use_cutedsl_bf16_gemm(
                 x.numel() // x.shape[-1],
                 layer.weight.shape[0],
                 layer.weight.shape[1],
             )
         ):
             x_shapes = x.shape
             output = _cutedsl_bf16_gemm(x.view(-1, x_shapes[-1]), layer.weight, bias)
             return output.view(*x_shapes[:-1], -1)
 
->>>>>>> official target@9a6f8e599204
+        if x.dtype != layer.weight.dtype:
+            x = x.to(layer.weight.dtype)
         return F.linear(x, layer.weight, bias)
 
 
 class UnquantizedFusedMoEMethod(FusedMoEMethodBase, MultiPlatformOp):
     """MoE method without quantization."""
 
     def __init__(
         self,
~~~~

</details>


<details>
<summary><code>python/sglang/srt/models/deepseek_v2.py</code> — 2 conflict hunks</summary>

**Resolution intent:** Retain DCU DSA helpers and add the official BF16 backend and Hash-MoE input_ids forwarding on the canonical model structure.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/models/deepseek_v2.py
+++ RESOLVED/python/sglang/srt/models/deepseek_v2.py
@@ -114,23 +114,19 @@
     create_per_token_group_quant_fp8_output_scale,
     fp8_dtype,
     per_tensor_quant_mla_fp8,
     per_token_group_quant_mla_deep_gemm_masked_fp8,
 )
 from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
     maybe_fuse_routed_scale_and_shared_add,
 )
-<<<<<<< DCU main@726ca92425c0
 from sglang.srt.layers.attention.dsa.dequant_k_cache import dequantize_k_cache_paged
 from sglang.srt.layers.attention.utils import concat_and_cast_mha_k_triton
-||||||| official previous@88db9e033a11
-=======
 from sglang.srt.layers.quantization.unquant import get_bf16_gemm_backend
->>>>>>> official target@9a6f8e599204
 from sglang.srt.layers.radix_attention import RadixAttention
 from sglang.srt.layers.rotary_embedding import get_rope_wrapper
 from sglang.srt.layers.utils import PPMissingLayer
 from sglang.srt.layers.utils.cp_utils import (
     can_cp_split,
     cp_all_gather_rerange_output,
     cp_split_and_rebuild_data,
     cp_split_and_rebuild_position,
@@ -1658,30 +1654,23 @@
         ):
             state.shared_output = self.shared_experts(hidden_states_mlp_input)
         else:
             state.shared_output = None
 
     def op_select_experts(self, state):
         router_logits = state.pop("router_logits")
         hidden_states = state.hidden_states_mlp_input
-<<<<<<< DCU main@726ca92425c0
-||||||| official previous@88db9e033a11
-
-=======
-
         # Hash MoE layers (e.g. DeepSeek-V4) route on input_ids; forward_deepep
         # passes them as a topk kwarg. The per-ubatch forward_batch.input_ids is
         # already sliced+padded to match hidden_states rows (and equals the
         # global ids under EP dp-attention). No-op for non-hash models.
         topk_kwargs = {}
         if getattr(self, "is_hash", False):
             topk_kwargs["input_ids"] = state.forward_batch.input_ids
-
->>>>>>> official target@9a6f8e599204
         if router_logits is not None:
             with get_global_expert_distribution_recorder().with_current_layer(
                 self.layer_id
             ):
                 state.topk_output = self.topk(
                     hidden_states=hidden_states,
                     router_logits=router_logits,
                     num_token_non_padded=state.forward_batch.num_token_non_padded,
~~~~

</details>


<details>
<summary><code>python/sglang/srt/models/deepseek_v4.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Accept the official get_flags LM-head structure while preserving the validated DCU early return from generic MHC prewarm.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/models/deepseek_v4.py
+++ RESOLVED/python/sglang/srt/models/deepseek_v4.py
@@ -2270,56 +2270,29 @@
         self.config = config
         self.tp_size = get_parallel().tp_size
         self.quant_config = quant_config
         self.determine_num_fused_shared_experts()
         self.model = DeepseekV4Model(
             config, quant_config, prefix=add_prefix("model", prefix)
         )
         self.pp_group = get_pp_group()
-<<<<<<< DCU main@726ca92425c0
-        if not self.pp_group.is_last_rank:
-||||||| official previous@88db9e033a11
-        if self.pp_group.is_last_rank:
-            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
-                self.lm_head = self.model.embed_tokens
-            else:
-                self.lm_head = ParallelLMHead(
-                    config.vocab_size,
-                    config.hidden_size,
-                    quant_config=quant_config,
-                    prefix=add_prefix("lm_head", prefix),
-                    use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
-                )
-        else:
-=======
         if self.pp_group.is_last_rank:
             if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                 self.lm_head = self.model.embed_tokens
             else:
                 self.lm_head = ParallelLMHead(
                     config.vocab_size,
                     config.hidden_size,
                     quant_config=quant_config,
                     prefix=add_prefix("lm_head", prefix),
                     use_attn_tp_group=get_flags().enable_dp_lm_head,
                 )
         else:
->>>>>>> official target@9a6f8e599204
             self.lm_head = PPMissingLayer()
-        elif config.tie_word_embeddings and self.pp_group.world_size == 1:
-            self.lm_head = self.model.embed_tokens
-        else:
-            self.lm_head = ParallelLMHead(
-                config.vocab_size,
-                config.hidden_size,
-                quant_config=quant_config,
-                prefix=add_prefix("lm_head", prefix),
-                use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
-            )
         self.logits_processor = LogitsProcessor(config)
         self.capture_aux_hidden_states = False
         get_attn_tp_context().init_context(config.q_lora_rank, is_dsa=True)
 
         self._routed_experts_weights_of_layer = LazyValue(
             lambda: {
                 layer_id: layer.mlp.get_moe_weights()
                 for layer_id, layer in enumerate(self.model.layers)
~~~~

</details>


<details>
<summary><code>python/sglang/srt/server_args.py</code> — 5 conflict hunks</summary>

**Resolution intent:** Use the official canonical field and override layout, then re-port DCU DSA, MLA, Mamba, LightOp, AITER, and graph settings.

~~~~diff
--- AUTO-CONFLICT/python/sglang/srt/server_args.py
+++ RESOLVED/python/sglang/srt/server_args.py
@@ -175,17 +175,17 @@
     "triton",
     "torch_native",
     "flex_attention",
     "dsa",
     "nsa",  # Deprecated alias for "dsa"
     "dsv4",
     "compressed",  # Deprecated alias for "dsv4"
     # ransplant from vllm
-    "dcu_mla", 
+    "dcu_mla",
     # NVIDIA specific
     "cutlass_mla",
     "fa3",
     "fa4",
     "flashinfer",
     "flashmla",
     "trtllm_mla",
     "cutedsl_mla",
@@ -3946,64 +3946,18 @@
                     else:
                         # Pure TP and partial DP Attention mode is active for DSA, logging a warning
                         if self.dp_size < self.tp_size:
                             logger.warning(
                                 f"DSA with TP mode is active, dp_size={self.dp_size}, tp_size={self.tp_size}, "
                                 f"attn_tp_size={self.tp_size}, attention weights will be sharded across {self.tp_size} ranks."
                             )
 
-<<<<<<< DCU main@726ca92425c0
-                    # Deferred import to avoid a circular import at module-load
-                    # time (dsa.utils imports get_global_server_args).
-                    from sglang.srt.layers.attention.dsa.utils import (
-                        aiter_can_use_preshuffle_paged_mqa,
-                    )
-
-                    if (
-                        is_hip()
-                        and not is_dcu()
-                        and not aiter_can_use_preshuffle_paged_mqa()
-                    ):
-                        # Legacy ROCm DSA path: aiter's gluon paged-MQA kernel is
-                        # unavailable (Triton<3.5 and AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS
-                        # not set, or SGLANG_DSA_HIP_DISABLE_PRESHUFFLE=1 / SGLANG_USE_AITER=0).
-                        self.page_size = 1
-                        logger.warning(
-                            "Setting page size to 1 for DeepSeek DSA on ROCm "
-                            "(aiter preshuffle paged-MQA path unavailable: "
-                            "needs Triton>=3.5.0 or AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1)."
-                        )
-                    else:
-                        self.page_size = 64
-                        logger.warning("Setting page size to 64 for DeepSeek DSA.")
-||||||| official previous@88db9e033a11
-                    # Deferred import to avoid a circular import at module-load
-                    # time (dsa.utils imports get_global_server_args).
-                    from sglang.srt.layers.attention.dsa.utils import (
-                        aiter_can_use_preshuffle_paged_mqa,
-                    )
-
-                    if is_hip() and not aiter_can_use_preshuffle_paged_mqa():
-                        # Legacy ROCm DSA path: aiter's gluon paged-MQA kernel is
-                        # unavailable (Triton<3.5 and AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS
-                        # not set, or SGLANG_DSA_HIP_DISABLE_PRESHUFFLE=1 / SGLANG_USE_AITER=0).
-                        self.page_size = 1
-                        logger.warning(
-                            "Setting page size to 1 for DeepSeek DSA on ROCm "
-                            "(aiter preshuffle paged-MQA path unavailable: "
-                            "needs Triton>=3.5.0 or AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1)."
-                        )
-                    else:
-                        self.page_size = 64
-                        logger.warning("Setting page size to 64 for DeepSeek DSA.")
-=======
                     # The DSA page-size selection moved to the override registry
                     # (arg_groups/overrides.py: _deepseek_family_overrides).
->>>>>>> official target@9a6f8e599204
 
                     import torch
 
                     major, _ = torch.cuda.get_device_capability()
                     self._set_default_dsa_kv_cache_dtype(
                         major, resolved_view(self).quantization
                     )
                     self._set_default_dsa_backends(self.kv_cache_dtype, major)
@@ -4351,29 +4305,19 @@
 
     def _validate_mamba_extra_buffer(self, view, model_arch: str):
         from sglang.srt.arg_groups.overrides import supports_mamba_cache_extra_buffer
 
         assert supports_mamba_cache_extra_buffer(
             view, model_arch
         ), f"extra_buffer is not supported for {model_arch}; use no_buffer."
         assert (
-<<<<<<< DCU main@726ca92425c0
             is_cuda() or is_musa() or is_npu() or is_dcu()
         ), "extra_buffer needs CUDA/MUSA/NPU/DCU (FLA)."
-        if self.speculative_num_draft_tokens is not None:
-||||||| official previous@88db9e033a11
-            is_cuda() or is_musa() or is_npu()
-        ), "extra_buffer needs CUDA/MUSA/NPU (FLA)."
-        if self.speculative_num_draft_tokens is not None:
-=======
-            is_cuda() or is_musa() or is_npu()
-        ), "extra_buffer needs CUDA/MUSA/NPU (FLA)."
         if view.speculative_num_draft_tokens is not None:
->>>>>>> official target@9a6f8e599204
             assert (
                 view.mamba_radix_cache_strategy != "extra_buffer_lazy"
             ), "extra_buffer_lazy unsupported with spec."
             assert view.mamba_track_interval >= view.speculative_num_draft_tokens
         if view.page_size is not None:
             assert view.mamba_track_interval % view.page_size == 0
             assert self.mamba_cache_chunk_size is not None
 
@@ -4525,81 +4469,21 @@
             model_config.is_encoder_decoder
             and not self.disable_radix_cache
             and "WhisperForConditionalGeneration"
             in (model_config.hf_config.architectures or [])
         ):
             logger.info("Radix cache is disabled for Whisper")
             self.disable_radix_cache = True
 
-<<<<<<< DCU main@726ca92425c0
-        # Major NVIDIA platforms backends
-        if (
-            self.attention_backend == "flashmla"
-            or self.decode_attention_backend == "flashmla"
-            or self.attention_backend == "dcu_mla"
-            or self.decode_attention_backend == "dcu_mla"
-        ):
-            logger.warning(
-                "FlashMLA/DCU MLA only supports a page_size of 64, change page_size to 64."
-            )
-            self.page_size = 64
-
-        if (
-            self.attention_backend == "cutlass_mla"
-            or self.decode_attention_backend == "cutlass_mla"
-        ):
-            logger.warning(
-                "Cutlass MLA only supports a page_size of 128, change page_size to 128."
-            )
-            self.page_size = 128
-
-        if (
-            self.attention_backend == "trtllm_mla"
-            or self.decode_attention_backend == "trtllm_mla"
-        ):
-            if not is_blackwell_supported():
-                raise ValueError(
-                    "TRTLLM MLA backend is only supported on Blackwell GPUs (SM100/SM12x). Please use a different backend."
-                )
-||||||| official previous@88db9e033a11
-        # Major NVIDIA platforms backends
-        if (
-            self.attention_backend == "flashmla"
-            or self.decode_attention_backend == "flashmla"
-        ):
-            logger.warning(
-                "FlashMLA only supports a page_size of 64, change page_size to 64."
-            )
-            self.page_size = 64
-
-        if (
-            self.attention_backend == "cutlass_mla"
-            or self.decode_attention_backend == "cutlass_mla"
-        ):
-            logger.warning(
-                "Cutlass MLA only supports a page_size of 128, change page_size to 128."
-            )
-            self.page_size = 128
-
-        if (
-            self.attention_backend == "trtllm_mla"
-            or self.decode_attention_backend == "trtllm_mla"
-        ):
-            if not is_blackwell_supported():
-                raise ValueError(
-                    "TRTLLM MLA backend is only supported on Blackwell GPUs (SM100/SM12x). Please use a different backend."
-                )
-=======
         # Major NVIDIA platforms backends: the page-size snaps of this family
         # moved to the resolution pipeline (arg_groups/overrides.py:
         # _mla_backend_page_constraints); the raises and the cutedsl prefill
         # fallback stay below.
         run_post_process_pass(self, _mla_backend_page_constraints)
->>>>>>> official target@9a6f8e599204
 
         # The TRT-LLM / tokenspeed MLA kv-dtype validations moved to the
         # resolution pipeline (arg_groups/overrides.py:
         # _mla_kv_cache_dtype_checks), invoked here at their legacy slot.
         from sglang.srt.arg_groups.overrides import _mla_kv_cache_dtype_checks
 
         run_post_process_pass(self, _mla_kv_cache_dtype_checks)
 
@@ -4618,45 +4502,17 @@
                 )
             if decode_backend == "trtllm_mha" and not (
                 is_sm90_supported() or is_sm100_supported() or is_sm120_supported()
             ):
                 raise ValueError(
                     "TRTLLM MHA backend for decode is only supported on Hopper (SM90), Blackwell (SM100) and (SM120) GPUs. Please use a different decode backend."
                 )
 
-<<<<<<< DCU main@726ca92425c0
-            if self.page_size not in [16, 32, 64]:
-                logger.warning(
-                    f"TensorRT-LLM MHA only supports page_size of 16, 32 or 64, changing page_size from {self.page_size} to 64."
-                )
-                self.page_size = 64
-
-        # if self.attention_backend == "fa3" and self.kv_cache_dtype == "fp8_e5m2":
-        #     logger.warning(
-        #         "FlashAttention3 only supports fp8_e4m3 if using FP8; "
-        #         "Setting attention backend to triton."
-        #     )
-        #     self.attention_backend = "triton"
-||||||| official previous@88db9e033a11
-            if self.page_size not in [16, 32, 64]:
-                logger.warning(
-                    f"TensorRT-LLM MHA only supports page_size of 16, 32 or 64, changing page_size from {self.page_size} to 64."
-                )
-                self.page_size = 64
-
-        if self.attention_backend == "fa3" and self.kv_cache_dtype == "fp8_e5m2":
-            logger.warning(
-                "FlashAttention3 only supports fp8_e4m3 if using FP8; "
-                "Setting attention backend to triton."
-            )
-            self.attention_backend = "triton"
-=======
         run_post_process_pass(self, _attention_backend_fa3_fp8_fallback)
->>>>>>> official target@9a6f8e599204
 
         run_post_process_pass(self, _fa4_page_constraint)
 
         # AMD platforms backends
         if resolved_view(self).attention_backend == "aiter":
             if model_config.context_len > 8192:
                 self.mem_fraction_static *= 0.85
 
@@ -5160,90 +5016,56 @@
 
         if view.moe_runner_backend == "flashinfer_trtllm_routed":
             assert view.quantization in [
                 "fp8",
                 "mxfp8",
                 "modelopt_fp4",
                 "nvfp4_online",
                 None,
-<<<<<<< DCU main@726ca92425c0
-            ], f"Invalid quantization '{self.quantization}'. \nFlashInfer TRTLLM routed MOE supports only: 'fp8', 'mxfp8', 'modelopt_fp4', 'nvfp4_online', or bfloat16 (None)."
-            self.disable_shared_experts_fusion = True
+            ], f"Invalid quantization '{view.quantization}'. \nFlashInfer TRTLLM routed MOE supports only: 'fp8', 'mxfp8', 'modelopt_fp4', 'nvfp4_online', or bfloat16 (None)."
+
+        if view.moe_runner_backend == "lightop":
             logger.warning(
-                "FlashInfer TRTLLM routed MoE is enabled. --disable-shared-experts-fusion is automatically set."
-            )
-
-        if self.moe_runner_backend == "lightop":
-            logger.warning(
-                "LightOp MoE runner is a transitional backend and may be deprecated in future releases. Please use AITER MoE runner."
+                "LightOp MoE runner is a transitional DCU backend and may be "
+                "deprecated in a future release."
             )
             assert is_dcu(), "lightop MoE runner backend is only supported on DCU."
-            assert (
-                self.quantization == "w8a8_int8"
-            ), "lightop MoE runner backend currently supports only w8a8_int8 quantization."
-            assert self.moe_a2a_backend == "none", (
+            assert view.quantization == "w8a8_int8", (
+                "lightop MoE runner backend currently supports only "
+                "w8a8_int8 quantization."
+            )
+            assert view.moe_a2a_backend == "none", (
                 "lightop MoE runner backend currently supports only "
                 "moe_a2a_backend='none'."
             )
 
-        if self.moe_runner_backend == "aiter" and self.quantization == "w8a8_int8":
+        if view.moe_runner_backend == "aiter" and view.quantization == "w8a8_int8":
             assert is_dcu(), (
                 "aiter MoE runner backend with w8a8_int8 quantization is only "
                 "supported on DCU."
             )
-            assert self.moe_a2a_backend == "none", (
+            assert view.moe_a2a_backend == "none", (
                 "aiter MoE runner backend with w8a8_int8 quantization currently "
                 "supports only moe_a2a_backend='none'."
             )
-
-        if envs.SGLANG_CUTLASS_MOE.get():
-            logger.warning(
-                "SGLANG_CUTLASS_MOE is deprecated, use --moe-runner-backend=cutlass and/or --speculative-moe-runner-backend=cutlass instead"
-            )
-            assert self.quantization in [
-                "fp8",
-                "mxfp8",
-            ], "cutlass MoE is only supported with fp8/mxfp8 quantization"
-            self.moe_runner_backend = "cutlass"
-        if self.moe_runner_backend == "cutlass" and self.quantization in [
-||||||| official previous@88db9e033a11
-            ], f"Invalid quantization '{self.quantization}'. \nFlashInfer TRTLLM routed MOE supports only: 'fp8', 'mxfp8', 'modelopt_fp4', 'nvfp4_online', or bfloat16 (None)."
-            self.disable_shared_experts_fusion = True
-            logger.warning(
-                "FlashInfer TRTLLM routed MoE is enabled. --disable-shared-experts-fusion is automatically set."
-            )
-
-        if envs.SGLANG_CUTLASS_MOE.get():
-            logger.warning(
-                "SGLANG_CUTLASS_MOE is deprecated, use --moe-runner-backend=cutlass and/or --speculative-moe-runner-backend=cutlass instead"
-            )
-            assert self.quantization in [
-                "fp8",
-                "mxfp8",
-            ], "cutlass MoE is only supported with fp8/mxfp8 quantization"
-            self.moe_runner_backend = "cutlass"
-        if self.moe_runner_backend == "cutlass" and self.quantization in [
-=======
-            ], f"Invalid quantization '{view.quantization}'. \nFlashInfer TRTLLM routed MOE supports only: 'fp8', 'mxfp8', 'modelopt_fp4', 'nvfp4_online', or bfloat16 (None)."
 
         # The runner-driven shared-experts fusion disables moved to the
         # pipeline (arg_groups/overrides.py: _moe_runner_fusion_disable),
         # invoked here at the legacy write slots.
         run_post_process_pass(self, _moe_runner_fusion_disable)
 
         # The deprecated SGLANG_CUTLASS_MOE override moved to the pipeline
         # (arg_groups/overrides.py: _cutlass_moe_env_override). It sits after
         # the fusion blocks above on purpose: they must observe the
         # pre-override runner value, exactly as they did imperatively.
         run_post_process_pass(self, _cutlass_moe_env_override)
         if resolved_view(self).moe_runner_backend == "cutlass" and resolved_view(
             self
         ).quantization in [
->>>>>>> official target@9a6f8e599204
             "fp8",
             "mxfp8",
         ]:
             assert (
                 resolved_view(self).ep_size == 1
             ), "FP8/MXFP8 Cutlass MoE is only supported with ep_size == 1"
 
     def cutedsl_moe_max_num_tokens(self) -> int:
~~~~

</details>


<details>
<summary><code>test/registered/debug_utils/test_dumper.py</code> — 1 conflict hunk</summary>

**Resolution intent:** Adopt the official temp_set_env import move and preserve the disabled DCU nightly registration.

~~~~diff
--- AUTO-CONFLICT/test/registered/debug_utils/test_dumper.py
+++ RESOLVED/test/registered/debug_utils/test_dumper.py
@@ -35,32 +35,29 @@
     _register_forward_hook_or_replace_fn,
     _SGLangPlugin,
     _torch_save,
     dumper,
     get_tensor_info,
     get_truncated_value,
 )
 from sglang.srt.utils import kill_process_tree
-<<<<<<< DCU main@726ca92425c0
-from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_dcu_ci
+from sglang.srt.utils.common import temp_set_env
+from sglang.test.ci.ci_register import (
+    register_amd_ci,
+    register_cuda_ci,
+    register_dcu_ci,
+)
 
 register_dcu_ci(
     est_time=120,
     suite="nightly-dcu",
     nightly=True,
     disabled='DCU Full Enabled run 26941698027 failed; keep disabled until BW1100 failure is fixed or revalidated.',
 )
-
-||||||| official previous@88db9e033a11
-from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
-=======
-from sglang.srt.utils.common import temp_set_env
-from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
->>>>>>> official target@9a6f8e599204
 from sglang.test.test_utils import (
     DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
     DEFAULT_URL_FOR_TEST,
     find_available_port,
     popen_launch_server,
     run_distributed_test,
 )
 
~~~~

</details>

