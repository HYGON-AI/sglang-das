# DCU CI PR 触发测试记录

## 背景

本次目标是在 GitHub 仓库 `HYGON-AI/dcu-sglang` 的 `dcu-ci-pr` 测试分支上验证 DCU PR workflow 的自动触发链路。

之前手动触发入口不稳定或不可见，因此改用 PR 触发方式：从 `dcu-ci-pr` 拉测试分支，提交一组最小 workflow 改动，再向 `dcu-ci-pr` 发起 PR，观察 `PR Test (DCU)` 是否被 GitHub Actions 自动触发。

## 操作基线

- 远端仓库：`git@github.com:HYGON-AI/dcu-sglang.git`
- 基线分支：`dcu-ci-pr`
- 本地隔离 worktree：`/public/home/dingxl/dcu-sglang-ci-trigger-test`
- 测试分支：`test-trigger-dcu-ci-0601`
- 基线 commit：`b41c7b181 ci: use explicit DCU checkout`

本次没有在原 `/public/home/dingxl/sglang` 脏工作区里直接操作，避免混入已有开发改动。

## 修改内容

### 1. 允许 PR 到 `dcu-ci-pr` 触发 DCU workflow

修改文件：`.github/workflows/pr-test-dcu.yml`

原先 `pull_request.branches` 只包含：

```yaml
branches:
  - v0.5.12_dev
  - main
```

本次新增：

```yaml
branches:
  - dcu-ci-pr
```

这样从测试分支向 `dcu-ci-pr` 发 PR 时，也能触发 `PR Test (DCU)`。

### 2. workflow 默认跳过重依赖安装和 editable build

修改文件：`.github/workflows/pr-test-dcu.yml`

`dcu_ci_install_dependency.sh` 已经支持以下变量：

```bash
DCU_CI_SKIP_DEPENDENCY_INSTALL
DCU_CI_SKIP_REQUIREMENTS_INSTALL
DCU_CI_SKIP_SGLANG_BUILD
```

但 workflow 里之前默认值为空，需要仓库 Variables 显式配置才会跳过。本次改为：

```yaml
DCU_CI_SKIP_REQUIREMENTS_INSTALL: ${{ vars.DCU_CI_SKIP_REQUIREMENTS_INSTALL || '1' }}
DCU_CI_SKIP_SGLANG_BUILD: ${{ vars.DCU_CI_SKIP_SGLANG_BUILD || '1' }}
```

含义：

- 默认不安装 `requirements_dcu.txt`，避免破坏当前镜像中已经验证过的依赖。
- 默认不做 editable `sglang` build，测试通过 `PYTHONPATH=/sglang-checkout/python` 使用当前 checkout 代码。
- `DCU_CI_SKIP_DEPENDENCY_INSTALL` 保持由仓库变量控制；如果后续确认镜像完全自足，可以在 GitHub Variables 中设为 `1`，连轻量依赖检查/安装也跳过。

## 未修改内容

- 未改 `dcu_ci_start_container.sh`。
- 未改 `dcu_ci_exec.sh`。
- 未改 `dcu_ci_install_dependency.sh`，因为脚本本身已经支持 skip 变量。
- 未改 Stage-A / Stage-B 的测试列表。
- 未改 runner label、镜像、matrix 分片策略。

## 验证方式

本地静态验证：

```bash
python3 - <<'PY'
from pathlib import Path
import yaml
yaml.safe_load(Path('.github/workflows/pr-test-dcu.yml').read_text())
print('workflow yaml ok')
PY

bash -n scripts/ci/dcu/dcu_ci_install_dependency.sh
bash -n scripts/ci/dcu/dcu_ci_start_container.sh
bash -n scripts/ci/dcu/dcu_ci_exec.sh
git diff --check
```

GitHub 侧验证：

1. 推送 `test-trigger-dcu-ci-0601` 分支。
2. 在 GitHub 页面创建 PR：
   - base：`dcu-ci-pr`
   - compare：`test-trigger-dcu-ci-0601`
3. 观察 Actions 是否出现 `PR Test (DCU)`。
4. 预期先进入 `check-changes`，然后走 `validate-config`、Stage-A、Stage-B。

## 当前执行结果

本地已完成提交：

```text
dc28314e2 ci: enable DCU PR trigger branch
```

尝试推送到 `HYGON-AI/dcu-sglang`：

```bash
git push git@github.com:HYGON-AI/dcu-sglang.git HEAD:refs/heads/test-trigger-dcu-ci-0601
```

结果失败：

```text
ERROR: Write access to repository not granted.
fatal: Could not read from remote repository.
```

结论：当前服务器上的 GitHub SSH 身份可以读取私有仓库，但没有向 `HYGON-AI/dcu-sglang` 直接推送分支的权限。后续需要二选一：

1. 给当前 GitHub 账号/SSH key 开通该仓库写权限后，重新执行 push。
2. 推送到个人 fork，再从 fork 向 `HYGON-AI/dcu-sglang:dcu-ci-pr` 创建 PR。

## 回滚方式

如果本次测试后不需要保留这些改动：

1. 关闭测试 PR。
2. 删除远端测试分支：

```bash
git push git@github.com:HYGON-AI/dcu-sglang.git --delete test-trigger-dcu-ci-0601
```

3. 如果改动已经合入测试分支，再 revert 对应 commit：

```bash
git revert <commit_sha>
```

4. 如果只想手动恢复文件，删除以下改动即可：
   - 从 `.github/workflows/pr-test-dcu.yml` 的 `pull_request.branches` 中移除 `dcu-ci-pr`。
   - 把 `DCU_CI_SKIP_REQUIREMENTS_INSTALL` 默认值从 `'1'` 改回空字符串。
   - 把 `DCU_CI_SKIP_SGLANG_BUILD` 默认值从 `'1'` 改回空字符串。
   - 删除本记录文档。

## 备注

这次 PR 不是为了合入功能，而是为了验证 `dcu-ci-pr` 分支上的 DCU workflow 是否能通过 PR 事件自动触发。测试完成后，可根据结果决定是否保留 `dcu-ci-pr` 作为临时触发目标分支。
