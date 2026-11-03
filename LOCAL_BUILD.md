<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# macOS 本地构建与入口验证

以下命令为 frontend 和 mocker 创建共用的 Python 环境，并编译 Rust 扩展。使用 zsh，在每一步成功后再继续；已有环境时，`uv venv` 不代表清理旧构建产物。

```bash
setopt interactivecomments  # 允许 zsh 识别行尾注释

cd /Users/hutingcong/Documents/GitHub/dynamo  # 进入仓库根目录；其他机器请替换路径

uv venv .venv  # 创建项目虚拟环境
source .venv/bin/activate  # 激活虚拟环境
unset CONDA_PREFIX  # 避免 maturin 检测到 Conda 和 venv 同时激活

uv pip install pip 'maturin[patchelf]'  # 安装构建工具

cd lib/bindings/python  # 进入 Rust/Python 绑定目录
maturin develop --uv  # 编译并安装 frontend 和 mocker 共用的 Rust 扩展
cd ../../..  # 返回仓库根目录

uv pip install -e lib/gpu_memory_service  # 以可编辑模式安装 GPU 内存服务包
uv pip install -e .  # 以可编辑模式安装 Dynamo 及其依赖

python -m dynamo.frontend --help  # 验证 frontend 入口能否加载
python -m dynamo.mocker --help  # 验证 mocker 入口能否加载
```

## 验证边界与常见问题

- 两个 `--help` 成功仅表示入口加载正常；etcd 连接、worker 注册和实际请求需要另外验证。
- 更新源码后出现 `AisPerfConfig` / `AicPerfConfig` 导入错误，可能是旧 `_core.abi3.so` 与源码不匹配，应重新编译绑定。
- `Both VIRTUAL_ENV and CONDA_PREFIX are set` 表示两种环境标记同时存在；以上 `unset CONDA_PREFIX` 只影响当前终端，不删除 Conda。
- 这些是本地 macOS 无 GPU 调试环境的构建步骤，不代表 GPU 后端运行已验证。
