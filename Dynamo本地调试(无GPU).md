# Dynamo 本地调试（无 GPU）

本文是基于本机 macOS、Dynamo 源码和无 GPU mocker 的实际验证流程。目标是调试 Python frontend、模型发现、tokenizer 预处理、KV 路由和 OpenAI 兼容 HTTP 接口；不启动真实 vLLM、SGLang 或 GPU worker。

验证环境：Python 3.12.13、Rust 1.96.1、`protoc`（Homebrew protobuf）、`uv`、本机代理 `127.0.0.1:7897`。

## 1. 拓扑与范围

```text
curl / IDE client
       |
       v
dynamo.frontend :8000  <--- etcd discovery --->  dynamo.mocker
       |                                             |
       +---------------- TCP request plane ----------+
```

- `dynamo.mocker` 只模拟请求处理与 KV 状态，不加载模型权重，也不做真实推理。
- 本文默认通过已运行的 Docker 三节点 etcd 做服务发现；frontend 和 mocker 必须使用同一组 `ETCD_ENDPOINTS`。
- `--event-plane zmq` 保持单机 mocker 调试的轻量事件通道；etcd 只负责服务发现。
- mocker 仍需要模型的配置和 tokenizer，以便 frontend 构造 Chat Completions 管线；本文使用 `Qwen/Qwen3-0.6B`，只下载配置/分词器文件，不下载权重。

不覆盖的内容：CUDA、真实 vLLM/SGLang/TensorRT-LLM 推理、GPU 性能和多机 KV 传输。这些需要 Linux + NVIDIA GPU。

### 为什么默认使用 etcd

`file` discovery 需要通过 `DYN_FILE_KV` 指定一个本地目录，适合零依赖的临时验证；它不使用现有 etcd，也只能在同一台机器上工作。本机已有可用的 etcd 集群，因此本文改为默认使用它：worker 注册有 lease，进程异常退出后 etcd 会在默认 TTL 后清理陈旧注册，且更接近实际部署的发现路径。

本机 Docker etcd 的 host 端口是 `32379`、`32479`、`32579`，三个 endpoint 统一写为：

```bash
export ETCD_ENDPOINTS=http://127.0.0.1:32379,http://127.0.0.1:32479,http://127.0.0.1:32579
```

在启动 Dynamo 前先验证 etcd；若容器已停止，在其 Compose 项目目录启动它们：

```bash
docker ps --format '{{.Names}} {{.Status}} {{.Ports}}' | rg '^etcd[123] '
curl --noproxy '*' -fsS http://127.0.0.1:32379/health

# 仅当上述检查显示 etcd 容器未运行时执行
cd /Users/hutingcong/Documents/services/etcd
docker compose up -d
```

第一条命令中三个容器应为 `Up`，第二条应返回 `{"health":"true"}`。请勿在 Dynamo 仓库中执行 `docker compose up -d`，因为该 Compose 项目位于 `/Users/hutingcong/Documents/services/etcd`。

如需完全不依赖 Docker etcd，可将第 7 节的两个 `--discovery-backend etcd` 改成 `file`，并在两个终端都设置相同的 `DYN_FILE_KV=/private/tmp/dynamo-frontend-debug`。

## 2. 每个终端先配置代理

本机外部依赖下载需要代理。每个用于安装或启动 mocker 的新终端先执行：

```bash
export https_proxy=http://127.0.0.1:7897
export http_proxy=http://127.0.0.1:7897
export all_proxy=socks5://127.0.0.1:7897
```

验证 GitHub：

```bash
curl -I -L https://github.com/swagger-api/swagger-ui/archive/refs/tags/v5.17.14.zip
```

应看到最终 `HTTP/2 200`。Dynamo 的 Rust 构建会由 `utoipa-swagger-ui` 下载这个 Swagger UI 归档；没有代理时这里可能无限等待。

## 3. 系统前置条件

```bash
cd /Users/hutingcong/Documents/GitHub/dynamo

python3 --version
uv --version
rustup --version
xcode-select -p
brew install protobuf
protoc --version
```

`protoc --version` 必须能输出版本号；否则 `etcd-client` 的 proto 编译会报 `Could not find protoc`。

不要同时激活 Conda 和 `.venv`：

```bash
echo "VIRTUAL_ENV=${VIRTUAL_ENV:-<empty>}"
echo "CONDA_PREFIX=${CONDA_PREFIX:-<empty>}"
```

创建 venv 前，两者都应为空；激活 `.venv` 后只应有 `VIRTUAL_ENV`。

## 4. 从零创建虚拟环境并安装构建工具

> 此节会删除仓库根目录的 `.venv`，不会删除源码、Cargo 缓存或 Docker etcd。

```bash
cd /Users/hutingcong/Documents/GitHub/dynamo
rm -rf .venv

uv venv --python 3.12 .venv
source .venv/bin/activate

uv pip install pip 'maturin[patchelf]'
python --version
maturin --version
protoc --version
```

本次验证的 Python 是 `3.12.13`，maturin 是 `1.14.1`。

## 5. 构建 Rust/PyO3 扩展

`maturin` 必须在 Python binding crate 目录运行，不能在仓库根目录运行；根目录 `Cargo.toml` 是 workspace 清单，没有 `package` 字段。

```bash
cd /Users/hutingcong/Documents/GitHub/dynamo
source .venv/bin/activate

cd lib/bindings/python
PROTOC="$(brew --prefix protobuf)/bin/protoc" maturin develop --uv
cd ../../..
```

首次构建会下载并编译大量 Rust crate，耗时较长。成功后验证原生扩展：

```bash
python -c 'import dynamo._core; print(dynamo._core.__file__)'
```

本机验证的扩展路径是：

```text
/Users/hutingcong/Documents/GitHub/dynamo/lib/bindings/python/src/dynamo/_core.abi3.so
```

### VS Code Rust Analyzer 与 Cargo 锁

若输出停在 `Blocking waiting for file lock on package cache`，先检查：

```bash
ps -axo pid,ppid,etime,command | rg 'cargo metadata|maturin|rust-analyzer'
```

常见持锁者是 VS Code Rust Analyzer 的 `cargo metadata`。等待索引结束即可；不要删除 `~/.cargo/.package-cache` 锁文件。若反复竞争，可临时关闭工作区的 Rust Analyzer 后重试。

## 6. 安装 Python editable 包

```bash
cd /Users/hutingcong/Documents/GitHub/dynamo
source .venv/bin/activate

uv pip install -e lib/gpu_memory_service
uv pip install -e .

python -m dynamo.frontend --help
python -m dynamo.mocker --help
```

两条 `--help` 都应以退出码 0 返回。mocker 可能提示 “PyTorch was not found”；对于本流程是预期行为，因为 mocker 不做真实推理。

## 7. 启动无 GPU 调试拓扑

使用三个终端；都进入仓库根目录、激活 `.venv`，并配置第 2 节代理。

统一使用已有的三节点 etcd：

```bash
export ETCD_ENDPOINTS=http://127.0.0.1:32379,http://127.0.0.1:32479,http://127.0.0.1:32579
```

### 终端 A：启动 mocker

先启动 mocker。它会下载 `Qwen/Qwen3-0.6B` 的 tokenizer/config，日志必须出现 `ignore_weights is set to true`，说明模型权重被跳过。

```bash
cd /Users/hutingcong/Documents/GitHub/dynamo
source .venv/bin/activate
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897 all_proxy=socks5://127.0.0.1:7897
export ETCD_ENDPOINTS=http://127.0.0.1:32379,http://127.0.0.1:32479,http://127.0.0.1:32579

python -m dynamo.mocker \
  --model-path Qwen/Qwen3-0.6B \
  --model-name debug-mocker \
  --speedup-ratio 0 \
  --discovery-backend etcd \
  --request-plane tcp \
  --event-plane zmq
```

`--speedup-ratio 0` 表示取消模拟执行延迟。mocker 正常启动后会显示 `All 1 mocker worker(s) created and running`。

### 终端 B：启动 frontend

Chat Completions 需要预处理路由，因此此无 GPU 拓扑使用 `--router-mode kv`，不要使用 `round-robin`。

```bash
cd /Users/hutingcong/Documents/GitHub/dynamo
source .venv/bin/activate
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897 all_proxy=socks5://127.0.0.1:7897
export ETCD_ENDPOINTS=http://127.0.0.1:32379,http://127.0.0.1:32479,http://127.0.0.1:32579

DYN_LOG='dynamo=debug' \
DYN_LOGGING_JSONL=false \
DYN_LOGGING_SPAN_EVENTS=true \
python -m dynamo.frontend \
  --http-host 127.0.0.1 \
  --http-port 8000 \
  --discovery-backend etcd \
  --request-plane tcp \
  --event-plane zmq \
  --router-mode kv
```

以下日志表明模型链路已就绪：

```text
Chat completions is ready
Committed discovered model group ... model_name=debug-mocker
chat endpoints enabled
```

### 终端 C：验证

先确认模型发现：

```bash
curl --noproxy '*' -sS -i http://127.0.0.1:8000/v1/models
```

预期列表中包含 `debug-mocker`。然后发请求：

```bash
curl --noproxy '*' -sS -i http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "debug-mocker",
    "messages": [{"role": "user", "content": "你好，请只回复 OK"}],
    "max_tokens": 8,
    "stream": false
  }'
```

本机实测返回 `HTTP/1.1 200 OK`。响应文字是模拟 token，不代表 Qwen 真实推理结果；它证明的是 HTTP → tokenizer → KV router → TCP mocker → OpenAI 响应的链路。

## 8. VS Code 断点调试

在 VS Code 安装微软的 Python 扩展；若还要调试 Rust，再安装 **CodeLLDB**（vadimcn.vscode-lldb）。使用命令面板执行 **Python: Select Interpreter**，选择：

```text
/Users/hutingcong/Documents/GitHub/dynamo/.venv/bin/python
```

### 创建启动配置

打开 VS Code 左侧的 **Run and Debug**（运行和调试）面板，点击 **create a launch.json file**。要创建 Python 调试配置时，在 “Select debugger” 选择 **Python Debugger**，下一步选择 **Python File**。要创建 Rust 调试配置时选择 **CodeLLDB**。

VS Code 随后会创建 `.vscode/launch.json`。将自动生成的内容替换为下面的两套配置；若仓库已有 `launch.json`，只将两个对象添加到其 `configurations` 数组，不要覆盖原有配置：

```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Frontend(Python, etcd, no GPU)",
            "type": "debugpy",
            "request": "launch",
            "module": "dynamo.frontend",
            "cwd": "${workspaceFolder}",
            "args": [
                "--http-host",
                "127.0.0.1",
                "--http-port",
                "8000",
                "--discovery-backend",
                "etcd",
                "--request-plane",
                "tcp",
                "--event-plane",
                "zmq",
                "--router-mode",
                "kv"
            ],
            "env": {
                "ETCD_ENDPOINTS": "http://127.0.0.1:32379,http://127.0.0.1:32479,http://127.0.0.1:32579",
                "DYN_LOG": "dynamo=debug",
                "DYN_LOGGING_JSONL": "false",
                "DYN_LOGGING_SPAN_EVENTS": "true"
            },
            "justMyCode": false
        },
        {
            "name": "Frontend(Rust, etcd, no GPU)",
            "type": "lldb",
            "request": "launch",
            "sourceLanguages": ["rust"],
            "program": "${workspaceFolder}/.venv/bin/python",
            "args": [
                "-m",
                "dynamo.frontend",
                "--http-host",
                "127.0.0.1",
                "--http-port",
                "8000",
                "--discovery-backend",
                "etcd",
                "--request-plane",
                "tcp",
                "--event-plane",
                "zmq",
                "--router-mode",
                "kv"
            ],
            "cwd": "${workspaceFolder}",
            "env": {
                "ETCD_ENDPOINTS": "http://127.0.0.1:32379,http://127.0.0.1:32479,http://127.0.0.1:32579",
                "DYN_LOG": "dynamo=debug",
                "DYN_LOGGING_JSONL": "true",
                "DYN_LOGGING_SPAN_EVENTS": "true"
            }
        },
        {
            "name": "Worker(Rust, etcd, no GPU)",
            "type": "lldb",
            "request": "launch",
            "program": "${workspaceFolder}/.venv/bin/python",
            "args": [
                "-m",
                "dynamo.mocker",
                "--model-path",
                "Qwen/Qwen3-0.6B",
                "--model-name",
                "debug-mocker",
                "--speedup-ratio",
                "0",
                "--discovery-backend",
                "etcd",
                "--request-plane",
                "tcp",
                "--event-plane",
                "zmq"
            ],
            "cwd": "${workspaceFolder}",
            "env": {
                "https_proxy": "http://127.0.0.1:7897",
                "http_proxy": "http://127.0.0.1:7897",
                "all_proxy": "socks5://127.0.0.1:7897",
                "ETCD_ENDPOINTS": "http://127.0.0.1:32379,http://127.0.0.1:32479,http://127.0.0.1:32579",
                "DYN_LOG": "dynamo=debug",
                "DYN_LOGGING_JSONL": "false"
            }
        }
    ]
}
```

### 选择正确的启动配置

- 调试 `components/src/dynamo/frontend/` 中的 Python 代码：选择 `Dynamo Frontend (etcd, no GPU)`。
- 调试 `lib/` 或 `lib/bindings/python/src/` 中的 Rust 代码：选择 `Dynamo Frontend Rust (etcd, no GPU)`。

两套配置均会启动同一个 frontend，并绑定 `127.0.0.1:8000`，**不能同时运行**。切换到 Rust 调试前，先在 VS Code 停止正在运行的 `Dynamo Frontend (etcd, no GPU)`；mocker 不要停止。Rust 调试配置会通过 CodeLLDB 启动 `.venv/bin/python`，再加载带调试符号的 PyO3 动态库。`initCommands` 中的 `target.process.stop-on-exec false` 是 macOS 上调试 Python 所必需：Python 启动时会触发 `exec`，CodeLLDB 若在此事件停住会导致调试会话异常结束。

在 VS Code 的 **Run and Debug** 面板选择其中一套配置，按 `F5` 启动 frontend；随后在终端 A 启动 mocker，最后在终端 C 发请求。

推荐断点：

- `components/src/dynamo/frontend/main.py`：启动和运行时初始化。
- `components/src/dynamo/frontend/openai.py`：OpenAI 请求/响应处理。
- `components/src/dynamo/frontend/frontend_args.py`：命令行参数。
- `components/src/dynamo/mocker/main.py`：mocker 启动、模型元数据与注册。
- `lib/llm/src/discovery/watcher.rs`：模型发现及 Chat pipeline 物化（Rust）。
- `lib/runtime/src/transports/etcd.rs`：etcd endpoint、连接与注册（Rust）。
- `lib/bindings/python/src/`：Python 调用 Rust 的 PyO3 边界。

Python frontend 源码改动后直接重启 frontend。Rust 或 PyO3 binding 改动后，需要先停止调试会话，再重新运行第 5 节的 `maturin develop --uv`，然后重新按 `F5` 启动 Rust 配置。

## 9. 常见错误

### `Both VIRTUAL_ENV and CONDA_PREFIX are set`

Conda 与 venv 同时激活。新开终端，确认 `CONDA_PREFIX` 为空后再执行：

```bash
cd /Users/hutingcong/Documents/GitHub/dynamo
source .venv/bin/activate
```

### 无法连接 etcd

确认三个容器仍运行，并检查 endpoint 健康状态：

```bash
docker ps --format '{{.Names}} {{.Ports}}' | rg '^etcd[123] '
curl --noproxy '*' -sS http://127.0.0.1:32379/health
```

三个 `etcd` 容器应分别映射到 `32379`、`32479`、`32579`；`/health` 应返回 JSON 且 `health` 为 `true`。frontend 和 mocker 必须使用完全相同的 `ETCD_ENDPOINTS`。

### `Could not find protoc`

```bash
brew install protobuf
PROTOC="$(brew --prefix protobuf)/bin/protoc" maturin develop --uv
```

注意必须在 `lib/bindings/python` 目录运行最后一条命令。

### 根目录运行 maturin 后提示 `missing field package`

原因：仓库根目录是 Cargo workspace，而非 Python extension crate。执行：

```bash
cd /Users/hutingcong/Documents/GitHub/dynamo/lib/bindings/python
maturin develop --uv
```

### 编译停在 Swagger UI 下载

确认该终端已配置第 2 节代理；构建脚本会访问 GitHub。不要为了绕过网络而修改 `Cargo.toml` 的依赖 feature。

### frontend 日志出现 `chat pipeline requires preprocessed routing`

原因通常是 mocker 没有 tokenizer 元数据，或 frontend 使用了 `--router-mode round-robin`。按第 7 节：mocker 传入 `--model-path Qwen/Qwen3-0.6B`，frontend 使用 `--router-mode kv`。

## 10. 停止与清理

在 frontend 和 mocker 终端分别按 `Ctrl+C`。Dynamo 使用 etcd lease 注册；正常退出会撤销注册，异常退出时 etcd 会在 lease TTL 后自动清理。因此不需要删除本地 discovery 目录，也不要清空现有 etcd 数据。

## 11. 日常最小启动命令

完成首次构建后，只需启动 mocker、frontend 和请求验证：

```bash
# 终端 A：mocker
source /Users/hutingcong/Documents/GitHub/dynamo/.venv/bin/activate
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897 all_proxy=socks5://127.0.0.1:7897
export ETCD_ENDPOINTS=http://127.0.0.1:32379,http://127.0.0.1:32479,http://127.0.0.1:32579
python -m dynamo.mocker --model-path Qwen/Qwen3-0.6B --model-name debug-mocker --speedup-ratio 0 --discovery-backend etcd --request-plane tcp --event-plane zmq
```

```bash
# 终端 B：frontend
source /Users/hutingcong/Documents/GitHub/dynamo/.venv/bin/activate
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897 all_proxy=socks5://127.0.0.1:7897
export ETCD_ENDPOINTS=http://127.0.0.1:32379,http://127.0.0.1:32479,http://127.0.0.1:32579
DYN_LOG='dynamo=debug' DYN_LOGGING_JSONL=false DYN_LOGGING_SPAN_EVENTS=true python -m dynamo.frontend --http-host 127.0.0.1 --http-port 8000 --discovery-backend etcd --request-plane tcp --event-plane zmq --router-mode kv
```

```bash
# 终端 C：验证
curl --noproxy '*' -sS -i http://127.0.0.1:8000/v1/models
```

```bash
curl --noproxy '*' -sS -i http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "debug-mocker",
    "messages": [{"role": "user", "content": "你好，请只回复 OK"}],
    "max_tokens": 8,
    "stream": true
  }'
```

```bash
curl --noproxy '*' -sS -i http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'x-dynamo-request-id: ths_request_id_test' \
  -d '{
    "model": "debug-mocker",
    "messages": [{"role": "user", "content": "你好，请只回复 OK"}],
    "max_tokens": 8,
    "stream": true
  }'
```
