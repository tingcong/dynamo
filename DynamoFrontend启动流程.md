# Dynamo frontend 启动流程

> **范围声明：本文说明 `python -m dynamo.frontend` 的源码启动路径。**
>
> **[通用]** 表示适用于 frontend 的公共启动/发现/HTTP 服务机制；**[当前 mocker 环境]** 只描述本机无 GPU、`dynamo.mocker + etcd + TCP + ZMQ` 调试拓扑下看到的具体表现，不应当被当作真实 vLLM、SGLang 或 TensorRT-LLM worker 的实现细节。

本文对应的本机启动命令：

```bash
python -m dynamo.frontend \
  --http-host 127.0.0.1 \
  --http-port 8000 \
  --discovery-backend etcd \
  --request-plane tcp \
  --event-plane zmq \
  --router-mode kv
```

## 1. frontend 的职责 [通用]

frontend 是 HTTP API、模型发现与请求路由层，不是推理 worker。它主要完成：

1. 暴露 OpenAI 兼容 HTTP 接口，例如 `/v1/models`、`/v1/chat/completions`。
2. 从 discovery backend（此处是 etcd）发现 worker 发布的 Model Deployment Card（MDC）。
3. 根据 MDC 创建对应模型的请求处理管线和 router。
4. 发现 worker Endpoint instance 后，通过 request plane（此处是 TCP）把预处理后的请求转发给 worker。
5. 在模型或 worker 上下线时持续更新内存中的路由拓扑。

正常的独立 frontend 启动不会把自己作为推理 worker 写入 `v1/mdc` 或 `v1/instances`；这些是 worker 注册的模型元数据和可路由 Endpoint。frontend 的角色是读取并 watch 这些记录。

```text
HTTP client
   |
   v
dynamo.frontend :8000
   |                    ^
   | HTTP / model watch | etcd discovery
   v                    |
in-memory ModelManager <-+-- worker MDC:      v1/mdc/...
   |                         worker instance: v1/instances/...
   v
router / TCP client
   |
   v
worker endpoint
```

## 2. Python 入口与参数解析 [通用]

模块入口 [components/src/dynamo/frontend/__main__.py](components/src/dynamo/frontend/__main__.py) 直接调用 `dynamo.frontend.main.main()`。

`main()` 位于 [components/src/dynamo/frontend/main.py](components/src/dynamo/frontend/main.py)，按顺序执行：

1. `_raise_fd_limit()` 检查/提升文件描述符限制。
2. `uvloop.run(async_main())` 创建并运行 asyncio event loop。
3. `async_main()` 调用 `parse_args()`，解析 frontend、router、HTTP、discovery 与 processor 参数。
4. 清除继承来的 `DYN_SYSTEM_PORT`，避免 frontend 误占用 worker system metrics port。
5. 如设置了 `--dump-config-to`，输出解析后的配置。

本机命令中的几个关键参数：

| 参数 | 启动阶段的作用 |
|---|---|
| `--http-host/--http-port` | 最终 HTTP API 的监听地址 `127.0.0.1:8000` |
| `--discovery-backend etcd` | 构造 discovery client，读取并 watch etcd 中 worker 发布的记录 |
| `--request-plane tcp` | router 到 worker 使用 TCP 请求通道 |
| `--event-plane zmq` | KV router 的事件通道选择；不是 HTTP 协议 |
| `--router-mode kv` | 创建 KV-aware router 配置 |

## 3. 创建 DistributedRuntime 与退出信号 [通用]

`async_main()` 使用 Python binding 创建 `DistributedRuntime`：

```python
runtime = DistributedRuntime(
    loop,
    config.discovery_backend,
    config.request_plane,
    event_plane=config.event_plane,
)
```

源码位置：[frontend/main.py](components/src/dynamo/frontend/main.py#L348-L355)。

当 discovery backend 为 etcd 时，runtime 从 `ETCD_ENDPOINTS` 读取 endpoint 并建立 discovery 能力；此刻 frontend 尚未发现或加载任何模型。

随后，frontend 为 `SIGINT` 与 `SIGTERM` 注册 handler。收到信号后调用 `runtime.shutdown()`，取消 runtime 的 primary cancellation token，使 HTTP server、model watch 与路由相关任务退出。

## 4. 选择路由模式并构造动态 frontend engine [通用]

frontend 根据 `--router-mode` 构造 `RouterConfig`：

| CLI 值 | Rust `RouterMode` |
|---|---|
| `kv` | `KV` |
| `random` | `Random` |
| `direct` | `Direct` |
| `power-of-two` | `PowerOfTwoChoices` |
| `least-loaded` | `LeastLoaded` |
| `device-aware-weighted` | `DeviceAwareWeighted` |
| 未指定/其他默认值 | `RoundRobin` |

当前无 GPU 调试命令使用 `kv`，因此会同时构造 `KvRouterConfig`。这属于 frontend/router 的通用模式，不依赖 mocker；只是本机用 mocker 作为可发现的 worker。

之后 Python 构造：

```python
e = EntrypointArgs(EngineType.Dynamic, **kwargs)
engine = await make_engine(runtime, e)
```

这里的 `EngineType.Dynamic` 很关键：frontend 没有静态绑定一个模型或一个本地推理 engine，而是等待 discovery 中的远端 MDC，动态添加或删除模型。PyO3 `make_engine()` 在 [lib/bindings/python/rust/llm/entrypoint.rs](lib/bindings/python/rust/llm/entrypoint.rs#L698)，`select_engine()` 会为 `Dynamic` 返回 `EngineConfig::Dynamic`。

若配置了 `--chat-processor vllm` 或 `--chat-processor sglang`，Python 会在此阶段设置 `chat_engine_factory`。它用于后续模型被发现时创建相应的 frontend 预处理/后处理逻辑；这不是启动本地 vLLM 或 SGLang worker。

## 5. 进入 HTTP 模式 [通用]

默认情况下（没有 `--interactive`，也没有 `--kserve-grpc-server`），Python 调用：

```python
await run_input(runtime, "http", engine, frontend_route_extensions)
```

`run_input()` 位于 [lib/bindings/python/rust/llm/entrypoint.rs](lib/bindings/python/rust/llm/entrypoint.rs#L965)，会进入 Rust 的 `HttpFrontend::run()`：[lib/llm/src/entrypoint/input/http.rs](lib/llm/src/entrypoint/input/http.rs#L71)。

启动过程先初始化 request-trace 等可选能力，再创建默认 `DefaultWorkerSelector`。如果项目静态链接了自定义 worker-selection policy，则该 policy 在构建每个 worker set 时使用；模型发现、调度与错误处理仍由 Dynamo 管理。

## 6. 创建 HTTP service [通用]

`run_with_worker_selector_factory()` 基于 `LocalModel` 中的 HTTP 配置构造 `HttpService`：

1. 设置 host、port，或在同时提供证书和私钥时启用 TLS。
2. 设置 request template、metrics、frontend API 开关、runtime metrics registry。
3. 把 runtime discovery client 注入 HTTP service，供 `/health` 与 router 指标等逻辑使用。
4. 加载配置的 frontend route extensions。

源码位置：[lib/llm/src/entrypoint/input/http.rs](lib/llm/src/entrypoint/input/http.rs#L131-L205)。

对于 `EngineConfig::Dynamic`，HTTP service 会先构建出来，再启动 discovery watcher；此时端口可以监听，但尚未必有任何模型 endpoint 被启用。

最终 `HttpService::run()` 在 [lib/llm/src/http/service/service_v2.rs](lib/llm/src/http/service/service_v2.rs#L732) 中绑定 `host:port` 并运行 Axum HTTP server。你本地会看到类似 `Starting HTTP service` 的日志。

## 7. 订阅 etcd 中的模型变化 [通用]

Dynamic frontend 调用 `run_watcher()`，创建 `ModelWatcher`，并对 discovery 发起：

```rust
discovery.list_and_watch(DiscoveryQuery::AllModels, ...).await
```

源码位置：[lib/llm/src/entrypoint/input/http.rs](lib/llm/src/entrypoint/input/http.rs#L294-L360)。这个操作由 discovery backend 返回：

1. 当前已有 MDC 的初始快照。
2. 后续 MDC Added/Removed 事件流。

使用 etcd 时，它观察的是 worker 写入的 `v1/mdc/...` 数据。frontend **不等待某个固定模型**才绑定 HTTP 端口；模型在稍后出现时会被动态加入。

`NamespaceFilter` 会根据 frontend 的 `--namespace` 或 `--namespace-prefix` 过滤发现到的模型；未设置时接受所有 namespace。

## 8. 收到 MDC 后：创建模型组、router 与 API 能力 [通用]

`ModelWatcher` 位于 [lib/llm/src/discovery/watcher.rs](lib/llm/src/discovery/watcher.rs)。收到有效的 `DiscoveryInstance::Model` 后，它会由 `ModelDiscoveryController` 汇总模型卡片，再交给 `ModelManager` 建立一致的 discovery group 和请求处理拓扑。

这个阶段的关键语义是：

1. MDC 提供模型名、endpoint、模型输入类型、worker type、需要的 peer worker 类型及 runtime 配置。
2. watcher 将同一模型/部署的发现信息聚合为可提交的 model group，而不是对每个 etcd 事件立即暴露半成品路由。
3. ModelManager 创建相应的 router、worker set 与预处理/后处理 pipeline。
4. `ModelUpdate::Added` 经 channel 送给 `update_http_endpoints()`；它依据 `ModelType` 启用 Chat、Completions、Embeddings 等 HTTP endpoint，并更新模型指标。

所以 `/v1/models` 能列出模型，并不必然表示每个 worker instance 都可马上处理请求；worker endpoint availability 与 worker-type readiness 仍会参与实际路由决策。

## 9. worker instance 发现与可路由性 [通用]

MDC 解决“提供什么模型、能力是什么”；Endpoint instance 解决“请求发往哪里”。ModelManager 创建的 router/worker set 会对对应 endpoint 的实例进行 discovery，并维护实例增删。

对于 TCP request plane，worker 在 etcd 的 `v1/instances/...` 记录包含 transport 地址，格式类似：

```text
<worker-host>:<port>/<instance-id-hex>/generate
```

router 选择一个满足条件的 instance 后，通过 TCP client 将预处理后的请求发送给它。

在 `--router-mode kv` 下，KV router 还会结合 worker 的 KV 状态/事件进行 worker 选择；其目标仍然是发现的 Endpoint instance，而不是 HTTP frontend 本身。

## 10. 当前无 GPU mocker 环境中的表现 [当前 mocker 环境]

当前本机 worker 是：

```text
dynamo.mocker --model-name debug-mocker
```

因此 frontend 看见的是 mocker 发布的 MDC 和 Endpoint instance。启动后的一条 Chat Completions 请求的链路是：

```text
curl /v1/chat/completions
  → frontend HTTP/Axum route
  → 由 ModelManager 找到 debug-mocker 的 pipeline
  → tokenizer / request preprocessing
  → KV router 选择 mocker Endpoint
  → TCP request plane
  → dynamo.mocker 的模拟 scheduler / token 输出
  → frontend 后处理并以 OpenAI 响应或 SSE 返回
```

这里的“模拟 scheduler / token 输出”只属于 mocker。真实 worker 会在 TCP 请求到达后交由 vLLM、SGLang 或 TensorRT-LLM backend 与 GPU 执行；frontend 的 HTTP、MDC discovery、Endpoint discovery 与路由前半段保持同类架构。

## 11. 关闭流程 [通用]

当收到 `Ctrl+C` 或 `SIGTERM`：

1. Python signal handler 调用 `runtime.shutdown()`。
2. primary cancellation token 取消。
3. `HttpService::run()` 停止 HTTP server。
4. `list_and_watch()`、ModelWatcher、路由任务因 cancellation 结束。
5. `HttpFrontend::run()` 返回后再次调用 runtime shutdown，保证相关任务被取消。

frontend 关闭本身不会主动注销 worker 的 MDC 或 Endpoint；worker 的注册记录由对应 worker 的 shutdown 流程负责清理。

## 12. 调试断点清单

| 顺序 | 文件 | 断点位置 | 类型 | 观察内容 |
|---|---|---|---|---|
| 1 | `components/src/dynamo/frontend/main.py` | `async_main()` | [通用] | CLI 配置、runtime、router mode |
| 2 | `components/src/dynamo/frontend/main.py` | `EntrypointArgs(EngineType.Dynamic, ...)` | [通用] | frontend 动态 engine 参数 |
| 3 | `lib/bindings/python/rust/llm/entrypoint.rs` | `make_engine()` | [通用] | Python → Rust engine 初始化边界 |
| 4 | `lib/llm/src/entrypoint/input/http.rs` | `HttpFrontend::run()` | [通用] | HTTP frontend 初始化 |
| 5 | `lib/llm/src/entrypoint/input/http.rs` | `run_watcher()` | [通用] | 开始 `AllModels` discovery watch |
| 6 | `lib/llm/src/discovery/watcher.rs` | `ModelWatcher::watch()` | [通用] | MDC Added/Removed 的处理 |
| 7 | `lib/llm/src/discovery/model_manager.rs` | discovery group 提交逻辑 | [通用] | 模型组、router、worker set 建立 |
| 8 | `lib/llm/src/http/service/service_v2.rs` | `HttpService::run_inner()` | [通用] | HTTP 监听地址与服务生命周期 |
| 9 | `lib/llm/src/http/service/openai.rs` | Chat Completions handler | [通用] | 单请求进入 OpenAI API 后的处理 |
| 10 | `lib/llm/src/mocker.rs` | `MockerExecutionContext` | [当前 mocker 环境] | 请求已经到达 mocker 后的模拟执行 |

排查“frontend 已启动但没有模型”时，先看第 5、6 步是否拿到了 MDC；排查“模型可见但请求无法发给 worker”时，再看 ModelManager/worker set 是否发现了对应 `v1/instances` 记录，以及 frontend 与 worker 是否使用相同的 `ETCD_ENDPOINTS`、request plane 和 namespace 过滤条件。
