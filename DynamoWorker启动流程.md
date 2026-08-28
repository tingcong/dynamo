# Dynamo worker 启动流程：mocker 实现

> **范围声明：本文追踪的是 `dynamo.mocker`，不是 vLLM、SGLang 或 TensorRT-LLM 的真实 GPU worker。**
>
> 文中以 **[通用]** 标记的步骤属于 Dynamo runtime / endpoint / discovery 的公共机制，真实 worker 也会复用或遵循同类流程；以 **[仅 mocker]** 标记的步骤只适用于无 GPU 的 mocker 实现，不能据此推断真实推理引擎的内部启动、模型加载、调度和 KV 管理行为。

本文基于当前源码说明无 GPU 环境下 `python -m dynamo.mocker` 启动一个 mocker worker 时，Python 与 Rust 分别执行的动作、etcd 中写入的内容，以及 worker 何时真正可被 frontend 路由。

适用启动命令：

```bash
python -m dynamo.mocker \
  --model-path Qwen/Qwen3-0.6B \
  --model-name debug-mocker \
  --speedup-ratio 0 \
  --discovery-backend etcd \
  --request-plane tcp \
  --event-plane zmq
```

其中，etcd 仅承担服务发现；TCP 是 frontend 到 worker 的请求通道；ZMQ 是 **[仅 mocker]** 的 KV 事件通道。

## 1. 总览 [仅 mocker]

```text
python -m dynamo.mocker
        |
        v
解析参数、创建 DistributedRuntime（etcd / TCP / ZMQ）
        |
        v
下载模型配置和 tokenizer（跳过权重）并创建 MockerExecutionContext
        |
        v
为 dyn://<namespace>.<component>.<endpoint> 创建请求处理流水线
        |
        +--> 注册 MDC 到 etcd：v1/mdc/...        （模型元数据）
        |
        +--> 绑定 TCP request-plane endpoint
        |
        +--> 注册 Endpoint 到 etcd：v1/instances/... （可路由 worker）
        |
        v
MockerExecutionContext 发现自身 Endpoint 已出现，启动模拟调度器和模拟 KV 事件发布
```

默认 endpoint 是参数 `--endpoint` 的默认值；当前本地调试通常是 `dyn://dynamo.mocker.generate`。一个 endpoint 由 namespace、component、endpoint name 三部分组成。

## 2. Python 入口与参数准备 [仅 mocker]

入口文件是 [components/src/dynamo/mocker/__main__.py](components/src/dynamo/mocker/__main__.py)，它调用 `dynamo.mocker.main.main()`；`main()` 用 `uvloop` 运行异步的 `worker()`。

在 [components/src/dynamo/mocker/main.py](components/src/dynamo/mocker/main.py) 中，worker 会依次：

1. 解析 mocker 参数，构造 `MockEngineArgs`。
2. 根据 `--num-workers` 循环创建 worker；默认只创建一个。
3. 每个 worker 分别创建一个 `DistributedRuntime`，因此多 worker 不是共享同一个 runtime。
4. 创建传给 Rust 的 `EntrypointArgs`：其中包含模型路径、模型名、endpoint、KV block size、prefill/decode 角色及 mocker 配置。
5. 调用 PyO3 暴露的 `make_engine()` 创建 Rust engine 配置。
6. 调用 `run_input()` 启动 Rust endpoint 服务；该 Future 会一直运行到进程退出。

对应核心调用：

```python
runtime, loop = create_runtime(...)
engine_config = await make_engine(runtime, entrypoint_args)
future = run_input(runtime, args.endpoint, engine_config)
```

源码位置：[main.py](components/src/dynamo/mocker/main.py#L160-L220)。

## 3. 创建 DistributedRuntime：连接 discovery 和传输层 [通用]

`create_runtime()` 在 [components/src/dynamo/common/utils/runtime.py](components/src/dynamo/common/utils/runtime.py) 中创建 Python binding 的 `DistributedRuntime`。

```python
runtime = DistributedRuntime(
    loop,
    discovery_backend,  # etcd
    request_plane,      # tcp
    event_plane=event_plane,  # zmq
)
```

当 `--discovery-backend etcd` 时，Rust runtime 使用 `ETCD_ENDPOINTS` 建立 etcd client，并把 discovery 操作交给 KV store 实现。此时还没有注册 worker；这里只是具备读写 discovery 的能力。

`--request-plane tcp` 使 worker 稍后启动本地 TCP 服务。绑定端口之后，端口和 worker 的 instance ID 会编码进 discovery 的 transport 地址。

## 4. `make_engine`：准备模型元数据与 mocker 引擎 [仅 mocker]

PyO3 的 `make_engine()` 在 [lib/bindings/python/rust/llm/entrypoint.rs](lib/bindings/python/rust/llm/entrypoint.rs#L698) 中。

它做以下事情：

1. 使用 `EntrypointArgs` 填充 `LocalModelBuilder`：模型名、endpoint、KV block size、router/runtime 配置等。
2. 处理 `--model-path`。
   - 本地路径存在则直接使用。
   - 路径不存在则调用 `LocalModel::fetch()` 从 Hugging Face 获取所需文件。
   - 对 `EngineType::Mocker`，`ignore_weights=true`，因此只下载模型配置和 tokenizer，不下载模型权重。**这是 mocker 特性；真实 GPU worker 会执行各自 backend 的模型权重加载。**
3. `builder.build()` 生成 `LocalModel` 和其 Model Deployment Card（MDC）。此时 MDC 只在内存中，尚未写入 etcd。
4. `select_engine()` 识别 `EngineType::Mocker`，调用 `make_mocker_engine()` 创建 `MockerExecutionContext`。

`make_mocker_engine()` 位于 [lib/llm/src/mocker.rs](lib/llm/src/mocker.rs#L1279)。它先创建 engine 对象，再启动一个后台任务等待 component 下出现可用 Endpoint instance。这个后台任务不会抢先将 worker 注册到 etcd；真正的注册由后面的 endpoint 启动步骤完成。

## 5. `run_input`：将 mocker engine 变为网络服务 [仅 mocker]

Python 调用的 `run_input()` 仍在 [entrypoint.rs](lib/bindings/python/rust/llm/entrypoint.rs#L965)，它把 endpoint 字符串解析为 Rust `Input::Endpoint`，再进入 [lib/llm/src/entrypoint/input/endpoint.rs](lib/llm/src/entrypoint/input/endpoint.rs)。

对于 mocker，`EngineConfig::InProcessTokens` 分支会：

1. 创建 `SegmentSource → Backend → ServiceBackend` 请求流水线。
2. 创建对应的 `Ingress`，使 TCP request-plane 收到的请求可进入 mocker engine。
3. 根据角色设置模型声明：
   - 未使用 `--is-prefill-worker` 和 `--is-decode-worker`：`WorkerType::Aggregated`，支持 Chat 和 Completions。
   - prefill worker：`WorkerType::Prefill`，不暴露 OpenAI surface，依赖 Decode worker。
   - decode worker：`WorkerType::Decode`，支持 Chat 和 Completions，依赖 Prefill worker。
4. 调用 `model.attach(...)`。
5. 调用 `endpoint.endpoint_builder().handler(ingress).start()` 启动 endpoint。

注意顺序：`attach()` 在 `start()` 之前，因此 MDC 会先于 endpoint 实例写入 etcd。

## 6. 第一次 etcd 注册：MDC（模型元数据）[通用]

`model.attach(...)` 在 [lib/llm/src/local_model.rs](lib/llm/src/local_model.rs#L619) 中完成。它先把 `model_type`、`model_input`、`worker_type`、`needs` 写入 `ModelDeploymentCard`，然后调用：

```rust
register_model_card(endpoint, &self.card).await
```

`register_model_card()` 将 card 转换为 `DiscoverySpec::Model`，并执行 `discovery.register(spec)`。

在 etcd KV discovery 中，模型元数据被写入 bucket：

```text
v1/mdc/<namespace>/<component>/<endpoint>/<instance-id>
```

这一步让 frontend 能发现 `debug-mocker` 模型、了解模型的输入类型和 worker 类型；但它**不代表 worker 已经可接收请求**。此时 TCP 服务还没有完成 endpoint 注册，router 不能据此向它转发请求。真实 worker 的 MDC 内容会随 backend、模型和部署模式变化，但“模型元数据注册不等于实例可路由”的语义相同。

## 7. 启动 TCP 服务并注册 Endpoint：worker 正式上线 [通用]

`endpoint_builder().start()` 最终进入 [lib/runtime/src/component/endpoint.rs](lib/runtime/src/component/endpoint.rs#L128) 的 `EndpointConfigBuilder::start_with_registration()`。

按顺序执行：

1. 获取或启动 request-plane server：`endpoint.drt().request_plane_server().await?`。
2. 根据 TCP server 已绑定的地址、端口、`connection_id` 和 endpoint 名生成 transport 信息，例如：

   ```text
   <host>:<port>/<instance-id-hex>/generate
   ```

3. 在 TCP server 中注册 handler：

   ```rust
   server.register_endpoint(..., handler, connection_id, ...).await?
   ```

   此时 worker 本地已经具备接收请求的能力，但外部 router 尚不知道它的地址。

4. 创建 `DiscoverySpec::Endpoint`，调用：

   ```rust
   discovery.register(discovery_spec).await
   ```

这就是“worker 注册到 etcd”的准确步骤，源码在 [endpoint.rs](lib/runtime/src/component/endpoint.rs#L231-L265)。注册成功之后，frontend/router 才能发现 worker，并通过保存的 TCP transport 向它发送请求。

在 etcd KV discovery 中，Endpoint 实例被写入：

```text
v1/instances/<namespace>/<component>/<endpoint>/<instance-id>
```

底层实现是 [lib/runtime/src/discovery/kv_store.rs](lib/runtime/src/discovery/kv_store.rs#L346-L470)：它把 `DiscoverySpec` 转为 `DiscoveryInstance`、序列化为 JSON，再通过 KV bucket 的 `insert()` 写入。

## 8. mocker 进入可工作状态 [仅 mocker]

Endpoint 注册后，`make_mocker_engine()` 中先前启动的后台任务可以通过 `component.list_instances()` 观察到该 component 至少有一个 Endpoint instance。随后它取得 endpoint 并调用 `MockerExecutionContext::start(...)`。

该阶段会启动 mocker 的运行逻辑，包括：

1. 初始化 mock scheduler 与 KV cache **模拟**状态。
2. 根据 `MockEngineArgs` 启动一个或多个模拟执行 engine。
3. 初始化 worker 指标发布。
4. 依据 `--event-plane zmq` 设置 KV 事件 sink，并把 KV block 的增删事件发布给 KV router。
5. 等待 frontend 经 TCP request-plane 转发的预处理请求；收到后由 mock scheduler 模拟排队、KV 分配和 token 生成。

真实 vLLM/SGLang/TensorRT-LLM worker 在此阶段会进入各自的 GPU backend 初始化、权重加载、CUDA/KV-cache 分配和 engine loop；这些不经过 `MockerExecutionContext`，也不由本文的 `lib/llm/src/mocker.rs` 覆盖。

因此，日志中即使已经看到 MDC 注册，也应以 Endpoint 注册成功或 frontend 发现到实例为准，判断 worker 是否真正 ready。

## 9. 关闭时的清理 [通用 endpoint；信号入口仅 mocker]

`mocker/main.py` 为 `SIGINT` 和 `SIGTERM` 注册了 graceful shutdown handler。runtime 的 cancellation token 被取消后，endpoint 启动时建立的清理任务会：

1. 调用 `discovery.unregister(discovery_instance)` 删除对应的 Endpoint instance。
2. 从 TCP request-plane server 注销 endpoint handler。
3. 注销 graceful shutdown tracker 中的 endpoint。

清理逻辑在 [lib/runtime/src/component/endpoint.rs](lib/runtime/src/component/endpoint.rs#L267-L303)。MDC 的清理由 discovery/runtime 的 shutdown 路径协同处理；排障时如果进程被强制终止，可检查 etcd 中是否仍有陈旧的 `v1/mdc` 或 `v1/instances` 记录。

## 10. 调试断点清单（mocker 路径）[仅 mocker，除标注项外]

按启动时间顺序设置这些断点，最容易看清完整流程：

| 顺序 | 文件 | 断点位置 | 观察内容 |
|---|---|---|---|
| 1 | `components/src/dynamo/mocker/main.py` | `create_runtime(...)` | 启动参数和每个 worker 的 runtime |
| 2 | `lib/bindings/python/rust/llm/entrypoint.rs` | `make_engine()` | `LocalModelBuilder`、是否跳过权重 |
| 3 | `lib/llm/src/mocker.rs` | `make_mocker_engine()` | mocker context 和后台启动任务 |
| 4 | `lib/llm/src/entrypoint/input/endpoint.rs` | `model.attach(...)` | Aggregated/Prefill/Decode 角色 |
| 5 | `lib/llm/src/local_model.rs` | `register_model_card(...)` | MDC 的第一次 discovery 注册 **[通用]** |
| 6 | `lib/runtime/src/component/endpoint.rs` | `server.register_endpoint(...)` | TCP handler 已绑定 **[通用]** |
| 7 | `lib/runtime/src/component/endpoint.rs` | `discovery.register(discovery_spec)` | worker 正式加入路由池 **[通用]** |
| 8 | `lib/runtime/src/discovery/kv_store.rs` | `bucket.insert(...)` | 实际写入 etcd 的 bucket、key、JSON **[通用 etcd backend]** |
| 9 | `lib/llm/src/mocker.rs` | `startup_engine.start(endpoint)` | endpoint 出现后 mocker engine 开始运行 |

如果目的是确认 frontend 为什么没有看到 worker，优先检查第 5、7、8 步：MDC 与 Endpoint instance 必须都注册成功，且 frontend 与 worker 必须使用同一个 `ETCD_ENDPOINTS`。
