# 带 K3 Execution Provider 的 RV64GC/musl 全静态 ONNX Runtime

`ort_learn` 是一个简短的 Rust 推理程序：它读取 `bus.png`，执行
`../yolov5n.onnx`，并打印第一个输出 Tensor 的形状。最终产物是面向
RV64GC + musl 的全静态可执行文件，并使用仓库内的 K3 Execution Provider
（以下简称 K3 EP）。

整体链接和调用关系如下：

```text
Rust 应用程序（ort crate）
  -> 本地 ort-sys 的静态链接配置
  -> libonnxruntime + libonnxruntime_providers_k3.a
  -> libk3_ai_runtime.a
  -> K3 AI runner 设备驱动
```

K3 EP 在 `src/main.rs` 中通过 C API
`OrtSessionOptionsAppendExecutionProvider_K3` 直接注册。上游 Rust `ort`
crate 目前没有 `ort::ep::K3`，因此这里直接调用 C API 是有意为之。

## K3 与 CPU fallback 的行为

在 Session 初始化阶段，K3 EP 会声明它支持的节点；没有被 K3 声明的节点由
CPU EP 执行，因此同一个 ONNX 模型可以一部分在 K3 上执行、一部分在 CPU 上
执行。

但这不是失败重试机制：节点一旦已被 K3 EP 声明，之后 K3 执行失败时，ONNX
Runtime 会直接返回错误，不会把该节点重新交给 CPU。例如
`/model.0/conv/Conv` 返回错误，表示这个 Conv 已经进入 K3，只是 K3 运行失败。

## 前置条件

- ONNX Runtime 源码 checkout（由 `ONNXRUNTIME_ROOT` 指向）
- K3 runtime 仓库根目录（由 `K3_RUNTIME_ROOT` 指向）
- RISC-V musl 工具链根目录（由 `RISCV_MUSL_TOOLCHAIN_ROOT` 指向，其 `bin` 目录需在 `PATH` 中）
- 当前 Rust toolchain 已安装 `riscv64gc-unknown-linux-musl` target
- 目标 RV64GC/musl 系统已加载 K3 AI runner 设备驱动

不要传入 `--build_shared_lib`：本项目需要链接 ORT 的静态库。

## 首次构建：ORT 与 K3 runtime

在 `ort_test` 目录设置以下环境变量；路径由使用者按本地目录结构提供：

```bash
export ORT_TEST_ROOT="$PWD"
export ONNXRUNTIME_ROOT='<ONNX Runtime source checkout>'
export K3_RUNTIME_ROOT='<K3 runtime repository root>'
export RISCV_MUSL_TOOLCHAIN_ROOT='<RISC-V musl toolchain root>'
export PATH="$RISCV_MUSL_TOOLCHAIN_ROOT/bin:$PATH"
```

随后在 ONNX Runtime 源码目录执行：

```bash
cd "$ONNXRUNTIME_ROOT"

python3 tools/ci_build/build.py \
  --config Release \
  --build_dir "$PWD/build/riscv64gc-musl" \
  --update --build \
  --skip_tests \
  --parallel \
  --rv64 \
  --riscv_toolchain_root "$RISCV_MUSL_TOOLCHAIN_ROOT" \
  --use_k3 \
  --k3_runtime_root "$K3_RUNTIME_ROOT" \
  --k3_runtime_mode device \
  --cmake_extra_defines \
    CMAKE_TOOLCHAIN_FILE="$PWD/cmake/riscv64-musl.toolchain.cmake" \
    RISCV_MUSL_TOOLCHAIN_ROOT="$RISCV_MUSL_TOOLCHAIN_ROOT" \
    onnxruntime_BUILD_UNIT_TESTS=OFF \
    onnxruntime_ENABLE_CPUINFO=OFF

# 这两个静态库被标记为 EXCLUDE_FROM_ALL；但 Rust 的静态链接仍需要它们。
cmake --build "$PWD/build/riscv64gc-musl/Release" \
  --target re2 model_package -- -j"$(nproc)"
```

K3 runtime 仅支持 `device` 模式。它会将 `k3_aiRuntime` 交叉编译成
`riscv64gc-unknown-linux-musl` 静态库，再链接进
`libonnxruntime_providers_k3.a`，所有算子都经 K3 AI runner 设备提交给内核。

本配置明确关闭 `cpuinfo`，因为它不支持当前的 RISC-V 构建。构建完成后可检查
CMake 缓存：

```bash
rg '^(onnxruntime_USE_K3|K3_RUNTIME_MODE|K3_RUNTIME_ROOT):' \
  build/riscv64gc-musl/Release/CMakeCache.txt
```

预期可见 `onnxruntime_USE_K3:BOOL=ON` 和
`K3_RUNTIME_MODE:STRING=device`。

## 构建 Rust 应用程序

```bash
cd "$ORT_TEST_ROOT"
cargo build --release
```

本项目故意使用 `third_party/ort-sys` 这个本地 `ort-sys` 副本。它包含全静态
ORT 链接所需的适配：`model_package`、K3 provider 静态库、K3 Rust runtime
静态库，以及系统库 `dl`、`pthread`、`m`。不要删除 `Cargo.toml` 中的
`[patch.crates-io]`；否则会退回上游的动态链接配置。

验证最终产物是全静态 RISC-V 文件：

```bash
file target/riscv64gc-unknown-linux-musl/release/ort_learn
readelf -d target/riscv64gc-unknown-linux-musl/release/ort_learn
```

第二条命令应输出 `There is no dynamic section in this file`。

## Host 构建和运行

默认 `cargo build` / `cargo run` 保持 RISC-V/musl 目标。要在开发主机上构建或运行
x86_64 GNU 版本，使用 Cargo aliases：

```bash
cargo b-host
cargo r-host -- -t   # 运行 concat.onnx 测试图
```

Cargo 不支持 `cargo b host` 这种位置参数形式，因此项目提供等价的
`b-host` / `r-host` aliases。host 首次构建会由 `ort` 下载匹配的 x86_64 ONNX
Runtime；它不会注册 K3 EP，也不会链接 RISC-V 的 K3 库。RISC-V 的 K3 库路径
使用项目私有的 `K3_ORT_LIB_PATH` 配置，而非全局 `ORT_LIB_PATH`，避免 host
误用交叉编译产物。

## 同步 release 二进制到 rootfs 镜像

先完成 RISC-V release 构建，再运行：

```bash
cargo build --release
python3 tools/sync_result.py
```

脚本会使用 `sudo losetup --find --show --partscan` 挂载指定的 Alpine ext4 rootfs，
将当前目录的 `concat.onnx`、`yolov5n.onnx` 和
`target/riscv64gc-unknown-linux-musl/release/ort_learn` 原子替换为镜像中的
rootfs 中测试目录的对应文件，最后自动卸载并 detach loop 设备；任一源文件缺失时
会在挂载镜像前失败，即使复制失败也会执行清理。

## 修改 K3 AI runtime 后如何重建

K3 runtime 是外部 Rust 项目。因此修改 K3 runtime 后，仅运行
`cargo build --release` 不足以把新代码放进最终程序：必须先重建 K3 静态库，
再重新链接 Rust 可执行文件。

如果只是编辑已有的 `k3_aiRuntime/src/` 或 `k3_aiUabi/src/` 中的 `.rs` 文件，
使用这个快速重建流程：

```bash
cd "$ONNXRUNTIME_ROOT"
cmake --build "$PWD/build/riscv64gc-musl/Release" \
  --target onnxruntime_providers_k3 -- -j"$(nproc)"

cd "$ORT_TEST_ROOT"
rm -f target/riscv64gc-unknown-linux-musl/release/ort_learn
cargo build --release
```

CMake 的 K3 规则现在会监视 device runtime 所需的全部 Rust 源文件。因此
上面的第一条构建命令会在修改 `fronted/kd_uring.rs`、
`fronted/tensormanager.rs` 或 `ort_ffi.rs` 等文件后重新调用 Cargo；它也会追踪
`k3_aiUabi`。

如果新增或删除 Rust 源文件、修改任意 K3 `Cargo.toml`、更换 Rust target /
工具链，请重新执行“首次构建”中的完整命令（带
`--update --build`），然后再构建 `re2` 和 `model_package`。这会刷新 CMake 的
源文件列表和所有构建配置。

上面 `rm -f` 只会删除生成的最终可执行文件，目的是强制 Cargo 重新进行最后的
链接，避免把旧二进制复制到设备上。

## 部署和运行

按 `main.rs` 当前使用的相对路径部署：

```text
deploy/
  ort_dir/
    ort_learn
    bus.png
    concat.onnx
    yolov5n.onnx
```

在 `deploy/ort_dir` 中执行时，程序从当前目录读取 `bus.png`、`concat.onnx` 和
`yolov5n.onnx`。

```bash
./ort_learn
```

启用详细 K3 runtime 调试日志：

```bash
K3_ORT_DEBUG=1 ./ort_learn 2>&1 | tee k3-ort.log
```

常见日志阶段含义：

| 日志 | 含义 |
| --- | --- |
| `stage=request` 到 `stage=descriptor` | ORT Tensor 已完成校验，并被转换为 K3 节点请求。 |
| `stage=graph-ready` | K3 图对象已准备完成。 |
| `stage=channel-build-*` | runtime 正在打开或配置 K3 设备通道。 |
| `stage=submit-*` | 图执行请求已提交给 K3 驱动。 |
| `stage=completion-*` | K3 已返回完成事件；失败时可能含有 `node_id`、`op`、`backend_err`。 |

遇到 `K3 node execution failed (-7)` 时，请开启该日志。最终的详细阶段可以区分
问题发生在构建设备通道、提交图，还是等待设备完成事件。

## 预期程序输出

对于兼容的 YOLOv5 模型，成功时输出类似：

```text
image load success
model load success
YOLOv5 output shape: [1, 25200, 85]
K3 executed nodes: <non-zero count>
```

实际输出形状及 K3 执行节点数取决于 ONNX 模型和 K3 当前支持的算子。
