# 怎么使用 / 复现本项目

本项目运行时依赖 StarryOS 中的 AI runner 内核驱动（`k3_airunner`）。需要特别说明的是：`k3_airunner` 驱动源码和 StarryOS 板级集成不在本仓库中，而是在配套的 `tgoskits` 仓库中维护。本仓库主要保存 AI runtime、UAPI、调度器、backend 和测试程序等 crate，这些 crate 会作为独立组件编译进 StarryOS 的用户态或内核态。

两种运行环境：QEMU 模拟和 SpacemiT K3 COM260 Kit 真板。

> **说明**：本复现目前最大的作用是让 StarryOS 能在 K3 板子上启动，并验证 AI runtime 相关模块的集成路径。`/dev/k3_airunner` 内核驱动和 K3 板级适配在 `tgoskits` 仓库中；本仓库附带用户态算子库、UAPI、调度器、backend 和测试程序等开发产物。多数模块仍处于阶段一早期，尚未形成完整的端到端推理通路。
>
> **板级驱动支持状态**：
>
> | 驱动 | 状态 |
> |---|---|
> | UFS | 测试版，有问题欢迎提 issue |
> | GMAC（网卡） | 开发中 |
> | SDMMC（SD 卡） | 开发中 |

## 环境准备

### 克隆项目

```bash
git clone https://github.com/Dirinkbottle/k3x-HERA-RT.git
cd k3x-HERA-RT
```

本项目依赖 `tgoskits` 仓库。**必须切换到 `dev` 分支**，K3 板级支持和 AI runner 设备驱动都在该分支上：

```bash
git clone https://github.com/Dirinkbottle/tgoskits.git
cd tgoskits
git checkout dev
```

两个仓库必须在同一目录!

### 安装 Rust 工具链

需要 nightly Rust 和 riscv64 目标：

```bash
rustup default nightly
rustup target add riscv64gc-unknown-linux-musl
rustup target add riscv64gc-unknown-none-elf
```

## QEMU 模拟运行

在 **tgoskits 仓库根目录** 下执行：

```bash
cd tgoskits
cargo starry qemu --arch riscv64
```

这会用 QEMU 启动 StarryOS


## SpacemiT K3 COM260 Kit 真板运行

### 1. 构建内核

在 tgoskits 仓库根目录下，使用 K3 板级配置构建：

```bash
cd tgoskits
cargo starry build --config os/StarryOS/configs/board/spacemitk3-com260kit.toml
```

构建产物位于 `target/riscv64gc-unknown-linux-musl/release/starryos.bin`。

板级配置默认开启：
- `k3-ufs`：UFS 块设备驱动
- `k3-gmac`：GMAC 网卡驱动（开发中）
- AI runner 设备 `/dev/k3_airunner`
- `/proc/set_ai_thread` 接口

### 2. 刷写 rootfs

K3 开发板使用官方 Yocto BSP 分区表。分区表路径：

```
riscv-yocto/riscv-yocto/layers/meta-riscv/recipes-core/images/titan-cfg/partition_universal.json
```

板上已有原厂 UFS 镜像时，只需刷写 rootfs 分区。将 Starry 构建产物中的 rootfs 镜像写入选定分区：

```
tgoskits/tmp/axbuild/rootfs/rootfs-riscv64-alpine.img/rootfs-riscv64-alpine.img
```

为方便比赛复现，配套 `tgoskits` 仓库根目录也会放置一份可用 rootfs 镜像：

```text
tgoskits/root_fs.img
```

该镜像用于 K3 真板启动和 AI runtime 相关模块验证。刷写时仍需确认目标分区是 rootfs 分区，避免覆盖 U-Boot、设备树或其他启动关键分区。

具体刷写方式取决于板子当前的启动介质和连接方式，常见做法是通过 U-Boot fastboot 或直接在 Linux 下 `dd` 写入对应 UFS 分区。

### 3. 上传内核与设备树

通过 USB 连接开发板，在 U-Boot 命令行和宿主机之间配合操作。

**方式一：fastboot**（当前使用的方式）

在 U-Boot 命令行中依次执行以下操作，每步配合宿主机端 `fastboot stage` 命令：

先是内核（34 MiB 加载地址，32 MiB 大小上限）：

```
# U-Boot
fastboot -l 0x140000000 -s 0x02000000 usb 0
```

```
# Host（宿主机）
fastboot stage target/riscv64gc-unknown-linux-musl/release/starryos.bin
```

然后是设备树（8 MiB 大小上限）：

```
# U-Boot（Ctrl-C 结束上一个 fastboot 会话后）
fastboot -l 0x138000000 -s 0x00800000 usb 0
```

```
# Host
fastboot stage os/StarryOS/configs/board/spacemit-k3-com260-ifx.dtb
```

最后启动：

```
# U-Boot
booti 0x140000000 - 0x138000000
```

**方式二：ostool**（备选）

也可以使用 `ostool` 工具配合 sftp 进行启动。

### 4. 验证启动

连接串口（默认 `/dev/ttyUSB0`，波特率 115200），启动后应看到 StarryOS shell：

```
root@starry:#
```


AI runner 设备会自动挂载，可以通过以下方式确认：

```bash
ls /dev/k3_airunner
```

## 注意事项

- **V 扩展上下文支持**：当前已补齐基础的 RISC-V V 扩展上下文保存/恢复，并通过同线程、跨线程和真板测试。A100/X100 IME backend 仍在继续接入，使用时仍需注意线程核心类型和向量宽度差异。
- **板级驱动状态**：UFS 驱动仍在测试，但可以满足基本启动需求；SD 卡驱动和 GMAC 网卡驱动仍在开发中。
