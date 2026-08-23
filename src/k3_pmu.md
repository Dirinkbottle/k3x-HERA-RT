# SpacemiT K3 / X100 PMU Raw Event 映射记录

> 状态：实验记录，2026-08-21  
> 目标：整理 K3/X100 当前公开的 `mhpmevent` raw selector → Linux `perf` event 映射，并记录已知映射问题。  
> 注意：本文优先依据 SpacemiT 公开 K3 Device Tree、Bianbu perf 文档和官方论坛问题帖。

## 资料来源

1. **K3 官方 EDK2 Device Tree：X100 PMU 映射**
   - https://github.com/spacemit-com/edk2-platforms/blob/k3/Platform/Spacemit/K3/DeviceTree/K3.dts
   - 其中 `pmu { compatible = "riscv,pmu"; ... }` 节点定义：
     - `riscv,event-to-mhpmevent`
     - `riscv,event-to-mhpmcounters`
     - `riscv,raw-event-to-mhpmcounters`

2. **Bianbu 官方 Perf Usage Note**
   - https://github.com/spacemit-com/docs-bianbu/blob/375fb8d5d8e97c30a555e0d3e499ed652d0a8b2c/en/development/perf.md
   - 说明 RISC-V Linux `perf`、SBI PMU、Device Tree `riscv,pmu` 节点与 raw hardware event 的关系。
   - 也说明 raw event 可通过 `perf -e rNNN` 或 PMU `event=...` 形式使用。

3. **K3 Datasheet**
   - https://github.com/spacemit-com/docs-chip/blob/main/en/key_stone/k3/k3_docs/k3_ds.md
   - Datasheet 明确 X100 与 A100 均集成 RISC-V PMU；A100 支持 1024-bit RVV。

---

## Perf Event映射链路

```text
Linux perf event
    ↓
SBI PMU generic/cache event ID
    ↓
OpenSBI 解析 Device Tree: riscv,event-to-mhpmevent
    ↓
mhpmeventN 写入 raw selector
    ↓
X100 PMU 硬件计数
```

下面表格里的 **Raw `mhpmevent`** 是最终写入硬件 HPM event selector 的值

例如：

```text
perf: stalled-cycles-frontend
        ↓
SBI/DT event ID: 0x00008
        ↓
mhpmevent raw selector: 0x03
```

---

## X100：Raw `mhpmevent` → `perf` Event 映射

| Raw `mhpmevent` | SBI / DT Event ID | Linux `perf` 语义 | 建议的 `perf` 名称 | 当前状态 |
|---:|---:|---|---|---|
| `0x01` | `0x00005` | Retired branch instructions | `branches` / `branch-instructions` | ✅ 已公开映射 |
| `0x02` | `0x00006` | Branch mispredictions | `branch-misses` | ✅ 已公开映射 |
| `0x03` | `0x00008` | Stalled cycles - frontend | `stalled-cycles-frontend` | ✅ 已公开映射 |
| `0x04` | `0x00009` | Stalled cycles - backend | `stalled-cycles-backend` | ✅ 已公开映射 |
| `0x05` | `0x10001` | L1D read miss | `L1-dcache-load-misses` | ✅ 已公开映射 |
| `0x06` | `0x10000` | L1D read access | `L1-dcache-loads` | ✅ 已公开映射 |
| `0x07` | `0x10005` | L1D prefetch miss | `L1-dcache-prefetch-misses` | ✅ 已公开映射 |
| `0x08` | `0x10004` | L1D prefetch access | `L1-dcache-prefetches` | ✅ 已公开映射 |
| `0x09` | `0x10003` | L1D write miss | `L1-dcache-store-misses` | ✅ 已公开映射 |
| `0x0a` | `0x10002` | L1D write access | `L1-dcache-stores` | ✅ 已公开映射 |
| `0x0b` | `0x10009` | L1I read miss | `L1-icache-load-misses` | ✅ 已公开映射 |
| `0x0c` | `0x10008` | L1I read access | `L1-icache-loads` | ✅ 已公开映射 |
| `0x16` | `0x10018` | DTLB read access | `dTLB-loads` | ✅ 已公开映射 |
| `0x19` | `0x1001b` | DTLB write miss | `dTLB-store-misses` | ✅ 已公开映射 |
| `0x1a` | `0x1001a` | DTLB write access | `dTLB-stores` | ✅ 已公开映射 |
| `0x1b` | `0x10021` | ITLB read miss | `iTLB-load-misses` | ✅ 已公开映射 |
| `0x1c` | `0x10020` | ITLB read access | `iTLB-loads` | ✅ 已公开映射 |
| `0x1e` | `0x00001` | CPU cycles | `cycles` | ✅ 已公开映射 |
| `0x1f` | `0x00002` | Retired instructions | `instructions` | ✅ 已公开映射 |


## 当前 DTS 允许使用、但公开资料尚未给出语义的 Raw Event

K3 官方 DTS 的 `riscv,raw-event-to-mhpmcounters` 还允许下列 raw selector 被配置到普通 HPM counter：

```text
0x01 0x02 0x03 0x04 0x05 0x06 0x07 0x08
0x09 0x0a 0x0b 0x0c
0x10 0x11 0x12 0x13 0x14 0x15 0x16 0x17
0x18 0x19 0x1a 0x1b 0x1c
0x1e 0x1f
0x20 0x21 0x22
0x24 0x25 0x26 0x27 0x28 0x29
0x2b 0x2d
0x32 0x33 0x34 0x35 0x36 0x37 0x38
0x3a 0x3b 0x3c 0x3d 0x3f
```

其中只有一部分已经通过 `riscv,event-to-mhpmevent` 给出公开语义。

官方论坛对“完整 PMU event 对照表”的公开答复是：

> 这个暂时不对外提供的

来源同上：

- https://forum.spacemit.com/t/topic/1500

---