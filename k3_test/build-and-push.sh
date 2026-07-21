#!/usr/bin/env bash
set -euo pipefail

# ── 配置 ──────────────────────────────────────────────────────────
CRATE_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN_NAME="k3_test"
ROOTFS_IMG="/home/inkbottle/othersrc/tgoskits/tmp/axbuild/rootfs/rootfs-riscv64-alpine.img/rootfs-riscv64-alpine.img"
TARGET_DIR="/root"

# ── 编译 ──────────────────────────────────────────────────────────
echo "==> cargo build --release"
cd "${CRATE_DIR}"
cargo build --release

# 找到真正的二进制 (riscv64gc-unknown-linux-musl)
BUILD_DIR="${CRATE_DIR}/target/riscv64gc-unknown-linux-musl/release"
BIN="${BUILD_DIR}/${BIN_NAME}"

if [ ! -f "${BIN}" ]; then
    echo "ERROR: binary not found: ${BIN}"
    echo "trying fallback: target/release/${BIN_NAME}"
    BIN="${CRATE_DIR}/target/release/${BIN_NAME}"
    if [ ! -f "${BIN}" ]; then
        echo "FATAL: cannot find compiled binary"
        exit 1
    fi
fi

echo "==> binary: ${BIN}"
file "${BIN}"
echo "   size: $(du -h "${BIN}" | cut -f1)"

# ── 写入 rootfs 镜像 ──────────────────────────────────────────────
echo "==> writing to ${ROOTFS_IMG}:${TARGET_DIR}/${BIN_NAME}"

if [ ! -f "${ROOTFS_IMG}" ]; then
    echo "FATAL: rootfs image not found: ${ROOTFS_IMG}"
    exit 1
fi

cat <<CMDS | debugfs -w "${ROOTFS_IMG}" -f /dev/stdin
cd ${TARGET_DIR}
rm ${BIN_NAME}
write ${BIN} ${BIN_NAME}
chmod 755 ${BIN_NAME}
CMDS

echo "==> done: ${TARGET_DIR}/${BIN_NAME}"
