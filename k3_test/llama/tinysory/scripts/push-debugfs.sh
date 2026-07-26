#!/usr/bin/env bash
set -euo pipefail

CRATE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BIN_NAME="tinysory"
ROOTFS_IMG="${ROOTFS_IMG:-/home/inkbottle/othersrc/tgoskits/tmp/axbuild/rootfs/rootfs-riscv64-alpine.img/rootfs-riscv64-alpine.img}"
TARGET_DIR="${TARGET_DIR:-/root}"

cd "${CRATE_DIR}"
echo "==> cargo build --release"
cargo build --release

BIN="${CRATE_DIR}/target/riscv64gc-unknown-linux-musl/release/${BIN_NAME}"
if [ ! -f "${BIN}" ]; then
    echo "ERROR: binary not found: ${BIN}"
    exit 1
fi

if [ ! -f "${ROOTFS_IMG}" ]; then
    echo "ERROR: rootfs image not found: ${ROOTFS_IMG}"
    exit 1
fi

echo "==> binary: ${BIN}"
file "${BIN}"

echo "==> writing ${TARGET_DIR}/${BIN_NAME} into ${ROOTFS_IMG}"
cat <<CMDS | debugfs -w "${ROOTFS_IMG}" -f /dev/stdin
cd ${TARGET_DIR}
rm ${BIN_NAME}
write ${BIN} ${BIN_NAME}
chmod 755 ${BIN_NAME}
CMDS

echo "==> done"
