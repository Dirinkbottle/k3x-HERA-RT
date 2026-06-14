//! k3 AI runtime 集成测试，验证 channel 建立和共享区内存保活。

use core::ptr;
use std::os::raw::{c_int, c_void};

use k3_aiRuntime::fronted::kd_uring::build_channel;

unsafe extern "C" {
    fn mmap(
        addr: *mut c_void,
        length: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        offset: isize,
    ) -> *mut c_void;
}

// ── mmap 常量（来自 Linux asm-generic/mman.h）──────────────────
const PROT_READ: c_int = 0x1;
const PROT_WRITE: c_int = 0x2;
const MAP_SHARED: c_int = 0x01;
const MAP_ANONYMOUS: c_int = 0x20;
const MAP_FIXED_NOREPLACE: c_int = 0x100000; // 不覆盖已有映射，冲突时返回 EEXIST
const MAP_FAILED: *mut c_void = !0 as *mut c_void;

fn main() {
    let channel = build_channel().expect("failed to build /dev/k3_airunner channel");
    let channel_va = channel.shared.user_va;
    let channel_size = channel.shared.size_bytes;

    println!(
        "k3_test: channel built, va={:#x}, size={:#x}",
        channel_va, channel_size
    );

    // 直接往 channel 共享区写入一段标记字符串，后面拿它验证这块共享区有没有保住。
    unsafe {
        let bytes = b"ovchannel";
        ptr::copy_nonoverlapping(bytes.as_ptr(), channel_va as *mut u8, bytes.len());
    }
    println!("k3_test: wrote marker string into shared channel memory");

    // 显式 drop 用户态 channel 句柄。
    // 这一步之后，这段共享区如果还能读出来，就说明 runtime 全局持有路径生效了。
    drop(channel);
    println!("k3_test: dropped user-facing UringChannel handle");

    // 先直接从原地址读回，验证 drop 用户态句柄后，这段共享区仍然活着。
    let mut readback = [0_u8; 9];
    unsafe {
        ptr::copy_nonoverlapping(channel_va as *const u8, readback.as_mut_ptr(), readback.len());
    }
    let text = core::str::from_utf8(&readback).expect("shared memory marker is not utf8");
    println!("k3_test: readback after drop=\"{}\"", text);
    assert_eq!(text, "ovchannel");

    // 再尝试把同一段虚拟地址重新 mmap 一次。
    // 这里不能用 MAP_FIXED；MAP_FIXED 成功会直接顶掉旧映射，测出来的就不是原内容了。
    // 用 MAP_FIXED_NOREPLACE 才能判断这一地址范围是否依然被原映射占着。
    let remap_ptr = unsafe {
        mmap(
            channel_va as *mut c_void,
            channel_size,
            PROT_READ | PROT_WRITE,
            MAP_SHARED | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE,
            -1,
            0,
        )
    };

    if remap_ptr == MAP_FAILED {
        println!("k3_test: MAP_FIXED_NOREPLACE failed as expected, original mapping is still occupying the same va range");
    } else {
        panic!(
            "k3_test: unexpected remap success at {:#x}, channel mapping was not kept alive",
            remap_ptr as usize
        );
    }
}
