//! k3 AI runtime 集成测试，验证 channel 建立和共享区内存保活。

use core::ptr;
use std::os::raw::{c_int, c_void};

use k3_aiRuntime::fronted::{
    AiDtype, AiKernelDesc, AiTargetHint, GraphManager, MatMulAttr, TensorManager,
    kd_uring::{build_channel, submit_graph, wait_graph_complete}
};

unsafe extern "C" {
    fn mmap(
        addr: *mut c_void,
        length: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        offset: isize,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, length: usize) -> c_int;
}

// ── mmap 常量（来自 Linux asm-generic/mman.h）──────────────────
const PROT_READ: c_int = 0x1;
const PROT_WRITE: c_int = 0x2;
const MAP_SHARED: c_int = 0x01;
const MAP_ANONYMOUS: c_int = 0x20;
const MAP_FAILED: *mut c_void = !0 as *mut c_void;

fn main() {
    let channel = build_channel().expect("failed to build /dev/k3_airunner channel");

    println!("k3_test: channel built, va={:#x}, size={:#x}",
        channel.shared.user_va, channel.shared.size_bytes);

    let tensor_mgr = TensorManager::new();
    let mut graph = GraphManager::new();

    // DAG: a 和 b 并行，d 依赖 a 和 b
    // a: 2×3 @ 3×2 = 2×2
    // b: 2×2 @ 2×3 = 2×3
    // d: 2×2 @ 2×3 = 2×3 (使用 a_out 和 b_out)

    // === node a ===
    let mut a_lhs = tensor_mgr.alloc_tensor(AiDtype::F32, &[2, 3]);
    let mut a_rhs = tensor_mgr.alloc_tensor(AiDtype::F32, &[3, 2]);
    let a_out = tensor_mgr.alloc_tensor(AiDtype::F32, &[2, 2]);

    {
        let lhs = a_lhs.as_f32_mut_slice();
        lhs[0] = 1.0; lhs[1] = 2.0; lhs[2] = 3.0;
        lhs[3] = 4.0; lhs[4] = 5.0; lhs[5] = 6.0;

        let rhs = a_rhs.as_f32_mut_slice();
        rhs[0] = 1.0; rhs[1] = 0.0;
        rhs[2] = 0.0; rhs[3] = 1.0;
        rhs[4] = 0.0; rhs[5] = 0.0;
    }

    let a_node = graph.push_kernel_no_depend(AiKernelDesc::new(
        &MatMulAttr {
            m: 2, n: 2, k: 3, batch: 0,
            lhs_row_stride: 3, lhs_col_stride: 1, lhs_batch_stride: 0,
            rhs_row_stride: 2, rhs_col_stride: 1, rhs_batch_stride: 0,
            out_row_stride: 2, out_col_stride: 1, out_batch_stride: 0,
            flags: 0, accum_dtype: AiDtype::F32, reserved: [0; 3],
        },
        AiTargetHint::PREFER_CPU, 2, 1,
        &[a_lhs.desc(), a_rhs.desc(), a_out.desc()]
    )).expect("failed to push node a");

    // === node b ===
    let mut b_lhs = tensor_mgr.alloc_tensor(AiDtype::F32, &[2, 2]);
    let mut b_rhs = tensor_mgr.alloc_tensor(AiDtype::F32, &[2, 3]);
    let b_out = tensor_mgr.alloc_tensor(AiDtype::F32, &[2, 3]);

    {
        let lhs = b_lhs.as_f32_mut_slice();
        lhs[0] = 2.0; lhs[1] = 0.0;
        lhs[2] = 0.0; lhs[3] = 2.0;

        let rhs = b_rhs.as_f32_mut_slice();
        rhs[0] = 1.0; rhs[1] = 2.0; rhs[2] = 3.0;
        rhs[3] = 4.0; rhs[4] = 5.0; rhs[5] = 6.0;
    }

    let b_node = graph.push_kernel_no_depend(AiKernelDesc::new(
        &MatMulAttr {
            m: 2, n: 3, k: 2, batch: 0,
            lhs_row_stride: 2, lhs_col_stride: 1, lhs_batch_stride: 0,
            rhs_row_stride: 3, rhs_col_stride: 1, rhs_batch_stride: 0,
            out_row_stride: 3, out_col_stride: 1, out_batch_stride: 0,
            flags: 0, accum_dtype: AiDtype::F32, reserved: [0; 3],
        },
        AiTargetHint::PREFER_CPU, 2, 1,
        &[b_lhs.desc(), b_rhs.desc(), b_out.desc()]
    )).expect("failed to push node b");

    // === node d: 依赖 a 和 b ===
    let d_out = tensor_mgr.alloc_tensor(AiDtype::F32, &[2, 3]);

    let _d_node = graph.push_kernel_depend_many(
        &[a_node, b_node],
        AiKernelDesc::new(
            &MatMulAttr {
                m: 2, n: 3, k: 2, batch: 0,
                lhs_row_stride: 2, lhs_col_stride: 1, lhs_batch_stride: 0,
                rhs_row_stride: 3, rhs_col_stride: 1, rhs_batch_stride: 0,
                out_row_stride: 3, out_col_stride: 1, out_batch_stride: 0,
                flags: 0, accum_dtype: AiDtype::F32, reserved: [0; 3],
            },
            AiTargetHint::PREFER_CPU, 2, 1,
            &[a_out.desc(), b_out.desc(), d_out.desc()]
        )
    ).expect("failed to push node d");

    let blob = graph.freeze().expect("failed to freeze graph");
    let entry = blob.submit_entry(42);

    println!("k3_test: submitting DAG graph (a || b) -> d");
    submit_graph(&channel, &entry).expect("failed to submit graph");

    if let Err(e) = wait_graph_complete(&entry, &channel) {
        println!("graph execute fail with err {}", e);
    }

    println!("d result (2×3):");
    let result = d_out.as_f32_slice();
    for i in 0..2 {
        print!("  [");
        for j in 0..3 {
            print!("{:6.1}", result[i * 3 + j]);
        }
        println!(" ]");
    }


    println!("continue , current avoid kernel access user addr");
    loop {
        
    }
}
