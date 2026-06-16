//! k3 AI runtime 集成测试，验证 channel 建立和共享区内存保活。

use core::ptr;
use std::os::raw::{c_int, c_void};

use k3_aiRuntime::fronted::{
    AiDtype, AiKernelDesc, AiTargetHint, GraphManager, MatMulAttr, TensorManager,
    kd_uring::{build_channel, submit_graph}
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

    // matmul: 2×3 @ 3×5 = 2×5
    let mut lhs = tensor_mgr.alloc_tensor(AiDtype::F32, &[2, 3]);
    let mut rhs = tensor_mgr.alloc_tensor(AiDtype::F32, &[3, 5]);
    let out = tensor_mgr.alloc_tensor(AiDtype::F32, &[2, 5]);

    // 填充 lhs 数据: [[1, 2, 3], [4, 5, 6]]
    let lhs_data = lhs.as_f32_mut_slice();
    lhs_data[0] = 1.0; lhs_data[1] = 2.0; lhs_data[2] = 3.0;
    lhs_data[3] = 4.0; lhs_data[4] = 5.0; lhs_data[5] = 6.0;

    // 填充 rhs 数据: [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]]
    let rhs_data = rhs.as_f32_mut_slice();
    for i in 0..15 { rhs_data[i] = 0.0; }
    rhs_data[0] = 1.0; rhs_data[6] = 1.0; rhs_data[12] = 1.0;

    let matmul_attr = MatMulAttr {
        m: 2,
        n: 5,
        k: 3,
        batch: 0,
        lhs_row_stride: 3,
        lhs_col_stride: 1,
        lhs_batch_stride: 0,
        rhs_row_stride: 5,
        rhs_col_stride: 1,
        rhs_batch_stride: 0,
        out_row_stride: 5,
        out_col_stride: 1,
        out_batch_stride: 0,
        flags: 0,
        accum_dtype: AiDtype::F32,
        reserved: [0; 3],
    };

    let matmul_desc = AiKernelDesc::new(
        &matmul_attr, AiTargetHint::AUTO, 2, 1,
        &[lhs.desc(), rhs.desc(), out.desc()]
    );
    let _node = graph.push_kernel_no_depend(matmul_desc).expect("failed to push matmul");

    // 冻结并提交
    let blob = graph.freeze().expect("failed to freeze graph");
    let entry = blob.submit_entry(0);

    println!("k3_test: submitting 2×5 matmul");
    submit_graph(&channel, &entry).expect("failed to submit graph");
    println!("k3_test: graph submitted successfully");

    // 打印结果
    let result = out.as_f32_slice();
    println!("result (2×5):");
    for i in 0..2 {
        print!("  [");
        for j in 0..5 {
            print!("{:6.1}", result[i * 5 + j]);
        }
        println!(" ]");
    }


    println!("continue , current avoid kernel access user addr");
    loop {
        
    }
}
