use std::{
    env, fs,
    path::{Path, PathBuf},
};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn hash_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn hash_file(hash: u64, manifest_dir: &Path, rel_path: &str) -> u64 {
    let path = manifest_dir.join(rel_path);
    println!("cargo:rerun-if-changed={}", path.display());

    let mut hash = hash_bytes(hash, rel_path.as_bytes());
    hash = hash_bytes(hash, &[0]);
    let bytes = fs::read(&path).unwrap_or_else(|err| {
        panic!("failed to read ABI input {}: {err}", path.display());
    });
    hash_bytes(hash, &bytes)
}

fn collect_dir_inputs(manifest_dir: &Path, rel_dir: &str, out: &mut Vec<String>) {
    let dir = manifest_dir.join(rel_dir);
    println!("cargo:rerun-if-changed={}", dir.display());

    let mut entries = fs::read_dir(&dir)
        .unwrap_or_else(|err| panic!("failed to read ABI dir {}: {err}", dir.display()))
        .map(|entry| entry.expect("failed to read ABI dir entry").path())
        .collect::<Vec<_>>();
    entries.sort();

    for path in entries {
        let rel_path = path
            .strip_prefix(manifest_dir)
            .expect("ABI input path should be under manifest dir")
            .to_string_lossy()
            .replace('\\', "/");
        if path.is_dir() {
            collect_dir_inputs(manifest_dir, &rel_path, out);
        } else {
            out.push(rel_path);
        }
    }
}

fn collect_abi_inputs(manifest_dir: &Path) -> Vec<String> {
    let mut inputs = vec!["Cargo.toml".to_owned(), "build.rs".to_owned()];
    collect_dir_inputs(manifest_dir, "src", &mut inputs);
    inputs
}

fn main() {
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());

    let abi_inputs = collect_abi_inputs(&manifest_dir);

    let source_hash = abi_inputs.iter().fold(FNV_OFFSET, |hash, rel_path| {
        hash_file(hash, &manifest_dir, rel_path)
    });

    let mut abi_version = (source_hash ^ (source_hash >> 32)) as u32;
    if abi_version == 0 {
        abi_version = 1;
    }

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let out_file = out_dir.join("ai_abi_version.rs");
    fs::write(
        out_file,
        format!(
            "\
/// 当前 AI runtime UAPI 构建版本。\n\
///\n\
/// 该值由 `k3_aiUabi/build.rs` 根据稳定 UABI 源码生成。\n\
/// 用户态和内核态必须完全相同，否则内核拒绝后续操作。\n\
pub const AI_ABI_VERSION: u32 = 0x{abi_version:08x};\n\
\n\
/// 稳定 UABI 源码内容哈希，用于定位版本来源。\n\
pub const AI_ABI_SOURCE_HASH: u64 = 0x{source_hash:016x};\n"
        ),
    )
    .expect("failed to write generated ABI version");
}
