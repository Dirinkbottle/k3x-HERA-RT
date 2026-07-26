//! 算子计算图。
//!
//! channel里提交的是 `AiGraphSubmitEntry`
//! graph blob 内部用 offset 组织节点和边。

use alloc::vec;
use alloc::vec::Vec;
use core::mem;

use crate::{
    AI_ABI_VERSION, AiKernelDesc, ByteOffset, ByteSize, EdgeCount, GraphFlags, NodeCount,
    SubmitFlags, UserToken, UserVa,
};

/// graph blob 魔数
pub const AI_GRAPH_MAGIC: u32 = 0x4845_5241; // "HERA"

/// 提交类型。
#[repr(transparent)]
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct GraphSubmitKind(pub u32);

impl GraphSubmitKind {
    /// 提交一次 graph 执行。
    pub const GRAPH_SUBMIT: Self = Self(1);
    /// 取消一次已提交的 graph。
    pub const CANCEL: Self = Self(2);
    /// 查询已提交 graph 的状态。
    pub const QUERY: Self = Self(3);

    /// 最小合法性检查：提交类型是否落在已知区间内。
    pub const fn is_known(self) -> bool {
        matches!(self.0, 1..=3)
    }
}

impl Default for GraphSubmitKind {
    fn default() -> Self {
        Self::GRAPH_SUBMIT
    }
}

/// 提交到 channel 的 graph 入口。
/// 一次算子操作链任务的描述
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct AiGraphSubmitEntry {
    /// ABI 版本，必须等于 `AI_ABI_VERSION`。
    pub abi_version: u32,

    /// 提交类型。默认 `GraphSubmitKind::GRAPH_SUBMIT`。
    pub submit_kind: GraphSubmitKind,

    /// 提交 fgs，保留。
    pub flags: SubmitFlags,

    /// 预留字段，保持 8 字节对齐。
    pub reserved0: u32,

    /// 用户态 completion cookie。
    /// 用户态用它匹配完成的 graph。
    pub user_token: UserToken,

    /// graph blob 的用户态虚拟地址。
    pub graph_user_va: UserVa,

    /// graph blob 总字节数。
    pub graph_size: ByteSize,
}

impl Default for AiGraphSubmitEntry {
    fn default() -> Self {
        Self {
            abi_version: AI_ABI_VERSION,
            submit_kind: GraphSubmitKind::GRAPH_SUBMIT,
            flags: SubmitFlags::new(0),
            reserved0: 0,
            user_token: UserToken::new(0),
            graph_user_va: UserVa::new(0),
            graph_size: ByteSize::new(0),
        }
    }
}

impl AiGraphSubmitEntry {
    /// 构造一个 graph 提交入口。
    ///
    /// `abi_version` 固定为 `AI_ABI_VERSION`，`flags`/`reserved0` 归零。
    pub fn new(
        user_token: UserToken,
        graph_user_va: UserVa,
        graph_size: ByteSize,
        submit_kind: GraphSubmitKind,
    ) -> Self {
        Self {
            abi_version: AI_ABI_VERSION,
            submit_kind,
            flags: SubmitFlags::new(0),
            reserved0: 0,
            user_token,
            graph_user_va,
            graph_size,
        }
    }
    /// 序列化提交
    pub fn to_le_byte(&self) -> Option<&[u8]> {
        let self_size = core::mem::size_of::<Self>();
        if self_size > 255 {
            return None;
        }
        unsafe {
            Some(core::slice::from_raw_parts(
                self as *const Self as *const u8,
                self_size,
            ))
        }
    }
}

/// graph blob 头部。
///
/// 后续 nodes/edges 都通过 offset 在同一块 blob 内定位。
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct AiGraphHeader {
    /// graph 魔数，必须等于 `AI_GRAPH_MAGIC`。
    pub magic: u32,

    /// 整个 graph blob 的字节数。
    pub total_size: ByteOffset,

    /// graph flags，阶段一先保留。
    pub flags: GraphFlags,

    /// 节点数量。
    pub node_count: NodeCount,

    /// 依赖边数量。
    pub edge_count: EdgeCount,

    /// node 数组在 graph blob 内的偏移。
    pub nodes_offset: ByteOffset,

    /// edge 数组在 graph blob 内的偏移。
    pub edges_offset: ByteOffset,
}

/// graph node 是 `AiKernelDesc` 的薄封装。
///
/// `desc` 描述这个节点要执行的语义级算子；`node_id` 只用于 graph 依赖关系。
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct AiGraphNode {
    /// graph 内稳定节点编号。
    pub node_id: AiGraphNodeId,
    /// 单个 lowered 算子的描述。
    pub desc: AiKernelDesc,
    /// Graph节点的状态
    pub state: AiGraphState,
}

/// 表示某个图节点的执行状态
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct AiGraphState {
    /// 该节点是否已执行完成：0=未完成，1=已完成。
    pub complete: u8,
    /// 错误码：0 表示无错误，非零为 `BackendErr as u8`。
    pub error_flag: u8,
}

/// graph 依赖边。
///
/// 表示 `from_node` 完成后，`to_node` 的一个前置依赖满足。
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct AiGraphEdge {
    /// 依赖边起点节点 id（前置节点）。
    pub from_node: AiGraphNodeId,
    /// 依赖边终点节点 id（后继节点）。
    pub to_node: AiGraphNodeId,
}

/// 用户态构图时返回的依赖标识。
///
/// 当前链尾 node id：继续向这条链追加算子时，
/// 把这个 id 传给 `push_kernel_depend` 。
#[repr(transparent)]
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct AiGraphNodeId(pub u32);

impl AiGraphNodeId {
    /// 用原始 node id 构造图节点 id。
    pub const fn new(raw: u32) -> Self {
        Self(raw)
    }

    /// 返回底层原始 node id。
    pub const fn get(self) -> u32 {
        self.0
    }

    /// 返回可用于索引的 `usize`。
    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl PartialEq<u32> for AiGraphNodeId {
    fn eq(&self, other: &u32) -> bool {
        self.0 == *other
    }
}

impl PartialEq<AiGraphNodeId> for u32 {
    fn eq(&self, other: &AiGraphNodeId) -> bool {
        *self == other.0
    }
}

impl From<u32> for AiGraphNodeId {
    fn from(value: u32) -> Self {
        Self::new(value)
    }
}

impl From<AiGraphNodeId> for u32 {
    fn from(value: AiGraphNodeId) -> Self {
        value.get()
    }
}

impl core::fmt::Display for AiGraphNodeId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

/// 一个 chain id 表示当前主图链的尾节点。
pub type AiGraphChainId = AiGraphNodeId;

/// 用户态可持有的 frozen graph blob。
///
/// 一整块连续字节流，
pub struct AiGraphBlob {
    /// 序列化后的连续字节流：header + nodes + edges。
    bytes: Vec<u8>,
}

impl AiGraphBlob {
    /// 从节点和边序列化成一整块 graph blob。
    pub fn from_parts(
        nodes: &[AiGraphNode],
        edges: &[AiGraphEdge],
    ) -> Result<Self, AiGraphBuildError> {
        let node_count = NodeCount::new(
            u32::try_from(nodes.len()).map_err(|_| AiGraphBuildError::TooManyNodes)?,
        );
        let edge_count = EdgeCount::new(
            u32::try_from(edges.len()).map_err(|_| AiGraphBuildError::TooManyEdges)?,
        );
        validate_graph(nodes, edges)?;

        let header_size = mem::size_of::<AiGraphHeader>();
        let nodes_size = mem::size_of_val(nodes);
        let edges_size = mem::size_of_val(edges);
        let reserve_size = header_size
            .checked_add(mem::align_of::<AiGraphNode>())
            .and_then(|v| v.checked_add(nodes_size))
            .and_then(|v| v.checked_add(mem::align_of::<AiGraphEdge>()))
            .and_then(|v| v.checked_add(edges_size))
            .ok_or(AiGraphBuildError::SizeOverflow)?;

        let mut bytes = Vec::with_capacity(reserve_size);
        append_repr(&mut bytes, &AiGraphHeader::default());

        let nodes_offset = pad_to(&mut bytes, mem::align_of::<AiGraphNode>())?;
        append_repr_slice(&mut bytes, nodes);

        let edges_offset = pad_to(&mut bytes, mem::align_of::<AiGraphEdge>())?;
        append_repr_slice(&mut bytes, edges);

        let total_size = ByteOffset::new(
            u32::try_from(bytes.len()).map_err(|_| AiGraphBuildError::SizeOverflow)?,
        );
        let header = AiGraphHeader {
            magic: AI_GRAPH_MAGIC,
            total_size,
            flags: GraphFlags::new(0),
            node_count,
            edge_count,
            nodes_offset,
            edges_offset,
        };
        write_repr_at(&mut bytes, 0, &header);

        Ok(Self { bytes })
    }

    /// 返回底层 graph blob 的字节切片。
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// 基于当前 blob 构造一个可提交到 channel 的 graph 入口。
    pub fn submit_entry(&self, user_token: UserToken) -> AiGraphSubmitEntry {
        AiGraphSubmitEntry::new(
            user_token,
            UserVa::new(self.bytes.as_ptr() as u64),
            ByteSize::new(self.bytes.len() as u64),
            GraphSubmitKind::GRAPH_SUBMIT,
        )
    }
}

/// 解析后的 graph。
pub struct AiParsedGraph {
    /// graph 头部元数据。
    pub header: AiGraphHeader,
    /// 解析出的节点列表。
    pub nodes: Vec<AiGraphNode>,
    /// 解析出的依赖边列表。
    pub edges: Vec<AiGraphEdge>,
}

/// 从字节流中解析 graph blob 的工具类型。
pub struct AiGraphParser;

impl AiGraphParser {
    /// 校验魔数、大小、分段范围后，把 graph blob 解析成 `AiParsedGraph`。
    pub fn parse(bytes: &[u8]) -> Result<AiParsedGraph, AiGraphParseError> {
        let header: AiGraphHeader = read_repr_at(bytes, 0)?;
        if header.magic != AI_GRAPH_MAGIC {
            return Err(AiGraphParseError::BadMagic(header.magic));
        }
        if header.total_size.get() as usize != bytes.len() {
            return Err(AiGraphParseError::SizeMismatch {
                header_size: header.total_size,
                actual_size: bytes.len(),
            });
        }

        let nodes_range = checked_section(
            "nodes",
            bytes.len(),
            header.nodes_offset,
            header.node_count.get() as usize,
            mem::size_of::<AiGraphNode>(),
        )?;
        let edges_range = checked_section(
            "edges",
            bytes.len(),
            header.edges_offset,
            header.edge_count.get() as usize,
            mem::size_of::<AiGraphEdge>(),
        )?;

        let nodes =
            read_repr_vec::<AiGraphNode>(&bytes[nodes_range], header.node_count.get() as usize)?;
        let edges =
            read_repr_vec::<AiGraphEdge>(&bytes[edges_range], header.edge_count.get() as usize)?;

        Ok(AiParsedGraph {
            header,
            nodes,
            edges,
        })
    }
}

/// 用户态 graph 管理器。
#[derive(Default)]
pub struct GraphManager {
    /// 已追加的节点，下标即节点 id。
    nodes: Vec<AiGraphNode>,
    /// 已追加的依赖边。
    edges: Vec<AiGraphEdge>,
}

impl GraphManager {
    /// 创建一个空的 graph 管理器。
    pub fn new() -> Self {
        Self::default()
    }

    /// 添加一个无依赖算子，返回当前链尾 id。
    pub fn push_kernel_no_depend(
        &mut self,
        desc: AiKernelDesc,
    ) -> Result<AiGraphChainId, AiGraphBuildError> {
        self.push_node(desc)
    }

    /// 添加一个依赖单条链尾的算子。
    ///
    /// 返回的新 id 表示新链尾；继续向后追加时传这个 id。
    pub fn push_kernel_depend(
        &mut self,
        depend: AiGraphChainId,
        desc: AiKernelDesc,
    ) -> Result<AiGraphChainId, AiGraphBuildError> {
        self.push_kernel_depend_many(&[depend], desc)
    }

    /// 添加一个依赖多条链尾的算子。
    ///
    /// 例如 `a -> b, c -> b` 可写成 `push_kernel_depend_many(&[a, c], b_desc)`。
    pub fn push_kernel_depend_many(
        &mut self,
        depends: &[AiGraphChainId],
        desc: AiKernelDesc,
    ) -> Result<AiGraphChainId, AiGraphBuildError> {
        for &depend in depends {
            self.validate_node_id(depend)?;
        }

        let edge_base = self.edges.len();
        let node_id = self.push_node(desc)?;

        for &depend in depends {
            if let Err(err) = self.push_edge_checked(depend, node_id) {
                self.edges.truncate(edge_base);
                self.nodes.pop();
                return Err(err);
            }
        }

        Ok(node_id)
    }

    /// 冻结成可提交的连续 graph blob。
    pub fn freeze(&self) -> Result<AiGraphBlob, AiGraphBuildError> {
        AiGraphBlob::from_parts(&self.nodes, &self.edges)
    }

    /// 追加一个节点，节点 id 等于其在数组中的下标，返回新链尾 id。
    fn push_node(&mut self, desc: AiKernelDesc) -> Result<AiGraphChainId, AiGraphBuildError> {
        let node_id = AiGraphNodeId::new(
            u32::try_from(self.nodes.len()).map_err(|_| AiGraphBuildError::TooManyNodes)?,
        );
        self.nodes.push(AiGraphNode {
            node_id,
            desc,
            state: AiGraphState::default(),
        });
        Ok(node_id)
    }

    /// 校验节点 id 是否指向本管理器内一个真实存在的节点。
    fn validate_node_id(&self, node_id: AiGraphNodeId) -> Result<(), AiGraphBuildError> {
        let idx = node_id.as_usize();
        if idx >= self.nodes.len() || self.nodes[idx].node_id != node_id {
            return Err(AiGraphBuildError::InvalidDepend(node_id));
        }
        Ok(())
    }

    /// 追加依赖边前先校验节点 id 合法性和是否引入环。
    fn push_edge_checked(
        &mut self,
        from: AiGraphNodeId,
        to: AiGraphNodeId,
    ) -> Result<(), AiGraphBuildError> {
        self.validate_node_id(from)?;
        self.validate_node_id(to)?;

        if from == to || self.reaches(to, from) {
            return Err(AiGraphBuildError::CycleDetected);
        }

        self.edges.push(AiGraphEdge {
            from_node: from,
            to_node: to,
        });
        Ok(())
    }

    /// 从 `start` 出发按依赖边做 DFS，判断能否到达 `target`（用于环检测）。
    fn reaches(&self, start: AiGraphNodeId, target: AiGraphNodeId) -> bool {
        let mut stack = vec![start];
        let mut visited = vec![false; self.nodes.len()];

        while let Some(node_id) = stack.pop() {
            if node_id == target {
                return true;
            }

            let idx = node_id.as_usize();
            if idx >= visited.len() || visited[idx] {
                continue;
            }

            visited[idx] = true;
            for edge in &self.edges {
                if edge.from_node == node_id {
                    stack.push(edge.to_node);
                }
            }
        }

        false
    }
}

/// 构图阶段的错误。
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum AiGraphBuildError {
    /// 节点数超过 `u32` 上限。
    TooManyNodes,
    /// 边数超过 `u32` 上限。
    TooManyEdges,
    /// 依赖引用了不存在的节点 id。
    InvalidDepend(AiGraphNodeId),
    /// 检测到依赖环。
    CycleDetected,
    /// blob 尺寸计算溢出。
    SizeOverflow,
}

/// 解析 graph blob 时的错误。
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum AiGraphParseError {
    /// 字节流长度不足以容纳所需结构。
    TooSmall,
    /// 魔数与 `AI_GRAPH_MAGIC` 不符，携带实际读到的值。
    BadMagic(u32),
    /// ABI 版本不匹配，携带实际读到的值。
    BadAbi(u32),
    /// header 声明的大小与实际字节数不一致。
    SizeMismatch {
        /// header 中声明的总大小。
        header_size: ByteOffset,
        /// 字节流的实际长度。
        actual_size: usize,
    },
    /// 某个分段的 offset/size 超出 blob 范围。
    SectionOutOfRange {
        /// 越界分段名称（如 "nodes"/"edges"）。
        section: &'static str,
        /// 分段起始偏移。
        offset: ByteOffset,
        /// 分段字节大小。
        size: usize,
        /// blob 总字节数。
        total_size: usize,
    },
    /// 元素数量与大小相乘时溢出。
    CountOverflow,
}

/// 把一个 `Copy` 值的原始字节追加到 buffer 末尾。
fn append_repr<T: Copy>(bytes: &mut Vec<u8>, value: &T) {
    let src = unsafe {
        core::slice::from_raw_parts((value as *const T).cast::<u8>(), mem::size_of::<T>())
    };
    bytes.extend_from_slice(src);
}

/// 把一个 `Copy` 切片的原始字节追加到 buffer 末尾。
fn append_repr_slice<T: Copy>(bytes: &mut Vec<u8>, values: &[T]) {
    let src = unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), mem::size_of_val(values))
    };
    bytes.extend_from_slice(src);
}

/// 在指定 offset 处原地覆盖写入一个 `Copy` 值的原始字节。
fn write_repr_at<T: Copy>(bytes: &mut [u8], offset: usize, value: &T) {
    let src = unsafe {
        core::slice::from_raw_parts((value as *const T).cast::<u8>(), mem::size_of::<T>())
    };
    bytes[offset..offset + src.len()].copy_from_slice(src);
}

/// 从指定 offset 处非对齐读取一个 `Copy` 值，越界时返回 `TooSmall`。
fn read_repr_at<T: Copy>(bytes: &[u8], offset: usize) -> Result<T, AiGraphParseError> {
    let end = offset
        .checked_add(mem::size_of::<T>())
        .ok_or(AiGraphParseError::CountOverflow)?;
    if end > bytes.len() {
        return Err(AiGraphParseError::TooSmall);
    }
    Ok(unsafe { core::ptr::read_unaligned(bytes.as_ptr().add(offset).cast::<T>()) })
}

/// 连续读取 `count` 个 `Copy` 值到一个 `Vec`。
fn read_repr_vec<T: Copy>(bytes: &[u8], count: usize) -> Result<Vec<T>, AiGraphParseError> {
    let expected = count
        .checked_mul(mem::size_of::<T>())
        .ok_or(AiGraphParseError::CountOverflow)?;
    if expected > bytes.len() {
        return Err(AiGraphParseError::TooSmall);
    }

    let mut out = Vec::with_capacity(count);
    for idx in 0..count {
        out.push(read_repr_at::<T>(bytes, idx * mem::size_of::<T>())?);
    }
    Ok(out)
}

/// 用 0 填充 buffer 到 `align` 对齐边界，返回对齐后的偏移。
fn pad_to(bytes: &mut Vec<u8>, align: usize) -> Result<ByteOffset, AiGraphBuildError> {
    let padding = (align - (bytes.len() % align)) % align;
    let new_len = bytes
        .len()
        .checked_add(padding)
        .ok_or(AiGraphBuildError::SizeOverflow)?;
    bytes.resize(new_len, 0);
    Ok(ByteOffset::new(
        u32::try_from(bytes.len()).map_err(|_| AiGraphBuildError::SizeOverflow)?,
    ))
}

/// 校验 `offset + count * item_size` 落在 blob 内，返回该分段的字节范围。
fn checked_section(
    section: &'static str,
    total_size: usize,
    offset: ByteOffset,
    count: usize,
    item_size: usize,
) -> Result<core::ops::Range<usize>, AiGraphParseError> {
    let size = count
        .checked_mul(item_size)
        .ok_or(AiGraphParseError::CountOverflow)?;
    match offset.checked_range(size, total_size) {
        Ok(range) => Ok(range),
        Err(crate::AbiTypeError::CountOverflow) => Err(AiGraphParseError::CountOverflow),
        Err(_) => Err(AiGraphParseError::SectionOutOfRange {
            section,
            offset,
            size,
            total_size,
        }),
    }
}

/// 序列化前校验：节点 id 连续、边端点合法、且整图无环。
fn validate_graph(nodes: &[AiGraphNode], edges: &[AiGraphEdge]) -> Result<(), AiGraphBuildError> {
    for (idx, node) in nodes.iter().enumerate() {
        let expected = AiGraphNodeId::new(idx as u32);
        if node.node_id != expected {
            return Err(AiGraphBuildError::InvalidDepend(node.node_id));
        }
    }

    for edge in edges {
        if edge.from_node.as_usize() >= nodes.len() {
            return Err(AiGraphBuildError::InvalidDepend(edge.from_node));
        }
        if edge.to_node.as_usize() >= nodes.len() {
            return Err(AiGraphBuildError::InvalidDepend(edge.to_node));
        }
    }

    if has_cycle(nodes.len(), edges) {
        return Err(AiGraphBuildError::CycleDetected);
    }

    Ok(())
}

/// 用 Kahn 拓扑排序判断整图是否存在环。
fn has_cycle(node_count: usize, edges: &[AiGraphEdge]) -> bool {
    let mut indegree = vec![0_usize; node_count];
    let mut outgoing = vec![Vec::new(); node_count];

    for edge in edges {
        let from = edge.from_node.as_usize();
        let to = edge.to_node.as_usize();
        outgoing[from].push(to);
        indegree[to] += 1;
    }

    let mut ready = Vec::new();
    for (idx, degree) in indegree.iter().enumerate() {
        if *degree == 0 {
            ready.push(idx);
        }
    }

    let mut visited = 0_usize;
    while let Some(node) = ready.pop() {
        visited += 1;
        for &next in &outgoing[node] {
            indegree[next] -= 1;
            if indegree[next] == 0 {
                ready.push(next);
            }
        }
    }

    visited != node_count
}

const _: () = assert!(core::mem::align_of::<AiGraphSubmitEntry>() == 64);
const _: () = assert!(core::mem::align_of::<AiGraphNode>() == 64);

/// ABI raw mirror layout checks for graph structures touched by transparent newtypes.
#[allow(dead_code, missing_docs, clippy::missing_docs_in_private_items)]
mod abi_layout {
    use super::*;

    #[repr(C, align(64))]
    struct RawAiGraphSubmitEntry {
        abi_version: u32,
        submit_kind: u32,
        flags: u32,
        reserved0: u32,
        user_token: u32,
        graph_user_va: u64,
        graph_size: u64,
    }

    #[repr(C)]
    struct RawAiGraphHeader {
        magic: u32,
        total_size: u32,
        flags: u32,
        node_count: u32,
        edge_count: u32,
        nodes_offset: u32,
        edges_offset: u32,
    }

    #[repr(C)]
    struct RawAiGraphState {
        complete: u8,
        error_flag: u8,
    }

    #[repr(C)]
    struct RawAiGraphNode {
        node_id: u32,
        desc: AiKernelDesc,
        state: RawAiGraphState,
    }

    #[repr(C)]
    struct RawAiGraphEdge {
        from_node: u32,
        to_node: u32,
    }

    const _: () = assert!(
        core::mem::size_of::<AiGraphSubmitEntry>() == core::mem::size_of::<RawAiGraphSubmitEntry>()
    );
    const _: () = assert!(
        core::mem::align_of::<AiGraphSubmitEntry>()
            == core::mem::align_of::<RawAiGraphSubmitEntry>()
    );
    const _: () = assert!(
        core::mem::offset_of!(AiGraphSubmitEntry, flags)
            == core::mem::offset_of!(RawAiGraphSubmitEntry, flags)
    );
    const _: () = assert!(
        core::mem::offset_of!(AiGraphSubmitEntry, user_token)
            == core::mem::offset_of!(RawAiGraphSubmitEntry, user_token)
    );
    const _: () = assert!(
        core::mem::offset_of!(AiGraphSubmitEntry, graph_user_va)
            == core::mem::offset_of!(RawAiGraphSubmitEntry, graph_user_va)
    );
    const _: () = assert!(
        core::mem::offset_of!(AiGraphSubmitEntry, graph_size)
            == core::mem::offset_of!(RawAiGraphSubmitEntry, graph_size)
    );

    const _: () =
        assert!(core::mem::size_of::<AiGraphHeader>() == core::mem::size_of::<RawAiGraphHeader>());
    const _: () = assert!(
        core::mem::align_of::<AiGraphHeader>() == core::mem::align_of::<RawAiGraphHeader>()
    );
    const _: () = assert!(
        core::mem::offset_of!(AiGraphHeader, total_size)
            == core::mem::offset_of!(RawAiGraphHeader, total_size)
    );
    const _: () = assert!(
        core::mem::offset_of!(AiGraphHeader, flags)
            == core::mem::offset_of!(RawAiGraphHeader, flags)
    );
    const _: () = assert!(
        core::mem::offset_of!(AiGraphHeader, node_count)
            == core::mem::offset_of!(RawAiGraphHeader, node_count)
    );
    const _: () = assert!(
        core::mem::offset_of!(AiGraphHeader, edge_count)
            == core::mem::offset_of!(RawAiGraphHeader, edge_count)
    );
    const _: () = assert!(
        core::mem::offset_of!(AiGraphHeader, nodes_offset)
            == core::mem::offset_of!(RawAiGraphHeader, nodes_offset)
    );
    const _: () = assert!(
        core::mem::offset_of!(AiGraphHeader, edges_offset)
            == core::mem::offset_of!(RawAiGraphHeader, edges_offset)
    );

    const _: () =
        assert!(core::mem::size_of::<AiGraphState>() == core::mem::size_of::<RawAiGraphState>());
    const _: () =
        assert!(core::mem::align_of::<AiGraphState>() == core::mem::align_of::<RawAiGraphState>());
    const _: () = assert!(
        core::mem::offset_of!(AiGraphState, complete)
            == core::mem::offset_of!(RawAiGraphState, complete)
    );
    const _: () = assert!(
        core::mem::offset_of!(AiGraphState, error_flag)
            == core::mem::offset_of!(RawAiGraphState, error_flag)
    );

    const _: () =
        assert!(core::mem::size_of::<AiGraphNode>() == core::mem::size_of::<RawAiGraphNode>());
    const _: () =
        assert!(core::mem::align_of::<AiGraphNode>() == core::mem::align_of::<RawAiGraphNode>());
    const _: () = assert!(
        core::mem::offset_of!(AiGraphNode, node_id)
            == core::mem::offset_of!(RawAiGraphNode, node_id)
    );
    const _: () = assert!(
        core::mem::offset_of!(AiGraphNode, desc) == core::mem::offset_of!(RawAiGraphNode, desc)
    );
    const _: () = assert!(
        core::mem::offset_of!(AiGraphNode, state) == core::mem::offset_of!(RawAiGraphNode, state)
    );

    const _: () =
        assert!(core::mem::size_of::<AiGraphEdge>() == core::mem::size_of::<RawAiGraphEdge>());
    const _: () =
        assert!(core::mem::align_of::<AiGraphEdge>() == core::mem::align_of::<RawAiGraphEdge>());
    const _: () = assert!(
        core::mem::offset_of!(AiGraphEdge, from_node)
            == core::mem::offset_of!(RawAiGraphEdge, from_node)
    );
    const _: () = assert!(
        core::mem::offset_of!(AiGraphEdge, to_node)
            == core::mem::offset_of!(RawAiGraphEdge, to_node)
    );
}

/// graph 构建与解析的单元测试。
#[cfg(test)]
mod tests {
    use super::*;

    /// 基础插入测试。
    #[test]
    fn build_parse_chain() {
        let mut graph = GraphManager::new();

        let a = graph
            .push_kernel_no_depend(AiKernelDesc::default())
            .unwrap();
        let b = graph
            .push_kernel_depend(a, AiKernelDesc::default())
            .unwrap();
        let _c = graph
            .push_kernel_depend(b, AiKernelDesc::default())
            .unwrap();

        let blob = graph.freeze().unwrap();
        let parsed = AiGraphParser::parse(blob.as_bytes()).unwrap();

        assert_eq!(parsed.header.node_count, 3);
        assert_eq!(parsed.header.edge_count, 2);
        assert_eq!(parsed.nodes[0].node_id, 0);
        assert_eq!(parsed.edges[0].from_node, 0);
        assert_eq!(parsed.edges[0].to_node, 1);
    }

    /// 坏节点检测。
    #[test]
    fn reject_bad_depend() {
        let mut graph = GraphManager::new();
        let err = graph
            .push_kernel_depend(AiGraphNodeId(99), AiKernelDesc::default())
            .unwrap_err();
        assert_eq!(err, AiGraphBuildError::InvalidDepend(AiGraphNodeId(99)));
    }

    /// 多依赖图测试。
    #[test]
    fn build_parse_join_and_fork() {
        let mut graph = GraphManager::new();

        let a = graph
            .push_kernel_no_depend(AiKernelDesc::default())
            .unwrap();
        let c = graph
            .push_kernel_no_depend(AiKernelDesc::default())
            .unwrap();
        let b = graph
            .push_kernel_depend_many(&[a, c], AiKernelDesc::default())
            .unwrap();
        let _d = graph
            .push_kernel_depend(b, AiKernelDesc::default())
            .unwrap();
        let _e = graph
            .push_kernel_depend(b, AiKernelDesc::default())
            .unwrap();

        let blob = graph.freeze().unwrap();
        let parsed = AiGraphParser::parse(blob.as_bytes()).unwrap();

        assert_eq!(parsed.header.node_count, 5);
        assert_eq!(parsed.header.edge_count, 4);
        assert_eq!(parsed.edges[0].from_node, a.0);
        assert_eq!(parsed.edges[0].to_node, b.0);
        assert_eq!(parsed.edges[1].from_node, c.0);
        assert_eq!(parsed.edges[1].to_node, b.0);
        assert_eq!(parsed.edges[2].from_node, b.0);
        assert_eq!(parsed.edges[3].from_node, b.0);
    }

    /// 环检测。
    #[test]
    fn reject_cycle_edge_insert() {
        let mut graph = GraphManager::new();

        let a = graph
            .push_kernel_no_depend(AiKernelDesc::default())
            .unwrap();
        let b = graph
            .push_kernel_depend(a, AiKernelDesc::default())
            .unwrap();

        let err = graph.push_edge_checked(b, a).unwrap_err();
        assert_eq!(err, AiGraphBuildError::CycleDetected);
    }
}
