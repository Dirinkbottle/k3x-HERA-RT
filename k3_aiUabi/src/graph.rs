//! 算子计算图。
//!
//! channel里提交的是 `AiGraphSubmitEntry`
//! graph blob 内部用 offset 组织节点和边。

use core::mem;
use alloc::vec::Vec;
use alloc::vec;
use bitflags::bitflags;

use crate::{AI_ABI_VERSION, AiKernelDesc};

/// graph blob 魔数
pub const AI_GRAPH_MAGIC: u32 = 0x4845_5241; // "HERA"

/// 提交类型。
#[repr(transparent)]
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct GraphSubmitKind(pub u32);

impl GraphSubmitKind {
    pub const GRAPH_SUBMIT: Self = Self(1);
    pub const CANCEL: Self = Self(2);
    pub const QUERY: Self = Self(3);

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
    pub flags: u32,

    /// 预留字段，保持 8 字节对齐。
    pub reserved0: u32,

    /// 用户态 completion cookie。
    /// 用户态用它匹配完成的 graph。
    pub user_token: u64,

    /// graph blob 的用户态虚拟地址。
    pub graph_user_va: u64,

    /// graph blob 总字节数。
    pub graph_size: u64,
}

impl Default for AiGraphSubmitEntry {
    fn default() -> Self {
        Self {
            abi_version: AI_ABI_VERSION,
            submit_kind: GraphSubmitKind::GRAPH_SUBMIT,
            flags: 0,
            reserved0: 0,
            user_token: 0,
            graph_user_va: 0,
            graph_size: 0,
        }
    }
}

impl AiGraphSubmitEntry {
    pub fn new(
        user_token: u64,
        graph_user_va: u64,
        graph_size: u64,
        submit_kind: GraphSubmitKind,
    ) -> Self {
        Self {
            abi_version: AI_ABI_VERSION,
            submit_kind,
            flags: 0,
            reserved0: 0,
            user_token,
            graph_user_va,
            graph_size,
        }
    }
    /// 序列化提交
    pub fn to_le_byte(&self)->Option<&[u8]>{
        let self_size = core::mem::size_of::<Self>();
        if self_size>255 {
            return None
        }
        unsafe {
            Some(core::slice::from_raw_parts(
                self as *const Self as *const u8,
                self_size
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
    pub total_size: u32,

    /// graph flags，阶段一先保留。
    pub flags: u32,

    /// 节点数量。
    pub node_count: u32,

    /// 依赖边数量。
    pub edge_count: u32,

    /// node 数组在 graph blob 内的偏移。
    pub nodes_offset: u32,

    /// edge 数组在 graph blob 内的偏移。
    pub edges_offset: u32,
}

/// graph node 是 `AiKernelDesc` 的薄封装。
///
/// `desc` 描述这个节点要执行的语义级算子；`node_id` 只用于 graph 依赖关系。
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct AiGraphNode {
    /// graph 内稳定节点编号。
    pub node_id: u32,
    /// 单个 lowered 算子的描述。
    pub desc: AiKernelDesc,
    /// Graph节点的状态
    pub state:AiGraphState
}

/// 表示某个图节点的执行状态
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct AiGraphState{
    complete:bool,
    // bitflag 按位解释错误原因
    error_flag:u8,
}

bitflags!{
    pub struct GraphAiErrorFlags: u32 {
        const A = 0b00000001;
        const B = 0b00000010;
        const C = 0b00000100;
    }
}

/// graph 依赖边。
///
/// 表示 `from_node` 完成后，`to_node` 的一个前置依赖满足。
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct AiGraphEdge {
    pub from_node: u32,
    pub to_node: u32,
}

/// 用户态构图时返回的依赖标识。
///
/// 当前链尾 node id：继续向这条链追加算子时，
/// 把这个 id 传给 `push_kernel_depend` 。
#[repr(transparent)]
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct AiGraphNodeId(pub u32);

/// 一个 chain id 表示当前主图链的尾节点。
pub type AiGraphChainId = AiGraphNodeId;

/// 用户态可持有的 frozen graph blob。
///
/// 一整块连续字节流，
pub struct AiGraphBlob {
    bytes: Vec<u8>,
}

impl AiGraphBlob {
    /// 从节点和边序列化成一整块 graph blob。
    pub fn from_parts(
        nodes: &[AiGraphNode],
        edges: &[AiGraphEdge],
    ) -> Result<Self, AiGraphBuildError> {
        let node_count = u32::try_from(nodes.len()).map_err(|_| AiGraphBuildError::TooManyNodes)?;
        let edge_count = u32::try_from(edges.len()).map_err(|_| AiGraphBuildError::TooManyEdges)?;
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

        let total_size = u32::try_from(bytes.len()).map_err(|_| AiGraphBuildError::SizeOverflow)?;
        let header = AiGraphHeader {
            magic: AI_GRAPH_MAGIC,
            total_size,
            flags: 0,
            node_count,
            edge_count,
            nodes_offset,
            edges_offset,
        };
        write_repr_at(&mut bytes, 0, &header);

        Ok(Self { bytes })
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn submit_entry(&self, user_token: u64) -> AiGraphSubmitEntry {
        AiGraphSubmitEntry::new(
            user_token,
            self.bytes.as_ptr() as u64,
            self.bytes.len() as u64,
            GraphSubmitKind::GRAPH_SUBMIT,
        )
    }
}

/// 解析后的 graph。
pub struct AiParsedGraph {
    pub header: AiGraphHeader,
    pub nodes: Vec<AiGraphNode>,
    pub edges: Vec<AiGraphEdge>,
}

pub struct AiGraphParser;

impl AiGraphParser {
    pub fn parse(bytes: &[u8]) -> Result<AiParsedGraph, AiGraphParseError> {
        let header: AiGraphHeader = read_repr_at(bytes, 0)?;
        if header.magic != AI_GRAPH_MAGIC {
            return Err(AiGraphParseError::BadMagic(header.magic));
        }
        if header.total_size as usize != bytes.len() {
            return Err(AiGraphParseError::SizeMismatch {
                header_size: header.total_size,
                actual_size: bytes.len(),
            });
        }

        let nodes_range = checked_section(
            "nodes",
            bytes.len(),
            header.nodes_offset,
            header.node_count,
            mem::size_of::<AiGraphNode>(),
        )?;
        let edges_range = checked_section(
            "edges",
            bytes.len(),
            header.edges_offset,
            header.edge_count,
            mem::size_of::<AiGraphEdge>(),
        )?;

        let nodes = read_repr_vec::<AiGraphNode>(&bytes[nodes_range], header.node_count as usize)?;
        let edges = read_repr_vec::<AiGraphEdge>(&bytes[edges_range], header.edge_count as usize)?;

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
    nodes: Vec<AiGraphNode>,
    edges: Vec<AiGraphEdge>,
}

impl GraphManager {
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

    fn push_node(&mut self, desc: AiKernelDesc) -> Result<AiGraphChainId, AiGraphBuildError> {
        let node_id =
            u32::try_from(self.nodes.len()).map_err(|_| AiGraphBuildError::TooManyNodes)?;
        self.nodes.push(AiGraphNode { node_id, desc,state:AiGraphState::default()});
        Ok(AiGraphNodeId(node_id))
    }

    fn validate_node_id(&self, node_id: AiGraphNodeId) -> Result<(), AiGraphBuildError> {
        let idx = node_id.0 as usize;
        if idx >= self.nodes.len() || self.nodes[idx].node_id != node_id.0 {
            return Err(AiGraphBuildError::InvalidDepend(node_id));
        }
        Ok(())
    }

    fn push_edge_checked(
        &mut self,
        from: AiGraphNodeId,
        to: AiGraphNodeId,
    ) -> Result<(), AiGraphBuildError> {
        self.validate_node_id(from)?;
        self.validate_node_id(to)?;

        if from == to || self.reaches(to.0, from.0) {
            return Err(AiGraphBuildError::CycleDetected);
        }

        self.edges.push(AiGraphEdge {
            from_node: from.0,
            to_node: to.0,
        });
        Ok(())
    }

    fn reaches(&self, start: u32, target: u32) -> bool {
        let mut stack = vec![start];
        let mut visited = vec![false; self.nodes.len()];

        while let Some(node_id) = stack.pop() {
            if node_id == target {
                return true;
            }

            let idx = node_id as usize;
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

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum AiGraphBuildError {
    TooManyNodes,
    TooManyEdges,
    InvalidDepend(AiGraphNodeId),
    CycleDetected,
    SizeOverflow,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum AiGraphParseError {
    TooSmall,
    BadMagic(u32),
    BadAbi(u32),
    SizeMismatch {
        header_size: u32,
        actual_size: usize,
    },
    SectionOutOfRange {
        section: &'static str,
        offset: u32,
        size: usize,
        total_size: usize,
    },
    CountOverflow,
}

fn append_repr<T: Copy>(bytes: &mut Vec<u8>, value: &T) {
    let src = unsafe {
        core::slice::from_raw_parts((value as *const T).cast::<u8>(), mem::size_of::<T>())
    };
    bytes.extend_from_slice(src);
}

fn append_repr_slice<T: Copy>(bytes: &mut Vec<u8>, values: &[T]) {
    let src = unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), mem::size_of_val(values))
    };
    bytes.extend_from_slice(src);
}

fn write_repr_at<T: Copy>(bytes: &mut [u8], offset: usize, value: &T) {
    let src = unsafe {
        core::slice::from_raw_parts((value as *const T).cast::<u8>(), mem::size_of::<T>())
    };
    bytes[offset..offset + src.len()].copy_from_slice(src);
}

fn read_repr_at<T: Copy>(bytes: &[u8], offset: usize) -> Result<T, AiGraphParseError> {
    let end = offset
        .checked_add(mem::size_of::<T>())
        .ok_or(AiGraphParseError::CountOverflow)?;
    if end > bytes.len() {
        return Err(AiGraphParseError::TooSmall);
    }
    Ok(unsafe { core::ptr::read_unaligned(bytes.as_ptr().add(offset).cast::<T>()) })
}

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

fn pad_to(bytes: &mut Vec<u8>, align: usize) -> Result<u32, AiGraphBuildError> {
    let padding = (align - (bytes.len() % align)) % align;
    let new_len = bytes
        .len()
        .checked_add(padding)
        .ok_or(AiGraphBuildError::SizeOverflow)?;
    bytes.resize(new_len, 0);
    u32::try_from(bytes.len()).map_err(|_| AiGraphBuildError::SizeOverflow)
}

fn checked_section(
    section: &'static str,
    total_size: usize,
    offset: u32,
    count: u32,
    item_size: usize,
) -> Result<core::ops::Range<usize>, AiGraphParseError> {
    let size = (count as usize)
        .checked_mul(item_size)
        .ok_or(AiGraphParseError::CountOverflow)?;
    let start = offset as usize;
    let end = start
        .checked_add(size)
        .ok_or(AiGraphParseError::CountOverflow)?;
    if end > total_size {
        return Err(AiGraphParseError::SectionOutOfRange {
            section,
            offset,
            size,
            total_size,
        });
    }
    Ok(start..end)
}

fn validate_graph(nodes: &[AiGraphNode], edges: &[AiGraphEdge]) -> Result<(), AiGraphBuildError> {
    for (idx, node) in nodes.iter().enumerate() {
        if node.node_id != idx as u32 {
            return Err(AiGraphBuildError::InvalidDepend(AiGraphNodeId(
                node.node_id,
            )));
        }
    }

    for edge in edges {
        if edge.from_node as usize >= nodes.len() {
            return Err(AiGraphBuildError::InvalidDepend(AiGraphNodeId(
                edge.from_node,
            )));
        }
        if edge.to_node as usize >= nodes.len() {
            return Err(AiGraphBuildError::InvalidDepend(AiGraphNodeId(
                edge.to_node,
            )));
        }
    }

    if has_cycle(nodes.len(), edges) {
        return Err(AiGraphBuildError::CycleDetected);
    }

    Ok(())
}

fn has_cycle(node_count: usize, edges: &[AiGraphEdge]) -> bool {
    let mut indegree = vec![0_usize; node_count];
    let mut outgoing = vec![Vec::new(); node_count];

    for edge in edges {
        let from = edge.from_node as usize;
        let to = edge.to_node as usize;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    ///基础插入测试
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

    #[test]
    /// 坏节点检测
    fn reject_bad_depend() {
        let mut graph = GraphManager::new();
        let err = graph
            .push_kernel_depend(AiGraphNodeId(99), AiKernelDesc::default())
            .unwrap_err();
        assert_eq!(err, AiGraphBuildError::InvalidDepend(AiGraphNodeId(99)));
    }

    #[test]
    /// 多依赖图测试
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

    #[test]
    /// 环检测
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
