use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use k3_ai_runtime::fronted::kd_uring::{build_channel, submit_graph, wait_graph_complete};
use k3_ai_runtime::fronted::{
    AiDtype, AiKernelDesc, AiTargetHint, BinaryAttr, CopyAttr, DimCount, DimSize, ElemStride,
    GetRowsAttr, GluAttr, GraphManager, KernelOp, MatMulAttr, OpFlags, RmsNormAttr, RopeAttr,
    SoftmaxAttr, Tensor, TensorAxis, TensorCount, TensorManager, TransposeAttr, UnaryAttr,
    UserToken,
};

const GGUF_MAGIC: u32 = 0x4655_4747;
const DEFAULT_ALIGNMENT: u64 = 32;
const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_Q8_0: u32 = 8;
const GGML_TYPE_Q3_K: u32 = 11;
const GGML_TYPE_IQ4_NL: u32 = 20;
const SOURCE: &str = "TINYSORY_K3";
const HIDDEN: usize = 288;
const N_LAYER: usize = 6;
const N_HEAD: usize = 6;
const HEAD_DIM: usize = 48;
const FFN: usize = 768;
const VOCAB: usize = 32_000;
const CONTEXT: usize = 256;
const RMS_EPS: f32 = 1.0e-5;
const ROPE_FREQ_BASE: f32 = 10_000.0;

#[derive(Debug)]
struct Args {
    model: PathBuf,
    prompt: Option<String>,
    tokens: Vec<usize>,
    temp: f32,
    n_predict: usize,
    dump: PathBuf,
    list_tensors: bool,
    interactive: bool,
}

#[derive(Debug)]
struct Gguf {
    alignment: u64,
    data_start: usize,
    tensors: Vec<TensorInfo>,
    tokenizer: Tokenizer,
}

#[derive(Debug)]
struct Tokenizer {
    pieces: Vec<String>,
    piece_to_id: HashMap<String, usize>,
    bos: usize,
    eos: usize,
    unk: usize,
    max_piece_bytes: usize,
}

#[derive(Debug)]
struct TensorInfo {
    name: String,
    shape: Vec<u64>,
    ggml_type: u32,
    offset: u64,
}

#[derive(Clone, Copy)]
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

struct TokenSource {
    next: u32,
}

struct ModelWeights {
    weights: HashMap<String, Tensor>,
}

#[derive(Clone, Copy)]
struct RunConfig {
    dump_stride: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = parse_args()?;
    let bytes = fs::read(&args.model)?;
    let gguf = parse_gguf(&bytes)?;

    if let Some(prompt) = &args.prompt {
        args.tokens = gguf.tokenizer.tokenize(prompt, true)?;
    }
    if args.tokens.is_empty() && !args.interactive {
        return Err("missing initial input: pass --tokens, --prompt, or use --interactive".into());
    }

    println!(
        "tinysory: model={}, tensors={}, alignment={}, data_start={}, temp={}, n_predict={}",
        args.model.display(),
        gguf.tensors.len(),
        gguf.alignment,
        gguf.data_start,
        args.temp,
        args.n_predict
    );
    println!(
        "tinysory: tokenizer bos={}, eos={}, unk={}",
        gguf.tokenizer.bos, gguf.tokenizer.eos, gguf.tokenizer.unk
    );
    if let Some(prompt) = &args.prompt {
        println!("tinysory: prompt={prompt:?}, tokens={:?}", args.tokens);
    }

    if args.list_tensors {
        for tensor in &gguf.tensors {
            println!(
                "tensor name={} type={} shape={:?} offset={}",
                tensor.name, tensor.ggml_type, tensor.shape, tensor.offset
            );
        }
    }

    let token_embd = gguf
        .tensors
        .iter()
        .find(|tensor| tensor.name == "token_embd.weight")
        .ok_or("missing token_embd.weight")?;

    validate_story_layout(token_embd)?;
    let weights = load_model_weights(&bytes, &gguf)?;

    let channel = build_channel().map_err(|err| format!("build_channel failed: {err}"))?;
    let mut token_source = TokenSource { next: 1 };
    File::create(&args.dump)?;
    let config = RunConfig { dump_stride: 1 };

    if args.interactive {
        run_interactive_tokens(
            &channel,
            &weights,
            &gguf.tokenizer,
            &args,
            &config,
            &mut token_source,
        )?;
    } else {
        run_token_stream(&channel, &weights, &args, &config, &mut token_source)?;
    }

    println!(
        "tinysory: K3 dumped current connected nodes for tokens {:?} -> {}",
        args.tokens,
        args.dump.display()
    );
    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let mut model = None;
    let mut prompt = None;
    let mut tokens = Vec::new();
    let mut temp = 0.0_f32;
    let mut n_predict = 1_usize;
    let mut dump = PathBuf::from("ggml.txt");
    let mut list_tensors = false;
    let mut interactive = false;

    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--model" | "-m" => model = Some(PathBuf::from(next_arg(&mut iter, "--model")?)),
            "--prompt" | "-p" => prompt = Some(next_arg(&mut iter, "--prompt")?),
            "--tokens" => tokens = parse_tokens(&next_arg(&mut iter, "--tokens")?)?,
            "--temp" => temp = next_arg(&mut iter, "--temp")?.parse()?,
            "-n" | "--n-predict" => n_predict = next_arg(&mut iter, "--n-predict")?.parse()?,
            "--dump" => dump = PathBuf::from(next_arg(&mut iter, "--dump")?),
            "--list-tensors" => list_tensors = true,
            "--interactive" | "-i" => interactive = true,
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}").into()),
        }
    }

    let model = model.ok_or("missing --model <story.gguf>")?;

    Ok(Args {
        model,
        prompt,
        tokens,
        temp,
        n_predict,
        dump,
        list_tensors,
        interactive,
    })
}

fn next_arg(
    iter: &mut impl Iterator<Item = String>,
    name: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    iter.next()
        .ok_or_else(|| format!("missing value after {name}").into())
}

fn parse_tokens(raw: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    raw.split(',')
        .map(|part| {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                Err("empty token id".into())
            } else {
                trimmed
                    .parse::<usize>()
                    .map_err(|err| format!("bad token id {trimmed}: {err}").into())
            }
        })
        .collect()
}

fn parse_token_or_text_input(
    raw: &str,
    tokenizer: &Tokenizer,
) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let numericish = raw
        .chars()
        .all(|ch| ch.is_ascii_digit() || ch == ',' || ch.is_ascii_whitespace());
    if numericish {
        return raw
            .split(|ch: char| ch == ',' || ch.is_ascii_whitespace())
            .filter(|part| !part.is_empty())
            .map(|part| {
                part.parse::<usize>()
                    .map_err(|err| format!("bad token id {part}: {err}").into())
            })
            .collect();
    }
    tokenizer.tokenize(raw, false)
}

fn print_usage() {
    eprintln!(
        "usage: tinysory --model story.gguf [--prompt TEXT | --tokens 1,2] [--temp 0] [-n 1] [--dump ggml.txt] [--interactive] [--list-tensors]"
    );
}

fn run_token_stream(
    channel: &k3_ai_runtime::fronted::kd_uring::UringChannel,
    weights: &ModelWeights,
    args: &Args,
    config: &RunConfig,
    token_source: &mut TokenSource,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut tokens = args.tokens.clone();
    let mut rng = seed_from_tokens(&tokens);
    for step in 0..args.n_predict {
        let logits = forward_logits(channel, weights, &tokens, &args.dump, config, token_source)?;
        let next = sample_next_token(&logits, args.temp, &mut rng)?;
        println!(
            "tinysory: step={step} input_len={} next_token={next}",
            tokens.len()
        );
        println!("{next}");
        tokens.push(next);
        if tokens.len() > CONTEXT {
            return Err(format!("context length exceeded: {} > {CONTEXT}", tokens.len()).into());
        }
    }
    Ok(())
}

fn run_interactive_tokens(
    channel: &k3_ai_runtime::fronted::kd_uring::UringChannel,
    weights: &ModelWeights,
    tokenizer: &Tokenizer,
    args: &Args,
    config: &RunConfig,
    token_source: &mut TokenSource,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut tokens = args.tokens.clone();
    let mut rng = seed_from_tokens(&tokens);
    println!("tinysory: starting tokens={tokens:?}");
    for step in 0..args.n_predict {
        if tokens.is_empty() {
            print!("tinysory enter initial token/text [{step}]> ");
        } else {
            print!("tinysory enter extra token/text or empty to generate [{step}]> ");
        }
        io::stdout().flush()?;
        let mut line = String::new();
        if io::stdin().read_line(&mut line)? == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed == "/exit" {
            break;
        }
        if trimmed.is_empty() && tokens.is_empty() {
            println!("tinysory: no initial input yet");
            continue;
        }
        if !trimmed.is_empty() {
            let added = parse_token_or_text_input(trimmed, tokenizer)?;
            let pieces = added
                .iter()
                .map(|&token| tokenizer.piece(token))
                .collect::<Vec<_>>();
            println!("tinysory: accepted input tokens={added:?} pieces={pieces:?}");
            tokens.extend(added);
        }
        let logits = forward_logits(channel, weights, &tokens, &args.dump, config, token_source)?;
        let next = sample_next_token(&logits, args.temp, &mut rng)?;
        println!(
            "tinysory: step={step} input_len={} next_token={next}",
            tokens.len()
        );
        println!("{next}");
        tokens.push(next);
        if tokens.len() > CONTEXT {
            return Err(format!("context length exceeded: {} > {CONTEXT}", tokens.len()).into());
        }
    }
    Ok(())
}

fn forward_logits(
    channel: &k3_ai_runtime::fronted::kd_uring::UringChannel,
    weights: &ModelWeights,
    tokens: &[usize],
    dump: &Path,
    config: &RunConfig,
    token_source: &mut TokenSource,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if tokens.is_empty() {
        return Err("forward needs at least one token".into());
    }
    if tokens.len() > CONTEXT {
        return Err(format!("context length exceeded: {} > {CONTEXT}", tokens.len()).into());
    }
    for &token in tokens {
        if token >= VOCAB {
            return Err(format!("token id {token} out of vocab {VOCAB}").into());
        }
    }

    let seq = tokens.len();
    let mut node = 0_usize;
    let mut x = run_embedding(
        channel,
        weights.tensor("token_embd.weight")?,
        tokens,
        token_source,
    )?;
    dump_node(dump, config, seq, node, "GET_ROWS+TRANSPOSE", "embd", &x)?;
    node += 1;

    for layer in 0..N_LAYER {
        let attn_norm = run_rms_norm(
            channel,
            &x,
            weights.tensor(&format!("blk.{layer}.attn_norm.weight"))?,
            HIDDEN,
            token_source,
            &format!("blk.{layer}/attn_norm"),
        )?;
        dump_node(
            dump,
            config,
            seq,
            node,
            "RMS_NORM",
            &format!("blk.{layer}.attn_norm"),
            &attn_norm,
        )?;
        node += 1;

        let q = run_quant_matmul_activation(
            channel,
            weights.tensor(&format!("blk.{layer}.attn_q.weight"))?,
            &attn_norm,
            HIDDEN,
            HIDDEN,
            seq,
            token_source,
            &format!("blk.{layer}/attn_q"),
        )?;
        let k = run_quant_matmul_activation(
            channel,
            weights.tensor(&format!("blk.{layer}.attn_k.weight"))?,
            &attn_norm,
            HIDDEN,
            HIDDEN,
            seq,
            token_source,
            &format!("blk.{layer}/attn_k"),
        )?;
        let v = run_quant_matmul_activation(
            channel,
            weights.tensor(&format!("blk.{layer}.attn_v.weight"))?,
            &attn_norm,
            HIDDEN,
            HIDDEN,
            seq,
            token_source,
            &format!("blk.{layer}/attn_v"),
        )?;

        let q4 = run_copy_reshape(
            channel,
            &q,
            &[1, seq as u32, N_HEAD as u32, HEAD_DIM as u32],
            token_source,
            &format!("blk.{layer}/q_reshape"),
        )?;
        let k4 = run_copy_reshape(
            channel,
            &k,
            &[1, seq as u32, N_HEAD as u32, HEAD_DIM as u32],
            token_source,
            &format!("blk.{layer}/k_reshape"),
        )?;
        let v4 = run_copy_reshape(
            channel,
            &v,
            &[1, seq as u32, N_HEAD as u32, HEAD_DIM as u32],
            token_source,
            &format!("blk.{layer}/v_reshape"),
        )?;

        let positions = make_positions(seq)?;
        let q_rope = run_rope(
            channel,
            &q4,
            &positions,
            seq,
            token_source,
            &format!("blk.{layer}/q_rope"),
        )?;
        let k_rope = run_rope(
            channel,
            &k4,
            &positions,
            seq,
            token_source,
            &format!("blk.{layer}/k_rope"),
        )?;

        let scores = run_attention_scores(
            channel,
            &q_rope,
            &k_rope,
            seq,
            token_source,
            &format!("blk.{layer}/attn_scores"),
        )?;
        let scaled = run_scale(
            channel,
            &scores,
            1.0 / (HEAD_DIM as f32).sqrt(),
            0.0,
            token_source,
            &format!("blk.{layer}/attn_scale"),
        )?;
        let mask = make_causal_mask(seq)?;
        let probs = run_softmax(
            channel,
            &scaled,
            Some(&mask),
            TensorAxis::new(2),
            1.0,
            token_source,
            &format!("blk.{layer}/attn_softmax"),
        )?;
        let context = run_attention_values(
            channel,
            &probs,
            &v4,
            seq,
            token_source,
            &format!("blk.{layer}/attn_values"),
        )?;
        let attn_out = run_quant_matmul_activation(
            channel,
            weights.tensor(&format!("blk.{layer}.attn_output.weight"))?,
            &context,
            HIDDEN,
            HIDDEN,
            seq,
            token_source,
            &format!("blk.{layer}/attn_output"),
        )?;
        x = run_binary(
            channel,
            KernelOp::ADD,
            &x,
            &attn_out,
            token_source,
            &format!("blk.{layer}/attn_residual"),
        )?;
        dump_node(
            dump,
            config,
            seq,
            node,
            "ADD",
            &format!("blk.{layer}.attn_residual"),
            &x,
        )?;
        node += 1;

        let ffn_norm = run_rms_norm(
            channel,
            &x,
            weights.tensor(&format!("blk.{layer}.ffn_norm.weight"))?,
            HIDDEN,
            token_source,
            &format!("blk.{layer}/ffn_norm"),
        )?;
        let gate = run_quant_matmul_activation(
            channel,
            weights.tensor(&format!("blk.{layer}.ffn_gate.weight"))?,
            &ffn_norm,
            HIDDEN,
            FFN,
            seq,
            token_source,
            &format!("blk.{layer}/ffn_gate"),
        )?;
        let up = run_quant_matmul_activation(
            channel,
            weights.tensor(&format!("blk.{layer}.ffn_up.weight"))?,
            &ffn_norm,
            HIDDEN,
            FFN,
            seq,
            token_source,
            &format!("blk.{layer}/ffn_up"),
        )?;
        let act = run_glu(
            channel,
            &gate,
            &up,
            token_source,
            &format!("blk.{layer}/swiglu"),
        )?;
        let down = run_quant_matmul_activation(
            channel,
            weights.tensor(&format!("blk.{layer}.ffn_down.weight"))?,
            &act,
            FFN,
            HIDDEN,
            seq,
            token_source,
            &format!("blk.{layer}/ffn_down"),
        )?;
        x = run_binary(
            channel,
            KernelOp::ADD,
            &x,
            &down,
            token_source,
            &format!("blk.{layer}/ffn_residual"),
        )?;
        dump_node(
            dump,
            config,
            seq,
            node,
            "ADD",
            &format!("blk.{layer}.ffn_residual"),
            &x,
        )?;
        node += 1;
    }

    let final_norm = run_rms_norm(
        channel,
        &x,
        weights.tensor("output_norm.weight")?,
        HIDDEN,
        token_source,
        "output_norm",
    )?;
    let logits = run_quant_matmul_activation(
        channel,
        weights.tensor("token_embd.weight")?,
        &final_norm,
        HIDDEN,
        VOCAB,
        seq,
        token_source,
        "output_logits",
    )?;
    let all_logits = tensor_f32_vec(&logits);
    let start = (seq - 1) * VOCAB;
    let last = all_logits[start..start + VOCAB].to_vec();
    append_dump_tensor(
        dump,
        seq,
        node,
        "MAT_MUL",
        "logits.last",
        &[1, 1, 1, VOCAB],
        Some(tokens),
        &last,
    )?;
    Ok(last)
}

fn run_embedding(
    channel: &k3_ai_runtime::fronted::kd_uring::UringChannel,
    embedding: &Tensor,
    tokens: &[usize],
    token_source: &mut TokenSource,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let seq = tokens.len();
    let tensor_mgr = TensorManager::new();
    let mut indices = tensor_mgr
        .alloc_tensor(AiDtype::I32, &[seq as u32])
        .map_err(|err| format!("alloc token indices failed: {err}"))?;
    let embd_raw = tensor_mgr
        .alloc_tensor(AiDtype::F32, &[HIDDEN as u32, seq as u32, 1, 1])
        .map_err(|err| format!("alloc embd raw failed: {err}"))?;
    copy_i32_to_tensor(
        &mut indices,
        &tokens.iter().map(|token| *token as i32).collect::<Vec<_>>(),
    );
    run_kernel(
        channel,
        KernelOp::GET_ROWS,
        &GetRowsAttr {
            flags: OpFlags::new(0),
            reserved: [0; 15],
        },
        AiTargetHint::AUTO,
        &[embedding, &indices],
        &[&embd_raw],
        token_source,
        "embd/get_rows",
    )?;

    let transposed = tensor_mgr
        .alloc_tensor(AiDtype::F32, &[seq as u32, HIDDEN as u32, 1, 1])
        .map_err(|err| format!("alloc embd transpose failed: {err}"))?;
    run_transpose(
        channel,
        &embd_raw,
        &transposed,
        [1, 0, 2, 3],
        token_source,
        "embd/transpose",
    )?;
    run_copy_reshape(
        channel,
        &transposed,
        &[1, 1, seq as u32, HIDDEN as u32],
        token_source,
        "embd/reshape",
    )
}

fn run_rms_norm(
    channel: &k3_ai_runtime::fronted::kd_uring::UringChannel,
    input: &Tensor,
    weight: &Tensor,
    hidden: usize,
    token_source: &mut TokenSource,
    label: &str,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let tensor_mgr = TensorManager::new();
    let shape = shape_u32(input);
    let out = tensor_mgr
        .alloc_tensor(AiDtype::F32, &shape)
        .map_err(|err| format!("alloc {label} failed: {err}"))?;
    run_kernel(
        channel,
        KernelOp::RMS_NORM,
        &RmsNormAttr {
            hidden_size: DimSize::new(hidden as u32),
            eps: RMS_EPS,
            flags: OpFlags::new(0),
            reserved: [0; 13],
        },
        AiTargetHint::AUTO,
        &[input, weight],
        &[&out],
        token_source,
        label,
    )?;
    Ok(out)
}

fn run_quant_matmul_activation(
    channel: &k3_ai_runtime::fronted::kd_uring::UringChannel,
    weight: &Tensor,
    activation: &Tensor,
    k: usize,
    out_dim: usize,
    seq: usize,
    token_source: &mut TokenSource,
    label: &str,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let tensor_mgr = TensorManager::new();
    let out = tensor_mgr
        .alloc_tensor(AiDtype::F32, &[1, 1, seq as u32, out_dim as u32])
        .map_err(|err| format!("alloc {label} failed: {err}"))?;
    run_kernel(
        channel,
        KernelOp::MAT_MUL,
        &MatMulAttr {
            m: DimSize::new(out_dim as u32),
            n: DimSize::new(seq as u32),
            k: DimSize::new(k as u32),
            batch: DimSize::new(0),
            lhs_row_stride: ElemStride::new(0),
            lhs_col_stride: ElemStride::new(0),
            lhs_batch_stride: ElemStride::new(0),
            rhs_row_stride: ElemStride::new(1),
            rhs_col_stride: ElemStride::new(k as u32),
            rhs_batch_stride: ElemStride::new(0),
            out_row_stride: ElemStride::new(1),
            out_col_stride: ElemStride::new(out_dim as u32),
            out_batch_stride: ElemStride::new(0),
            flags: OpFlags::new(0),
            accum_dtype: AiDtype::F32,
            reserved: [0; 3],
        },
        AiTargetHint::AUTO,
        &[weight, activation],
        &[&out],
        token_source,
        label,
    )?;
    Ok(out)
}

fn run_attention_scores(
    channel: &k3_ai_runtime::fronted::kd_uring::UringChannel,
    q: &Tensor,
    k: &Tensor,
    seq: usize,
    token_source: &mut TokenSource,
    label: &str,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let tensor_mgr = TensorManager::new();
    let out = tensor_mgr
        .alloc_tensor(AiDtype::F32, &[N_HEAD as u32, seq as u32, seq as u32])
        .map_err(|err| format!("alloc {label} failed: {err}"))?;
    run_kernel(
        channel,
        KernelOp::MAT_MUL,
        &MatMulAttr {
            m: DimSize::new(seq as u32),
            n: DimSize::new(seq as u32),
            k: DimSize::new(HEAD_DIM as u32),
            batch: DimSize::new(N_HEAD as u32),
            lhs_row_stride: ElemStride::new(HIDDEN as u32),
            lhs_col_stride: ElemStride::new(1),
            lhs_batch_stride: ElemStride::new(HEAD_DIM as u32),
            rhs_row_stride: ElemStride::new(1),
            rhs_col_stride: ElemStride::new(HIDDEN as u32),
            rhs_batch_stride: ElemStride::new(HEAD_DIM as u32),
            out_row_stride: ElemStride::new(seq as u32),
            out_col_stride: ElemStride::new(1),
            out_batch_stride: ElemStride::new((seq * seq) as u32),
            flags: OpFlags::new(0),
            accum_dtype: AiDtype::F32,
            reserved: [0; 3],
        },
        AiTargetHint::AUTO,
        &[q, k],
        &[&out],
        token_source,
        label,
    )?;
    Ok(out)
}

fn run_attention_values(
    channel: &k3_ai_runtime::fronted::kd_uring::UringChannel,
    probs: &Tensor,
    v: &Tensor,
    seq: usize,
    token_source: &mut TokenSource,
    label: &str,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let tensor_mgr = TensorManager::new();
    let out = tensor_mgr
        .alloc_tensor(
            AiDtype::F32,
            &[1, seq as u32, N_HEAD as u32, HEAD_DIM as u32],
        )
        .map_err(|err| format!("alloc {label} failed: {err}"))?;
    run_kernel(
        channel,
        KernelOp::MAT_MUL,
        &MatMulAttr {
            m: DimSize::new(seq as u32),
            n: DimSize::new(HEAD_DIM as u32),
            k: DimSize::new(seq as u32),
            batch: DimSize::new(N_HEAD as u32),
            lhs_row_stride: ElemStride::new(seq as u32),
            lhs_col_stride: ElemStride::new(1),
            lhs_batch_stride: ElemStride::new((seq * seq) as u32),
            rhs_row_stride: ElemStride::new(HIDDEN as u32),
            rhs_col_stride: ElemStride::new(1),
            rhs_batch_stride: ElemStride::new(HEAD_DIM as u32),
            out_row_stride: ElemStride::new(HIDDEN as u32),
            out_col_stride: ElemStride::new(1),
            out_batch_stride: ElemStride::new(HEAD_DIM as u32),
            flags: OpFlags::new(0),
            accum_dtype: AiDtype::F32,
            reserved: [0; 3],
        },
        AiTargetHint::AUTO,
        &[probs, v],
        &[&out],
        token_source,
        label,
    )?;
    Ok(out)
}

fn run_binary(
    channel: &k3_ai_runtime::fronted::kd_uring::UringChannel,
    op: KernelOp,
    lhs: &Tensor,
    rhs: &Tensor,
    token_source: &mut TokenSource,
    label: &str,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let tensor_mgr = TensorManager::new();
    let shape = shape_u32(lhs);
    let out = tensor_mgr
        .alloc_tensor(AiDtype::F32, &shape)
        .map_err(|err| format!("alloc {label} failed: {err}"))?;
    run_kernel(
        channel,
        op,
        &BinaryAttr {
            broadcast_kind: 0,
            alpha: 1.0,
            beta: 1.0,
            flags: OpFlags::new(0),
            reserved: [0; 12],
        },
        AiTargetHint::AUTO,
        &[lhs, rhs],
        &[&out],
        token_source,
        label,
    )?;
    Ok(out)
}

fn run_scale(
    channel: &k3_ai_runtime::fronted::kd_uring::UringChannel,
    input: &Tensor,
    alpha: f32,
    beta: f32,
    token_source: &mut TokenSource,
    label: &str,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let tensor_mgr = TensorManager::new();
    let shape = shape_u32(input);
    let out = tensor_mgr
        .alloc_tensor(AiDtype::F32, &shape)
        .map_err(|err| format!("alloc {label} failed: {err}"))?;
    run_kernel(
        channel,
        KernelOp::SCALE,
        &UnaryAttr {
            alpha,
            beta,
            flags: OpFlags::new(0),
            reserved: [0; 13],
        },
        AiTargetHint::AUTO,
        &[input],
        &[&out],
        token_source,
        label,
    )?;
    Ok(out)
}

fn run_softmax(
    channel: &k3_ai_runtime::fronted::kd_uring::UringChannel,
    input: &Tensor,
    mask: Option<&Tensor>,
    axis: TensorAxis,
    scale: f32,
    token_source: &mut TokenSource,
    label: &str,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let tensor_mgr = TensorManager::new();
    let shape = shape_u32(input);
    let out = tensor_mgr
        .alloc_tensor(AiDtype::F32, &shape)
        .map_err(|err| format!("alloc {label} failed: {err}"))?;
    let mut inputs = vec![input];
    if let Some(mask) = mask {
        inputs.push(mask);
    }
    run_kernel(
        channel,
        KernelOp::SOFTMAX,
        &SoftmaxAttr {
            axis,
            scale,
            max_bias: 0.0,
            flags: OpFlags::new(if mask.is_some() {
                SoftmaxAttr::HAS_MASK
            } else {
                0
            }),
            reserved: [0; 12],
        },
        AiTargetHint::AUTO,
        &inputs,
        &[&out],
        token_source,
        label,
    )?;
    Ok(out)
}

fn run_rope(
    channel: &k3_ai_runtime::fronted::kd_uring::UringChannel,
    input: &Tensor,
    positions: &Tensor,
    seq: usize,
    token_source: &mut TokenSource,
    label: &str,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let tensor_mgr = TensorManager::new();
    let shape = shape_u32(input);
    let out = tensor_mgr
        .alloc_tensor(AiDtype::F32, &shape)
        .map_err(|err| format!("alloc {label} failed: {err}"))?;
    run_kernel(
        channel,
        KernelOp::ROPE,
        &RopeAttr {
            n_dims: DimSize::new(HEAD_DIM as u32),
            mode: RopeAttr::MODE_NEOX,
            n_ctx: DimSize::new(CONTEXT as u32),
            head_count: DimSize::new(N_HEAD as u32),
            freq_base: ROPE_FREQ_BASE,
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 0.0,
            beta_slow: 0.0,
            flags: OpFlags::new(0),
            reserved: [0; 5],
        },
        AiTargetHint::AUTO,
        &[input, positions],
        &[&out],
        token_source,
        label,
    )?;
    let _ = seq;
    Ok(out)
}

fn run_glu(
    channel: &k3_ai_runtime::fronted::kd_uring::UringChannel,
    gate: &Tensor,
    up: &Tensor,
    token_source: &mut TokenSource,
    label: &str,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let tensor_mgr = TensorManager::new();
    let shape = shape_u32(gate);
    let out = tensor_mgr
        .alloc_tensor(AiDtype::F32, &shape)
        .map_err(|err| format!("alloc {label} failed: {err}"))?;
    run_kernel(
        channel,
        KernelOp::GLU,
        &GluAttr {
            op: GluAttr::OP_SWIGLU,
            swapped: 0,
            flags: OpFlags::new(0),
            reserved: [0; 13],
        },
        AiTargetHint::AUTO,
        &[gate, up],
        &[&out],
        token_source,
        label,
    )?;
    Ok(out)
}

fn run_copy_reshape(
    channel: &k3_ai_runtime::fronted::kd_uring::UringChannel,
    input: &Tensor,
    shape: &[u32],
    token_source: &mut TokenSource,
    label: &str,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let tensor_mgr = TensorManager::new();
    let out = tensor_mgr
        .alloc_tensor(AiDtype::F32, shape)
        .map_err(|err| format!("alloc {label} failed: {err}"))?;
    run_kernel(
        channel,
        KernelOp::COPY,
        &CopyAttr {
            flags: OpFlags::new(0),
            reserved: [0; 15],
        },
        AiTargetHint::AUTO,
        &[input],
        &[&out],
        token_source,
        label,
    )?;
    Ok(out)
}

fn run_transpose(
    channel: &k3_ai_runtime::fronted::kd_uring::UringChannel,
    input: &Tensor,
    output: &Tensor,
    perm: [i32; 4],
    token_source: &mut TokenSource,
    label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    run_kernel(
        channel,
        KernelOp::TRANSPOSE,
        &TransposeAttr {
            rank: DimCount::new(4),
            perm: [
                TensorAxis::new(perm[0]),
                TensorAxis::new(perm[1]),
                TensorAxis::new(perm[2]),
                TensorAxis::new(perm[3]),
                TensorAxis::new(4),
                TensorAxis::new(5),
                TensorAxis::new(6),
                TensorAxis::new(7),
            ],
            flags: OpFlags::new(0),
            reserved: [0; 6],
        },
        AiTargetHint::AUTO,
        &[input],
        &[output],
        token_source,
        label,
    )
}

fn make_positions(seq: usize) -> Result<Tensor, Box<dyn std::error::Error>> {
    let tensor_mgr = TensorManager::new();
    let mut positions = tensor_mgr
        .alloc_tensor(AiDtype::I64, &[seq as u32])
        .map_err(|err| format!("alloc positions failed: {err}"))?;
    let values = (0..seq).map(|idx| idx as i64).collect::<Vec<_>>();
    copy_i64_to_tensor(&mut positions, &values);
    Ok(positions)
}

fn make_causal_mask(seq: usize) -> Result<Tensor, Box<dyn std::error::Error>> {
    let tensor_mgr = TensorManager::new();
    let mut mask = tensor_mgr
        .alloc_tensor(AiDtype::F32, &[1, seq as u32, seq as u32])
        .map_err(|err| format!("alloc causal mask failed: {err}"))?;
    let mut values = vec![0.0_f32; seq * seq];
    for query in 0..seq {
        for key in query + 1..seq {
            values[query * seq + key] = -1.0e30;
        }
    }
    copy_f32_to_tensor(&mut mask, &values);
    Ok(mask)
}

fn dump_node(
    dump: &Path,
    config: &RunConfig,
    seq: usize,
    node: usize,
    op: &str,
    name: &str,
    tensor: &Tensor,
) -> Result<(), Box<dyn std::error::Error>> {
    if config.dump_stride == 0 || !seq.is_multiple_of(config.dump_stride) {
        return Ok(());
    }
    let mut shape = [1_usize; 4];
    for (idx, dim) in tensor.shape().iter().enumerate().take(4) {
        shape[idx] = dim.get() as usize;
    }
    append_dump_tensor(
        dump,
        seq,
        node,
        op,
        name,
        &shape,
        None,
        &tensor_f32_vec(tensor),
    )
}

fn load_model_weights(
    bytes: &[u8],
    gguf: &Gguf,
) -> Result<ModelWeights, Box<dyn std::error::Error>> {
    let tensor_mgr = TensorManager::new();
    let mut weights = HashMap::with_capacity(gguf.tensors.len());
    for info in &gguf.tensors {
        let shape = info
            .shape
            .iter()
            .map(|dim| {
                u32::try_from(*dim).map_err(|_| format!("tensor {} dim overflow", info.name))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let nbytes = ggml_tensor_nbytes(info)?;
        let raw = checked_tensor_bytes(bytes, gguf, info, nbytes)?;
        let mut tensor = if info.ggml_type == GGML_TYPE_F32 {
            tensor_mgr
                .alloc_tensor(AiDtype::F32, &shape)
                .map_err(|err| format!("alloc {} failed: {err}", info.name))?
        } else {
            tensor_mgr
                .alloc_ggml_quant_tensor(ggml_type_to_ai_dtype(info.ggml_type)?, &shape, 0)
                .map_err(|err| format!("alloc {} failed: {err}", info.name))?
        };
        tensor.as_mut_slice().copy_from_slice(raw);
        weights.insert(info.name.clone(), tensor);
    }
    Ok(ModelWeights { weights })
}

fn validate_story_layout(token_embd: &TensorInfo) -> Result<(), Box<dyn std::error::Error>> {
    if token_embd.ggml_type != GGML_TYPE_Q8_0 || token_embd.shape != [HIDDEN as u64, VOCAB as u64] {
        return Err(format!(
            "story.gguf layout mismatch: token_embd type={} shape={:?}, expected Q8_0 [{HIDDEN}, {VOCAB}]",
            token_embd.ggml_type, token_embd.shape
        )
        .into());
    }
    Ok(())
}

fn ggml_type_to_ai_dtype(ggml_type: u32) -> Result<AiDtype, Box<dyn std::error::Error>> {
    match ggml_type {
        GGML_TYPE_Q8_0 => Ok(AiDtype::Q8_0),
        GGML_TYPE_Q3_K => Ok(AiDtype::Q3_K),
        GGML_TYPE_IQ4_NL => Ok(AiDtype::IQ4_NL),
        other => Err(format!("unsupported GGML tensor type {other}").into()),
    }
}

fn ggml_tensor_nbytes(info: &TensorInfo) -> Result<usize, Box<dyn std::error::Error>> {
    if info.shape.is_empty() {
        return Err(format!("tensor {} has empty shape", info.name).into());
    }
    let first = info.shape[0] as usize;
    let rows = info.shape[1..]
        .iter()
        .try_fold(1_usize, |product, &dim| product.checked_mul(dim as usize))
        .ok_or_else(|| format!("tensor {} size overflow", info.name))?;
    let row_bytes = match info.ggml_type {
        GGML_TYPE_F32 => first
            .checked_mul(4)
            .ok_or_else(|| format!("tensor {} row overflow", info.name))?,
        GGML_TYPE_Q8_0 => quant_row_bytes(first, 32, 34, &info.name)?,
        GGML_TYPE_Q3_K => quant_row_bytes(first, 256, 110, &info.name)?,
        GGML_TYPE_IQ4_NL => quant_row_bytes(first, 32, 18, &info.name)?,
        other => {
            return Err(format!("unsupported GGML tensor type {other} in {}", info.name).into());
        }
    };
    row_bytes
        .checked_mul(rows)
        .ok_or_else(|| format!("tensor {} size overflow", info.name).into())
}

fn quant_row_bytes(
    first: usize,
    block: usize,
    bytes: usize,
    name: &str,
) -> Result<usize, Box<dyn std::error::Error>> {
    if first == 0 || !first.is_multiple_of(block) {
        return Err(format!("tensor {name} first dim {first} is not multiple of {block}").into());
    }
    Ok(first / block * bytes)
}

impl ModelWeights {
    fn tensor(&self, name: &str) -> Result<&Tensor, Box<dyn std::error::Error>> {
        self.weights
            .get(name)
            .ok_or_else(|| format!("missing tensor {name}").into())
    }
}

fn sample_next_token(
    logits: &[f32],
    temp: f32,
    rng: &mut u64,
) -> Result<usize, Box<dyn std::error::Error>> {
    if logits.len() != VOCAB {
        return Err(format!("logits len mismatch: {} != {VOCAB}", logits.len()).into());
    }
    if temp <= 0.0 {
        return logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx)
            .ok_or_else(|| "empty logits".into());
    }
    let inv_temp = 1.0 / temp;
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f64;
    for &logit in logits {
        sum += ((logit - max) * inv_temp).exp() as f64;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err("bad logits distribution".into());
    }
    let mut threshold = next_unit_f64(rng) * sum;
    for (idx, &logit) in logits.iter().enumerate() {
        threshold -= ((logit - max) * inv_temp).exp() as f64;
        if threshold <= 0.0 {
            return Ok(idx);
        }
    }
    Ok(VOCAB - 1)
}

fn seed_from_tokens(tokens: &[usize]) -> u64 {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    for &token in tokens {
        state ^= token as u64;
        state = state.wrapping_mul(0xbf58_476d_1ce4_e5b9).rotate_left(27);
    }
    state.max(1)
}

fn next_unit_f64(state: &mut u64) -> f64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x.max(1);
    ((*state >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
}

fn shape_u32(tensor: &Tensor) -> Vec<u32> {
    tensor.shape().iter().map(|dim| dim.get()).collect()
}

fn copy_f32_to_tensor(tensor: &mut Tensor, values: &[f32]) {
    let bytes = tensor.as_mut_slice();
    assert_eq!(bytes.len(), std::mem::size_of_val(values));
    for (chunk, value) in bytes.chunks_exact_mut(4).zip(values) {
        chunk.copy_from_slice(&value.to_ne_bytes());
    }
}

fn copy_i64_to_tensor(tensor: &mut Tensor, values: &[i64]) {
    let bytes = tensor.as_mut_slice();
    assert_eq!(bytes.len(), std::mem::size_of_val(values));
    for (chunk, value) in bytes.chunks_exact_mut(8).zip(values) {
        chunk.copy_from_slice(&value.to_ne_bytes());
    }
}

fn run_kernel<T: Copy>(
    channel: &k3_ai_runtime::fronted::kd_uring::UringChannel,
    op: KernelOp,
    attr: &T,
    target: AiTargetHint,
    inputs: &[&Tensor],
    outputs: &[&Tensor],
    tokens: &mut TokenSource,
    label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let tensors = inputs
        .iter()
        .chain(outputs.iter())
        .map(|tensor| tensor.desc())
        .collect::<Vec<_>>();
    let desc = AiKernelDesc::new_with_op(
        op,
        attr,
        target,
        TensorCount::new(inputs.len() as u32),
        TensorCount::new(outputs.len() as u32),
        &tensors,
    );

    let mut graph = GraphManager::new();
    graph
        .push_kernel_no_depend(desc)
        .unwrap_or_else(|err| panic!("failed to push {label} node: {err:?}"));
    let blob = graph
        .freeze()
        .unwrap_or_else(|err| panic!("failed to freeze {label} graph: {err:?}"));
    let entry = blob.submit_entry(UserToken::new(tokens.next()));

    println!("tinysory: submit {label}, op={op:?}, target={target:?}");
    submit_graph(channel, &entry).map_err(|err| format!("submit {label} failed: {err}"))?;
    wait_graph_complete(&entry, channel).map_err(|err| format!("wait {label} failed: {err}"))?;
    Ok(())
}

fn checked_tensor_bytes<'a>(
    bytes: &'a [u8],
    gguf: &Gguf,
    tensor: &TensorInfo,
    nbytes: usize,
) -> Result<&'a [u8], Box<dyn std::error::Error>> {
    let base = gguf
        .data_start
        .checked_add(tensor.offset as usize)
        .ok_or("tensor offset overflow")?;
    let end = base.checked_add(nbytes).ok_or("tensor end overflow")?;
    bytes
        .get(base..end)
        .ok_or_else(|| format!("tensor {} exceeds file size", tensor.name).into())
}

fn copy_i32_to_tensor(tensor: &mut Tensor, values: &[i32]) {
    let bytes = tensor.as_mut_slice();
    assert_eq!(bytes.len(), std::mem::size_of_val(values));
    for (chunk, value) in bytes.chunks_exact_mut(4).zip(values) {
        chunk.copy_from_slice(&value.to_ne_bytes());
    }
}

fn tensor_f32_vec(tensor: &Tensor) -> Vec<f32> {
    tensor
        .as_slice()
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn append_dump_tensor(
    dump_path: &Path,
    seq: usize,
    node: usize,
    op: &str,
    name: &str,
    shape: &[usize; 4],
    tokens: Option<&[usize]>,
    values: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::options().append(true).create(true).open(dump_path)?;
    write!(
        file,
        "{{\"source\":\"{SOURCE}\",\"graph\":0,\"seq\":{seq},\"node\":{node},\
         \"op\":\"{op}\",\"name\":\"{name}\",\"type\":\"f32\",\
         \"shape\":[{},{},{},{}],",
        shape[0], shape[1], shape[2], shape[3]
    )?;
    if let Some(tokens) = tokens {
        write!(file, "\"tokens\":[")?;
        for (idx, token) in tokens.iter().enumerate() {
            if idx != 0 {
                write!(file, ",")?;
            }
            write!(file, "{token}")?;
        }
        write!(file, "],")?;
    }
    write!(file, "\"values\":[")?;
    for (idx, value) in values.iter().enumerate() {
        if idx != 0 {
            write!(file, ",")?;
        }
        write!(file, "{value:.9}")?;
    }
    writeln!(file, "]}}")?;
    Ok(())
}

impl TokenSource {
    fn next(&mut self) -> u32 {
        let token = self.next;
        self.next = self.next.checked_add(1).expect("user token overflow");
        token
    }
}

fn parse_gguf(bytes: &[u8]) -> Result<Gguf, Box<dyn std::error::Error>> {
    let mut reader = Reader {
        data: bytes,
        pos: 0,
    };
    let magic = reader.u32()?;
    if magic != GGUF_MAGIC {
        return Err(format!("bad GGUF magic: 0x{magic:08x}").into());
    }
    let version = reader.u32()?;
    if version != 2 && version != 3 {
        return Err(format!("unsupported GGUF version: {version}").into());
    }
    let tensor_count = reader.u64()? as usize;
    let kv_count = reader.u64()? as usize;

    let mut alignment = DEFAULT_ALIGNMENT;
    let mut tokenizer_pieces = None;
    let mut bos = 1_usize;
    let mut eos = 2_usize;
    let mut unk = 0_usize;
    for _ in 0..kv_count {
        let key = reader.string()?;
        let value_type = reader.u32()?;
        match key.as_str() {
            "general.alignment" if value_type == 4 => alignment = reader.u32()? as u64,
            "tokenizer.ggml.tokens" if value_type == 9 => {
                tokenizer_pieces = Some(reader.string_array()?)
            }
            "tokenizer.ggml.bos_token_id" if value_type == 4 => bos = reader.u32()? as usize,
            "tokenizer.ggml.eos_token_id" if value_type == 4 => eos = reader.u32()? as usize,
            "tokenizer.ggml.unknown_token_id" if value_type == 4 => unk = reader.u32()? as usize,
            _ => reader.skip_value(value_type)?,
        }
    }
    let tokenizer = Tokenizer::new(
        tokenizer_pieces.ok_or("missing tokenizer.ggml.tokens")?,
        bos,
        eos,
        unk,
    )?;

    let mut tensors = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = reader.string()?;
        let ndim = reader.u32()? as usize;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(reader.u64()?);
        }
        let ggml_type = reader.u32()?;
        let offset = reader.u64()?;
        tensors.push(TensorInfo {
            name,
            shape,
            ggml_type,
            offset,
        });
    }

    let data_start = align_up(reader.pos as u64, alignment)? as usize;
    Ok(Gguf {
        alignment,
        data_start,
        tensors,
        tokenizer,
    })
}

fn align_up(value: u64, alignment: u64) -> Result<u64, Box<dyn std::error::Error>> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(format!("bad GGUF alignment: {alignment}").into());
    }
    Ok((value + alignment - 1) & !(alignment - 1))
}

impl Tokenizer {
    fn new(
        pieces: Vec<String>,
        bos: usize,
        eos: usize,
        unk: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if pieces.len() != VOCAB {
            return Err(format!("tokenizer vocab mismatch: {} != {VOCAB}", pieces.len()).into());
        }
        let mut piece_to_id = HashMap::with_capacity(pieces.len());
        let mut max_piece_bytes = 0_usize;
        for (idx, piece) in pieces.iter().enumerate() {
            max_piece_bytes = max_piece_bytes.max(piece.len());
            piece_to_id.entry(piece.clone()).or_insert(idx);
        }
        Ok(Self {
            pieces,
            piece_to_id,
            bos,
            eos,
            unk,
            max_piece_bytes,
        })
    }

    fn tokenize(
        &self,
        text: &str,
        add_bos: bool,
    ) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
        let mut out = Vec::new();
        if add_bos {
            out.push(self.bos);
        }
        let normalized = normalize_llama_text(text);
        let mut pos = 0_usize;
        while pos < normalized.len() {
            let max_end = normalized.len().min(pos + self.max_piece_bytes);
            let mut matched = None;
            let mut end = max_end;
            while end > pos {
                if normalized.is_char_boundary(end) {
                    if let Some(&token) = self.piece_to_id.get(&normalized[pos..end]) {
                        matched = Some((token, end));
                        break;
                    }
                }
                end -= 1;
            }
            if let Some((token, next)) = matched {
                out.push(token);
                pos = next;
                continue;
            }

            let ch = normalized[pos..]
                .chars()
                .next()
                .ok_or("tokenizer stalled")?;
            let next = pos + ch.len_utf8();
            for byte in normalized[pos..next].as_bytes() {
                let piece = format!("<0x{byte:02X}>");
                out.push(*self.piece_to_id.get(&piece).unwrap_or(&self.unk));
            }
            pos = next;
        }
        Ok(out)
    }

    fn piece(&self, token: usize) -> &str {
        self.pieces
            .get(token)
            .map(String::as_str)
            .unwrap_or("<bad-token>")
    }
}

fn normalize_llama_text(text: &str) -> String {
    let mut normalized = String::with_capacity(text.len() + 4);
    normalized.push('▁');
    for ch in text.chars() {
        if ch == ' ' {
            normalized.push('▁');
        } else {
            normalized.push(ch);
        }
    }
    normalized
}

impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> io::Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "offset overflow"))?;
        if end > self.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "short GGUF file",
            ));
        }
        let bytes = &self.data[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }

    fn u32(&mut self) -> io::Result<u32> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn u64(&mut self) -> io::Result<u64> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn string(&mut self) -> io::Result<String> {
        let len = self.u64()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))
    }

    fn string_array(&mut self) -> io::Result<Vec<String>> {
        let item_type = self.u32()?;
        if item_type != 8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("tokenizer tokens array item type must be string, got {item_type}"),
            ));
        }
        let count = self.u64()? as usize;
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(self.string()?);
        }
        Ok(values)
    }

    fn skip_value(&mut self, value_type: u32) -> io::Result<()> {
        match value_type {
            0 | 1 | 7 => self.skip(1),
            2 | 3 => self.skip(2),
            4 | 5 | 6 => self.skip(4),
            8 => {
                let len = self.u64()? as usize;
                self.skip(len)
            }
            9 => {
                let item_type = self.u32()?;
                let count = self.u64()? as usize;
                for _ in 0..count {
                    self.skip_value(item_type)?;
                }
                Ok(())
            }
            10 | 11 | 12 => self.skip(8),
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown GGUF value type {other}"),
            )),
        }
    }

    fn skip(&mut self, len: usize) -> io::Result<()> {
        self.take(len).map(|_| ())
    }
}
