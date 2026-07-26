# tinysory

`tinysory` is a tiny, model-specific K3 runner for `story.gguf`.

This is intentionally not a generic `llama.cpp` replacement.  It is a narrow
debug tool that dumps stable checkpoints while submitting operators through
`build_channel -> submit_graph -> wait_graph_complete`.

Current chunk:

- parses GGUF metadata and tensor table
- loads the fixed `story.gguf` 6-layer TinyLlama layout
- recomputes the full prefix through K3 graph ops up to final logits
- samples and prints the next token id for each `-n/--n-predict` step
- writes `ggml.txt` JSON lines for layer checkpoints and final logits
- accepts `--tokens`, `--prompt`, `--temp`, and `-n`
- `--tokens` is exact: the first token is whatever id you pass
- `--prompt` uses the GGUF llama tokenizer and prepends BOS
- interactive mode can start empty and accept either token ids or text

Host compile check:

```sh
cd /home/inkbottle/othersrc/k3x-HERA-RT/k3_test/llama/tinysory
cargo check --target x86_64-unknown-linux-gnu
```

Starry/rootfs build and push:

```sh
cd /home/inkbottle/othersrc/k3x-HERA-RT/k3_test/llama/tinysory
./scripts/push-debugfs.sh
```

Run on Starry:

```sh
cd /root
./tinysory --model story.gguf --prompt "Once upon" -n 16 --temp 0 --dump ggml.txt
```

Exact token-id entry:

```sh
cd /root
./tinysory --model story.gguf --tokens 1,9038,2501 -n 16 --temp 0 --dump ggml.txt
```

Interactive token/text entry:

```sh
cd /root
./tinysory --model story.gguf -n 16 --temp 0 --interactive --dump ggml.txt
```
