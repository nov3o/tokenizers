# tokenizers (toki fork)

> **This is a fork**, not the official [huggingface/tokenizers](https://github.com/huggingface/tokenizers).
> - **Training** is based on Yikai Liao's parallel BPE trainer,
>   [huggingface/tokenizers#2501](https://github.com/huggingface/tokenizers/pull/2501) (head `e4f787dc`), with
>   [toki](https://github.com/nov3o/toki)'s word counting in front of it.
> - **Encoding** is HF tokenizers `main` at [`ff8ae1ba`](https://github.com/huggingface/tokenizers/commit/ff8ae1baabca0707b7fa9f22854652e231a933de)
>   (the 1.0 rewrite), with one patch so a single long input is encoded on every thread.
>
> Everything else is upstream, unchanged. Results are the same as upstream, bit for bit: every merge, the vocab,
> every token id.

## Install

```bash
pip install "git+https://github.com/nov3o/tokenizers#subdirectory=bindings/python"
```

Builds from source, so it needs a Rust toolchain ([rustup](https://rustup.rs)).

## Use

```python
import tokenizers

# train: a text file, one document or paragraph per line
vocab, merges = tokenizers.train_bpe("corpus.txt", vocab_size=131072, pre_tokenizer="bytelevel", threads=8)

# encode: the upstream API; one long text is now split across all threads
tok = tokenizers.Tokenizer.from_file("tokenizer.json")
ids = tok.encode(open("big.txt").read()).ids
```

`train_bpe` gives the same vocab and merges as HF's `BpeTrainer` fed the file line by line. Pre-tokenizers:
bytelevel, llama3, gpt4o, whitespace, whitespace_split, metaspace, bert, punctuation, digits, unicode_scripts,
char_delimiter. `normalizers`: a list from nfc, nfd, nfkc, nfkd, lowercase, strip_accents. Also `min_frequency`
and `strip_newlines`.

From Rust, the [`tokenizers/toki`](tokenizers/toki) crate has `toki::train::train_file` and two command-line tools:
`toki-train <corpus> <vocab_size> <out_dir>` and `toki-encode <tokenizer.json> <corpus> <pretok> <ids_out>`
(streaming, every thread reads, encodes and writes its own chunk).

## What changed

| commit | what |
|---|---|
| Encode one long input on every thread | `tk-encode`: `main` splits a long input across threads only at special tokens; this also cuts long text before a `\n` between ASCII letters/digits when the pre-tokenizer always ends a word there (GPT regexes, whitespace, BERT) and the normalizers work char by char |
| `BpeTrainer::do_train_entries` | `tk-train`: train from `(word, count)` entries (6 lines) |
| toki | `tokenizers/toki`: parallel chunked reads, bounded per-thread word maps flushed into a sharded map, entries straight into PR #2501's trainer; the `toki-train` / `toki-encode` tools |
| Python: `train_bpe` | `bindings/python`: binds `toki::train::train_file` |

## Results

Default setting: 1 GB, 8 threads, GPT-2 byte-level, vocab 131 072, no normalizer, 2× AMD EPYC 7742, pinned cores.
Each table swaps one axis; average of 4 datasets (english, multilingual, code, mix). Training vs PR #2501, encoding
vs HF `main` (on a held-out half). Time = median of 2 runs, MEM = peak GB. Setup, data, per-dataset tables and the
scripts: [nov3o/toki](https://github.com/nov3o/toki).

Equivalence: 214 training runs of ours, 214 with merges identical to PR #2501's; 212 encoding runs, 212 with ids identical to HF main's (token by token).

### Default setting

| dataset | PR #2501 train s | bytes/s | MEM | **ours** train s | bytes/s | MEM | HF main enc s | bytes/s | MEM | **ours** enc s | bytes/s | MEM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| english | 5.21 | 192 MB/s | 0.37 | **2.48** (2.1×) | **402 MB/s** | **0.31** | 1.80 | 278 MB/s | 0.69 | **0.99** (1.8×) | **503 MB/s** | **0.31** |
| multilingual | 16.2 | 62 MB/s | 4.04 | **13.5** (1.2×) | **74 MB/s** | **3.98** | 2.99 | 167 MB/s | 0.77 | **1.83** (1.6×) | **272 MB/s** | **0.37** |
| code | 9.34 | 107 MB/s | 1.19 | **4.71** (2.0×) | **212 MB/s** | **0.87** | 2.33 | 215 MB/s | 0.87 | **1.23** (1.9×) | **407 MB/s** | **0.37** |
| mix | 11.1 | 90 MB/s | 2.19 | **7.88** (1.4×) | **127 MB/s** | **2.14** | 2.75 | 182 MB/s | 0.80 | **1.46** (1.9×) | **342 MB/s** | **0.36** |
| **average** | 10.5 | 95 MB/s | 1.95 | **7.15** (1.5×) | **140 MB/s** | **1.83** | 2.47 | 203 MB/s | 0.78 | **1.38** (1.8×) | **362 MB/s** | **0.35** |

### Threads

| threads | PR #2501 train s | bytes/s | MEM | **ours** train s | bytes/s | MEM | HF main enc s | bytes/s | MEM | **ours** enc s | bytes/s | MEM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 27.9 | 36 MB/s | 1.84 | **17.3** (1.6×) | **58 MB/s** | **1.84** | 4.47 | 112 MB/s | 0.73 | **3.71** (1.2×) | **135 MB/s** | **0.15** |
| 4 | 15.7 | 64 MB/s | 1.87 | **10.5** (1.5×) | **96 MB/s** | **1.86** | 3.12 | 160 MB/s | 0.74 | **2.20** (1.4×) | **227 MB/s** | **0.21** |
| 8 | 10.5 | 95 MB/s | 1.95 | **7.15** (1.5×) | **140 MB/s** | **1.83** | 2.47 | 203 MB/s | 0.78 | **1.38** (1.8×) | **362 MB/s** | **0.35** |
| 16 | 8.44 | 118 MB/s | 2.17 | **5.43** (1.6×) | **184 MB/s** | **1.92** | 2.04 | 245 MB/s | 0.84 | **0.99** (2.1×) | **507 MB/s** | **0.60** |

### Vocab

| vocab | PR #2501 train s | bytes/s | MEM | **ours** train s | bytes/s | MEM | HF main enc s | bytes/s | MEM | **ours** enc s | bytes/s | MEM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32k | 9.22 | 108 MB/s | 1.58 | **5.99** (1.5×) | **167 MB/s** | **1.60** | 2.03 | 247 MB/s | 0.74 | **1.10** (1.8×) | **456 MB/s** | **0.34** |
| 64k | 9.78 | 102 MB/s | 1.76 | **6.57** (1.5×) | **152 MB/s** | **1.71** | 2.09 | 239 MB/s | 0.78 | **1.17** (1.8×) | **427 MB/s** | **0.34** |
| 128k | 10.5 | 95 MB/s | 1.95 | **7.15** (1.5×) | **140 MB/s** | **1.83** | 2.47 | 203 MB/s | 0.78 | **1.38** (1.8×) | **362 MB/s** | **0.35** |
| 256k | 11.2 | 89 MB/s | 2.15 | **7.93** (1.4×) | **126 MB/s** | **1.98** | 3.21 | 156 MB/s | 0.80 | **1.86** (1.7×) | **268 MB/s** | **0.37** |

### Pre-tokenizer

| pre-tokenizer | PR #2501 train s | bytes/s | MEM | **ours** train s | bytes/s | MEM | HF main enc s | bytes/s | MEM | **ours** enc s | bytes/s | MEM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bytelevel | 10.5 | 95 MB/s | 1.95 | **7.15** (1.5×) | **140 MB/s** | **1.83** | 2.47 | 203 MB/s | 0.78 | **1.38** (1.8×) | **362 MB/s** | **0.35** |
| llama3 | 10.8 | 93 MB/s | 2.18 | **7.45** (1.4×) | **134 MB/s** | **2.06** | 2.39 | 209 MB/s | 0.78 | **1.41** (1.7×) | **355 MB/s** | **0.34** |
| gpt4o | 10.6 | 95 MB/s | 2.02 | **7.54** (1.4×) | **133 MB/s** | **1.88** | 2.44 | 205 MB/s | 0.77 | **1.43** (1.7×) | **349 MB/s** | **0.34** |
| whitespace | 8.56 | 117 MB/s | 1.44 | **5.34** (1.6×) | **187 MB/s** | **1.28** | 2.16 | 231 MB/s | 0.72 | **1.25** (1.7×) | **401 MB/s** | **0.33** |
| whitespace_split | 12.2 | 82 MB/s | 2.71 | **8.98** (1.4×) | **111 MB/s** | **2.55** | 2.68 | 186 MB/s | 0.72 | **1.79** (1.5×) | **279 MB/s** | **0.36** |
| metaspace | 13.8 | 73 MB/s | 3.19 | **10.5** (1.3×) | **95 MB/s** | **3.09** | 3.59 | 139 MB/s | 0.84 | **2.69** (1.3×) | **186 MB/s** | **0.57** |
| bert | 8.16 | 123 MB/s | 1.53 | **7.66** (1.1×) | **131 MB/s** | **1.33** | 2.17 | 230 MB/s | 0.73 | **1.20** (1.8×) | **417 MB/s** | **0.34** |
| punctuation | 30.6 | 33 MB/s | 8.05 | **30.6** (1.0×) | **33 MB/s** | **7.68** | 5.09 | 98 MB/s | 0.76 | **4.13** (1.2×) | **121 MB/s** | **0.36** |
| digits | 37.8 | 26 MB/s | 9.72 | **34.3** (1.1×) | **29 MB/s** | **9.38** | 6.51 | 77 MB/s | 0.77 | **5.69** (1.1×) | **88 MB/s** | **0.42** |
| unicode_scripts | 29.3 | 34 MB/s | 7.90 | **29.9** (1.0×) | **33 MB/s** | **7.64** | 4.83 | 104 MB/s | 0.73 | **3.95** (1.2×) | **127 MB/s** | **0.26** |
| char_delimiter | 12.6 | 80 MB/s | 3.12 | **9.42** (1.3×) | **106 MB/s** | **2.98** | 2.92 | 171 MB/s | 0.74 | **2.03** (1.4×) | **246 MB/s** | **0.38** |

### Normalizer

| normalizer | PR #2501 train s | bytes/s | MEM | **ours** train s | bytes/s | MEM | HF main enc s | bytes/s | MEM | **ours** enc s | bytes/s | MEM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 10.5 | 95 MB/s | 1.95 | **7.15** (1.5×) | **140 MB/s** | **1.83** | 2.47 | 203 MB/s | 0.78 | **1.38** (1.8×) | **362 MB/s** | **0.35** |
| nfc | 10.6 | 94 MB/s | 1.93 | **10.5** (1.0×) | **95 MB/s** | **1.83** | 2.53 | 197 MB/s | 0.78 | **1.95** (1.3×) | **256 MB/s** | **0.36** |
| nfd | 10.8 | 92 MB/s | 1.74 | **8.90** (1.2×) | **112 MB/s** | **1.65** | 2.99 | 167 MB/s | 0.79 | **2.02** (1.5×) | **247 MB/s** | **0.41** |
| nfkc | 10.9 | 91 MB/s | 1.95 | **10.3** (1.1×) | **97 MB/s** | **1.80** | 3.41 | 147 MB/s | 0.78 | **2.39** (1.4×) | **209 MB/s** | **0.37** |
| nfkd | 11.0 | 91 MB/s | 1.75 | **8.91** (1.2×) | **112 MB/s** | **1.65** | 3.02 | 165 MB/s | 0.79 | **2.00** (1.5×) | **250 MB/s** | **0.41** |
| lowercase | 10.7 | 93 MB/s | 1.82 | **7.74** (1.4×) | **129 MB/s** | **1.72** | 2.65 | 189 MB/s | 0.78 | **1.62** (1.6×) | **308 MB/s** | **0.39** |
| strip_accents | 11.3 | 88 MB/s | 1.95 | **9.46** (1.2×) | **106 MB/s** | **1.80** | 3.20 | 156 MB/s | 0.79 | **2.20** (1.5×) | **227 MB/s** | **0.41** |
| nfkc_lowercase | 11.2 | 89 MB/s | 1.82 | **10.5** (1.1×) | **95 MB/s** | **1.68** | 3.63 | 138 MB/s | 0.78 | **2.63** (1.4×) | **190 MB/s** | **0.37** |
| replace | 10.6 | 95 MB/s | 1.95 | **7.66** (1.4×) | **131 MB/s** | **1.83** | 2.50 | 200 MB/s | 0.79 | **1.41** (1.8×) | **355 MB/s** | **0.37** |
| bert | 13.7 | 73 MB/s | 0.82 | **10.5** (1.3×) | **96 MB/s** | **0.66** | 5.11 | 98 MB/s | 0.78 | **4.08** (1.3×) | **123 MB/s** | **0.32** |

### Size

| size | PR #2501 train s | bytes/s | MEM | **ours** train s | bytes/s | MEM | HF main enc s | bytes/s | MEM | **ours** enc s | bytes/s | MEM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 GB | 10.5 | 95 MB/s | 1.95 | **7.15** (1.5×) | **140 MB/s** | **1.83** | 2.47 | 203 MB/s | 0.78 | **1.38** (1.8×) | **362 MB/s** | **0.35** |
| 10 GB | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 | 🕐 |

## History

I started by implementing models and ML algorithms from scratch. At the same time I got very excited about
pre-training, partly through my project [chsm8](https://huggingface.co/LegumMagister/chsm8): seeing which data leads
to which result made me read a lot about pre-training and its best practices. After a couple of models I moved on to
tokenizers, to figure out how they work.

I wrote my own BPE in C++ without looking at how it is done in practice; my competitive programming background helped
me work out how I would implement it. My implementation (not BPE itself) turned out a bit different, and with an
agent I made it more parallel: a 10-20× speedup for training and 10-33× for encoding.

When I was about to open a PR, I found [Yikai](https://github.com/Yikai-Liao)'s [PR](https://github.com/huggingface/tokenizers/pull/2501) and realized
I had compared against a year-old release: HF [`main`](https://github.com/huggingface/tokenizers) had moved on, Yikai's trainer is much faster than mine, and
`main`'s encoder is a bit faster than mine. Since we got our speedups in different ways, combining them should beat
both. So with an agent I built (HF `main`, mine) for encoding and (PR #2501, mine) for training.

I'm really glad [HF tokenizers](https://github.com/huggingface/tokenizers) made such a leap in just one year: compared with the [0.22 release](https://github.com/huggingface/tokenizers/releases/tag/v0.22.0), `main` encodes 12-20×
faster with about the same memory and trains 1.1-2.4× faster, and Yikai's PR #2501 trains 8-20× faster with 4-5× less
memory. Combined with mine, 10 GB is now encoded in seconds, not minutes.

In the beginning I could follow; by the end I no longer understood the changes going on:
[this](https://www.youtube.com/watch?v=W_dxteeedgs).
