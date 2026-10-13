//! Fast word counting for BPE training, feeding PR #2501's merge loop (`BpeTrainer::do_train_entries`).
//! Workers pread chunks of the file cut at line starts, pre-tokenize every line (GPT-2 byte-level with bitcannon's
//! SIMD scanner, or the other pre-tokenizers of `crate::pretok`) and count words in bounded per-thread maps, flushed
//! into one global map split in 64 shards. The unique (word, count) entries go straight to the trainer.
use compact_str::CompactString;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicUsize, Ordering};
use crate::pretok;
use tk_train::trainers::BpeTrainer;

pub type Vocab = std::collections::HashMap<String, u32>;
pub type Merges = Vec<(String, String)>;

/// Training options. `pre_tokenizer`: a `crate::pretok` name (bytelevel, llama3, gpt4o, whitespace, ...);
/// `normalizer`: a `crate::normalize` spec ("" = none); `strip_newlines`: drop each line's "\n" / "\r\n" (as a feed
/// of `BufRead::lines()`), otherwise the "\n" is part of the line (as HF's file feed).
#[derive(Clone, Debug)]
pub struct Options {
    pub vocab_size: usize,
    pub min_frequency: u64,
    pub threads: usize,
    pub pre_tokenizer: String,
    pub normalizer: String,
    pub strip_newlines: bool,
    pub block_kb: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options { vocab_size: 30000, min_frequency: 0, threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
                  pre_tokenizer: "bytelevel".into(), normalizer: String::new(), strip_newlines: false, block_kb: 4096 }
    }
}

/// Seconds per stage and the number of unique words.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub count_s: f64,
    pub feed_s: f64,
    pub train_s: f64,
    pub words: usize,
}

const SHARDS: usize = 64;

#[derive(Clone, Copy, Default)]
struct Slot {
    h: u64,
    cl: u64,       // count in the low 56 bits, word length in the top 8 (words >= 255 bytes: 255, length in the arena)
    key: [u8; 16], // the word itself when <= 16 bytes, else its arena offset (u64 LE) and length (u64 LE)
}

const CNT: u64 = (1 << 56) - 1;

/// unique word -> count, open addressing on the word's hash; words of <= 16 bytes live in the slot (one memory access
/// per lookup), longer ones in the map's own arena
struct WordMap {
    s: Vec<Slot>,
    arena: Vec<u8>,
    n: usize,
    mask: usize,
}

impl WordMap {
    fn new() -> Self {
        WordMap { s: vec![Slot::default(); 1 << 10], arena: Vec::new(), n: 0, mask: (1 << 10) - 1 }
    }
    #[inline]
    fn word<'a>(&'a self, x: &'a Slot) -> &'a [u8] {
        let l = (x.cl >> 56) as usize;
        if l <= 16 {
            &x.key[..l]
        } else {
            let off = u64::from_le_bytes(x.key[..8].try_into().unwrap()) as usize;
            let len = u64::from_le_bytes(x.key[8..].try_into().unwrap()) as usize;
            &self.arena[off..off + len]
        }
    }
    #[inline]
    fn add(&mut self, w: &[u8], h: u64, c: u64) {
        let l8 = (w.len().min(255) as u64) << 56;
        let mut i = h as usize & self.mask;
        while self.s[i].h != 0 {
            let x = &self.s[i];
            if x.h == h && x.cl & !CNT == l8 && self.word(x) == w {
                self.s[i].cl += c;
                return;
            }
            i = (i + 1) & self.mask;
        }
        let mut key = [0u8; 16];
        if w.len() <= 16 {
            key[..w.len()].copy_from_slice(w);
        } else {
            key[..8].copy_from_slice(&(self.arena.len() as u64).to_le_bytes());
            key[8..].copy_from_slice(&(w.len() as u64).to_le_bytes());
            self.arena.extend_from_slice(w);
        }
        self.s[i] = Slot { h, cl: l8 | c, key };
        self.n += 1;
        if self.n * 2 > self.s.len() {
            let old = std::mem::take(&mut self.s);
            self.s = vec![Slot::default(); old.len() * 2];
            self.mask = self.s.len() - 1;
            for x in old {
                if x.h != 0 {
                    let mut j = x.h as usize & self.mask;
                    while self.s[j].h != 0 {
                        j = (j + 1) & self.mask;
                    }
                    self.s[j] = x;
                }
            }
        }
    }
    /// Empty the map, keeping its allocations (a worker's local map is flushed and reused).
    fn clear(&mut self) {
        self.s.fill(Slot::default());
        self.arena.clear();
        self.n = 0;
    }
    fn words(&self) -> impl Iterator<Item = (&[u8], u64, u64)> {
        self.s.iter().filter(|x| x.h != 0).map(|x| (self.word(x), x.h, x.cl & CNT))
    }
}

/// Small direct-mapped front for WordMap: words of <= 16 bytes with the key inline, counts accumulated here and
/// flushed to the big map on eviction and at the end. Frequent words never touch the big map (tens of MB, misses the
/// CPU caches). env HB_FRONT_BITS (12; 0 = off).
#[derive(Clone, Copy, Default)]
struct FSlot {
    key: u128,
    h: u64,
    cnt: u32,
    len: u32,
}

struct Front {
    s: Vec<FSlot>,
    mask: usize,
}

impl Front {
    fn new(bits: u32) -> Self {
        Front { s: vec![FSlot::default(); 1 << bits], mask: (1 << bits) - 1 }
    }
    #[inline]
    fn add(&mut self, w: &[u8], h: u64, back: &mut WordMap) {
        if w.len() > 16 || self.mask == 0 {
            back.add(w, h, 1);
            return;
        }
        let mut b = [0u8; 16];
        b[..w.len()].copy_from_slice(w);
        let key = u128::from_le_bytes(b);
        let s = &mut self.s[(h >> 24) as usize & self.mask];
        if s.h == h && s.key == key && s.len as usize == w.len() {
            s.cnt += 1;
            if s.cnt == u32::MAX {
                Self::flush(s, back);
            }
            return;
        }
        if s.h != 0 {
            Self::flush(s, back);
        }
        *s = FSlot { key, h, cnt: 1, len: w.len() as u32 };
    }
    #[inline]
    fn prefetch(&self, h: u64) {
        crate::prefetch(&self.s[(h >> 24) as usize & self.mask]);
    }
    #[inline]
    fn flush(s: &mut FSlot, back: &mut WordMap) {
        if s.h != 0 && s.cnt > 0 {
            back.add(&s.key.to_le_bytes()[..s.len as usize], s.h, s.cnt as u64);
        }
        s.cnt = 0;
        if s.h != 0 {
            s.h = 0;
        }
    }
    fn drain(&mut self, back: &mut WordMap) {
        for s in self.s.iter_mut() {
            Self::flush(s, back);
        }
    }
}

fn byte_chars() -> [char; 256] {
    // GPT-2 bytes_to_unicode
    let mut bs: Vec<u32> = (33..127).chain(161..173).chain(174..256).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut t = ['\0'; 256];
    for (b, c) in bs.iter().zip(cs) {
        t[*b as usize] = char::from_u32(c).unwrap();
    }
    t
}

/// Count the words of the file at `corpus`, then train: the vocab and merges of HF's `BpeTrainer` on the same input.
/// `before_train` runs between counting and training (the CLI hands freed memory back to the system there).
pub fn train_file(corpus: &str, o: &Options, before_train: impl FnOnce()) -> Result<(Vocab, Merges, Stats), String> {
    let t_n = o.threads.max(1);
    tk_encode::parallelism::set_num_threads(t_n);
    let block = o.block_kb.max(1) << 10;
    let kind = pretok::Kind::from_name(&o.pre_tokenizer).ok_or(format!("unknown pre-tokenizer {}", o.pre_tokenizer))?;
    let norm = crate::normalize::parse(&o.normalizer).map_err(|e| format!("normalizer: {e:?}"))?;
    let byte_level = kind.byte_level();
    let t0 = std::time::Instant::now();
    let f = File::open(corpus).map_err(|e| format!("{corpus}: {e}"))?;
    let size = f.metadata().unwrap().len() as usize;
    let cut = |x: usize, scratch: &mut Vec<u8>| -> usize {
        if x == 0 || x >= size {
            return x.min(size);
        }
        let mut from = x - 2;
        loop {
            let n = (64usize << 10).min(size - from);
            scratch.resize(n, 0);
            f.read_exact_at(scratch, from as u64).unwrap();
            for c in (x.max(from + 2))..from + n {
                if scratch[c - from - 1] == b'\n' {
                    return c;
                }
            }
            if from + n >= size {
                return size;
            }
            from += n - 2;
        }
    };
    let n_chunks = size.div_ceil(block).max(1);
    let strip_nl = o.strip_newlines;
    let next = AtomicUsize::new(0);
    // Counts: each worker counts into a local map; when it holds more than HB_LOCAL_WORDS distinct words
    // it is flushed into one global map split in SHARDS parts by hash, each behind its own lock, and reused. So a
    // word is stored once globally, plus a bounded local map per worker (memory does not grow with the workers).
    // the locals together hold at most ~512k words (65536 each at 8 workers), so memory does not grow with workers
    let local_limit = if t_n == 1 { usize::MAX } else { (1usize << 19) / t_n };
    let global: Vec<std::sync::Mutex<WordMap>> = (0..SHARDS).map(|_| std::sync::Mutex::new(WordMap::new())).collect();
    let flush = |m: &mut WordMap, by: &mut Vec<Vec<u32>>| {
        by.iter_mut().for_each(Vec::clear);
        for (i, x) in m.s.iter().enumerate() {
            if x.h != 0 {
                by[(x.h >> 58) as usize % SHARDS].push(i as u32);
            }
        }
        for (k, idx) in by.iter().enumerate() {
            if idx.is_empty() {
                continue;
            }
            let mut g = global[k].lock().unwrap();
            for &i in idx {
                let x = &m.s[i as usize];
                g.add(m.word(x), x.h, x.cl & CNT);
            }
        }
        m.clear();
    };
    std::thread::scope(|sc| {
        let hs: Vec<_> = (0..t_n)
            .map(|_| {
                let (next, cut, f, norm, flush) = (&next, &cut, &f, &norm, &flush);
                sc.spawn(move || {
                    let (mut data, mut scratch) = (Vec::new(), Vec::new());
                    let mut m = WordMap::new();
                    let mut by: Vec<Vec<u32>> = (0..SHARDS).map(|_| Vec::new()).collect();
                    let mut front = Front::new(0);
                    let (mut tags, mut starts, mut flag, mut spans) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                    let (mut nbuf, mut wbuf) = (String::new(), Vec::new());
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n_chunks {
                            front.drain(&mut m);
                            flush(&mut m, &mut by);
                            break;
                        }
                        if m.n > local_limit {
                            flush(&mut m, &mut by);
                        }
                        let (lo, hi) = (cut(i * block, &mut scratch), cut((i + 1) * block, &mut scratch));
                        data.resize(hi - lo, 0);
                        f.read_exact_at(&mut data, lo as u64).unwrap();
                        // per line (with its '\n'), as HF's feed and ours: a pre-token never spans a line end.
                        // GPT-2: HF's bitcannon (SIMD classify + bitstream GPT-2 grammar) finds the token starts
                        for line in data.split_inclusive(|&c| c == b'\n') {
                            // invalid UTF-8 (e.g. a file cut mid-character) becomes U+FFFD, as HF's feed reads it
                            let lossy;
                            let line: &[u8] = match std::str::from_utf8(line) {
                                Ok(_) => line,
                                Err(_) => {
                                    lossy = String::from_utf8_lossy(line).into_owned();
                                    lossy.as_bytes()
                                }
                            };
                            // strip_newlines: drop the line's "\n" / "\r\n", as a feed of `BufRead::lines()` does
                            let line = if strip_nl { line.strip_suffix(b"\n").map_or(line, |l| l.strip_suffix(b"\r").unwrap_or(l)) } else { line };
                            let line: &[u8] = if norm.is_empty() {
                                line
                            } else {
                                crate::normalize::apply(&norm, std::str::from_utf8(line).unwrap(), &mut nbuf);
                                nbuf.as_bytes()
                            };
                            if kind == pretok::Kind::Gpt2 {
                                let n = line.len();
                                tags.resize(n.max(tags.len()), 0);
                                starts.resize(n.div_ceil(64).max(starts.len()), 0);
                                flag.resize(n.div_ceil(64).max(flag.len()), 0);
                                spans.resize(n.max(spans.len()), bitcannon::Span { start: 0, end: 0 });
                                bitcannon::classify::classify(line, &mut tags[..n]);
                                let k = bitcannon::bitcannon_byte_level(line, &tags[..n], &mut starts, &mut flag, &mut spans);
                                for sp in &spans[..k] {
                                    let w = &line[sp.start as usize..sp.end as usize];
                                    front.add(w, crate::hbytes(w) | 1, &mut m);
                                }
                                continue;
                            }
                            pretok::for_each_word(kind, line, &mut wbuf, |w| {
                                if !w.is_empty() {
                                    front.add(w, crate::hbytes(w) | 1, &mut m);
                                }
                            });
                        }
                    }
                })
            })
            .collect();
        hs.into_iter().for_each(|h| h.join().unwrap());
    });
    let t_count = t0.elapsed().as_secs_f64();
    // each global shard is turned into (word, count) entries on its own thread, mapping byte-level words to chars
    let map = byte_chars();
    let parts: Vec<Vec<(CompactString, u64)>> = std::thread::scope(|sc| {
        let hs: Vec<_> = global
            .into_iter()
            .map(|g| {
                let map = &map;
                sc.spawn(move || {
                    let all = g.into_inner().unwrap();
                    all.words()
                        .map(|(w, _, c)| {
                            let k: CompactString = if byte_level { w.iter().map(|&b| map[b as usize]).collect() } else { std::str::from_utf8(w).unwrap().into() };
                            (k, c)
                        })
                        .collect()
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    // unique (word, count) entries straight to the trainer, no map of them (PR #2501's own feed does the same)
    let mut wc: Vec<(CompactString, u64)> = Vec::with_capacity(parts.iter().map(|p| p.len()).sum());
    for p in parts {
        wc.extend(p);
    }
    let n_words = wc.len();
    before_train();
    let feed_s = t0.elapsed().as_secs_f64();
    let t1 = std::time::Instant::now();
    // non-byte-level pre-tokenizers train with an [UNK] special token, as train_hf.py / train_ours.py
    let specials = if byte_level { vec![] } else { vec![tk_encode::vocab::bucket_added_vocabulary::AddedToken::from("[UNK]", true)] };
    let trainer = BpeTrainer::builder().vocab_size(o.vocab_size).min_frequency(o.min_frequency).special_tokens(specials).show_progress(false).build();
    let (vocab, merges, _) = trainer.do_train_entries(wc).map_err(|e| e.to_string())?;
    let train_s = t1.elapsed().as_secs_f64();
    let vocab: Vocab = vocab.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    let merges: Merges = merges.into_iter().map(|(l, r)| (l.to_string(), r.to_string())).collect();
    Ok((vocab, merges, Stats { count_s: t_count, feed_s, train_s, words: n_words }))
}
