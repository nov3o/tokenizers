//! toki-train <corpus> <vocab_size> <out_dir>: `toki::train::train_file`, writes out_dir/merges.txt ("l r" per line)
//! and out_dir/vocab.txt ("id\ttoken" per line). env: HB_THREADS (all cores), HB_PRETOK (bytelevel), HB_NORMALIZER
//! (a toki::normalize spec, "" = none), HB_MIN_FREQ (0), HB_STRIP_NL (0), HB_BLOCK_KB (4096).
//! Run with MIMALLOC_PURGE_DELAY=0, so memory freed after counting is returned before training peaks.
use std::io::Write;
use toki::env_or;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 4 {
        eprintln!("usage: toki-train <corpus> <vocab_size> <out_dir>");
        std::process::exit(2);
    }
    let (corpus, out) = (&a[1], &a[3]);
    let d = toki::train::Options::default();
    let o = toki::train::Options {
        vocab_size: a[2].parse().expect("vocab_size"),
        min_frequency: env_or("HB_MIN_FREQ", 0u64),
        threads: env_or("HB_THREADS", d.threads),
        pre_tokenizer: std::env::var("HB_PRETOK").unwrap_or(d.pre_tokenizer),
        normalizer: std::env::var("HB_NORMALIZER").unwrap_or_default(),
        strip_newlines: env_or("HB_STRIP_NL", 0usize) == 1,
        block_kb: env_or("HB_BLOCK_KB", d.block_kb),
    };
    std::fs::create_dir_all(out).unwrap();
    let (vocab, merges, st) =
        toki::train::train_file(corpus, &o, || unsafe { libmimalloc_sys::mi_collect(true) }).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1)
        });
    let mut f = std::io::BufWriter::new(std::fs::File::create(format!("{out}/merges.txt")).unwrap());
    for (l, r) in &merges {
        writeln!(f, "{l} {r}").unwrap();
    }
    let mut v: Vec<_> = vocab.iter().collect();
    v.sort_by_key(|x| *x.1);
    let mut g = std::io::BufWriter::new(std::fs::File::create(format!("{out}/vocab.txt")).unwrap());
    for (t, i) in v {
        writeln!(g, "{i}\t{t}").unwrap();
    }
    println!("count_s={:.3} feed_s={:.3}", st.count_s, st.feed_s);
    println!("train_s={:.3} words={}", st.train_s, st.words);
    println!("peak_rss_mb: {:.0}", toki::peak_rss_mb());
}
