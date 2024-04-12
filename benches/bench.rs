fn args() -> Vec<usize> {
    vec![4, 32, 256, 2048, 65536, 1048576]
}

fn count_ones_simple(bytes: &[u8]) -> u64 {
    let (head, body, tail) = bytemuck::pod_align_to::<u8, u64>(bytes);
    let head = head.iter().map(|x| x.count_ones() as u64).sum::<u64>();
    let body = body.iter().map(|x| x.count_ones() as u64).sum::<u64>();
    let tail = tail.iter().map(|x| x.count_ones() as u64).sum::<u64>();

    head + body + tail
}

#[divan::bench(args = args())]
fn bench_simple(bencher: divan::Bencher, n: usize) {
    let bytes = &*vec![41u8; n];
    bencher.bench(|| count_ones_simple(bytes))
}

#[divan::bench(args = args())]
fn bench_popcnt(bencher: divan::Bencher, n: usize) {
    let bytes = &*vec![41u8; n];
    bencher.bench(|| popcnt::count_ones(bytes))
}

fn main() {
    divan::main();
}
