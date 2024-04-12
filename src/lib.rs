#![cfg_attr(
    all(feature = "nightly", any(target_arch = "x86", target_arch = "x86_64")),
    feature(stdarch_x86_avx512, avx512_target_feature)
)]

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    use pulp::x86::V3;
    use pulp::*;

    simd_type! {
        #[cfg(feature = "nightly")]
        pub struct V4PopCnt {
            avx512f: "avx512f",
            avx512vpopcntdq: "avx512vpopcntdq",
        }

        pub struct PopCnt {
            popcnt: "popcnt",
        }
    }

    #[cfg(feature = "nightly")]
    pub fn count_ones_simd_avx512(simd: V4PopCnt, bytes: &[u8]) -> u64 {
        struct Impl<'a> {
            simd: V4PopCnt,
            bytes: &'a [u8],
        }

        impl pulp::NullaryFnOnce for Impl<'_> {
            type Output = u64;

            #[inline(always)]
            fn call(self) -> Self::Output {
                let Self { simd, bytes } = self;

                let (head, body, tail) = bytemuck::pod_align_to::<u8, __m512i>(bytes);
                let (body4, body1) = pulp::as_arrays::<4, _>(body);

                let mut acc0 = simd.avx512f._mm512_setzero_si512();
                let mut acc1 = simd.avx512f._mm512_setzero_si512();
                let mut acc2 = simd.avx512f._mm512_setzero_si512();
                let mut acc3 = simd.avx512f._mm512_setzero_si512();

                let head = head.iter().map(|x| x.count_ones() as u64).sum::<u64>();

                for &[x0, x1, x2, x3] in body4 {
                    unsafe {
                        acc0 = simd.avx512f._mm512_add_epi64(acc0, _mm512_popcnt_epi64(x0));
                        acc1 = simd.avx512f._mm512_add_epi64(acc1, _mm512_popcnt_epi64(x1));
                        acc2 = simd.avx512f._mm512_add_epi64(acc2, _mm512_popcnt_epi64(x2));
                        acc3 = simd.avx512f._mm512_add_epi64(acc3, _mm512_popcnt_epi64(x3));
                    }
                }

                for &x0 in body1 {
                    unsafe {
                        acc0 = simd.avx512f._mm512_add_epi64(acc0, _mm512_popcnt_epi64(x0));
                    }
                }

                acc0 = simd.avx512f._mm512_add_epi64(acc0, acc1);
                acc2 = simd.avx512f._mm512_add_epi64(acc2, acc3);

                acc0 = simd.avx512f._mm512_add_epi64(acc0, acc2);
                let body: [u64; 8] = pulp::cast(acc0);
                let body = body.iter().sum::<u64>();

                let tail = tail.iter().map(|x| x.count_ones() as u64).sum::<u64>();

                head + body + tail
            }
        }

        simd.vectorize(Impl { simd, bytes })
    }

    pub fn count_ones_simd_avx2(simd: V3, bytes: &[u8]) -> u64 {
        struct Impl<'a> {
            simd: V3,
            bytes: &'a [u8],
        }

        impl pulp::NullaryFnOnce for Impl<'_> {
            type Output = u64;

            #[inline(always)]
            fn call(self) -> Self::Output {
                let Self { simd, bytes } = self;

                let (head_, bytes, tail_) = bytemuck::pod_align_to::<u8, __m256i>(bytes);
                let mut acc = 0u64;

                acc += head_.iter().map(|x| x.count_ones() as u64).sum::<u64>();

                let low_mask = simd.splat_u8x32(0x0F);
                let lookup = u8x32(
                    /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2, /* 4 */ 1,
                    /* 5 */ 2, /* 6 */ 2, /* 7 */ 3, /* 8 */ 1, /* 9 */ 2,
                    /* a */ 2, /* b */ 3, /* c */ 2, /* d */ 3, /* e */ 3,
                    /* f */ 4, /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
                    /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3, /* 8 */ 1,
                    /* 9 */ 2, /* a */ 2, /* b */ 3, /* c */ 2, /* d */ 3,
                    /* e */ 3, /* f */ 4,
                );

                let (head4, head1) = pulp::as_arrays::<4, _>(bytes);

                for chunk in head4.chunks(8) {
                    let mut acc0 = simd.splat_u8x32(0);
                    let mut acc1 = simd.splat_u8x32(0);
                    let mut acc2 = simd.splat_u8x32(0);
                    let mut acc3 = simd.splat_u8x32(0);

                    for &[x0, x1, x2, x3] in chunk {
                        let x0: u8x32 = pulp::cast(x0);
                        let x1: u8x32 = pulp::cast(x1);
                        let x2: u8x32 = pulp::cast(x2);
                        let x3: u8x32 = pulp::cast(x3);

                        let lo0 = simd.and_u8x32(x0, low_mask);
                        let lo1 = simd.and_u8x32(x1, low_mask);
                        let lo2 = simd.and_u8x32(x2, low_mask);
                        let lo3 = simd.and_u8x32(x3, low_mask);

                        acc0 = simd.wrapping_add_u8x32(
                            acc0,
                            pulp::cast(
                                simd.avx2
                                    ._mm256_shuffle_epi8(pulp::cast(lookup), pulp::cast(lo0)),
                            ),
                        );
                        acc1 = simd.wrapping_add_u8x32(
                            acc1,
                            pulp::cast(
                                simd.avx2
                                    ._mm256_shuffle_epi8(pulp::cast(lookup), pulp::cast(lo1)),
                            ),
                        );
                        acc2 = simd.wrapping_add_u8x32(
                            acc2,
                            pulp::cast(
                                simd.avx2
                                    ._mm256_shuffle_epi8(pulp::cast(lookup), pulp::cast(lo2)),
                            ),
                        );
                        acc3 = simd.wrapping_add_u8x32(
                            acc3,
                            pulp::cast(
                                simd.avx2
                                    ._mm256_shuffle_epi8(pulp::cast(lookup), pulp::cast(lo3)),
                            ),
                        );

                        let hi0 = simd.and_u8x32(
                            pulp::cast(simd.shr_const_u16x16::<4>(pulp::cast(x0))),
                            low_mask,
                        );
                        let hi1 = simd.and_u8x32(
                            pulp::cast(simd.shr_const_u16x16::<4>(pulp::cast(x1))),
                            low_mask,
                        );
                        let hi2 = simd.and_u8x32(
                            pulp::cast(simd.shr_const_u16x16::<4>(pulp::cast(x2))),
                            low_mask,
                        );
                        let hi3 = simd.and_u8x32(
                            pulp::cast(simd.shr_const_u16x16::<4>(pulp::cast(x3))),
                            low_mask,
                        );

                        acc0 = simd.wrapping_add_u8x32(
                            acc0,
                            pulp::cast(
                                simd.avx2
                                    ._mm256_shuffle_epi8(pulp::cast(lookup), pulp::cast(hi0)),
                            ),
                        );
                        acc1 = simd.wrapping_add_u8x32(
                            acc1,
                            pulp::cast(
                                simd.avx2
                                    ._mm256_shuffle_epi8(pulp::cast(lookup), pulp::cast(hi1)),
                            ),
                        );
                        acc2 = simd.wrapping_add_u8x32(
                            acc2,
                            pulp::cast(
                                simd.avx2
                                    ._mm256_shuffle_epi8(pulp::cast(lookup), pulp::cast(hi2)),
                            ),
                        );
                        acc3 = simd.wrapping_add_u8x32(
                            acc3,
                            pulp::cast(
                                simd.avx2
                                    ._mm256_shuffle_epi8(pulp::cast(lookup), pulp::cast(hi3)),
                            ),
                        );
                    }

                    let acc0 = simd.sum_of_absolute_differences_u8x32(acc0, simd.splat_u8x32(0));
                    let acc1 = simd.sum_of_absolute_differences_u8x32(acc1, simd.splat_u8x32(0));
                    let acc2 = simd.sum_of_absolute_differences_u8x32(acc2, simd.splat_u8x32(0));
                    let acc3 = simd.sum_of_absolute_differences_u8x32(acc3, simd.splat_u8x32(0));

                    let acc0 = simd.wrapping_add_u64x4(acc0, acc1);
                    let acc2 = simd.wrapping_add_u64x4(acc2, acc3);

                    let acc0 = simd.wrapping_add_u64x4(acc0, acc2);
                    let acc0: [u64; 4] = pulp::cast(acc0);
                    acc += acc0.iter().sum::<u64>();
                }

                let mut acc0 = simd.splat_u8x32(0);
                for &x0 in head1 {
                    let x0: u8x32 = pulp::cast(x0);
                    let lo0 = simd.and_u8x32(x0, low_mask);

                    acc0 = simd.wrapping_add_u8x32(
                        acc0,
                        pulp::cast(
                            simd.avx2
                                ._mm256_shuffle_epi8(pulp::cast(lookup), pulp::cast(lo0)),
                        ),
                    );
                    let hi0 = simd.and_u8x32(
                        pulp::cast(simd.shr_const_u16x16::<4>(pulp::cast(x0))),
                        low_mask,
                    );

                    acc0 = simd.wrapping_add_u8x32(
                        acc0,
                        pulp::cast(
                            simd.avx2
                                ._mm256_shuffle_epi8(pulp::cast(lookup), pulp::cast(hi0)),
                        ),
                    );
                }

                let acc0 = simd.sum_of_absolute_differences_u8x32(acc0, simd.splat_u8x32(0));
                let acc0: [u64; 4] = pulp::cast(acc0);
                acc += acc0.iter().sum::<u64>();

                for &x in tail_ {
                    acc += x.count_ones() as u64;
                }

                acc
            }
        }

        simd.vectorize(Impl { simd, bytes })
    }

    pub fn count_ones_popcnt(simd: PopCnt, bytes: &[u8]) -> u64 {
        struct Impl<'a> {
            simd: PopCnt,
            bytes: &'a [u8],
        }

        impl pulp::NullaryFnOnce for Impl<'_> {
            type Output = u64;

            #[inline(always)]
            fn call(self) -> Self::Output {
                let Self { simd, bytes } = self;
                _ = simd;

                let (head, body, tail) = bytemuck::pod_align_to::<u8, u64>(bytes);
                let head = head.iter().map(|x| x.count_ones() as u64).sum::<u64>();
                let body = body.iter().map(|x| x.count_ones() as u64).sum::<u64>();
                let tail = tail.iter().map(|x| x.count_ones() as u64).sum::<u64>();

                head + body + tail
            }
        }

        simd.vectorize(Impl { simd, bytes })
    }
}

/// Countes the number of bits equal to `1` in the provided slice, and returns the accumulated result.
pub fn count_ones(bytes: &[u8]) -> u64 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "nightly")]
        if let Some(simd) = x86::V4PopCnt::try_new() {
            return x86::count_ones_simd_avx512(simd, bytes);
        }

        if let Some(simd) = pulp::x86::V3::try_new() {
            return x86::count_ones_simd_avx2(simd, bytes);
        }

        if let Some(simd) = x86::PopCnt::try_new() {
            return x86::count_ones_popcnt(simd, bytes);
        }
    }

    let (head, body, tail) = bytemuck::pod_align_to::<u8, u64>(bytes);
    let head = head.iter().map(|x| x.count_ones() as u64).sum::<u64>();
    let body = body.iter().map(|x| x.count_ones() as u64).sum::<u64>();
    let tail = tail.iter().map(|x| x.count_ones() as u64).sum::<u64>();

    head + body + tail
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_count_ones() {
        let bytes = &*(0..1001).map(|_| rand::random::<u8>()).collect::<Vec<_>>();

        let target = {
            let (head, body, tail) = bytemuck::pod_align_to::<u8, u64>(bytes);
            let head = head.iter().map(|x| x.count_ones() as u64).sum::<u64>();
            let body = body.iter().map(|x| x.count_ones() as u64).sum::<u64>();
            let tail = tail.iter().map(|x| x.count_ones() as u64).sum::<u64>();

            head + body + tail
        };

        #[cfg(target_arch = "x86_64")]
        if let Some(simd) = pulp::x86::V3::try_new() {
            let actual = crate::x86::count_ones_simd_avx2(simd, bytes);
            assert_eq!(actual, target);
        }

        #[cfg(all(feature = "nightly", target_arch = "x86_64"))]
        if let Some(simd) = crate::x86::V4PopCnt::try_new() {
            let actual = crate::x86::count_ones_simd_avx512(simd, bytes);
            assert_eq!(actual, target);
        }

        let actual = crate::count_ones(bytes);
        assert_eq!(actual, target);
    }
}
