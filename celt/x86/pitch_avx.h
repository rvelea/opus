// Copyright 2015 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the Chromium LICENSE file.

#ifndef PITCH_AVX_H
#define PITCH_AVX_H

#if defined(HAVE_CONFIG_H)
#include "config.h"
#endif

#include "pitch_sse.h"

#if defined(OPUS_X86_PRESUME_AVX) && !defined(FIXED_POINT)

void xcorr_kernel_avx(
		const opus_val16 *x,
		const opus_val16 *y,
		opus_val32       sum[4],
		int              len);
#ifndef OVERRIDE_XCORR_KERNEL
#define OVERRIDE_XCORR_KERNEL
#endif
#undef xcorr_kernel
#define xcorr_kernel(x, y, sum, len, arch) \
    ((void)arch, xcorr_kernel_avx(x, y, sum, len))

opus_val32 celt_inner_prod_avx(
		const opus_val16 *x,
		const opus_val16 *y,
		int               N);
#ifndef OVERRIDE_CELT_INNER_PROD
#define OVERRIDE_CELT_INNER_PROD
#endif
#undef celt_inner_prod
#define celt_inner_prod(x, y, N, arch) \
	((void)arch, celt_inner_prod_avx(x, y, N))

#ifndef OVERRIDE_DUAL_INNER_PROD
#define OVERRIDE_DUAL_INNER_PROD
#endif
#undef dual_inner_prod
void dual_inner_prod_avx(const opus_val16 *x,
		const opus_val16 *y01,
		const opus_val16 *y02,
		int               N,
		opus_val32       *xy1,
		opus_val32       *xy2);
# define dual_inner_prod(x, y01, y02, N, xy1, xy2, arch) \
    ((void)(arch),dual_inner_prod_avx(x, y01, y02, N, xy1, xy2))

#ifndef OVERRIDE_COMB_FILTER_CONST
#define OVERRIDE_COMB_FILTER_CONST
#endif
#undef comb_filter_const
void comb_filter_const_avx(opus_val32 *y,
		opus_val32 *x,
		int         T,
		int         N,
		opus_val16  g10,
		opus_val16  g11,
		opus_val16  g12);
#define comb_filter_const(y, x, T, N, g10, g11, g12, arch) \
    ((void)(arch),comb_filter_const_avx(y, x, T, N, g10, g11, g12))

#endif // defined(OPUS_X86_PRESUME_AVX) && !defined(FIXED_POINT)

#endif // PITCH_AVX_H
