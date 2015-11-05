#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "macros.h"
#include "celt_lpc.h"
#include "stack_alloc.h"
#include "mathops.h"
#include "pitch.h"

#if defined(OPUS_X86_MAY_HAVE_AVX) && !defined(FIXED_POINT)

#include <immintrin.h>
#include "arch.h"

void xcorr_kernel_avx(const opus_val16 *x, const opus_val16 *y, opus_val32 sum[4], int len) {
	int j;
	__m256 xsum1, xsum2;

	xsum1 = _mm256_setzero_ps();
	xsum2 = _mm256_setzero_ps();

	int avx_len = (len / 8) * 8;
	int remainder = len - avx_len;

	for (j = 0; j < avx_len; j += 8) {
		__m256 x0 = _mm256_loadu_ps(x + j);
		__m256 yj = _mm256_loadu_ps(y + j);
		__m256 y3 = _mm256_loadu_ps(y + j + 3);

		xsum1 = _mm256_add_ps(xsum1, _mm256_mul_ps(_mm256_shuffle_ps(x0, x0, 0x00), yj));
		xsum2 = _mm256_add_ps(xsum2,_mm256_mul_ps(_mm256_shuffle_ps(x0, x0, 0x55),
				_mm256_shuffle_ps(yj, y3, 0x49)));
		xsum1 = _mm256_add_ps(xsum1, _mm256_mul_ps(_mm256_shuffle_ps(x0, x0, 0xaa),
				_mm256_shuffle_ps(yj, y3, 0x9e)));
		xsum2 = _mm256_add_ps(xsum2, _mm256_mul_ps(_mm256_shuffle_ps(x0, x0, 0xff), y3));
	}

	xsum1 = _mm256_add_ps(xsum1, xsum2);

	if (remainder > 0) {
		x += avx_len;
		y += avx_len;

		opus_val16 y_0, y_1, y_2, y_3, tmp;
		y_3 = 0;
		y_0 = *y++;
		y_1 = *y++;
		y_2 = *y++;

		for (; j < len; j++) {
			int type = j & 0b11;
			switch(type) {
			case 0:
				tmp = *x++;
				y_3 = *y++;
				sum[0] = MAC16_16(sum[0], tmp, y_0);
				sum[1] = MAC16_16(sum[1], tmp, y_1);
				sum[2] = MAC16_16(sum[2], tmp, y_2);
				sum[3] = MAC16_16(sum[3], tmp, y_3);
				break;
			case 1:
				tmp = *x++;
				y_0 = *y++;
				sum[0] = MAC16_16(sum[0],tmp, y_1);
				sum[1] = MAC16_16(sum[1],tmp, y_2);
				sum[2] = MAC16_16(sum[2],tmp, y_3);
				sum[3] = MAC16_16(sum[3],tmp, y_0);
				break;
			case 2:
				tmp = *x++;
				y_1 = *y++;
				sum[0] = MAC16_16(sum[0], tmp, y_2);
				sum[1] = MAC16_16(sum[1], tmp, y_3);
				sum[2] = MAC16_16(sum[2], tmp ,y_0);
				sum[3] = MAC16_16(sum[3], tmp, y_1);
				break;
			case 3:
				tmp = *x++;
				y_2 = *y++;
				sum[0] = MAC16_16(sum[0], tmp, y_3);
				sum[1] = MAC16_16(sum[1], tmp, y_0);
				sum[2] = MAC16_16(sum[2], tmp, y_1);
				sum[3] = MAC16_16(sum[3], tmp, y_2);
				break;
			default:
				break;
			}
		}
	}

	xsum2 = _mm256_broadcast_ps((__m128 const *)sum);
	xsum1 = _mm256_add_ps(xsum1, _mm256_permute2f128_ps(xsum1, xsum1, 0x1));

	xsum1 = _mm256_add_ps(xsum1, xsum2);
	_mm_maskstore_ps(sum, _mm_set1_epi32(0x80000000), _mm256_extractf128_ps(xsum1, 0x0));
}

void dual_inner_prod_avx(const opus_val16 *x, const opus_val16 *y01, const opus_val16 *y02,
		int N, opus_val32 *xy1, opus_val32 *xy2) {
	int i;
	__m256 xsum1, xsum2;
	xsum1 = _mm256_setzero_ps();
	xsum2 = _mm256_setzero_ps();
	int avx_len = (N / 8) * 8;
	float xy[4];

	for (i = 0; i < avx_len; i += 8) {
		__m256 xi = _mm256_loadu_ps(x + i);
		__m256 y1i = _mm256_loadu_ps(y01 + i);
		__m256 y2i = _mm256_loadu_ps(y02 + i);
		xsum1 = _mm256_add_ps(xsum1, _mm256_mul_ps(xi, y1i));
		xsum2 = _mm256_add_ps(xsum2, _mm256_mul_ps(xi, y2i));
	}

	xsum1 = _mm256_hadd_ps(xsum1, xsum2);
	xsum1 = _mm256_hadd_ps(xsum1, xsum1);
	xsum1 = _mm256_add_ps(xsum1, _mm256_permute2f128_ps(xsum1, xsum1, 0x1));
	_mm_maskstore_ps(xy, _mm_set1_epi32(0x80000000), _mm256_extractf128_ps(xsum1, 0x0));

	*xy1 = xy[0];
	*xy2 = xy[1];

	for (; i < N; i++) {
		*xy1 = MAC16_16(*xy1, x[i], y01[i]);
		*xy2 = MAC16_16(*xy2, x[i], y02[i]);
	}
}

float celt_inner_prod_avx(const opus_val16 *x, const opus_val16 *y, int N) {
	int i;
	float xy;
	__m256 sum;
	sum = _mm256_setzero_ps();
	int avx_len = (N / 8) * 8;

	for (i = 0; i < avx_len; i += 8) {
		__m256 xi = _mm256_loadu_ps(x + i);
		__m256 yi = _mm256_loadu_ps(y + i);
		sum = _mm256_add_ps(sum, _mm256_mul_ps(xi, yi));
	}

	/* Horizontal sum */
	sum = _mm256_add_ps(sum, _mm256_permute2f128_ps(sum, sum, 0x1));
	sum = _mm256_add_ps(sum, _mm256_shuffle_ps(sum, sum, 0xB1));
	sum = _mm256_add_ps(sum, _mm256_shuffle_ps(sum, sum, 0x4B));
	_mm_store_ss(&xy, _mm256_extractf128_ps(sum, 0x0));

	for (; i < N; i++) {
		xy = MAC16_16(xy, x[i], y[i]);
	}

	return xy;
}

void comb_filter_const_avx(opus_val32 *y, opus_val32 *x, int T, int N,
		opus_val16 g10, opus_val16 g11, opus_val16 g12) {
	int i;
	__m256 x0v;
	__m256 g10v, g11v, g12v;
	int avx_len = (N / 8) * 8;

	g10v = _mm256_broadcast_ss(&g10);
	g11v = _mm256_broadcast_ss(&g11);
	g12v = _mm256_broadcast_ss(&g12);

	x0v = _mm256_loadu_ps(&x[-T - 2]);

	for (i = 0; i < avx_len; i += 8) {
		__m256 yi, yi2, x1v, x2v, x3v, x4v;
		const opus_val32 *xp = &x[i - T - 2];
		yi = _mm256_loadu_ps(x + i);
		x4v = _mm256_loadu_ps(xp + 4);
#if 0
		x1v = _mm256_loadu_ps(xp + 1);
		x2v = _mm256_loadu_ps(xp + 2);
		x3v = _mm256_loadu_ps(xp + 3);
#else
		x2v = _mm256_shuffle_ps(x0v, x4v, 0x4e);
		x1v = _mm256_shuffle_ps(x0v, x2v, 0x99);
		x3v = _mm256_shuffle_ps(x2v, x4v, 0x99);
#endif
		yi = _mm256_add_ps(yi, _mm256_mul_ps(g10v, x2v));
#if 0
		yi = _mm256_add_ps(yi, _mm256_mul_ps(g11v, _mm256_add_ps(x3v, x1v)));
		yi = _mm256_add_ps(yi, _mm256_mul_ps(g12v, _mm256_add_ps(x4v, x0v)));
#else
		yi2 = _mm256_add_ps(_mm256_mul_ps(g11v,_mm256_add_ps(x3v, x1v)),
				_mm256_mul_ps(g12v,_mm256_add_ps(x4v, x0v)));
		yi = _mm256_add_ps(yi, yi2);
#endif
		x0v = _mm256_loadu_ps(xp + 8);
		_mm256_storeu_ps(y + i, yi);
	}

	for (; i < N; i++) {
		y[i] = x[i]
		         + MULT16_32_Q15(g10, x[i-T])
		         + MULT16_32_Q15(g11, ADD32(x[i-T+1], x[i-T-1]))
		         + MULT16_32_Q15(g12, ADD32(x[i-T+2], x[i-T-2]));
	}
}

#endif
