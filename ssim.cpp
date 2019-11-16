/** SSIM using Halide
 *  Copyright 2019 Takayuki Minegishi
 *
 *  Permission is hereby granted, free of charge, to any person
 *  obtaining a copy of this software and associated documentation
 *  files (the "Software"), to deal in the Software without
 *  restriction, including without limitation the rights to use, copy,
 *  modify, merge, publish, distribute, sublicense, and/or sell copies
 *  of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *  
 *  The above copyright notice and this permission notice shall be
 *  included in all copies or substantial portions of the Software.
 *  
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 *  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 *  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 *  NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 *  HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 *  WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *  DEALINGS IN THE SOFTWARE.
 */

#include <vector>
#include <algorithm>

#define HALIDE

#if defined(_M_IX86) || defined(_M_AMD64)
#include <crtdbg.h>
#define VC_CHECK (assert(_CrtCheckMemory()))
#else
#define VC_CHECK
#endif

#ifdef _MSC_VER
#define _CRT_ALIGN(x) __declspec(align(x))
#define __attribute__(x)
#else
#ifndef _CRT_ALIGN
#define _CRT_ALIGN(x) __attribute__((aligned(x)))
#endif
#endif

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <cmath>
#include <numeric>
#include <string.h>
#include "switches.h"
#include "ssim_halide.h"
#include "mmap_file.h"
#include "getopt.h"
#ifdef HALIDE
#include "HalideBuf.h"
#endif

#ifdef VARIABLE_GAUSSIAN
#define RAD rad
#else
#define RAD 5
#endif

template <typename T>
static void extend_frame(const T src[], T dst[], int width, int height, int gap) {
	int dst_stride = width + gap * 2;
	int src_stride = width;
	T* dst_inner = dst + dst_stride * gap + gap;
	for (int y = 0; y < height; ++y) {
		std::copy(src, src + width, dst_inner);
		std::fill(dst_inner - gap, dst_inner, src[0]);
		std::fill(dst_inner + width, dst_inner + width + gap, src[width - 1]);
		dst_inner += dst_stride;
		src += src_stride;
	}
	const T* top = dst + dst_stride * gap;
	const T* bottom = dst + (dst_stride * (height + gap - 1));
	T* tail = dst + (dst_stride * (height + gap));
	for (int i = 0; i < gap; ++i) {
		std::copy(bottom, bottom + dst_stride, tail + dst_stride * i);
		std::copy(top, top + dst_stride, dst + dst_stride * i);
	}
}

template <typename T>
static void build_gaussian(T knl[], int radius, double sigma) {
	int dia = radius * 2 + 1;
	int dia2 = dia * dia;
	std::vector<double> not_normalized(dia2);
	double sum = 0.0;
	double* k = &not_normalized[0];
	for (int ry = -radius; ry <= radius; ++ry) {
		double s = 0.0;
		for (int rx = -radius; rx <= radius; ++rx) {
			double e = std::exp(-(rx * rx + ry * ry) / (2.0 * sigma * sigma));
			s += e;
			*k++ = e;
		}
		sum += s;
	}
	for (int i = 0; i < dia2; ++i) {
		knl[i] = static_cast<T>(not_normalized[i] / sum);
	}
}

typedef double real_t;

template <typename T>
static void conv2d(const T src[], const real_t knl[], real_t dst[], int width, int height, int rad) {
	static std::vector<T> src_extended((width + RAD * 2) * (height + RAD * 2));
	extend_frame(&src[0], &src_extended[0], width, height, RAD);
	int stride = width + RAD * 2;
	const T* src_ex = &src_extended[stride * RAD + RAD];
#pragma omp parallel for
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			real_t sum = static_cast<real_t>(0.0);
			const real_t* k = &knl[0];
			for (int ry = -RAD; ry <= RAD; ++ry) {
				real_t t = static_cast<real_t>(0.0);
				for (int rx = -RAD; rx <= RAD; ++rx) {
					t += *k++ * src_ex[(y + ry) * stride + x + rx];
				}
				sum += t;
			}
			dst[y * width + x] = sum;
		}
	}
}

template <typename T>
static void mul2d(const T src0[], const T src2[], real_t dst[], int width, int height) {
	int len = width * height;
	for (int i = 0; i < len; ++i) {
		dst[i] = static_cast<real_t>(src0[i]) * src2[i];
	}
}

template <typename T0, typename T2>
static void sub2d(const T0 src0[], const T2 src2[], real_t dst[], int width, int height) {
	int len = width * height;
	for (int i = 0; i < len; ++i) {
		dst[i] = static_cast<real_t>(src0[i]) - src2[i];
	}
}

void ssimFrame(const uint8_t img1[], const uint8_t img2[], const real_t knl[], real_t dstsum[], int width, int height, int rad) {
	static std::vector<real_t> mu1(width * height);
	static std::vector<real_t> mu2(width * height);
	static std::vector<real_t> mu1_sq(width * height);
	static std::vector<real_t> mu2_sq(width * height);
	static std::vector<real_t> mu1_mu2(width * height);

	static std::vector<real_t> i1m1(width * height);
	static std::vector<real_t> i2m2(width * height);
	static std::vector<real_t> i1m1_sq(width * height);
	static std::vector<real_t> i2m2_sq(width * height);
	static std::vector<real_t> i1m1_i2m2(width * height);

	static std::vector<real_t> sigma1_sq(width * height);
	static std::vector<real_t> sigma2_sq(width * height);
	static std::vector<real_t> sigma12(width * height);
	conv2d(img1, knl, &mu1[0], width, height, rad);
	conv2d(img2, knl, &mu2[0], width, height, rad);
	mul2d(&mu1[0], &mu1[0], &mu1_sq[0], width, height);
	mul2d(&mu2[0], &mu2[0], &mu2_sq[0], width, height);
	mul2d(&mu1[0], &mu2[0], &mu1_mu2[0], width, height);

	sub2d(&img1[0], &mu1[0], &i1m1[0], width, height);
	sub2d(&img2[0], &mu2[0], &i2m2[0], width, height);
	mul2d(&i1m1[0], &i1m1[0], &i1m1_sq[0], width, height);
	mul2d(&i2m2[0], &i2m2[0], &i2m2_sq[0], width, height);
	mul2d(&i1m1[0], &i2m2[0], &i1m1_i2m2[0], width, height);
	conv2d(&i1m1_sq[0], knl, &sigma1_sq[0], width, height, rad);
	conv2d(&i2m2_sq[0], knl, &sigma2_sq[0], width, height, rad);
	conv2d(&i1m1_i2m2[0], knl, &sigma12[0], width, height, rad);

	real_t k1 = 0.01;
	real_t k2 = 0.03;
	int bitdepth = 8;
	int L = (1 << bitdepth) - 1;
	real_t C1 = std::pow(k1 * L, 2.0);
	real_t C2 = std::pow(k2 * L, 2.0);
	int i = 0;
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x, ++i) {
			real_t d0 = ((2.0 * mu1_mu2[i] + C1) * (2.0 * sigma12[i] + C2)) / ((mu1_sq[i] + mu2_sq[i] + C1) * (sigma1_sq[i] + sigma2_sq[i] + C2));
			dstsum[i] += d0;
		}
	}
}

typedef enum {
	YUV420,
	YUV422,
	YUV444,
	YUV400
} format_t;

int format_dimension(format_t fmt) {
	if (fmt == YUV400) {
		return 1;
	} else {
		return 3;
	}
}

int format_usize(format_t fmt, int len, bool is_width) {
	switch (fmt) {
	case YUV420:
		len = len >> 1;
		break;
	case YUV422:
		len = is_width ? (len >> 1) : len;
		break;
	case YUV444:
		break;
	default:
		return 0;
	}
	return len;
}

template <typename PIX_T>
class SsimCalc {
#ifdef HALIDE
	HalideBuf<uint8_t> srcbuf_;
	HalideBuf<uint8_t> srcbufb_;
	HalideBuf<uint8_t> src2buf_;
	HalideBuf<uint8_t> src2bufb_;
	HalideBuf<float> gaussianbuf_;
	HalideBuf<float> dstbuf_y_;
	HalideBuf<float> dstbuf_u_;
	HalideBuf<float> dstbuf_v_;
	std::vector<std::vector<float> > ssim_sum_;
#else
	std::vector<std::vector<real_t> > ssim_sum_;
	std::vector<real_t> gaussian_;
#endif
	const PIX_T* img0_;
	const PIX_T* img2_;
	int width_, height_;
	int width_u_, height_u_;

	int ysize_, usize_, yuvsize_;
	size_t total_pixels_;
	size_t frames_;
	size_t processed_frames_;

	format_t fmt_;
	int radius_;
	real_t stddev_;

public:
	SsimCalc(const filemap_t<PIX_T>& file0, const filemap_t<PIX_T>& file2, int width, int height, format_t fmt, int rad, double stddev)
		: img0_(static_cast<const PIX_T*>(file0.data())),
		  img2_(static_cast<const PIX_T*>(file2.data())),
		  width_(width), height_(height),
		  width_u_(format_usize(fmt, width, true)),
		  height_u_(format_usize(fmt, height, false)),

		  ysize_(width_ * height_),
		  usize_(format_usize(fmt, width, true) * format_usize(fmt, height, false)),
		  yuvsize_(ysize_ + usize_ * 2),
		  total_pixels_(std::min(file0.size(), file2.size()) / sizeof(PIX_T)),
		  frames_(total_pixels_ / yuvsize_),
		  ssim_sum_(format_dimension(fmt)),

		  fmt_(fmt),
		  radius_(RAD), stddev_(stddev),
#ifdef HALIDE
		  srcbuf_(width, height),
		  srcbufb_(format_usize(fmt, width, true), format_usize(fmt, height, false)),
		  src2buf_(width, height),
		  src2bufb_(format_usize(fmt, width, true), format_usize(fmt, height, false)),
		  gaussianbuf_((RAD * 2 + 1), (RAD * 2 + 1)),
		  dstbuf_y_(width, height),
		  dstbuf_u_(format_usize(fmt, width, true), format_usize(fmt, height, false)),
		  dstbuf_v_(format_usize(fmt, width, true), format_usize(fmt, height, false))
#else
		  gaussian_((RAD * 2 + 1) * (RAD * 2 + 1))
#endif
	{
		ssim_sum_[0].resize(ysize_);
#ifdef HALIDE
		dstbuf_y_.set(&ssim_sum_[0][0]);
#endif
		if (1 < ssim_sum_.size()) {
			ssim_sum_[1].resize(usize_);
			ssim_sum_[2].resize(usize_);
#ifdef HALIDE
			dstbuf_u_.set(&ssim_sum_[1][0]);
			dstbuf_v_.set(&ssim_sum_[2][0]);
#endif
		}
#ifdef HALIDE
		build_gaussian(gaussianbuf_.get().data(), RAD, stddev);
#else
		build_gaussian(&gaussian_[0], RAD, stddev);
#endif
		zeroclear_sum();
	}

	void zeroclear_sum() {
		processed_frames_ = 0;
		for (size_t i = 0; i < ssim_sum_.size(); ++i) {
			std::fill(ssim_sum_[i].begin(), ssim_sum_[i].end(), 0.0);
		}
	}

	void calc_frame(size_t pos) {
		processed_frames_ += 1;
#ifdef HALIDE
		ssim_halide(srcbuf_.set(&img0_[pos * yuvsize_]), src2buf_.set(&img2_[pos * yuvsize_]), gaussianbuf_.get(), dstbuf_y_.get(), dstbuf_y_.get());
#else
		ssimFrame(&img0_[pos * yuvsize_], &img2_[pos * yuvsize_], &gaussian_[0], &ssim_sum_[0][0], width_, height_, radius_);
#endif
		if (ssim_sum_.size() == 1) {
			return;
		}
#ifdef HALIDE
		ssim_halide(srcbufb_.set(&img0_[pos * yuvsize_ + ysize_]), src2bufb_.set(&img2_[pos * yuvsize_ + ysize_]), gaussianbuf_.get(), dstbuf_u_.get(), dstbuf_u_.get());

		ssim_halide(srcbufb_.set(&img0_[pos * yuvsize_ + ysize_ + usize_]), src2bufb_.set(&img2_[pos * yuvsize_ + ysize_ + usize_]), gaussianbuf_.get(), dstbuf_v_.get(), dstbuf_v_.get());
#else
		ssimFrame(&img0_[pos * yuvsize_ + ysize_], &img2_[pos * yuvsize_ + ysize_], &gaussian_[0], &ssim_sum_[1][0], width_u_, height_u_, radius_);
		ssimFrame(&img0_[pos * yuvsize_ + ysize_ + usize_], &img2_[pos * yuvsize_ + ysize_ + usize_], &gaussian_[0], &ssim_sum_[2][0], width_u_, height_u_, radius_);
#endif
	}

	size_t frames() const {
		return frames_;
	}

	real_t result(int idx_bitmap) const {
		real_t sum = 0.0;
		size_t frame_size = 0;
		for (size_t i = 0; i < ssim_sum_.size(); ++i) {
			if (idx_bitmap & (1 << i)) {
				sum += std::accumulate(ssim_sum_[i].begin(), ssim_sum_[i].end(), 0.0);
				frame_size += (i == 0) ? ysize_ : usize_;
			}
		}
		return sum / (frame_size * processed_frames_);
	}
};

static int readint(const char* arg, const char* what) {
	char* endp;
	long val = strtol(arg, &endp, 0);
	if (!arg || *endp || (val == 0) || (INT_MAX < val)) {
		throw std::runtime_error(what);
	}
	return static_cast<int>(val);
}

static double readf(const char* arg, const char* what) {
	char* endp;
	double val = strtod(arg, &endp);
	if (!arg || *endp || (val <= 0.0)) {
		throw std::runtime_error(what);
	}
	return val;
}

class Options {
	static void BlameUser(const char* what = 0) {
		if (what) {
			fprintf(stderr, "Error %s\n", what);
		}
		fprintf(stderr,
			"Usage:\n"
			"ssim [options] <width> <height> <infile1> <infile2>\n"
			"options:\n"
			"\t-f yuv format\n"
			"\t\t0: 420(default)\n"
			"\t\t0: 422\n"
			"\t\t0: 444\n"
			"\t\t0: 400\n"
			"\t-r radius of gaussian (default: 5)\n"
#ifndef VARIABLE_GAUSSIAN
			"\t(-r are to be ignored, always 5)\n"
#endif
			"\t-s stddev of gaussian (default: 1.5)\n"
			);
		exit(1);
	}

public:
	Options()
		: format_(0), radius_(5), stddev_(1.5) {
	}

	void read(int argc, char* argv[]) {
		try {
			int opt;
			while ((opt = getopt(argc, argv, "f:r:s:")) != -1) {
				switch (opt) {
				case 'f':
					format_ = readint(optarg, "format");
					break;
				case 'r':
					radius_ = readint(optarg, "radius");
					break;
				case 's':
					stddev_ = readf(optarg, "stddev");
					break;
				default:
					BlameUser("invalid option");
					/* NOTREACHED */
				}
			}
			if (argc - optind < 4) {
				BlameUser();
				/* NOTREACHED */
			}

			width_ = readint(argv[optind++], "width");
			height_ = readint(argv[optind++], "height");
			fi0_.open(argv[optind++]);
			fi2_.open(argv[optind++]);
		} catch (std::runtime_error& e) {
			BlameUser(e.what());
		}
	}
	std::string infile1_, infile2_;
	int width_;
	int height_;
	int format_;
	int radius_;
	double stddev_;
	filemap_t<uint8_t> fi0_, fi2_;
};

int main(int argc, char **argv) {
#if defined(_MSC_VER) && !defined(NDEBUG)
	_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_WNDW);
//	_CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_WNDW);
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF|_CRTDBG_LEAK_CHECK_DF);
	atexit((void (*)(void))_CrtCheckMemory);
#endif
	Options opt;
	opt.read(argc, argv);

	if (opt.fi0_.size() != opt.fi2_.size()) {
		fprintf(stderr, "Length of Two Files Differ (%ld != %ld)\n", opt.fi0_.size(), opt.fi2_.size());
	}

	SsimCalc<uint8_t> ssim(opt.fi0_, opt.fi2_, opt.width_, opt.height_, static_cast<format_t>(opt.format_), opt.radius_, opt.stddev_);
	size_t frames = ssim.frames();
	for (size_t pos = 0; pos < frames; ++pos) {
		ssim.calc_frame(pos);
	}
	printf(
		"Y\tUV\tAll\n"
		"%.12e\t%.12e\t%.12e\n",
		ssim.result(1), ssim.result(6), ssim.result(7));

	return 0;
}
