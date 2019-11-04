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

#include "switches.h"
#include "Halide.h"
using namespace Halide;

#ifdef VARIABLE_GAUSSIAN
#define RAD (kernel.width() >> 1)
#define DIA kernel.width()
#else
#define RAD 5
#define DIA 11
#endif

class SsimGenerator : public Generator<SsimGenerator> {
	Var x, y;
	Var yi, yo;

	Func conv2d(Func src, Input<Buffer<float>>& kernel) {
		Func dst("filtered");
		RDom r(-RAD, DIA, -RAD, DIA);

		dst(x, y) = sum(kernel(RAD + r.x, RAD + r.y) * src(x + r.x, y + r.y));

		return dst;
	}

	Func mulconv2d(Func src1, Func src2, Input<Buffer<float>>& kernel) {
		Func mul;
		Func dst("filtered2");
		RDom r(-RAD, DIA, -RAD, DIA);

		mul(x, y) = src1(x, y) * src2(x, y);
		dst(x, y) = sum(kernel(RAD + r.x, RAD + r.y) * mul(x + r.x, y + r.y));
		return dst;
	}
public:
	Input<Buffer<uint8_t>> img1{"input1", 2};
	Input<Buffer<uint8_t>> img2{"input2", 2};
	Input<Buffer<float>> G{"kernel", 2};
	Input<Buffer<float>> output_in{"outputin", 2};
	Output<Buffer<float>> output{"output", 2};
	void generate() {
		Func mu1, mu2, mu1_sq, mu2_sq, mu1_mu2;
		Func sigma1_sq, sigma2_sq, sigma12;
		Func img1_sqm, img2_sqm, img12m;
		Func img1_ = BoundaryConditions::repeat_edge(img1);
		Func img2_ = BoundaryConditions::repeat_edge(img2);
		Func src1, src2;
		src1(x, y) = cast<float>(img1_(x, y));
		src2(x, y) = cast<float>(img2_(x, y));

		mu1 = conv2d(src1, G);
		mu2 = conv2d(src2, G);
		mu1_sq(x, y) = mu1(x, y) * mu1(x, y);
		mu2_sq(x, y) = mu2(x, y) * mu2(x, y);
		mu1_mu2(x, y) = mu1(x, y) * mu2(x, y);
		img1_sqm = mulconv2d(src1, src1, G);
		img2_sqm = mulconv2d(src2, src2, G);
		img12m = mulconv2d(src1, src2, G);
		sigma1_sq(x, y) = img1_sqm(x, y) - mu1_sq(x, y);
		sigma2_sq(x, y) = img2_sqm(x, y) - mu2_sq(x, y);
		sigma12(x, y) = img12m(x, y) - mu1_mu2(x, y);
		double k1 = 0.01;
		double k2 = 0.03;
		float C1 = static_cast<float>(std::pow(k1 * 255, 2.0));
		float C2 = static_cast<float>(std::pow(k2 * 255, 2.0));

		output(x, y) = output_in(x, y)
			+ (((2.0f * mu1_mu2(x, y) + C1) * (2.0f * sigma12(x, y) + C2))
			   /
			   ((mu1_sq(x, y) + mu2_sq(x, y) + C1) * (sigma1_sq(x, y) + sigma2_sq(x, y) + C2)));
	}

	void schedule() {
#define VEC 16
		output.split(y, yo, yi, 4).parallel(yo);
		output.vectorize(x, VEC);
	}
};

HALIDE_REGISTER_GENERATOR(SsimGenerator, ssim_halide)
