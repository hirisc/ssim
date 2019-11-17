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
	Func mu1, mu2;
	Func sigma11, sigma22, sigma12;
	Func mu1x, mu2x;
	Func sigma11x, sigma22x, sigma12x;
public:
	Input<Buffer<uint8_t>> img1{"input1", 2};
	Input<Buffer<uint8_t>> img2{"input2", 2};
	Input<Buffer<float>> G{"kernel", 1};
	Input<Buffer<float>> output_in{"outputin", 2};
	Output<Buffer<float>> output{"output", 2};
	void generate() {
		Func mu1_sq, mu2_sq, mu1_mu2;
		Func i1m1, i2m2, i1m1_sq, i2m2_sq, i1m1_i2m2;
		Func img1_ = BoundaryConditions::repeat_edge(img1);
		Func img2_ = BoundaryConditions::repeat_edge(img2);
		Func src1, src2;

		src1(x, y) = cast<float>(img1_(x, y));
		src2(x, y) = cast<float>(img2_(x, y));

		RDom r(-RAD, DIA);
		mu1x(x, y) = sum(G(RAD + r) * src1(x + r, y));
		mu2x(x, y) = sum(G(RAD + r) * src2(x + r, y));
		mu1(x, y) = sum(G(RAD + r) * mu1x(x, y + r));
		mu2(x, y) = sum(G(RAD + r) * mu2x(x, y + r));

		mu1_sq(x, y) = mu1(x, y) * mu1(x, y);
		mu2_sq(x, y) = mu2(x, y) * mu2(x, y);
		mu1_mu2(x, y) = mu1(x, y) * mu2(x, y);

		i1m1(x, y) = src1(x, y) - mu1(x, y);
		i2m2(x, y) = src2(x, y) - mu2(x, y);
		i1m1_sq(x, y) = i1m1(x, y) * i1m1(x, y);
		i2m2_sq(x, y) = i2m2(x, y) * i2m2(x, y);
		i1m1_i2m2(x, y) = i1m1(x, y) * i2m2(x, y);

		sigma11x(x, y) = sum(G(RAD + r) * i1m1_sq(x + r, y));
		sigma22x(x, y) = sum(G(RAD + r) * i2m2_sq(x + r, y));
		sigma12x(x, y) = sum(G(RAD + r) * i1m1_i2m2(x + r, y));
		sigma11(x, y) = sum(G(RAD + r) * sigma11x(x, y + r));
		sigma22(x, y) = sum(G(RAD + r) * sigma22x(x, y + r));
		sigma12(x, y) = sum(G(RAD + r) * sigma12x(x, y + r));

		double k1 = 0.01;
		double k2 = 0.03;
		float C1 = static_cast<float>(std::pow(k1 * 255, 2.0));
		float C2 = static_cast<float>(std::pow(k2 * 255, 2.0));

		output(x, y) = output_in(x, y)
			+ (((2.0f * mu1_mu2(x, y) + C1) * (2.0f * sigma12(x, y) + C2))
			   /
			   ((mu1_sq(x, y) + mu2_sq(x, y) + C1) * (sigma11(x, y) + sigma22(x, y) + C2)));
	}

	void schedule() {
#define VEC 16
		output.split(y, yo, yi, 16).parallel(yo);
		output.vectorize(x, VEC);
		mu1.split(y, yo, yi, 64).parallel(yo).vectorize(x, VEC);
		mu2.split(y, yo, yi, 64).parallel(yo).vectorize(x, VEC);
		mu1.store_root().compute_root();
		mu2.store_root().compute_root();

		mu1x.store_at(mu1, yo).compute_at(mu1, yi);
		mu2x.store_at(mu2, yo).compute_at(mu2, yi);

		sigma11x.store_at(output, yo).compute_at(output, yi).vectorize(x, VEC);
		sigma22x.store_at(output, yo).compute_at(output, yi).vectorize(x, VEC);
		sigma12x.store_at(output, yo).compute_at(output, yi).vectorize(x, VEC);
	}
};

HALIDE_REGISTER_GENERATOR(SsimGenerator, ssim_halide)
