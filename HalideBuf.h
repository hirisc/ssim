/** Wrapper of Halide::Runtime::Buffer
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

#ifndef HALIDEBUF_H
#define HALIDEBUF_H
#include "HalideBuffer.h"

template <typename T>
class HalideBuf {
	Halide::Runtime::Buffer<T> buf_;
	uint8_t* host_backup_;
	bool noswap_;
public:
	HalideBuf(int width, int height)
		: buf_(width, height), host_backup_(0), noswap_(true) {
	}

	HalideBuf(int len)
		: buf_(len), host_backup_(0), noswap_(true) {
	}

	~HalideBuf() {
		if (!noswap_) {
			std::swap(host_backup_, ((halide_buffer_t*)&buf_)->host);
		}
	}

	const Halide::Runtime::Buffer<T>& get() const {
		return buf_;
	}

	Halide::Runtime::Buffer<T>& get() {
		return buf_;
	}

	Halide::Runtime::Buffer<T>& set(const T src[]) {
		if (noswap_ && !host_backup_) {
			host_backup_ = ((halide_buffer_t*)&buf_)->host;
			noswap_ = false;
		}
		((halide_buffer_t*)&buf_)->host = (uint8_t*)src;
		return buf_;
	}
};
#endif /* HALIDEBUF_H */
