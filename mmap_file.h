/** file reader
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

#ifndef MMAP_FILE_H
#define MMAP_FILE_H
#ifdef _MSC_VER
#define		WIN32_LEAN_AND_MEAN
#include	<windows.h>
#else
#include	<sys/mman.h>
#include	<sys/stat.h>
#include	<fcntl.h>
#include	<unistd.h>
#endif
#include	<stdexcept>
#include <stdint.h>
#include <string.h>
#include <algorithm>

#ifdef min
#undef min
#endif

/** mmaped file reader
 */
template <typename T>
struct filemap_t {
	filemap_t()
		:
#ifndef _MSC_VER
		fd_(-1),
#endif
		size_(0)
	{
	}
	~filemap_t() {
		close();
	}
	void open(const char* path) {
#ifdef _MSC_VER
		hfile_ = CreateFileA(path, GENERIC_READ, 0, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, 0);
		hmap_ = CreateFileMapping(hfile_, 0, PAGE_READONLY, 0, 0, 0);
		if (!hmap_) {
			throw std::runtime_error("file open");
		}
		buf_ = (T*)MapViewOfFile(hmap_, FILE_MAP_READ, 0, 0, 0);
		uint32_t high;
		uint32_t low = GetFileSize(hfile_, (LPDWORD)&high);
		length_ = ((uint64_t)high << 32) | low;
#else
		fd_ = ::open(path, O_RDONLY);
		struct stat s;
		if ((fd_ < 0) || (fstat(fd_, &s) < 0)) {
			throw std::runtime_error("file open");
		}
		length_ = s.st_size;
		buf_ = (T*)mmap(0, s.st_size, PROT_READ, MAP_PRIVATE, fd_, 0);
#endif
		if (!buf_) {
			throw std::runtime_error("file map");
		}
		size_ = length_ / sizeof(T);
	}
	void close() {
#ifdef _MSC_VER
		if (buf_) {
			UnmapViewOfFile((LPCVOID)buf_);
			buf_ = 0;
		}
		if (hmap_) {
			CloseHandle(hmap_);
			hmap_ = 0;
		}
		if (hfile_) {
			CloseHandle(hfile_);
			hfile_ = 0;
		}
#else
		if (buf_) {
			munmap(buf_, (size_t)length_);
			buf_ = 0;
		}
		if (0 <= fd_) {
			::close(fd_);
			fd_ = -1;
		}
#endif
	}

	const uint8_t* data(uint64_t pos = 0) const {
		return buf_ + pos;
	}

	uint64_t size() const {
		return size_;
	}

private:
	T* buf_;
	uint64_t size_;
	uint64_t length_;
#ifdef _MSC_VER
	std::string fname_;
	HANDLE hfile_;
	HANDLE hmap_;
#else
	int fd_;
#endif
};
#endif /* MMAP_FILE_H */
