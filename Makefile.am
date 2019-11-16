EXTRA_DIST = ssim.sln ssim.vcxproj gen_ssim.vcxproj
# INCLUDES = -I$(includedir) -I../lib
# DEFS = $(DEFDISP) -D'_ANSI_ARGS_(x)=x'
# DEFS = -DUSE_ZLIB
noinst_PROGRAMS = gen_ssim ssim
gen_ssim_SOURCES = gen_ssim.cpp ../halide/tools/GenGen.cpp
gen_ssim_CPPFLAGS = -std=c++11 -fno-rtti -I../halide/include -UVERSION
gen_ssim_LDADD = -L../halide/lib -lHalide -lz -lpthread -ldl
ssim_SOURCES = ssim.cpp ssim_halide.h mmap_file.h HalideBuf.h getopt.h my_getopt.h my_getopt.c switches.h
ssim_CPPFLAGS = -std=c++11 -I../halide/include -UVERSION -fopenmp
ssim_LDADD = -L../halide/lib ssim_halide.a -lz -lpthread -ldl -lomp

ssim_halide.a: gen_ssim
	./gen_ssim -g ssim_halide -o . target=host

ssim_halide.h: gen_ssim
	./gen_ssim -g ssim_halide -o . target=host

