all:
	nvcc -ptx --expt-relaxed-constexpr -o vkmmc_optix_core.ptx vkmmc_optix_core.cu -I/usr/local/optix/include -I/usr/local/optix/SDK/ --use_fast_math -std=c++14
	g++ -std=c++14 -O2 -o vkmmc_optix vkmmc_optix.cpp miniz/miniz.c     -I. -Iminiz -I/usr/local/optix/include -I/usr/local/cuda/include     -L/usr/local/cuda/lib64 -lcudart -ldl -I../src -I../src/miniz
