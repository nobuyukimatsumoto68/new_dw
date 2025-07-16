CXX=g++
INCLUDES=-I"/home/nobu/opt/eigen/"

NVCC=/usr/local/cuda-12.6/bin/nvcc # nvcc
NVCCFLAGS=-arch=sm_70 -pg -O3 -std=c++20 -lcublas -lcusolver -lcusparse -lgomp -Xcompiler -fopenmp,-march=native # -diag-suppress<1650-D>

INCLUDES_CUDA=-I/usr/local/cuda-12.6/include/
LDFLAGS_CUDA=-L/usr/local/cuda-12.6/lib64/
INCLUDES_CUDA += $(INCLUDES)

SRCS := $(wildcard *.cu)
OBJS := $(SRCS:%.cu=%.o)

all: $(OBJS)

%.d: %.cu
	$(NVCC) $(INCLUDES_CUDA) -M $< -o $@

include $(SRCS:.cu=.d)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES_CUDA) $(LDFLAGS_CUDA) $< -o $@ # $(<:.d=.cu)

