#!/bin/sh
# PROTOTYPE (sm8x dense prefill, phase 2e; GitHub #474), throwaway: build the attention bench for
# linux/amd64 from a clean archive in /work/src (kernel/include, kernel/src, kernel/tests,
# kernel/vendor/src, kernel/vendor/include), inside nvidia/cuda:13.1.1-devel-ubuntu22.04 (glibc
# 2.35, so the binary's floor is 2.34). One translation unit: the bench includes the attention's
# source; the CUDA runtime, libstdc++ and libgcc are linked statically.
set -e
cd /work/src
ARCHS="-gencode arch=compute_86,code=[sm_86,compute_86] -gencode arch=compute_120a,code=[sm_120a,compute_120a]"
mkdir -p /work/obj
nvcc -O3 -DNDEBUG -std=c++20 $ARCHS -Ikernel/include -Ikernel/src -Ikernel/vendor/src -Ikernel/vendor/include \
  -c kernel/tests/bench_sm8x_attn.cu -o /work/obj/bench_sm8x_attn.o
g++ /work/obj/bench_sm8x_attn.o -L/usr/local/cuda/lib64 -lcudart_static -ldl -lrt -lpthread -static-libstdc++ -static-libgcc \
  -o /work/ignis_sm8x_attn_bench
ls -l /work/ignis_sm8x_attn_bench
echo "needs:"; objdump -p /work/ignis_sm8x_attn_bench | grep NEEDED
echo "glibc floor:"; objdump -T /work/ignis_sm8x_attn_bench | grep -o 'GLIBC_[0-9.]*' | sort -u -V | tail -1
echo "cubins:"; cuobjdump -lelf /work/ignis_sm8x_attn_bench
