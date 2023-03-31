all: neon_test run


CROSS_COMPILE ?= /data/home/yifan.bai/code/test/arm-toolchain/arm-gnu-toolchain-12.2.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-
CC=$(CROSS_COMPILE)gcc
CXXFLAGS= -march=armv8.6-a+crypto -static -g -Og

neon_test: neon_test.c
	${CC} $^ -o $@ ${CXXFLAGS}

run:
	-./qemu-aarch64 -cpu max neon_test

clean:
	-rm neon_test