# Build demo WASM: frames in -> detections (NDJSON) to stdout only.
# Requires kafu_clang and repo include/ (wasi_nn, kafu.h).

TOP := ../..
SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c

CFLAGS := -I$(TOP)/include

.PHONY: all clean

all: detect-kafu-main.wasm

detect-kafu-main.wasm: detect_objects.c
	kafu clang -- $(CFLAGS) -O3 $< -o $@

clean:
	rm -f *.wasm *.wat
