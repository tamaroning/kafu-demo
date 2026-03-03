CC = $(KAFU_SDK_PATH)/libexec/kafu_clang

all: kafu-main.wasm

kafu-main.wasm: kafu-main.c
	$(CC) $< -o $@ -O3

clean:
	rm -f *.wasm

.PHONY: all clean
