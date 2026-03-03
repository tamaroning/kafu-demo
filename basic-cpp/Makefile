CC = $(KAFU_SDK_PATH)/libexec/kafu_clang

all: main.wasm

main.wasm: main.cpp
	$(CC) $< -o $@ -lstdc++

clean:
	rm -f *.wasm

.PHONY: all clean
