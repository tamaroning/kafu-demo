ASSETS_DIR := assets
DEMO_MP4_URL := https://kafu-demo-20260107.s3.ap-northeast-1.amazonaws.com/P01-20240203-152323.mp4
DEMO_MP4 := $(ASSETS_DIR)/P01-20240203-152323.mp4
STORE_AISLE_MP4_URL := https://kafu-demo-20260107.s3.ap-northeast-1.amazonaws.com/store-aisle-detection.mp4
STORE_AISLE_MP4 := $(ASSETS_DIR)/store-aisle-detection.mp4
BOTTLE_DETECTION_MP4_URL := https://kafu-demo-20260107.s3.ap-northeast-1.amazonaws.com/bottle-detection.mp4
BOTTLE_DETECTION_MP4 := $(ASSETS_DIR)/bottle-detection.mp4
CLASSROOM_MP4_URL := https://kafu-demo-20260107.s3.ap-northeast-1.amazonaws.com/classroom.mp4
CLASSROOM_MP4 := $(ASSETS_DIR)/classroom.mp4
FRUIT_VEGETABLE_MP4_URL := https://kafu-demo-20260107.s3.ap-northeast-1.amazonaws.com/fruit-and-vegetable-detection.mp4
FRUIT_VEGETABLE_MP4 := $(ASSETS_DIR)/fruit-and-vegetable-detection.mp4
WORKER_ZONE_MP4_URL := https://kafu-demo-20260107.s3.ap-northeast-1.amazonaws.com/worker-zone-detection.mp4
WORKER_ZONE_MP4 := $(ASSETS_DIR)/worker-zone-detection.mp4
MOV_62C2A288_URL := https://kafu-demo-20260107.s3.ap-northeast-1.amazonaws.com/0000f77c-62c2a288.mov
MOV_62C2A288 := $(ASSETS_DIR)/0000f77c-62c2a288.mov
MOV_6257BE58_URL := https://kafu-demo-20260107.s3.ap-northeast-1.amazonaws.com/0000f77c-6257be58.mov
MOV_6257BE58 := $(ASSETS_DIR)/0000f77c-6257be58.mov
MOV_CB820C98_URL := https://kafu-demo-20260107.s3.ap-northeast-1.amazonaws.com/0000f77c-cb820c98.mov
MOV_CB820C98 := $(ASSETS_DIR)/0000f77c-cb820c98.mov

SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: all clean wasm assets

all: wasm assets

# Build WASM detector (requires kafu_clang; see demo/README.md).
wasm:
	$(MAKE) -C wasm all

assets: $(DEMO_MP4) $(STORE_AISLE_MP4) $(BOTTLE_DETECTION_MP4) $(CLASSROOM_MP4) $(FRUIT_VEGETABLE_MP4) $(WORKER_ZONE_MP4) $(MOV_62C2A288) $(MOV_6257BE58) $(MOV_CB820C98)

$(ASSETS_DIR):
	mkdir -p "$(ASSETS_DIR)"

# Generic download rule: $(call download_asset,url)
define download_asset
	@url="$(1)"; \
	out="$@"; \
	if command -v curl >/dev/null 2>&1; then \
	  curl -fL --retry 3 --retry-delay 1 -o "$$out" "$$url"; \
	elif command -v wget >/dev/null 2>&1; then \
	  wget -O "$$out" "$$url"; \
	else \
	  echo "Error: need curl or wget to download assets" >&2; \
	  exit 127; \
	fi; \
	test -s "$$out"
endef

$(DEMO_MP4): | $(ASSETS_DIR)
	$(call download_asset,$(DEMO_MP4_URL))

$(STORE_AISLE_MP4): | $(ASSETS_DIR)
	$(call download_asset,$(STORE_AISLE_MP4_URL))

$(BOTTLE_DETECTION_MP4): | $(ASSETS_DIR)
	$(call download_asset,$(BOTTLE_DETECTION_MP4_URL))

$(CLASSROOM_MP4): | $(ASSETS_DIR)
	$(call download_asset,$(CLASSROOM_MP4_URL))

$(FRUIT_VEGETABLE_MP4): | $(ASSETS_DIR)
	$(call download_asset,$(FRUIT_VEGETABLE_MP4_URL))

$(WORKER_ZONE_MP4): | $(ASSETS_DIR)
	$(call download_asset,$(WORKER_ZONE_MP4_URL))

$(MOV_62C2A288): | $(ASSETS_DIR)
	$(call download_asset,$(MOV_62C2A288_URL))

$(MOV_6257BE58): | $(ASSETS_DIR)
	$(call download_asset,$(MOV_6257BE58_URL))

$(MOV_CB820C98): | $(ASSETS_DIR)
	$(call download_asset,$(MOV_CB820C98_URL))

clean:
	$(MAKE) -C wasm clean
