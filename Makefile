BUILD_DIR   := build
CMAKE_FLAGS := -DCMAKE_BUILD_TYPE=Release

# Optional: make build CLI=OFF
ifdef CLI
    CMAKE_FLAGS += -DPARAKEET_BUILD_CLI=$(CLI)
endif

# Optional: make build SERVER=ON
ifdef SERVER
    CMAKE_FLAGS += -DPARAKEET_BUILD_SERVER_EXAMPLE=$(SERVER)
endif

# Use Ninja if available, otherwise Unix Makefiles
ifneq ($(shell which ninja 2>/dev/null),)
    GENERATOR    := Ninja
    BUILD_CMD    := ninja -C $(BUILD_DIR)
else
    GENERATOR    := "Unix Makefiles"
    NPROC        := $(shell sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)
    BUILD_CMD    := $(MAKE) -C $(BUILD_DIR) -j$(NPROC)
endif

.PHONY: all configure build run test bench bench-single debug install format format-check clean samples

all: build

configure:
	cmake -B $(BUILD_DIR) -G $(GENERATOR) $(CMAKE_FLAGS)

build: configure
	$(BUILD_CMD)

debug: CMAKE_FLAGS := -DCMAKE_BUILD_TYPE=Debug
debug: configure
	$(BUILD_CMD)

run: build
	./$(BUILD_DIR)/parakeet

test: build
	./$(BUILD_DIR)/parakeet_tests

bench: build
	./$(BUILD_DIR)/parakeet_bench $(ARGS)

bench-single: build
	./$(BUILD_DIR)/parakeet_bench $(ARGS)

install: build
	cmake --install $(BUILD_DIR)

format:
	find src include examples -name '*.cpp' -o -name '*.hpp' -o -name '*.c' | xargs clang-format -i

format-check:
	find src include examples -name '*.cpp' -o -name '*.hpp' -o -name '*.c' | xargs clang-format --dry-run --Werror

clean:
	rm -rf $(BUILD_DIR)

samples:
	@echo "Downloading samples..."
	@mkdir -p samples
	@wget --quiet --show-progress -O samples/gb0.ogg "https://upload.wikimedia.org/wikipedia/commons/2/22/George_W._Bush%27s_weekly_radio_address_%28November_1%2C_2008%29.oga"
	@wget --quiet --show-progress -O samples/gb1.ogg "https://upload.wikimedia.org/wikipedia/commons/1/1f/George_W_Bush_Columbia_FINAL.ogg"
	@wget --quiet --show-progress -O samples/hp0.ogg "https://upload.wikimedia.org/wikipedia/en/d/d4/En.henryfphillips.ogg"
	@wget --quiet --show-progress -O samples/mm1.wav "https://cdn.openai.com/whisper/draft-20220913a/micro-machines.wav"
	@wget --quiet --show-progress -O samples/a13.mp3 "https://upload.wikimedia.org/wikipedia/commons/transcoded/6/6f/Apollo13-wehaveaproblem.ogg/Apollo13-wehaveaproblem.ogg.mp3"
	@wget --quiet --show-progress -O samples/diffusion2023-07-03.flac "https://archive.org/download/diffusion2023-07-03/diffusion2023-07-03.flac"
