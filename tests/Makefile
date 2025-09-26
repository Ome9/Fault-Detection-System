# Makefile for Refined Model Testing
# Supports both Windows (MSVC/MinGW) and Linux (GCC) environments

# Compiler settings
CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2 -g
LDFLAGS = -lm

# Directories
SRC_DIR = .
BUILD_DIR = build
DEPLOY_DIR = ../refined_deployment

# Source files
SOURCES = refined_model_test.c
OBJECTS = $(BUILD_DIR)/refined_model_test.o
TARGET = $(BUILD_DIR)/refined_model_test

# Include paths
INCLUDES = -I$(DEPLOY_DIR)

# Platform detection
ifeq ($(OS),Windows_NT)
    TARGET = $(BUILD_DIR)/refined_model_test.exe
    RM = del /Q
    MKDIR = mkdir
    PATHSEP = \\
else
    RM = rm -f
    MKDIR = mkdir -p
    PATHSEP = /
endif

# Default target
all: $(TARGET)

# Create build directory
$(BUILD_DIR):
	$(MKDIR) $(BUILD_DIR)

# Compile object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Link executable
$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

# Run tests
test: $(TARGET)
	@echo "Running refined model test suite..."
	@$(TARGET)

# Run interactive test
interactive: $(TARGET)
	@echo "Starting interactive test mode..."
	@$(TARGET) interactive

# Clean build files
clean:
	$(RM) $(BUILD_DIR)$(PATHSEP)*

# Check if model data exists
check-model:
	@echo "Checking model data files..."
	@if [ -f "$(DEPLOY_DIR)/refined_model_data.h" ]; then \
		echo "✅ Model header file found"; \
		echo "   Size: $$(wc -c < $(DEPLOY_DIR)/refined_model_data.h) bytes"; \
	else \
		echo "❌ Model header file not found at $(DEPLOY_DIR)/refined_model_data.h"; \
		exit 1; \
	fi
	@if [ -f "$(DEPLOY_DIR)/refined_model.tflite" ]; then \
		echo "✅ TensorFlow Lite model found"; \
		echo "   Size: $$(wc -c < $(DEPLOY_DIR)/refined_model.tflite) bytes"; \
	else \
		echo "❌ TensorFlow Lite model not found"; \
	fi

# Install dependencies (for development)
install-deps:
	@echo "Installing development dependencies..."
	@if command -v apt-get >/dev/null 2>&1; then \
		sudo apt-get update && sudo apt-get install -y build-essential; \
	elif command -v yum >/dev/null 2>&1; then \
		sudo yum groupinstall -y "Development Tools"; \
	elif command -v pacman >/dev/null 2>&1; then \
		sudo pacman -S base-devel; \
	else \
		echo "Please install a C compiler (gcc) manually"; \
	fi

# Performance benchmark
benchmark: $(TARGET)
	@echo "Running performance benchmark..."
	@time $(TARGET)

# Memory check (if valgrind is available)
memcheck: $(TARGET)
	@if command -v valgrind >/dev/null 2>&1; then \
		echo "Running memory check..."; \
		valgrind --leak-check=full --error-exitcode=1 $(TARGET); \
	else \
		echo "Valgrind not available, skipping memory check"; \
	fi

# Help
help:
	@echo "Refined Model Testing Makefile"
	@echo "Available targets:"
	@echo "  all         - Build the test executable"
	@echo "  test        - Run the test suite"
	@echo "  interactive - Run interactive test mode"
	@echo "  clean       - Clean build files"
	@echo "  check-model - Verify model files exist"
	@echo "  install-deps- Install development dependencies"
	@echo "  benchmark   - Run performance benchmark"
	@echo "  memcheck    - Run memory leak check (requires valgrind)"
	@echo "  help        - Show this help message"

.PHONY: all test interactive clean check-model install-deps benchmark memcheck help