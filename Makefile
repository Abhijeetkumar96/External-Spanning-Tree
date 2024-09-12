# Makefile for CUDA project

# Compiler and flags
NVCC = nvcc
CXXFLAGS = -std=c++17 -arch=sm_80 -O3

# Source files
SRC = main.cu spanning_tree.cu eulerian_tour.cu list_ranking.cu graph.cu

# Object files
OBJ = $(SRC:.cu=.o)

# Executable name
EXEC = main

# Default target
all: CXXFLAGS += -O3
all: $(EXEC)

# Link object files to create the executable
$(EXEC): $(OBJ)
	$(NVCC) $(CXXFLAGS) -o $@ $^

# Compile .cu files to .o object files
%.o: %.cu
	$(NVCC) $(CXXFLAGS) -c $< -o $@

# Clean up object files and the executable
clean:
	rm -f $(OBJ) $(EXEC)

# Debug build target to add debugging flags
debug: CXXFLAGS += -DDEBUG
debug: $(EXEC)

# Phony targets
.PHONY: all clean
