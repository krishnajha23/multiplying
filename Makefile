TARGET = matmul
NVCC = nvcc
SRC = main.cu cpu_reference.cu kernels.cu utils.cu benchmark.cu analysis.cu

all:
	$(NVCC) $(SRC) -lcublas -o $(TARGET)

run: all
	./$(TARGET)

clean:
	rm -f $(TARGET)