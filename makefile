CC = nvcc
CFLAGS = -I/$(IDIR) -lboost_program_options -Xcompiler -fopenmp -gencode=arch=compute_75,code=sm_75 -Xcompiler -O3

ODIR = src
IDIR = include

_DEPS = matrix_op.cuh conv_op.cuh
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = main.cpp matrix_op.cu conv_op.cu
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

Tensor_Bench: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm -f src/conv/*.o
	rm -f src/mm/*.o
	rm -f src/*.o
	rm -f Tensor_Bench

test:
	@echo $(CC)
	@echo $(CFLAGS)
	@echo $(DEPS)
	@echo $(OBJ)