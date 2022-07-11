CC = nvcc
CFLAGS = -I/$(IDIR) -Xcompiler -fopenmp -Xcompiler -O3 -lboost_program_options -gencode=arch=compute_75,code=sm_75 

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