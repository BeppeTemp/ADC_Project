CC = g++
CFLAGS = -I$(IDIR)

ODIR = src
IDIR = include
LDIR = libs

LIBS=-lm

_DEPS = conv_cpu.h mm_cpu.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = main.o mm_cpu.o conv_cpu.o 
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

Tensor_Bench: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~ 