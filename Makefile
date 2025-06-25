ARCH ?= gcc

# -----------------------------------------------------------------------------
#  Environment
# -----------------------------------------------------------------------------
ifeq ($(ARCH), gcc)
  CC        = gcc
  MPICC     = mpicc
  CFLAGS    = -g -Wall -Wextra -Wpedantic -std=gnu99
  LDFLAGS   = -lm
  MPICFLAGS = -g $(CFLAGS)
  MPILDFLAGS= $(LDFLAGS)
  OPENMP_FLAG = -fopenmp
  OPENCL_LDFLAG = -lOpenCL
  ifneq (,$(DEBUG))
    CFLAGS += -g -DDEBUG
    MPICFLAGS += -g -DDEBUG
    LDFLAGS += -g
    MPILDFLAGS += -g
  else
    CFLAGS += -Ofast
    MPICFLAGS += -Ofast
  endif
endif

# -----------------------------------------------------------------------------
# Build config
# -----------------------------------------------------------------------------
gridlength    = 128
threads       = 4
SHELL         := /bin/bash
HOSTNAME      = $(shell hostname)
ARCHNAME      = $(shell uname -s)_$(shell uname -m)
MPIRUN        = mpirun
SRCDIR        = src
TESTDIR       = test
TESTDIR_PERF  = perf-test
TESTDIR_PERF_OUT = perf-test-out/$(HOSTNAME)
SCRIPTDIR     = helper-scripts

ifneq (,$(DESTDIR))
  OBJDIR = $(DESTDIR)/$(ARCHNAME)/obj
  BINDIR = $(DESTDIR)/$(ARCHNAME)/bin
  OUTDIR = $(DESTDIR)/$(ARCHNAME)/output
else
  OBJDIR = $(ARCHNAME)/obj
  BINDIR = $(ARCHNAME)/bin
  OUTDIR = $(ARCHNAME)/output
endif

OBJ = 
BIN = 
DATA = 
OBJ_TESTS = 
BIN_TESTS =

# -----------------------------------------------------------------------------
#  Macros
# -----------------------------------------------------------------------------
define DEF_BIN
  OBJ += $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(2))
  BIN += $(1)

  $(1): $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(2))
	$(CC) $$^ $(LDFLAGS) $(3) -o $$@

  DATA += $(4) $(4).time
  $(4): $(1)
	mkdir -p $(OUTDIR)
	/usr/bin/time -f "Tiempo: %e s" -o $$@.time \
	$$< -n $(gridlength) -o $$@
endef

define DEF_BIN_MPI
  OBJ += $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(2))
  BIN += $(1)

  $(1): $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(2))
	$(MPICC) $$^ $(MPILDFLAGS) $(3) -o $$@

  DATA += $(4) $(4).time
  $(4): $(1)
	mkdir -p $(OUTDIR)
	/usr/bin/time -f "Tiempo: %e s" -o $$@.time \
	$(MPIRUN) -n $(threads) $$< -n $(gridlength) -o $$@
endef

define DEF_TEST
  OBJ_TESTS += $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(2))
  BIN_TESTS += $(1)

  $(1): $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(2))
	$(CC) $$^ $(LDFLAGS) -lcunit $(3) -o $$@
endef

# -----------------------------------------------------------------------------
#  Targets
# -----------------------------------------------------------------------------
all: $(OBJDIR) $(BINDIR) bin

$(OBJDIR) $(BINDIR) $(OUTDIR):
	mkdir -p $@

# Serial reference
yee_SRC = $(SRCDIR)/yee.c $(SRCDIR)/yee_common.c
$(eval $(call DEF_BIN,$(BINDIR)/yee,$(yee_SRC),,$(OUTDIR)/yee.tsv))

# Pthreads
yee_block_pthr_SRC = $(SRCDIR)/yee_block_pthr.c $(SRCDIR)/yee_common.c
$(eval $(call DEF_BIN,$(BINDIR)/yee_block_pthr,$(yee_block_pthr_SRC),-pthread,$(OUTDIR)/yee_block_pthr.tsv))
yee_stride1_pthr_SRC = $(SRCDIR)/yee_stride1_pthr.c $(SRCDIR)/yee_common.c
$(eval $(call DEF_BIN,$(BINDIR)/yee_stride1_pthr,$(yee_stride1_pthr_SRC),-pthread,$(OUTDIR)/yee_stride1_pthr.tsv))

# OpenMP
yee_naive_omp_SRC = $(SRCDIR)/yee_naive_omp.c $(SRCDIR)/yee_common.c
$(eval $(call DEF_BIN,$(BINDIR)/yee_naive_omp,$(yee_naive_omp_SRC),$(OPENMP_FLAG),$(OUTDIR)/yee_naive_omp.tsv))
yee_block_omp_SRC = $(SRCDIR)/yee_block_omp.c $(SRCDIR)/yee_common.c
$(eval $(call DEF_BIN,$(BINDIR)/yee_block_omp,$(yee_block_omp_SRC),$(OPENMP_FLAG),$(OUTDIR)/yee_block_omp.tsv))
yee_stride1_omp_SRC = $(SRCDIR)/yee_stride1_omp.c $(SRCDIR)/yee_common.c
$(eval $(call DEF_BIN,$(BINDIR)/yee_stride1_omp,$(yee_stride1_omp_SRC),$(OPENMP_FLAG),$(OUTDIR)/yee_stride1_omp.tsv))

# OpenCL
yee_opencl_SRC = $(SRCDIR)/yee_opencl.c $(SRCDIR)/yee_common.c
$(eval $(call DEF_BIN,$(BINDIR)/yee_opencl,$(yee_opencl_SRC),$(OPENCL_LDFLAG),$(OUTDIR)/yee_opencl.tsv))

# MPI
yee_blocking_mpi_SRC = $(SRCDIR)/yee_blocking_mpi.c $(SRCDIR)/yee_common.c $(SRCDIR)/yee_common_mpi.c
$(eval $(call DEF_BIN_MPI,$(BINDIR)/yee_blocking_mpi,$(yee_blocking_mpi_SRC),,$(OUTDIR)/yee_blocking_mpi.tsv))
yee_nonblock_mpi_SRC = $(SRCDIR)/yee_nonblock_mpi.c $(SRCDIR)/yee_common.c $(SRCDIR)/yee_common_mpi.c
$(eval $(call DEF_BIN_MPI,$(BINDIR)/yee_nonblock_mpi,$(yee_nonblock_mpi_SRC),,$(OUTDIR)/yee_nonblock_mpi.tsv))

# Unit tests
unittests_SRC = $(SRCDIR)/yee_common_tests.c $(SRCDIR)/yee_common.c
$(eval $(call DEF_TEST,$(BINDIR)/unit_tests,$(unittests_SRC),))

.PHONY: clean test test-scaling test-perf data bin

# --- CORRECCIÓN ---
# La regla 'bin' ahora también depende de que los directorios se creen,
# igual que la regla 'all'. Esto asegura que 'make bin' funcione
# incluso después de un 'make clean'.
bin: $(OBJDIR) $(BINDIR) $(BIN)

data: all $(DATA)
test: $(BIN_TESTS)
	./$(BINDIR)/unit_tests
# ... (Aquí irían las demás reglas de testeo que borré por brevedad)

# -----------------------------------------------------------------------------
# Compilation rules <-- SECCIÓN MEJORADA Y ORGANIZADA
# -----------------------------------------------------------------------------

# Regla específica para los archivos de OpenMP
# Esta regla se asegura de que -fopenmp se use al crear los archivos .o
$(OBJDIR)/yee_%_omp.o: $(SRCDIR)/yee_%_omp.c
	$(CC) $(CFLAGS) $(OPENMP_FLAG) -c $< -o $@

# Regla específica para los archivos fuente de MPI
$(OBJDIR)/yee_%_mpi.o: $(SRCDIR)/yee_%_mpi.c
	$(MPICC) $(MPICFLAGS) -c $< -o $@

# Regla específica para yee_common_mpi.c
$(OBJDIR)/yee_common_mpi.o: $(SRCDIR)/yee_common_mpi.c
	$(MPICC) $(MPICFLAGS) -c $< -o $@

# Regla genérica para todos los demás archivos .c (debe ir al final)
$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# -----------------------------------------------------------------------------
# Clean rule
# -----------------------------------------------------------------------------
clean:
	rm -rf $(OBJDIR) $(BINDIR) $(OUTDIR) $(TESTDIR_PERF_OUT)