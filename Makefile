# OUTPUTDIR := bin/

# CFLAGS := -std=c++14 -fvisibility=hidden -lpthread -Wall -Wextra

# # ifeq (,$(CONFIGURATION))
# # 	CONFIGURATION := seq
# # endif

# # ifeq (debug,$(CONFIGURATION))
# # CFLAGS += -g
# # else
# # CFLAGS += -O2
# # endif

# CFLAGS += -g

# HEADERS := src/*.h

# CXX = mpic++

# .SUFFIXES:
# .PHONY: all clean

# # TODO: CHANGE THESE TO BE FOR OUR CODE
# # all: nbody-$(CONFIGURATION)-v1 nbody-$(CONFIGURATION)-v2

# # nbody-$(CONFIGURATION)-v1: $(HEADERS) src/mpi-simulator-v1.cpp
# # 	$(CXX) -o $@ $(CFLAGS) src/mpi-simulator-v1.cpp

# # nbody-$(CONFIGURATION)-v2: $(HEADERS) src/mpi-simulator-v2.cpp
# # 	$(CXX) -o $@ $(CFLAGS) src/mpi-simulator-v2.cpp

# # all: gol-$(CONFIGURATION)

# # gol-$(CONFIGURATION): $(HEADERS) src/gol-sequential.cpp
# # 	$(CXX) -o $@ $(CFLAGS) src/gol-sequential.cpp

# # clean:
# # 	rm -rf ./gol-$(CONFIGURATION)*

# all: gol-seq

# gol-seq: $(HEADERS) src/gol-sequential.cpp
# 	$(CXX) -o $@ $(CFLAGS) src/gol-sequential.cpp

# clean:
# 	rm -rf ./gol-seq*

# FILES = src/*.cpp 
# 		src/*.h

EXECUTABLE := gol
LDFLAGS=-L/usr/local/cuda-11.7/lib64/ -lcudart
CU_FILES   := golCuda.cu
CU_DEPS    :=
CC_FILES   := src/main.cpp 
LOGS	   := logs
OUTPUT     := output-files

all: $(EXECUTABLE)

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')
OBJDIR=objs
CXX=g++ -m64 -std=c++17
# CXXFLAGS=-O3 -Wall -g -lglfw -lGLEW -framework OpenGL
CXXFLAGS=-O3 -Wall -g
HOSTNAME=$(shell hostname)

LIBS       :=
FRAMEWORKS :=

NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc
LIBS += GL glut cudart

LDLIBS  := $(addprefix -l, $(LIBS))
LDFRAMEWORKS := $(addprefix -framework , $(FRAMEWORKS))

NVCC=nvcc

$(OBJDIR)/%.o: src/%.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

OBJS=$(OBJDIR)/main.o 


.PHONY: dirs clean

all: $(EXECUTABLE) 

dirs:
		mkdir -p $(OBJDIR)/ $(OUTPUT)/

clean:
		rm -rf $(OBJDIR) *~ $(EXECUTABLE) $(LOGS) $(OUTPUT)

export: $(EXFILES)
	cp -p $(EXFILES) $(STARTER)


$(EXECUTABLE): clean dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)


$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
