//do stuff for global constants up top


#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <string>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <vector>
#include "cube.h"
#include "fileLoader.h"
#include "golCuda.h"

#define BLOCK_SIZE 256

struct GlobalConstants {
    int sideLength;
    bool isMoore;
    int numStates;
    bool* ruleset;
    int* inputData;
    int* outputData;
};

__constant__ GlobalConstants cuConstIterationParams;

__global__ void kernelDoIterationMoore() {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int n = *(int*)&cuConstIterationParams.sideLength;
    if (index < 0 || index >= n * n * n) {
        return;
    }
    
    int z = (index / (n * n)) % n;
    int y = (index / n) % n;
    int x = index % n;
    int linIndex;
    int numAlive = 0;
    // printf("doin index %d, x: %d, y: %d, z: %d\n", index, x, y, z);
    for (int i = x - 1; i <= x + 1; i++) {
        for (int j = y - 1; j <= y + 1; j++) {
            for (int k = z - 1; k <= z + 1; k++) {
                if (i >= 0 && j >= 0 && k >= 0 && i < n && j < n && k < n) {
                    if (!(x == i && y == j && z == k)) { //don't include self
                        linIndex = (k * n * n) + (j * n) + i;
                        numAlive += *(int*)&cuConstIterationParams.inputData[linIndex] ? 1 : 0;
                    }
                }
            }
        }
    }
    // printf("x: %d, y: %d, z: %d has value: %d \n", x, y, z, *(int*)&cuConstIterationParams.inputData[index]);
    if (*(int*)&cuConstIterationParams.inputData[index]) {
        // printf("x: %d, y: %d, z: %d is alive rn with %d neighbors \n", x, y, z, numAlive);
        *(int*)(&cuConstIterationParams.outputData[index]) = (*(bool*)(&cuConstIterationParams.ruleset[27 + numAlive])) ? 1 : 0;
    } else {
        // printf("x: %d, y: %d, z: %d is dead rn \n");
        *(int*)(&cuConstIterationParams.outputData[index]) = (*(bool*)(&cuConstIterationParams.ruleset[numAlive])) ? 1 : 0;
    }
}

__global__ void kernelDoIterationVonNeumann() {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    *(int*)(&cuConstIterationParams.outputData[index]) = 5;
}



GolCuda::GolCuda() {
    sideLength = 0;
    isMoore = true;
    numStates = 0;

    cube = NULL;
    ruleset = NULL;
    inputData = NULL;

    cudaDeviceInputData = NULL;
    cudaDeviceOutputData = NULL;
    cudaDeviceRuleset = NULL;
}

GolCuda::~GolCuda() {

    if (cube) {
        delete [] cube;
    }
    if (ruleset) {
        delete [] ruleset;
    } 
    if (inputData) {
        delete [] inputData;
    }
    if (cudaDeviceInputData) {
        cudaFree(cudaDeviceRuleset);
        cudaFree(cudaDeviceInputData);
        cudaFree(cudaDeviceOutputData);
    }
}

void
GolCuda::clearOutputCube() {
    cube->clear();
}

void
GolCuda::allocOutputCube(int sideLength) {

    if (cube)
        delete cube;
        cube = new Cube(sideLength);
}
 
Cube*
GolCuda::getCube() {

    // Need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    //printf("Copying image data from device\n");

    cudaMemcpy(cube->data,
               cudaDeviceOutputData,
               sizeof(int) * sideLength * sideLength * sideLength,
               cudaMemcpyDeviceToHost);
    
    return cube;
}

int
GolCuda::loadInput(char* file, int n, char* outputDir) {
    return loadCubeInput(file, sideLength, ruleset, numStates, isMoore, inputData, n, outputDir);
}

void
GolCuda::setup() {

    int deviceCount = 0;
    bool isFastGPU = false;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;
        if (name.compare("NVIDIA GeForce RTX 2080") == 0)
        {
            isFastGPU = true;
        }

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    if (!isFastGPU)
    {
        printf("WARNING: "
               "You're not running on a fast GPU, please consider using "
               "NVIDIA RTX 2080.\n");
        printf("---------------------------------------------------------\n");
    }
    
    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy
    int cubeSize = sideLength * sideLength * sideLength;

    cudaMalloc(&cudaDeviceRuleset, sizeof(bool) * 54);
    cudaMalloc(&cudaDeviceInputData, sizeof(int) * cubeSize);
    cudaMalloc(&cudaDeviceOutputData, sizeof(int) * cubeSize);

    cudaMemcpy(cudaDeviceRuleset, ruleset, sizeof(bool) * 54, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceInputData, inputData, sizeof(int) * cubeSize, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sideLength = sideLength;
    params.isMoore = isMoore;
    params.numStates = numStates;
    params.inputData = cudaDeviceInputData;
    params.outputData = cudaDeviceOutputData;
    params.ruleset = cudaDeviceRuleset;
    
    cudaMemcpyToSymbol(cuConstIterationParams, &params, sizeof(GlobalConstants));
}


void 
GolCuda::advanceFrame() {
    cudaMemcpy(inputData,
        cudaDeviceOutputData,
        sizeof(int) * sideLength * sideLength * sideLength,
        cudaMemcpyDeviceToHost);

    cudaMemcpy(cudaDeviceInputData, inputData, sizeof(int) * sideLength * sideLength * sideLength, cudaMemcpyHostToDevice);
    //cudaMemcpyToSymbol(cuConstIterationParams, &params, sizeof(GlobalConstants));
}

void
GolCuda::doIteration() {    
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(((sideLength * sideLength * sideLength) + blockDim.x - 1) / blockDim.x);

    if (isMoore) {
        kernelDoIterationMoore<<<gridDim, blockDim>>>(); 
    } else {
        kernelDoIterationVonNeumann<<<gridDim, blockDim>>>(); 
    }
      
}  