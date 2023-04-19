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

struct GlobalConstants {
    int sideLength;
    int* outputData;
    int* ruleset;
    int* inputData;
};

__constant__ GlobalConstants cuConstIterationParams;

GolCuda::GolCuda() {
    sideLength = 0;
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
    // 256 threads per block is a healthy number
    // dim3 blockDim(16, 16, 1);
    // dim3 gridDim(
    //     (image->width + blockDim.x - 1) / blockDim.x,
    //     (image->height + blockDim.y - 1) / blockDim.y);

    // kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);

}

void
GolCuda::allocOutputCube(int sideLength) {

    if (cube)
        delete cube;
        cube = new Cube(sideLength);
}
 
const Cube*
GolCuda::getCube() {

    // Need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(cube->data,
               cudaDeviceOutputData,
               sizeof(int) * sideLength * sideLength * sideLength,
               cudaMemcpyDeviceToHost);

    return cube;
}

void
GolCuda::loadInput(char* file, int n) {
    loadCubeInput(file, sideLength, ruleset, inputData, n);
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

    cudaMalloc(&cudaDeviceRuleset, sizeof(int) * 56);
    cudaMalloc(&cudaDeviceInputData, sizeof(int) * cubeSize);
    cudaMalloc(&cudaDeviceOutputData, sizeof(int) * cubeSize);

    cudaMemcpy(cudaDeviceRuleset, ruleset, sizeof(int) * 56, cudaMemcpyHostToDevice);
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
    params.inputData = cudaDeviceInputData;
    params.outputData = cudaDeviceOutputData;
    params.ruleset = cudaDeviceRuleset;
    
    cudaMemcpyToSymbol(cuConstIterationParams, &params, sizeof(GlobalConstants));
}

void
GolCuda::doIteration() {
    // printf("HEY im in cuda and i also loaded the sideLength of: %d!\n", sideLength);
    return;
}