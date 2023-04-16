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


#include "golCuda.h"

GolCuda::GolCuda() {
    outputCube = NULL;
    sideLength = 0;
    ruleset = NULL;

    cudaDeviceOutputData = NULL;
    cudaDeviceSideLength = NULL;
    cudaDeviceRuleset = NULL;
}

GolCuda::~GolCuda() {

    if (outputCube) {
        delete outputCube;
    }
    if (ruleset) {
        delete ruleset;
    }
    if (cudaDeviceOutputData) {
        cudaFree(cudaDeviceOutputData);
        cudaFree(cudaDeviceSideLength);
        cudaFree(cudaDeviceRuleset);
    }
}

void
GolCuda::clearResultCube() {

    // 256 threads per block is a healthy number
    // dim3 blockDim(16, 16, 1);
    // dim3 gridDim(
    //     (image->width + blockDim.x - 1) / blockDim.x,
    //     (image->height + blockDim.y - 1) / blockDim.y);

    // kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);

}

void
GolCuda::allocResultCube(int sideLength) {

    // if (image)
    //     delete image;
    // image = new Image(width, height);
}

const Cube*
GolCuda::getResultCube() {

    // Need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    // printf("Copying image data from device\n");

    // cudaMemcpy(image->data,
    //            cudaDeviceImageData,
    //            sizeof(float) * 4 * image->width * image->height,
    //            cudaMemcpyDeviceToHost);

    // return image;
}

void
GolCuda::loadInput(char* file) {
    // sceneName = scene;
    // loadCircleScene(sceneName, numberOfCircles, position, velocity, color, radius);
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
        if (name.compare("GeForce RTX 2080") == 0)
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

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numberOfCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numberOfCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numberOfCircles = numberOfCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));
}

void
GolCuda::doIteration() {
    return;
}