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
#include <iostream>
#include <cmath>
#include <bitset>
#include "cube.h"
#include "fileLoader.h"
#include "golCuda.h"

#define BLOCK_SIZE 256

struct GlobalConstants {
    uint64_t sideLength;
    bool isMoore;
    int numStates;
    bool* ruleset;
    uint8_t* inputData;
    uint8_t* outputData;
};

__constant__ GlobalConstants cuConstIterationParams;

__global__ void kernelDoIterationMoore() {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = cuConstIterationParams.sideLength;
    if (index >= (n * n * n + 7) / 8) {
        return;
    }
    /* printf("Thread idx: %d\n", index); */
    
    // index of the first thing in the bit array
    uint64_t bitIndex = index * 8;
    /* printf("Bit idx: %d\n", bitIndex); */
    uint64_t neighborBitIndex;
    uint64_t neighborLinIndex;
    int neighborBit;
    uint8_t mask = 1;
    int numAlive = 0;
    int status;
    // printf("doin index %d, x: %d, y: %d, z: %d\n", index, x, y, z);
    for (int bit = 0; bit < 8; bit++) {
        /* printf("Bit idx: %d\n", bitIndex + bit); */
        uint64_t z = ((bitIndex + bit) / (n * n)) % n;
        uint64_t y = ((bitIndex + bit) / n) % n;
        uint64_t x = (bitIndex + bit) % n;
        /* printf("x: %d, y: %d, z: %d\n", x, y, z); */
        for (uint64_t i = (x == 0) ? 0 : x - 1; i <= x + 1; i++) {
            for (uint64_t j = (y == 0) ? 0 : y - 1; j <= y + 1; j++) {
                for (uint64_t k = (z == 0) ? 0 : z - 1; k <= z + 1; k++) {
                    /* printf("Hello from index %d and bit %d\n", index, bit); */
                    if (i < n && j < n && k < n) {
                        if (!(x == i && y == j && z == k)) { //don't include self
                            neighborBitIndex = (k * n * n) + (j * n) + i;
                            neighborLinIndex = neighborBitIndex / 8;
                            neighborBit = neighborBitIndex % 8;
                            /* if (index == 3 && bit == 0) { */
                            /*     printf("i: %d, j: %d, k: %d\n", i, j, k); */
                            /* } */
                            numAlive += (cuConstIterationParams.inputData[neighborLinIndex] >> (7 - neighborBit)) & mask;
                        }
                    }
                }
            }
        }
        /* printf("x: %d, y: %d, z: %d has value: %d \n", x, y, z, numAlive); */
        /* printf("Helloooo\n"); */
        if ((cuConstIterationParams.inputData[index] >> (7 - bit)) & mask) {
            // OLD STATE IS ALIVE
            /* printf("x: %d, y: %d, z: %d is alive rn with %d neighbors \n", x, y, z, numAlive); */
            status = cuConstIterationParams.ruleset[27 + numAlive] ? 1 : 0;
            if (status) {
                // alive
                cuConstIterationParams.outputData[index] = cuConstIterationParams.outputData[index] | (status << (7 - bit));
            } else {
                // dead
                cuConstIterationParams.outputData[index] = cuConstIterationParams.outputData[index] & ~(1 << (7 - bit));
            }
        } else {
            // OLD STATE IS DEAD
            /* printf("x: %d, y: %d, z: %d is dead rn \n", x, y, z); */
            status = cuConstIterationParams.ruleset[numAlive] ? 1 : 0;
            cuConstIterationParams.outputData[index] = cuConstIterationParams.outputData[index] | (status << (7 - bit));
        }
        numAlive = 0;
    }
}

__global__ void kernelDoIterationVonNeumann() {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = cuConstIterationParams.sideLength;
    if (index >= (n * n * n + 7) / 8) {
        return;
    }
    
    uint64_t bitIndex = index * 8;
    uint64_t neighborBitIndex;
    uint64_t neighborLinIndex;
    int neighborBit;
    uint8_t mask = 1;
    int numAlive = 0;
    int status;

    for (int bit = 0; bit < 8; bit++) {
        uint64_t z = ((bitIndex + bit) / (n * n)) % n;
        uint64_t y = ((bitIndex + bit) / n) % n;
        uint64_t x = (bitIndex + bit) % n;

        neighborBitIndex = (z * n * n) + (y * n) + x - 1;
        neighborLinIndex = neighborBitIndex / 8;
        neighborBit = neighborBitIndex % 8;
        if (x > 0 && x < n) numAlive += (cuConstIterationParams.inputData[neighborLinIndex] >> (7 - neighborBit)) & mask;
        
        neighborBitIndex = (z * n * n) + (y * n) + x + 1;
        neighborLinIndex = neighborBitIndex / 8;
        neighborBit = neighborBitIndex % 8;
        if (x + 1 < n) numAlive += (cuConstIterationParams.inputData[neighborLinIndex] >> (7 - neighborBit)) & mask;

        neighborBitIndex = (z * n * n) + ((y-1) * n) + x;
        neighborLinIndex = neighborBitIndex / 8;
        neighborBit = neighborBitIndex % 8;
        if (y > 0 && y < n) numAlive += (cuConstIterationParams.inputData[neighborLinIndex] >> (7 - neighborBit)) & mask;

        neighborBitIndex = (z * n * n) + ((y+1) * n) + x;
        neighborLinIndex = neighborBitIndex / 8;
        neighborBit = neighborBitIndex % 8;
        if (y + 1 < n) numAlive += (cuConstIterationParams.inputData[neighborLinIndex] >> (7 - neighborBit)) & mask;

        neighborBitIndex = ((z-1) * n * n) + (y * n) + x;
        neighborLinIndex = neighborBitIndex / 8;
        neighborBit = neighborBitIndex % 8;
        if (z > 0 && z < n) numAlive += (cuConstIterationParams.inputData[neighborLinIndex] >> (7 - neighborBit)) & mask;

        neighborBitIndex = ((z+1) * n * n) + (y * n) + x;
        neighborLinIndex = neighborBitIndex / 8;
        neighborBit = neighborBitIndex % 8;
        if (z + 1 < n) numAlive += (cuConstIterationParams.inputData[neighborLinIndex] >> (7 - neighborBit)) & mask;

        // printf("x: %d, y: %d, z: %d has value: %d \n", x, y, z, *(int*)&cuConstIterationParams.inputData[index]);
        if ((cuConstIterationParams.inputData[index] >> (7 - bit)) & mask) {
            // printf("x: %d, y: %d, z: %d is alive rn with %d neighbors \n", x, y, z, numAlive);
            status = cuConstIterationParams.ruleset[27 + numAlive] ? 1 : 0;
            if (status) {
                // alive
                cuConstIterationParams.outputData[index] = cuConstIterationParams.outputData[index] | (status << (7 - bit));
            } else {
                // dead
                cuConstIterationParams.outputData[index] = cuConstIterationParams.outputData[index] & ~(1 << (7 - bit));
            }
        } else {
            // printf("x: %d, y: %d, z: %d is dead rn \n");
            status = cuConstIterationParams.ruleset[numAlive] ? 1 : 0;
            cuConstIterationParams.outputData[index] = cuConstIterationParams.outputData[index] | (status << (7 - bit));
        }

        numAlive = 0;
    }
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
GolCuda::allocOutputCube(uint64_t sideLength) {
    /* printf("%lu\n", (sideLength * sideLength * sideLength + 7)); */
    /* printf("Side length: %d\n", sideLength); */
    printf("Size of data: %f MB\n", sizeof(uint8_t) * (((sideLength * sideLength * sideLength + 7) / 8) / (1024.f * 1024.f)));
    if (cube) delete cube;
    cube = new Cube(sideLength);
}
 
Cube*
GolCuda::getCube() {

    // Need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    //printf("Copying image data from device\n");

    cudaMemcpy(cube->data,
               cudaDeviceOutputData,
               sizeof(uint8_t) * ((sideLength * sideLength * sideLength + 7) / 8),
               cudaMemcpyDeviceToHost);
    
    return cube;
}

int
GolCuda::loadInput(char* file, uint64_t n, char* outputDir) {
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
    uint64_t cubeSize = (sideLength * sideLength * sideLength + 7) / 8;

    cudaMalloc(&cudaDeviceRuleset, sizeof(bool) * 54);
    cudaMalloc(&cudaDeviceInputData, sizeof(uint8_t) * cubeSize);
    cudaMalloc(&cudaDeviceOutputData, sizeof(uint8_t) * cubeSize);

    cudaMemcpy(cudaDeviceRuleset, ruleset, sizeof(bool) * 54, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceInputData, inputData, sizeof(uint8_t) * cubeSize, cudaMemcpyHostToDevice);

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
        sizeof(uint8_t) * ((sideLength * sideLength * sideLength + 7) / 8),
        cudaMemcpyDeviceToHost);

    cudaMemcpy(cudaDeviceInputData, inputData, sizeof(uint8_t) * ((sideLength * sideLength * sideLength + 7) / 8), cudaMemcpyHostToDevice);
    //cudaMemcpyToSymbol(cuConstIterationParams, &params, sizeof(GlobalConstants));
}

void
GolCuda::doIteration() {    

    /* printf("sideLength: %d, numStates: %d, isMoore %d\n", sideLength, numStates, isMoore); */
    /* printf("ruleset: "); */
    /* for (int i = 0; i < 54; i++) { */
    /*     printf("%d, ", ruleset[i]); */
    /* } */
    /* printf("\n, inputData: "); */
    /* for (uint64_t i = 0; i < (sideLength * sideLength * sideLength + 7) / 8; i++) { */
    /*     std::cout << i << ": " << std::bitset<8>(inputData[i]) << std::endl; */
    /* } */

    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((((sideLength * sideLength * sideLength + 7) / 8) + blockDim.x - 1) / blockDim.x);

    if (isMoore) {
        kernelDoIterationMoore<<<gridDim, blockDim>>>(); 
    } else {
        kernelDoIterationVonNeumann<<<gridDim, blockDim>>>(); 
    }
      
}  
