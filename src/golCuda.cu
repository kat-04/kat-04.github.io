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
// Comment for faster runtimes
//#define DEBUG

#ifdef DEBUG
#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n",
        cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#else
#define cudaCheckError(ans) ans
#endif

struct GlobalConstants {
    uint64_t sideLength;
    bool isMoore;
    int numStates;

    bool* ruleset;
    uint32_t* minMaxs;

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
    uint64_t boundedX = 1 + cuConstIterationParams.minMaxs[3] - cuConstIterationParams.minMaxs[0];
    uint64_t boundedY = 1 + cuConstIterationParams.minMaxs[4] - cuConstIterationParams.minMaxs[1];
    uint64_t boundedZ = 1 + cuConstIterationParams.minMaxs[5] - cuConstIterationParams.minMaxs[2];

    if (index >= (boundedX * boundedY * boundedZ + 7) / 8) {
        return;
    }

    // index of the first thing in the bit array
    uint64_t bitIndex = 8 * index;
    uint64_t neighborBitIndex;
    uint64_t neighborLinIndex;
    int neighborBit;
    uint8_t mask = 1;
    int numAlive = 0;
    int status;
    
    //printf("bounds: %lu, %lu, %lu\n", boundedX, boundedY, boundedZ);
    for (int bit = 0; bit < 8; bit++) {
        if (bitIndex + bit >= boundedX * boundedY * boundedZ) {
            break;
        }
        uint64_t x = (((bitIndex + bit) % boundedX) + cuConstIterationParams.minMaxs[0]);
        uint64_t y = (((bitIndex + bit ) / boundedX) % boundedY) + cuConstIterationParams.minMaxs[1];
        uint64_t z = (((bitIndex + bit ) / (boundedX * boundedY)) % boundedZ) + cuConstIterationParams.minMaxs[2];
        //printf("doin index %lu, x: %lu, y: %lu, z: %lu\n", index, x, y, z);

        for (uint64_t i = (x == 0) ? 0 : x - 1; i <= x + 1; i++) {
            for (uint64_t j = (y == 0) ? 0 : y - 1; j <= y + 1; j++) {
                for (uint64_t k = (z == 0) ? 0 : z - 1; k <= z + 1; k++) {
                    if (i < n && j < n && k < n) {
                        if (!(x == i && y == j && z == k)) { //don't include self
                            neighborBitIndex = (k * n * n) + (j * n) + i;
                            
                            neighborLinIndex = neighborBitIndex / 8;
                            neighborBit = neighborBitIndex % 8;
                            //printf("index bit %lu has neighbor bit idx %lu\n", index, neighborLinIndex);
                            numAlive += (cuConstIterationParams.inputData[neighborLinIndex] >> (7 - neighborBit)) & mask;
                        }
                    }
                }
            }
        }
        //printf("idx: %lu, x: %lu, y: %lu, z: %lu has value: %d \n", index, x, y, z, numAlive);
        /* printf("Helloooo\n"); */
        uint64_t shiftedLinIndex = (z * n * n + y * n + x) / 8;
        uint64_t shiftedBit = (z * n * n + y * n + x) % 8;
        //printf("x: %lu, y: %lu,, z: %lu,  shiftedLinIndex: %lu, shiftedBit: %lu\n", x, y, z, shiftedLinIndex, shiftedBit);
        if ((cuConstIterationParams.inputData[shiftedLinIndex] >> (7 - shiftedBit)) & mask) {
            // OLD STATE IS ALIVE
            /* printf("x: %d, y: %d, z: %d is alive rn with %d neighbors \n", x, y, z, numAlive); */
            status = cuConstIterationParams.ruleset[27 + numAlive] ? 1 : 0;
            if (status) {
                // alive
                cuConstIterationParams.outputData[shiftedLinIndex] = cuConstIterationParams.outputData[shiftedLinIndex] | (status << (7 - shiftedBit));
            } else {
                // dead
                cuConstIterationParams.outputData[shiftedLinIndex] = cuConstIterationParams.outputData[shiftedLinIndex] & ~(1 << (7 - shiftedBit));
            }
        } else {
            // OLD STATE IS DEAD
            /* printf("x: %d, y: %d, z: %d is dead rn \n", x, y, z); */
            status = cuConstIterationParams.ruleset[numAlive] ? 1 : 0;
            cuConstIterationParams.outputData[shiftedLinIndex] = cuConstIterationParams.outputData[shiftedLinIndex] | (status << (7 - shiftedBit));
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

    uint64_t boundedX = 1 + cuConstIterationParams.minMaxs[3] - cuConstIterationParams.minMaxs[0];
    uint64_t boundedY = 1 + cuConstIterationParams.minMaxs[4] - cuConstIterationParams.minMaxs[1];
    uint64_t boundedZ = 1 + cuConstIterationParams.minMaxs[5] - cuConstIterationParams.minMaxs[2];

    if (index >= (boundedX * boundedY * boundedZ + 7) / 8) {
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
        if (bitIndex + bit >= boundedX * boundedY * boundedZ) {
            break;
        }
        uint64_t x = (((bitIndex + bit) % boundedX) + cuConstIterationParams.minMaxs[0]);
        uint64_t y = (((bitIndex + bit ) / boundedX) % boundedY) + cuConstIterationParams.minMaxs[1];
        uint64_t z = (((bitIndex + bit ) / (boundedX * boundedY)) % boundedZ) + cuConstIterationParams.minMaxs[2];

        neighborBitIndex = (z * n * n) + (y * n) + x - 1;
        neighborLinIndex = neighborBitIndex / 8;
        neighborBit = neighborBitIndex % 8;
        if (x > 0 && x < n) numAlive += (cuConstIterationParams.inputData[neighborLinIndex] >> (7 - neighborBit)) & mask;
        
        neighborBitIndex = (z * n * n) + (y * n) + x + 1;
        neighborLinIndex = neighborBitIndex / 8;
        neighborBit = neighborBitIndex % 8;
        if (x + 1 < n) numAlive += (cuConstIterationParams.inputData[neighborLinIndex] >> (7 - neighborBit)) & mask;

        neighborBitIndex = (z * n * n) + ((y - 1) * n) + x;
        neighborLinIndex = neighborBitIndex / 8;
        neighborBit = neighborBitIndex % 8;
        if (y > 0 && y < n) numAlive += (cuConstIterationParams.inputData[neighborLinIndex] >> (7 - neighborBit)) & mask;

        neighborBitIndex = (z * n * n) + ((y + 1) * n) + x;
        neighborLinIndex = neighborBitIndex / 8;
        neighborBit = neighborBitIndex % 8;
        if (y + 1 < n) numAlive += (cuConstIterationParams.inputData[neighborLinIndex] >> (7 - neighborBit)) & mask;

        neighborBitIndex = ((z - 1) * n * n) + (y * n) + x;
        neighborLinIndex = neighborBitIndex / 8;
        neighborBit = neighborBitIndex % 8;
        if (z > 0 && z < n) numAlive += (cuConstIterationParams.inputData[neighborLinIndex] >> (7 - neighborBit)) & mask;

        neighborBitIndex = ((z + 1) * n * n) + (y * n) + x;
        neighborLinIndex = neighborBitIndex / 8;
        neighborBit = neighborBitIndex % 8;
        if (z + 1 < n) numAlive += (cuConstIterationParams.inputData[neighborLinIndex] >> (7 - neighborBit)) & mask;

        uint64_t shiftedLinIndex = (z * n * n + y * n + x) / 8;
        uint64_t shiftedBit = (z * n * n + y * n + x) % 8;
        if ((cuConstIterationParams.inputData[index] >> (7 - shiftedBit)) & mask) {
            // printf("x: %d, y: %d, z: %d is alive rn with %d neighbors \n", x, y, z, numAlive);
            status = cuConstIterationParams.ruleset[27 + numAlive] ? 1 : 0;
            if (status) {
                // alive
                cuConstIterationParams.outputData[shiftedLinIndex] = cuConstIterationParams.outputData[shiftedLinIndex] | (status << (7 - shiftedBit));
            } else {
                // dead
                cuConstIterationParams.outputData[shiftedLinIndex] = cuConstIterationParams.outputData[shiftedLinIndex] & ~(1 << (7 - shiftedBit));
            }
        } else {
            // printf("x: %d, y: %d, z: %d is dead rn \n");
            status = cuConstIterationParams.ruleset[numAlive] ? 1 : 0;
            cuConstIterationParams.outputData[shiftedLinIndex] = cuConstIterationParams.outputData[shiftedLinIndex] | (status << (7 - shiftedBit));
        }

        numAlive = 0;
    }
}

__global__ void kernelGetGlobalBounds() {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = cuConstIterationParams.sideLength;
    if (index >= n) {
        return;
    } 
    uint64_t localMinX = n - 1;
    uint64_t localMinY = n - 1;
    uint64_t localMinZ = n - 1;
    uint64_t localMaxX = 0;
    uint64_t localMaxY = 0;
    uint64_t localMaxZ = 0;

    uint64_t bit_index;
    // Bit and index in array of neighbor
    uint64_t linIndex;
    uint8_t bit;
    for (uint64_t y = 0; y < n; y++) {
        for (uint64_t z = 0; z < n; z++) {
            if (y >= n || z >= n) {
                continue;
            }

            bit_index = index + y * n + z * n * n;
            linIndex = bit_index / 8;
            bit = bit_index % 8;

            uint8_t mask = 1;
            uint8_t alive = ((cuConstIterationParams.inputData[linIndex] >> (7 - bit))) & mask;
            if (alive) {
                // update local values
                localMinX = min(localMinX, (index == 0) ? (uint64_t)0 : index - 1);
                localMinY = min(localMinY, (y == 0) ? (uint64_t)0 : y - 1);
                localMinZ = min(localMinZ, (z == 0) ? (uint64_t)0 : z - 1);
                localMaxX = min(max(localMaxX, index + 1), n - 1);
                localMaxY = min(max(localMaxY, y + 1), n - 1);
                localMaxZ = min(max(localMaxZ, z + 1), n - 1);
            }
        }
    }
    
    // atomically update global values
    atomicMin(&(cuConstIterationParams.minMaxs[0]), (uint32_t)localMinX);
    atomicMin(&(cuConstIterationParams.minMaxs[1]), (uint32_t)localMinY);
    atomicMin(&(cuConstIterationParams.minMaxs[2]), (uint32_t)localMinZ);
    atomicMax(&(cuConstIterationParams.minMaxs[3]), (uint32_t)localMaxX);
    atomicMax(&(cuConstIterationParams.minMaxs[4]), (uint32_t)localMaxY);
    atomicMax(&(cuConstIterationParams.minMaxs[5]), (uint32_t)localMaxZ);    
}



GolCuda::GolCuda() {
    sideLength = 0;
    isMoore = true;
    numStates = 0;

    cube = NULL;
    ruleset = NULL;
    inputData = NULL;
    minMaxs = NULL;

    cudaDeviceMinMaxs = NULL;
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
    if (minMaxs) {
        delete [] minMaxs;
    }
    if (cudaDeviceInputData) {
        cudaCheckError(cudaFree(cudaDeviceRuleset));
        cudaCheckError(cudaFree(cudaDeviceInputData));
        cudaCheckError(cudaFree(cudaDeviceOutputData));
        cudaCheckError(cudaFree(cudaDeviceMinMaxs));
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
    if (!cube) {
        std::cerr << "Cube allocation failed" << std::endl;
    }
}
 
Cube*
GolCuda::getCube() {

    // Need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    //printf("Copying image data from device\n");

    cudaCheckError(cudaMemcpy(cube->data,
               cudaDeviceOutputData,
               sizeof(uint8_t) * ((sideLength * sideLength * sideLength + 7) / 8),
               cudaMemcpyDeviceToHost));
    
    return cube;
}

int
GolCuda::loadInput(char* file, uint64_t n, char* outputDir) {
    return loadCubeInput(file, sideLength, ruleset, numStates, isMoore, inputData, n, outputDir, minMaxs);
}

void
GolCuda::setup() {

    int deviceCount = 0;
    bool isFastGPU = false;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    // printf("---------------------------------------------------------\n");
    // printf("Initializing CUDA for CudaRenderer\n");
    // printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;
        if (name.compare("NVIDIA GeForce RTX 2080") == 0)
        {
            isFastGPU = true;
        }

        // printf("Device %d: %s\n", i, deviceProps.name);
        // printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        // printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        // printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    // printf("---------------------------------------------------------\n");
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

    // std::cout << "Cube size: " << cubeSize << std::endl;

    cudaCheckError(cudaMalloc(&cudaDeviceRuleset, sizeof(bool) * 54));
    cudaCheckError(cudaMalloc(&cudaDeviceInputData, sizeof(uint8_t) * cubeSize));
    cudaCheckError(cudaMalloc(&cudaDeviceOutputData, sizeof(uint8_t) * cubeSize));
    cudaCheckError(cudaMalloc(&cudaDeviceMinMaxs, sizeof(uint32_t) * 6));

    /* if (!cudaDeviceInputData || !cudaDeviceOutputData) { */
    /*     std::cerr << "Cuda malloc failed" << std::endl; */
    /* } */

    cudaCheckError(cudaMemcpy(cudaDeviceRuleset, ruleset, sizeof(bool) * 54, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(cudaDeviceInputData, inputData, sizeof(uint8_t) * cubeSize, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(cudaDeviceMinMaxs, minMaxs, sizeof(uint32_t) * 6, cudaMemcpyHostToDevice));
    

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
    params.minMaxs = cudaDeviceMinMaxs;
    params.inputData = cudaDeviceInputData;
    params.outputData = cudaDeviceOutputData;
    params.ruleset = cudaDeviceRuleset;
    
    cudaCheckError(cudaMemcpyToSymbol(cuConstIterationParams, &params, sizeof(GlobalConstants)));
    
}
void GolCuda::updateBounds() {
    dim3 blockDim(sideLength);
    dim3 gridDim(1);
    printf("before mins: %d, %d, %d\n", minMaxs[0], minMaxs[1], minMaxs[2]);
    printf("before maxs: %d, %d, %d\n", minMaxs[3], minMaxs[4], minMaxs[5]);
    // update minMaxs array to maxs with 0 and mins with n - 1
    minMaxs[0] = sideLength - 1;
    minMaxs[1] = sideLength - 1;
    minMaxs[2] = sideLength - 1;
    minMaxs[3] = 0;
    minMaxs[4] = 0;
    minMaxs[5] = 0;
    
    cudaCheckError(cudaMemcpy(cudaDeviceMinMaxs, minMaxs, 6 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    
    kernelGetGlobalBounds<<<gridDim, blockDim>>>();
    
    cudaCheckError(cudaMemcpy(minMaxs, cudaDeviceMinMaxs, 6 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    printf("mins: %d, %d, %d\n", minMaxs[0], minMaxs[1], minMaxs[2]);
    printf("maxs: %d, %d, %d\n", minMaxs[3], minMaxs[4], minMaxs[5]);
}

void 
GolCuda::advanceFrame() {
    cudaCheckError(cudaMemcpy(inputData,
        cudaDeviceOutputData,
        sizeof(uint8_t) * ((sideLength * sideLength * sideLength + 7) / 8),
        cudaMemcpyDeviceToHost)); 

    cudaCheckError(cudaMemcpy(cudaDeviceInputData, inputData, sizeof(uint8_t) * ((sideLength * sideLength * sideLength + 7) / 8), cudaMemcpyHostToDevice));
    //cudaMemcpyToSymbol(cuConstIterationParams, &params, sizeof(GlobalConstants));
}

void
GolCuda::doIteration() {    
    
    uint64_t inBoundsVolume = (1 + minMaxs[3] - minMaxs[0]) * (1 + minMaxs[4] - minMaxs[1]) * (1 + minMaxs[5] - minMaxs[2]);
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((((inBoundsVolume + 7) / 8) + blockDim.x - 1) / blockDim.x);
    if (isMoore) {
        kernelDoIterationMoore<<<gridDim, blockDim>>>(); 
    } else {
        kernelDoIterationVonNeumann<<<gridDim, blockDim>>>(); 
    }
      
}  

//TODO: have them in global consts, but first alloc local host var and pass in to 
//kernel (atomic update this), then back in cpu copy to global consts when done