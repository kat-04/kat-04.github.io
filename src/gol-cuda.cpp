// TODO: write function to load data into cuda memory
#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <iostream>
#include <map>
#include <cstring>
#include <cmath>
#include <vector>
#include <tuple>
#include <fstream>
#include <iostream> //for cout
#include "timing.h"
#include "cube.h"
#include "golCuda.h"

// change to 1 to implement the bounding box optimization
#define DO_BOUNDING_BOX 0

// runs the full Cuda game of life algorithm
int gol_cuda(int argc, char** argv, std::string fileDir) {
    
    //parse args
    string inputFile = argv[1];
    int numFrames = stoi(argv[2]);
    uint64_t sideLength = stoi(argv[3]);
    string outputPath =  fileDir + "/frame";

    GolCuda* golCuda;
    golCuda = new GolCuda();

    // alloc space for frame output and load in input file
    golCuda->allocOutputCube(sideLength);
    int fileLoadRes = golCuda->loadInput(const_cast<char*>(inputFile.c_str()), sideLength, const_cast<char*>(outputPath.c_str()));
    if (fileLoadRes != 0) {
        std::cerr << "could not open input file: " << inputFile << std::endl;
        return 1;
    }

    // cuda set up
    golCuda->setup();
    golCuda->clearOutputCube();


    Cube* resultCube;
    double totalSimulationTime = 0.0;
    
    // main loop, does each frame with Cuda
    for (int f = 0; f < numFrames; f++) {
        Timer frameTimer; // timer start
        if (DO_BOUNDING_BOX) {
            golCuda->updateBounds();
        }
        golCuda->doIteration(DO_BOUNDING_BOX);
        totalSimulationTime += frameTimer.elapsed(); // timer end

        // get Cuda results and begin setup for next frame
        resultCube = golCuda->getCube();
        golCuda->advanceFrame();

        //write to output file
        ofstream output; 
        std::string frameOutputFile = outputPath + to_string(f+1) + ".txt";
        output.open(frameOutputFile);
        for (uint64_t i = 0; i < sideLength * sideLength * sideLength; i++) {
            uint64_t z = (i / (sideLength * sideLength)) % sideLength;
            uint64_t y = (i / sideLength) % sideLength;
            uint64_t x = i % sideLength;
            int bit = i % 8;
            if (((resultCube->data[i / 8]) >> (7 - bit)) & 1) {
                output << x << " " << y << " " << z << endl;
            }
        }
        output.close();
        golCuda->clearOutputCube();
    }
    
    printf("total simulation time: %.6fs\n", totalSimulationTime);
    return 0;
}
