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


int gol_cuda(int argc, char** argv, std::string fileDir) {
    
    //parse args (argc has been checked in main)
    string inputFile = argv[1];
    int numFrames = stoi(argv[2]);
    uint64_t sideLength = stoi(argv[3]);
    string outputPath =  fileDir + "/frame";

    GolCuda* golCuda;
    golCuda = new GolCuda();

    golCuda->allocOutputCube(sideLength);
    int fileLoadRes = golCuda->loadInput(const_cast<char*>(inputFile.c_str()), sideLength, const_cast<char*>(outputPath.c_str()));
    if (fileLoadRes != 0) {
        std::cerr << "could not open input file: " << inputFile << std::endl;
        return 1;
    }
    golCuda->setup();
    golCuda->clearOutputCube();

    Cube* resultCube;

    double totalSimulationTime = 0.0;

    //time start
    for (int f = 0; f < numFrames; f++) {
        Timer frameTimer;
        golCuda->updateBounds();
        golCuda->doIteration();
        totalSimulationTime += frameTimer.elapsed();

        resultCube = golCuda->getCube();
        golCuda->advanceFrame();
        //write to output file
        ofstream output; 
        std::string frameOutputFile = outputPath + to_string(f+1) + ".txt";
        output.open(frameOutputFile);
       
        
        //output << sideLength << std::endl;
        for (uint64_t i = 0; i < sideLength * sideLength * sideLength; i++) {
            uint64_t z = (i / (sideLength * sideLength)) % sideLength;
            uint64_t y = (i / sideLength) % sideLength;
            uint64_t x = i % sideLength;
            int bit = i % 8;
            /* std::cout << "."; */
            if (((resultCube->data[i / 8]) >> (7 - bit)) & 1) {
                /* std::cout << "Alive cell exists" << std::endl; */
                output << x << " " << y << " " << z << endl;
            }
        }
        output.close();
        golCuda->clearOutputCube();
    }
    
    printf("total simulation time: %.6fs\n", totalSimulationTime);
    return 0;
}
