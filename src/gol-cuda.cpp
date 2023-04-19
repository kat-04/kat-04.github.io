// TODO: write function to load data into cuda memory
#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <iostream>
#include <map>
#include <cstring>
#include <vector>
#include <tuple>
#include <fstream>
#include <iostream> //for cout
#include "timing.h"
#include "cube.h"
#include "golCuda.h"


//TODO: move to iterationLoader separate file so can include in cuda file
// void loadIterationInput(char* file, int*& sideLength, int*& ruleset, int*& cubeData) {

// }

int gol_cuda(int argc, char** argv, std::string fileDir) {
    
    //parse args (argc has been checked in main)
    string inputFile = argv[1];
    int numFrames = stoi(argv[2]);
    int sideLength = stoi(argv[3]);
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
    //golCuda->clearOutputCube();
    //time start
    golCuda->doIteration();
    //time end
    //golCuda->getCube();
    //write to output file

    return 0;
}