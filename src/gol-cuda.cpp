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
    golCuda->clearOutputCube();

    Cube* resultCube;

    //time start
    for (int f = 0; f < numFrames; f++) {
        golCuda->doIteration();
        //time end
        resultCube = golCuda->getCube();
        golCuda->advanceFrame();
        //write to output file
        ofstream output; 
        std::string frameOutputFile = outputPath + to_string(f+1) + ".txt";
        output.open(frameOutputFile);
       
        
        output << sideLength << std::endl;
        for (int i = 0; i < sideLength * sideLength * sideLength; i++) {
            int z = (i / (sideLength * sideLength)) % sideLength;
            int y = (i / sideLength) % sideLength;
            int x = i % sideLength;
            if (resultCube->data[i]) {
                output << x << " " << y << " " << z << endl;
            }
        }
        output.close();
    }
    
    

    return 0;
}