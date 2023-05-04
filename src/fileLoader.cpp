#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <cstring>
#include <vector>
#include <getopt.h>
#include <tuple>
#include <map>
#include <fstream>
#include <iostream> //for cout
#include <cmath>

#include "golCuda.h"
#include "parse.h"
#include "fileLoader.h"
#include "cube.h"

// load in data used for the Cuda implementation
int loadCubeInput(char* file, uint64_t& sideLength, bool*& ruleset, int& numStates, bool& isMoore,
                  uint8_t*& inputData, uint64_t n, char* outputPath, uint32_t*& minMaxs) {
    std::fstream input;
    std::string line;
    std::vector<std::string> coords;
    input.open(file, std::ios::in);
    
    uint64_t curLine = 0;
    const char* spaceDelim = " ";

    inputData = new uint8_t[(n*n*n + 7) / 8];
    memset(inputData, 0, sizeof(uint8_t) * (n*n*n + 7) / 8);
    minMaxs = new uint32_t[6];
    sideLength = n;

    //write frame0 to be same status as input (updates start at frame1)
    std::string outputDir = outputPath;
    std::string frameOutputFile = outputDir + "0.txt";
    std::ofstream outputInit; 
    outputInit.open(frameOutputFile);

    uint64_t linearIndex = 0;
    int bit = 0;
    uint8_t mask = 0;

    while (getline(input, line)) {
        if (curLine == 0) {
            std::tie(isMoore, numStates) = parseRulesCuda(line, ruleset);   

            //write frame_init
            std::ofstream frameInit;
            std::string frameInitFile = outputDir + "_init.txt";
            frameInit.open(frameInitFile);
            frameInit << n << std::endl;
            frameInit << numStates << std::endl;
            frameInit.close();
        } else {
            //set voxel to on
            coords = tokenizeLine(line, spaceDelim);
            linearIndex = (stoi(coords[0]) + (n * stoi(coords[1])) + (n * n * stoi(coords[2]))) / 8;
            bit = (stoi(coords[0]) + (n * stoi(coords[1])) + (n * n * stoi(coords[2]))) % 8;
            mask = 1 << (7 - bit);
            inputData[linearIndex] = inputData[linearIndex] | mask;
            outputInit << coords[0] << " " << coords[1] << " " << coords[2] << std::endl;
        }
        mask = 0;
        curLine++;
    }
    input.close();
    outputInit.close();

    return 0;
}
