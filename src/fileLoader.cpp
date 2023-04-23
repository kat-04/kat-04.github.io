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


int loadCubeInput(char* file, uint32_t& sideLength, bool*& ruleset, int& numStates, bool& isMoore,
                  uint8_t*& inputData, uint32_t n, char* outputPath) {
    std::fstream input;
    input.open(file, std::ios::in);
    std::string line;
    uint32_t curLine = 0;
    std::vector<std::string> coords;
    const char* spaceDelim = " ";
    inputData = new uint8_t[(n*n*n + 7) / 8];
    memset(inputData, 0, sizeof(inputData));

    //write frame0 to be same status as input (updates start at frame1)
    std::string outputDir = outputPath;
    std::string frameOutputFile = outputDir + "0.txt";
    std::ofstream outputInit; 
    outputInit.open(frameOutputFile);

    uint32_t linearIndex = 0;
    int bit = 0;
    uint8_t mask = 0;

    while (getline(input, line)) {
        if (curLine == 0) {
            std::tie(isMoore, numStates) = parseRulesCuda(line, ruleset);   
            outputInit << n << std::endl;
        } else {
            //set voxel to on
            coords = tokenizeLine(line, spaceDelim);
            //TODO: should probably check for out of bounds input here and error
            //TODO: parse input cell's state when this is implemented and replace auto set state to 1
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

    sideLength = n;
    return 0;
}
