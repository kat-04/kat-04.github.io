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

#include "golCuda.h"
#include "parse.h"
#include "fileLoader.h"
#include "cube.h"


int loadCubeInput(char* file, int& sideLength, bool* ruleset, int& numStates, bool& isMoore, int* inputData, int n, char* outputPath) {
    std::fstream input;
    input.open(file, std::ios::in);
    std::string line;
    int curLine = 0;
    std::vector<std::string> coords;
    const char* spaceDelim = " ";
    inputData = new int[n*n*n];

    //write frame0 to be same status as input (updates start at frame1)
    std::string outputDir = outputPath;
    std::string frameOutputFile = outputDir + "0.txt";
    std::ofstream outputInit; 
    outputInit.open(frameOutputFile);

    while (getline(input, line)) {
        if (curLine == 0) {
            std::tie(isMoore, numStates) = parseRulesCuda(line, ruleset);   
            outputInit << n << std::endl;
        } else {
            //set voxel to on
            coords = tokenizeLine(line, spaceDelim);
            //TODO: should probably check for out of bounds input here and error
            //TODO: parse input cell's state when this is implemented and replace auto set state to 1
            inputData[stoi(coords[0]) + (n * stoi(coords[1])) + (n * n * stoi(coords[2]))] = 1;
            outputInit << coords[0] << " " << coords[1] << " " << coords[2] << std::endl;
        }
        curLine++;
    }
    input.close();
    outputInit.close();

    sideLength = n;
    return 0;
}
