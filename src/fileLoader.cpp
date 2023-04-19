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
#include "fileLoader.h"
#include "cube.h"

std::vector<std::string> cTokenizeLine(std::string &line, const char* delim) {
    std::vector<std::string> out;
    char *token = std::strtok(const_cast<char*>(line.c_str()), delim); 
    while (token != nullptr) 
    { 
        out.push_back(std::string(token)); 
        token = strtok(nullptr, delim); 
    } 
    return out;
}

std::tuple<bool, int> cParseRules(std::string line, bool*&ruleset) {
    const char* slashDelim = "/";
    const char* commaDelim = ",";
    const char* dashDelim = "-";

    ruleset = new bool[54];

    std::vector<std::string> rules = cTokenizeLine(line, slashDelim);
    std::string survival = rules[0];
    std::string birth = rules[1];
    int numStates = stoi(rules[2]);
    bool isMoore = (rules[3] == "M");
    for (int i = 0; i < 54; i++) {
        ruleset[i] = false;
    }
    //parse survival and birth rules
    std::vector<std::string> survivalSubsets = cTokenizeLine(survival, commaDelim);
    std::vector<std::string> birthSubsets = cTokenizeLine(birth, commaDelim);
    
    for (int i = 0; i < (int)birthSubsets.size(); i++) {
        if (birthSubsets[i].find('-') == std::string::npos) {    
            if (birthSubsets[i] != "x") {
                ruleset[stoi(birthSubsets[i])] = true;
            }
            
        } else {
            std::vector<std::string> range = cTokenizeLine(birthSubsets[i], dashDelim);
            for (int j = stoi(range[0]); j <= stoi(range[1]); j++) {
                ruleset[j] = true;
            }
        }
    }

    for (int i = 0; i < (int)survivalSubsets.size(); i++) {
        if (survivalSubsets[i].find('-') == std::string::npos) {  
            if (survivalSubsets[i] != "x") {
                ruleset[27 + stoi(survivalSubsets[i])] = true;
            }  
        } else {
            std::vector<std::string> range = cTokenizeLine(survivalSubsets[i], dashDelim);
            for (int j = 27 + stoi(range[0]); j <= 27 + stoi(range[1]); j++) {
                ruleset[j] = true;
            }
        }
    }      
    return std::make_tuple(isMoore, numStates);
}

int loadCubeInput(char* file, int& sideLength, bool*& ruleset, int& numStates, bool& isMoore, int*& inputData, int n, char* outputPath) {
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
            std::tie(isMoore, numStates) = cParseRules(line, ruleset);   
            outputInit << n << std::endl;
        } else {
            //set voxel to on
            coords = cTokenizeLine(line, spaceDelim);
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
