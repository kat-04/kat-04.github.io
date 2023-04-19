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

vector<string> cTokenizeLine(string &line, const char* delim) {
    vector<string> out;
    char *token = std::strtok(const_cast<char*>(line.c_str()), delim); 
    while (token != nullptr) 
    { 
        out.push_back(string(token)); 
        token = strtok(nullptr, delim); 
    } 
    return out;
}

tuple<map<int, bool>, bool, int> cParseRules(string line) {
    const char* slashDelim = "/";
    const char* commaDelim = ",";
    const char* dashDelim = "-";
    map<int, bool> ruleMap;

    vector<string> rules = cTokenizeLine(line, slashDelim);
    string survival = rules[0];
    string birth = rules[1];
    int numStates = stoi(rules[2]);
    bool isMoore = (rules[3] == "M");

    //init map
    for (int i = 0; i < 54; i++) {
        ruleMap[i] = false;
    }

    //parse survival and birth rules
    vector<string> survivalSubsets = cTokenizeLine(survival, commaDelim);
    vector<string> birthSubsets = cTokenizeLine(birth, commaDelim);
    
    for (int i = 0; i < (int)birthSubsets.size(); i++) {
        if (birthSubsets[i].find('-') == string::npos) {    
            if (birthSubsets[i] != "x") {
                ruleMap[stoi(birthSubsets[i])] = true;
            }
            
        } else {
            vector<string> range = cTokenizeLine(birthSubsets[i], dashDelim);
            for (int j = stoi(range[0]); j <= stoi(range[1]); j++) {
                ruleMap[j] = true;
            }
        }
    }

    for (int i = 0; i < (int)survivalSubsets.size(); i++) {
        if (survivalSubsets[i].find('-') == string::npos) {  
            if (survivalSubsets[i] != "x") {
                ruleMap[27 + stoi(survivalSubsets[i])] = true;
            }  
        } else {
            vector<string> range = cTokenizeLine(survivalSubsets[i], dashDelim);
            for (int j = 27 + stoi(range[0]); j <= 27 + stoi(range[1]); j++) {
                ruleMap[j] = true;
            }
        }
    }  

    return make_tuple(ruleMap, isMoore, numStates);
}

//TODO: move to iterationLoader separate file so can include in cuda file
// void loadIterationInput(char* file, int*& sideLength, int*& ruleset, int*& cubeData) {

// }

int gol_cuda(int argc, char** argv, std::string fileDir) {
    
    //parse args (argc has been checked in main)
    string inputFile = argv[1];
    int numFrames = stoi(argv[2]);
    int sideLength = stoi(argv[3]);
    string outputPath =  fileDir + "/frame";
    string frameOutputFile; 
    const char* spaceDelim = " ";

    //init cube structure
    vector<vector<vector<bool>>> cube(sideLength);
    vector<vector<vector<bool>>> newCube(sideLength);
    for (int i = 0; i < sideLength; i++) {
        cube[i] = vector<vector<bool>>(sideLength); 
        newCube[i] = vector<vector<bool>>(sideLength); 
        for (int j = 0; j < sideLength; j++) {
            cube[i][j] = vector<bool>(sideLength);
            newCube[i][j] = vector<bool>(sideLength);
        }
    }

    //parse input file
    fstream input;
    input.open(inputFile, ios::in);
    if (!input.is_open()) {
        cerr << "could not open input file: " << inputFile << endl;
        return 1;
    }
    string line;
    int curLine = 0;
    vector<string> coords;

    //write frame0 to be same status as input (updates start at frame1)
    frameOutputFile = outputPath + "0.txt";
    ofstream outputInit; 
    outputInit.open(frameOutputFile);
    map<int, bool> ruleMap;
    int numStates;
    bool isMoore = false;
    while (getline(input, line)) {
        if (curLine == 0) {
            tie(ruleMap, isMoore, numStates) = cParseRules(line);
            
            outputInit << sideLength << endl;
        } else {
            //set voxel to on
            coords = cTokenizeLine(line, spaceDelim);
            //TODO: should probably check for out of bounds input here and error
            cube[stoi(coords[0])][stoi(coords[1])][stoi(coords[2])] = true;
            outputInit << coords[0] << " " << coords[1] << " " << coords[2] << endl;
        }
        curLine++;
    }
    input.close();
    outputInit.close();

    GolCuda* golCuda;
    golCuda = new GolCuda();

    golCuda->allocOutputCube(sideLength);
    golCuda->loadInput(const_cast<char*>(inputFile.c_str()), sideLength);
    golCuda->setup();
    //golCuda->clearOutputCube();
    //time start
    golCuda->doIteration();
    //time end
    //golCuda->getCube();
    //write to output file

    return 0;
}