// Sequential version of 3D gol
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
#include "omp.h"
// #include <GL/glew.h> // include GLEW and new version of GL on Windows
// #include <GLFW/glfw3.h> // GLFW helper library for window management
#include <iostream> //for cout
#include "timing.h"

//RULE FORMAT: survival/birth/numStates/isMoore
//MAP FORMAT: keys 0-26 -> birth for that num neighbors,
//            keys 27-54 -> survival for that num neighbors

using namespace std; 

void printVoxel(int x, int y, int z, bool alive) {
    cout << x << ", " << y << ", " << z << ", alive: " << alive << endl;
}

vector<string> tokenizeLine(string &line, const char* delim) {
    vector<string> out;
    char *token = std::strtok(const_cast<char*>(line.c_str()), delim); 
    while (token != nullptr) 
    { 
        out.push_back(string(token)); 
        token = strtok(nullptr, delim); 
    } 
    return out;
}

tuple<map<int, bool>, bool, int> parseRules(string line) {
    const char* slashDelim = "/";
    const char* commaDelim = ",";
    const char* dashDelim = "-";
    map<int, bool> ruleMap;

    vector<string> rules = tokenizeLine(line, slashDelim);
    string survival = rules[0];
    string birth = rules[1];
    int numStates = stoi(rules[2]);
    bool isMoore = (rules[3] == "M");

    //init map
    for (int i = 0; i < 54; i++) {
        ruleMap[i] = false;
    }

    //parse survival and birth rules
    vector<string> survivalSubsets = tokenizeLine(survival, commaDelim);
    vector<string> birthSubsets = tokenizeLine(birth, commaDelim);
    for (int i = 0; i < (int)birthSubsets.size(); i++) {
        if (birthSubsets[i].find('-') == string::npos) {    
            ruleMap[stoi(birthSubsets[i])] = true;
        } else {
            vector<string> range = tokenizeLine(birthSubsets[i], dashDelim);
            for (int j = stoi(range[0]); j <= stoi(range[1]); j++) {
                ruleMap[j] = true;
            }
        }
    }

    for (int i = 0; i < (int)survivalSubsets.size(); i++) {
        if (survivalSubsets[i].find('-') == string::npos) {    
            ruleMap[27 + stoi(survivalSubsets[i])] = true;
        } else {
            vector<string> range = tokenizeLine(survivalSubsets[i], dashDelim);
            for (int j = 27 + stoi(range[0]); j <= 27 + stoi(range[1]); j++) {
                ruleMap[j] = true;
            }
        }
    }  

    return make_tuple(ruleMap, isMoore, numStates);
}


int golParallel(int argc, char** argv)
{
    //TODO: timing code (output to log and/or print)

    //parse args
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " input_file number_of_frames side_length" << endl;
        return 1;
    }

    string inputFile = argv[1];
    int numFrames = stoi(argv[2]);
    int sideLength = stoi(argv[3]);
    string outputPath = "./output-files/frame";
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
            tie(ruleMap, isMoore, numStates) = parseRules(line);
            
            outputInit << sideLength << endl;
        } else {
            //set voxel to on
            coords = tokenizeLine(line, spaceDelim);
            //TODO: should probably check for out of bounds input here and error
            cube[stoi(coords[0])][stoi(coords[1])][stoi(coords[2])] = true;
            outputInit << coords[0] << " " << coords[1] << " " << coords[2] << endl;
        }
        curLine++;
    }
    input.close();
    outputInit.close();

    int numAlive; 
    bool voxelStatus;

    double totalSimulationTime = 0.0;

    //for each frame
    for (int f = 0; f < numFrames; f++) {
        frameOutputFile = outputPath + to_string(f+1) + ".txt";
        ofstream output; 
        output.open(frameOutputFile);
        output << sideLength << endl;
        //timer start
        double frameTime = 0.0;
        Timer frameTimer;
        //Loop through all voxels
        for (int x = 0; x < sideLength; x++) {
            for (int y = 0; y < sideLength; y++) {
                for (int z = 0; z < sideLength; z++) {
                    //get neighbor sum
                    frameTimer.reset();
                    numAlive = 0;
                    if (isMoore) {
                        for (int i = x - 1; i <= x + 1; i++) {
                            for (int j = y - 1; j <= y + 1; j++) {
                                for (int k = z - 1; k <= z + 1; k++) {
                                    if (i >= 0 && j >= 0 && k >= 0 && i < sideLength && j < sideLength && k < sideLength) {
                                        if (!(x == i && y == j && z == k)) { //don't include self
                                            numAlive += cube[i][j][k] ? 1 : 0;
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        //von neumann neighborhood
                        numAlive += (x - 1 >= 0 && cube[x-1][y][z]) ? 1 : 0;
                        numAlive += (x + 1 < sideLength && cube[x+1][y][z]) ? 1 : 0;
                        numAlive += (y - 1 >= 0 && cube[x][y-1][z]) ? 1 : 0;
                        numAlive += (y + 1 < sideLength && cube[x][y+1][z]) ? 1 : 0;
                        numAlive += (z - 1 >= 0 && cube[x][y][z-1]) ? 1 : 0;
                        numAlive += (z + 1 < sideLength && cube[x][y][z+1]) ? 1 : 0;
                    }

                    //update voxel based on rules
                    voxelStatus = ruleMap[(cube[x][y][z] ? 27 : 0) + numAlive];
                    newCube[x][y][z] = voxelStatus;  
                    //output for frame if on
                    frameTime += frameTimer.elapsed();
                    if (voxelStatus) {
                        output << x << " " << y << " " << z << endl;
                    }
                    
                    //printVoxel(x, y, z, cube[x][y][z]);              
                }
            }
        }
        cube.swap(newCube);
        totalSimulationTime += frameTime;
        output.close();
    }
    printf("total simulation time: %.6fs\n", totalSimulationTime);
    return 0;
}
