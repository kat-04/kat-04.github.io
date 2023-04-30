// Sequential version of 3D gol
#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <map>
#include <cstring>
#include <vector>
#include <tuple>
#include <fstream>
#include <iostream> //for cout
#include "timing.h"
#include "gol-sequential.h"
#include "parse.h"

//RULE FORMAT: survival/birth/numStates/isMoore
//MAP FORMAT: keys 0-26 -> birth for that num neighbors,
//            keys 27-54 -> survival for that num neighbors

using namespace std; 

int golSequential(int argc, char** argv, string outputDir)
{
    //TODO: timing code (output to log and/or print)
    string inputFile = argv[1];
    int numFrames = stoi(argv[2]);
    uint64_t sideLength = stoi(argv[3]);
    string outputPath =  outputDir + "/frame";
    string frameOutputFile; 
    const char* spaceDelim = " ";
    

    //init cube structure
    vector<vector<vector<bool>>> cube(sideLength);
    vector<vector<vector<bool>>> newCube(sideLength);
    for (uint64_t i = 0; i < sideLength; i++) {
        cube[i] = vector<vector<bool>>(sideLength); 
        newCube[i] = vector<vector<bool>>(sideLength); 
        for (uint64_t j = 0; j < sideLength; j++) {
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
            
            //write frame_init
            ofstream frameInit;
            frameInit.open(string(outputPath + "_init.txt"));
            frameInit << sideLength << endl;
            frameInit << numStates << endl;
            frameInit.close();
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
        //output << sideLength << endl;
        //timer start
        double frameTime = 0.0;
        Timer frameTimer;
        //Loop through all voxels
        for (uint64_t x = 0; x < sideLength; x++) {
            for (uint64_t y = 0; y < sideLength; y++) {
                for (uint64_t z = 0; z < sideLength; z++) {
                    //get neighbor sum
                    frameTimer.reset();
                    numAlive = 0;
                    if (isMoore) {
                        for (uint64_t i = (x == 0) ? 0 : x - 1; i <= x + 1; i++) {
                            for (uint64_t j = (y == 0) ? 0 : y - 1; j <= y + 1; j++) {
                                for (uint64_t k = (z == 0) ? 0 : z - 1; k <= z + 1; k++) {
                                    if (i < sideLength && j < sideLength && k < sideLength) {
                                        if (!(x == i && y == j && z == k)) { //don't include self
                                            numAlive += cube[i][j][k] ? 1 : 0;
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        //von neumann neighborhood
                        numAlive += (x > 0 && x < sideLength && cube[x-1][y][z]) ? 1 : 0;
                        numAlive += (x + 1 < sideLength && cube[x+1][y][z]) ? 1 : 0;
                        numAlive += (y > 0 && y < sideLength && cube[x][y-1][z]) ? 1 : 0;
                        numAlive += (y + 1 < sideLength && cube[x][y+1][z]) ? 1 : 0;
                        numAlive += (z > 0 && z < sideLength && cube[x][y][z-1]) ? 1 : 0;
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

