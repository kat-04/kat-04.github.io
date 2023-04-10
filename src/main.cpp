#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <iostream>
#include <fstream>
#include <GL/glew.h> // include GLEW and new version of GL on Windows
#include <GLFW/glfw3.h> // GLFW helper library for window management
#include <iostream> //for cout

using namespace std; 

void printVoxel(int x, int y, int z, bool alive) {
    cout << x << ", " << y << ", " << z << ", alive: " << alive << endl;
}

vector<int> tokenizeVoxelCoords(string &line) {
    vector<int> out;
    const char* delim = " ";
    char *token = strtok(const_cast<char*>(line.c_str()), delim); 
    while (token != nullptr) 
    { 
        out.push_back(stoi(string(token))); 
        token = strtok(nullptr, delim); 
    } 
    return out;
}

//TODO: actually deal with rulesets, return true iff should be alive
bool voxelUpdateStatus(int numNeighbors) {
    return true;
}


int main(int argc, char** argv)
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

    //init cube structure
    bool cube[sideLength][sideLength][sideLength];
    memset(cube, 0, sizeof(cube));

    //parse input file
    fstream input;
    input.open(inputFile, ios::in);
    if (!input.is_open()) {
        cerr << "could not open input file: " << inputFile << endl;
        return 1;
    }
    string line;
    int curLine = 0;
    vector<int> coords;

    //write frame0 to be same status as input (updates start at frame1)
    frameOutputFile = outputPath + "0.txt";
    ofstream outputInit; 
    outputInit.open(frameOutputFile);

    while (getline(input, line)) {
        if (curLine == 0) {
            //TODO: parse ruleset
        } else {
            //set voxel to on
            coords = tokenizeVoxelCoords(line);
            cube[coords[0]][coords[1]][coords[2]] = true;
            outputInit << coords[0] << " " << coords[1] << " " << coords[2] << endl;
        }
        curLine++;
    }
    input.close();
    outputInit.close();

    int numAlive; 
    bool voxelStatus;

    //for each frame
    for (int f = 0; f < numFrames; f++) {
        frameOutputFile = outputPath + to_string(f+1) + ".txt";
        ofstream output; 
        output.open(frameOutputFile);
        
        //Loop through all voxels
        for (int x = 0; x < sideLength; x++) {
            for (int y = 0; y < sideLength; y++) {
                for (int z = 0; z < sideLength; z++) {
                    //get neighbor sum
                    numAlive = 0;
                    for (int i = x - 1; i <= x + 1; i++) {
                        for (int j = y - 1; j <= y + 1; j++) {
                            for (int k = z - 1; k <= z + 1; k++) {
                                if (i >= 0 && j >= 0 && k >= 0 && i < sideLength && j < sideLength && k < sideLength) {
                                    numAlive += cube[i][j][k] ? 1 : 0;
                                }
                            }
                        }
                    }

                    //update voxel based on rules
                    voxelStatus = voxelUpdateStatus(numAlive);   
                    cube[x][y][z] = voxelStatus;  

                    //output for frame if on
                    if (voxelStatus) {
                        output << x << " " << y << " " << z << endl;
                    }

                    //printVoxel(x, y, z, cube[x][y][z]);              
                }
            }
        }
        output.close();
    }
    return 0;
}
