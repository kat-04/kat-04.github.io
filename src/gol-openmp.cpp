// Sequential version of 3D gol
#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <iostream>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <cstring>
#include <tuple>
#include <fstream>
#include <bitset>
#include <omp.h>
#include <iostream> //for cout
#include "timing.h"
#include "parse.h"
#include "bits.h"
#include "neighbors.h"
#include "gol-openmp.h"

using namespace std; 

uint64_t n = 0;

// performs one frame of game of life
void increment_frame(map<int, bool> *rules, vector<Vec3> *curAlive, vector<Vec3> *newAlive, 
                     uint8_t *states, uint8_t *states_tmp, get_neighbors_t get_neighbors) {

    #pragma omp parallel
    {
        // Neighbor alive count
        uint8_t num_alive = 0;

        // Iterate through all the alive voxels
        vector<Vec3> neighbors;
        vector<Vec3> n_neighbors;

        #pragma omp for nowait schedule(guided) private(neighbors, n_neighbors, num_alive)
        for (auto voxel : *curAlive) {

            num_alive = 0;

            // If current voxel is dead, we want to skip over it
            // We don't want to search cells that are not neighbors of alive cells
            if (!is_alive(voxel, states, n)) {
                continue;
            }

            // -----------------------------------------------------------------
            // From here, we are working with alive cells and their neighbors
            // -----------------------------------------------------------------
            
            // Check current voxel
            neighbors.clear();
            get_neighbors(voxel, &neighbors, n);
            
            for (auto neighbor : neighbors) {
                if (is_alive(neighbor, states, n)) {
                    num_alive++;
                }
            }
            // Check survival rule
            #pragma omp critical
            {
            if ((*rules)[27 + num_alive]) {
                // If it survives
                (*newAlive).push_back(voxel);
            } else {
                toggle_state(voxel, states_tmp, n);
            }
            }

            // -----------------------------------------------------------------
            // From here, we are working with dead neighbors
            // -----------------------------------------------------------------

            // Check neighbors of voxel if they are not yet in newAlive
            for (auto neighbor : neighbors) {
                // If neighbor is alive, skip because we do not want to work
                // with alive neighbors as they will be covered in a later for loop
                if (is_alive(neighbor, states, n)) continue;

                // If neighbor has already been updated, do not update again
                if (is_alive(neighbor, states_tmp, n)) continue;

                num_alive = 0;
                n_neighbors.clear();

                // Iterate through dead neighbors' neighbors
                get_neighbors(neighbor, &n_neighbors, n);
                for (auto n_neighbor : n_neighbors) {
                    if (is_alive(n_neighbor, states, n)) {
                        num_alive++;
                    }
                }

                // Check birth rule
                if ((*rules)[num_alive]) {
                    #pragma omp critical
                    {
                    if (!is_alive(neighbor, states_tmp, n)) {
                        (*newAlive).push_back(neighbor);
                        toggle_state(neighbor, states_tmp, n);
                    }
                    }
                }
            }
        }
        #pragma omp barrier
    }
}

// do all input parsing
void parse_input(string outputDir, string inputFile, uint64_t sideLength, map<int, bool> *rules, 
                 bool *isMoore, int *numStates, vector<Vec3> *voxels, uint8_t *states, uint8_t *states_tmp)
{
    // Open input file
    fstream input;
    input.open(inputFile, ios::in);
    if (!input.is_open()) {
        cerr << "could not open input file: " << inputFile << endl;
    }

    // Initialize variables for parsing
    string line;
    int curLine = 0;
    vector<string> coords;

    // Initialize frame 0 output path
    string outputPath = outputDir + "/frame";
    string frameOutputFile = outputPath + "0.txt";
    const char* spaceDelim = " ";
    ofstream outputInit; 
    outputInit.open(frameOutputFile);

    // Read lines of input file
    while (getline(input, line)) {
        if (curLine == 0) {
            // Read in initial rule set and store it in variables
            tie((*rules), (*isMoore), (*numStates)) = parseRules(line);

            //write frame_init
            ofstream frameInit;
            frameInit.open(string(outputPath + "_init.txt"));
            frameInit << sideLength << endl;
            frameInit << (*numStates) << endl;
            frameInit.close();
        } else {
            // Get voxel coordinates
            coords = tokenizeLine(line, spaceDelim);
            uint32_t x = stoi(coords[0]);
            uint32_t y = stoi(coords[1]);
            uint32_t z = stoi(coords[2]);

            // Check if and input voxel coordinates are out of side bounds
            if (x >= sideLength || y >= sideLength || z >= sideLength) {
                cerr << "Input coordinates out of bounds" << std::endl;
            }

            // Add voxel to map with "alive" state
            (*voxels).push_back(Vec3(x, y, z));

            // Toggle state to 1
            toggle_state(Vec3(x, y, z), states, n);
            toggle_state(Vec3(x, y, z), states_tmp, n);

            // Write it to output file
            outputInit << coords[0] << " " << coords[1] << " " << coords[2] << endl;
        }
        curLine++;
    }
    input.close();
    outputInit.close();
}

// write the frame's output to a file
void write_output(string frameOutputFile, vector<Vec3> *voxels, uint8_t *states) {
    // Read and write to output file
    ofstream output;
    output.open(frameOutputFile);
    for (auto v : *voxels) {
        // If voxel is alive
        if (is_alive(v, states, n)) {
            output << v.x << " " << v.y << " " << v.z << endl;
        }
    }
    output.close();
}

// performs the entire OpenMP game of life algorithm for a given number of frames
int golOpenMP(int argc, char** argv, string outputDir) {
    string inputFile = argv[1];
    int numFrames = stoi(argv[2]);
    uint64_t sideLength = stoi(argv[3]);

    n = sideLength;

    map<int, bool> rules;
    vector<Vec3> voxels, newVoxels;
    uint8_t *states;
    uint8_t *states_tmp;
    states = (uint8_t *)calloc(sizeof(uint8_t), ((sideLength * sideLength * sideLength + 7) / 8));
    states_tmp = (uint8_t *)calloc(sizeof(uint8_t), ((sideLength * sideLength * sideLength + 7) / 8));
    if (!states) {
        cerr << "Malloc states failed" << endl;
        return 1;
    }
    if (!states_tmp) {
        cerr << "Malloc states_tmp failed" << endl;
        return 1;
    }

    bool isMoore;
    int numStates;
    parse_input(outputDir, inputFile, sideLength, &rules, &isMoore, &numStates, &voxels, states, states_tmp);

    string outputPath = outputDir + "/frame";
    string frameOutputFile;

    double frameTime = 0.0;
    double totalSimulationTime = 0.0;
    Timer frameTimer;

    // main frame loop
    for (int f = 0; f < numFrames; f++) {
        frameOutputFile = outputPath + to_string(f + 1) + ".txt";
        frameTimer.reset(); // start timer

        // Calculate next frame
        if (isMoore) {
            increment_frame(&rules, &voxels, &newVoxels, states, states_tmp, &get_moore_neighbors);
        } else {
            increment_frame(&rules, &voxels, &newVoxels, states, states_tmp, &get_vn_neighbors);
        }

        frameTime = frameTimer.elapsed(); // end timer
        totalSimulationTime += frameTime;
        
        #pragma omp single
        {
        std::memcpy(states, states_tmp, (n * n * n + 7) / 8);
        }

        // Write to output
        write_output(frameOutputFile, &newVoxels, states);
        newVoxels.swap(voxels);
        newVoxels.clear();
    }
    printf("total simulation time: %.6fs\n", totalSimulationTime);

    free(states);
    return 0;
}
