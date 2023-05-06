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
#include <iostream> 
#include <iomanip>
#include "timing.h"
#include "parse.h"
#include "neighbors.h"
#include "bits.h"
#include "gol-openmp-states.h"

using namespace std; 

uint64_t n_states = 0;

// performs one frame of game of life with states
void increment_frame_states(map<int, bool> *rules, vector<Vec3> *curAlive,
                     vector<Vec3> *newAlive, vector<Vec3> *decayVoxels, int numStates, 
                     uint8_t *states, uint8_t *states_tmp, get_neighbors_t get_neighbors) {

    vector<Vec3> temp_decay;

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

            // If current voxel is dead/decaying, we want to skip over it
            // We don't want to search cells that are not neighbors of alive cells
            if (is_dead(voxel, states, n_states)) {
                continue;
            }

            // -----------------------------------------------------------------
            // From here, we are working with alive cells and their neighbors
            // -----------------------------------------------------------------
            
            // Check current voxel
            neighbors.clear();
            get_neighbors(voxel, &neighbors, n_states);
            
            for (auto neighbor : neighbors) {
                if (is_alive_states(neighbor, numStates, states, n_states)) {
                    num_alive++;
                }
            }
            // Check survival rule
            #pragma omp critical
            {
            if ((*rules)[27 + num_alive] && get_state(voxel, states, n_states) == numStates - 1) {
                // If it survives
                (*newAlive).push_back(voxel);
            } else {
                // It is now decaying
                (*decayVoxels).push_back(voxel);
            }
            }

            // -----------------------------------------------------------------
            // From here, we are working with dead neighbors
            // -----------------------------------------------------------------

            // Check neighbors of voxel if they are not yet in newAlive
            for (auto neighbor : neighbors) {
                // If neighbor is alive, skip because we do not want to work
                // with alive neighbors as they will be covered in a later for loop
                if (!is_dead(neighbor, states, n_states)) continue;

                // If neighbor has already been updated, do not update again
                if (!is_dead(neighbor, states_tmp, n_states)) continue;

                num_alive = 0;
                n_neighbors.clear();

                // Iterate through dead neighbors' neighbors
                get_neighbors(neighbor, &n_neighbors, n_states);
                for (auto n_neighbor : n_neighbors) {
                    if (is_alive_states(n_neighbor, numStates, states, n_states)) {
                        num_alive++;
                    }
                }

                // Check birth rule
                if ((*rules)[num_alive]) {
                    #pragma omp critical
                    {
                    if (is_dead(neighbor, states_tmp, n_states)) {
                        (*newAlive).push_back(neighbor);
                        init_state(neighbor, numStates, states_tmp, n_states);
                    }
                    }
                }
            }
        }

        #pragma omp barrier
        #pragma omp for schedule(guided)
        for (auto voxel : (*decayVoxels)) {
            #pragma omp critical
            {
            decrement_state(voxel, states_tmp, n_states);
            if (!is_dead(voxel, states_tmp, n_states)) {
                temp_decay.push_back(voxel);
            }
            }
        }
    }

    // swap data for frame over
    temp_decay.swap(*decayVoxels);
    std::memcpy(states, states_tmp, (n_states * n_states * n_states + 1) / 2);
}

// do all input parsing
void parse_input_states(string outputDir, string inputFile, uint64_t sideLength,
                 map<int, bool> *rules, bool *isMoore, int *numStates, 
                 vector<Vec3> *voxels, vector<Vec3> *decayVoxels, uint8_t *states, 
                 uint8_t *states_tmp)
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
        } else {
            // Get voxel coordinates
            coords = tokenizeLine(line, spaceDelim);
            uint32_t x = stoi(coords[0]);
            uint32_t y = stoi(coords[1]);
            uint32_t z = stoi(coords[2]);
            uint8_t v_state = (uint8_t)stoi(coords[3]);

            // Check if and input voxel coordinates are out of side bounds
            if (x >= sideLength || y >= sideLength || z >= sideLength) {
                cerr << "Input coordinates out of bounds" << endl;
            }

            // Check if state is in state bounds
            if (v_state >= *numStates) {
                cerr << "Input state out of bounds" << endl;
            }

            // Add voxel to map with "alive" state
            if (v_state == *numStates - 1) {
                (*voxels).push_back(Vec3(x, y, z));
            } else {
                (*decayVoxels).push_back(Vec3(x, y, z));
            }

            // Set state
            set_state(Vec3(x, y, z), v_state, states, n_states);
            set_state(Vec3(x, y, z), v_state, states_tmp, n_states);

            // Write it to output file
            outputInit << coords[0] << " " << coords[1] << " " << coords[2] << " " << unsigned(v_state) << endl;
        }
        curLine++;
    }
    input.close();
    outputInit.close();
}

// write the frame_init file
void write_init_states(string frameOutputFile, uint64_t sideLength, int numStates) {
    ofstream output;
    output.open(frameOutputFile);
    output << sideLength << endl;
    output << numStates << endl;
    output.close();
}

// write the frame's output to a file
void write_output_states(string frameOutputFile, vector<Vec3> *voxels, vector<Vec3> *decayVoxels, uint8_t *states) {
    // Read and write to output file
    ofstream output;
    output.open(frameOutputFile);
    for (auto v : *decayVoxels) {
        // If voxel is alive
        output << v.x << " " << v.y << " " << v.z << " " << unsigned(get_state(v, states, n_states)) << endl;
    }
    for (auto v : *voxels) {
        // If voxel is alive
        output << v.x << " " << v.y << " " << v.z << " " << unsigned(get_state(v, states, n_states)) << endl;
    }
    output.close();
}

// performs the entire OpenMP game of life algorithm with states for a given number of frames
int golOpenMPStates(int argc, char** argv, string outputDir) {
    string inputFile = argv[1];
    int numFrames = stoi(argv[2]);
    uint64_t sideLength = stoi(argv[3]);
    n_states = sideLength;

    map<int, bool> rules;
    vector<Vec3> voxels, newVoxels, decayVoxels;
    uint8_t *states;
    uint8_t *states_tmp;
    states = (uint8_t *)calloc(sizeof(uint8_t), ((sideLength * sideLength * sideLength + 1) / 2));
    states_tmp = (uint8_t *)calloc(sizeof(uint8_t), ((sideLength * sideLength * sideLength + 1) / 2));
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
    parse_input_states(outputDir, inputFile, sideLength, &rules, &isMoore, &numStates, &voxels, &decayVoxels, states, states_tmp);


    string outputPath = outputDir + "/frame";
    string frameOutputFile;

    write_init_states(outputPath + "_init.txt", sideLength, numStates);

    double frameTime = 0.0;
    double totalSimulationTime = 0.0;
    Timer frameTimer;

    // for each frame
    for (int f = 0; f < numFrames; f++) {
        frameOutputFile = outputPath + to_string(f + 1) + ".txt";
        frameTimer.reset(); // start timer

        // Calculate next frame
        if (isMoore) {
            increment_frame_states(&rules, &voxels, &newVoxels, &decayVoxels, numStates, states, states_tmp, &get_moore_neighbors);
        } else {
            increment_frame_states(&rules, &voxels, &newVoxels, &decayVoxels, numStates, states, states_tmp, &get_vn_neighbors);
        }

        #pragma omp barrier
        frameTime = frameTimer.elapsed(); // end timer
        totalSimulationTime += frameTime;

        // Write to output
        write_output_states(frameOutputFile, &newVoxels, &decayVoxels, states);
        newVoxels.swap(voxels);
        newVoxels.clear();
    }
    printf("total simulation time: %.6fs\n", totalSimulationTime);

    free(states);
    return 0;
}
