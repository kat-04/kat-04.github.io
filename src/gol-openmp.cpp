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
#include <vector>
#include <tuple>
#include <fstream>
#include <bitset>
#include <omp.h>
#include <iostream> //for cout
#include "timing.h"
#include "parse.h"
#include "gol-openmp.h"

//RULE FORMAT: survival/birth/numStates/isMoore
//MAP FORMAT: keys 0-26 -> birth for that num neighbors,
//            keys 27-54 -> survival for that num neighbors

// IDEA:
// have a map of voxels and state
// start with only alive ones and add in increment_
// set states to alive/dead by iterating through like a wavefront -- make the for loop inside parallel but for loop must be sequential O(n)
// NEW THOUGHT: instead of wavefront, can just check all neighbors for each alive cell -- map deals with duplicates and overwriting
// Have each thread read from map and append alive voxels to their own private vectors and atomically write to output
// each thread can query map based on (x, y, z) so assign blocks per thread

using namespace std; 

uint64_t n = 0;

class hashVec3 {
public:
    size_t operator()(const Vec3& v) const
    {
        /* std::cout << "n: " << n << std::endl; */
        return v.x + v.y * n + v.z * n * n;
    }
};

typedef void (*get_neighbors_t) (Vec3, vector<Vec3>*);

void get_moore_neighbors(Vec3 v, vector<Vec3> *neighbors) {
    for (uint32_t x_n = (v.x == 0) ? 0 : v.x - 1; x_n < v.x + 2; x_n++) {
        for (uint32_t y_n = (v.y == 0) ? 0 : v.y - 1; y_n < v.y + 2; y_n++) {
            for (uint32_t z_n = (v.z == 0) ? 0 : v.z - 1; z_n < v.z + 2; z_n++) {
                // If neighbor out of bounds of cube size
                if (x_n >= n || y_n >= n || z_n >= n) {
                    continue;
                }
                // Don't include itself
                if (x_n == v.x && y_n == v.y && z_n == v.z) {
                    continue;
                }
                (*neighbors).push_back(Vec3(x_n, y_n, z_n));
            }
        }
    }
}

void get_vn_neighbors(Vec3 v, vector<Vec3> *neighbors) {
    for (uint32_t x_n = (v.x == 0) ? 0 : v.x - 1; x_n < v.x + 2; x_n++) {
        for (uint32_t y_n = (v.y == 0) ? 0 : v.y - 1; y_n < v.y + 2; y_n++) {
            for (uint32_t z_n = (v.z == 0) ? 0 : v.z - 1; z_n < v.z + 2; z_n++) {
                // If neighbor out of bounds of cube size
                if (x_n >= n || y_n >= n || z_n >= n) {
                    continue;
                }
                // Don't include itself
                if (x_n == v.x && y_n == v.y && z_n == v.z) {
                    continue;
                }
                if (((v.x + v.y + v.z) % 2) != ((x_n + y_n + z_n) % 2)) continue;
                (*neighbors).push_back(Vec3(x_n, y_n, z_n));
            }
        }
    }
}


bool is_alive(Vec3 v, uint8_t *states) {
    // Index in terms of bits (as if the bits were a whole array) 
    uint64_t bit_index;
    // Bit and index in array of neighbor
    uint64_t index;
    uint8_t bit;

    bit_index = v.x + v.y * n + v.z * n * n;
    index = bit_index / 8;
    bit = bit_index % 8;

    uint8_t mask = 1;
    uint8_t alive = (states[index] >> (7 - bit)) & mask;

    return (alive == 1);
}


// Update an alive cell as dead or dead as alive
void toggle_state(Vec3 v, uint8_t *states) {
    // Index in terms of bits (as if the bits were a whole array) 
    uint64_t bit_index;
    // Bit and index in array of neighbor
    uint64_t index;
    uint8_t bit;

    bit_index = v.x + v.y * n + v.z * n * n;
    index = bit_index / 8;
    bit = bit_index % 8;

    uint8_t mask = 1 << (7 - bit);
    states[index] ^= mask;
}


void increment_frame(map<int, bool> *rules, vector<Vec3> *curAlive, vector<Vec3> *newAlive, uint8_t *states, uint8_t *states_tmp, get_neighbors_t get_neighbors) {

    /* unordered_map<Vec3, bool, hashVec3> to_toggle; */
    /* int num_threads, chunk; */

    #pragma omp parallel
    {
        /* num_threads = omp_get_num_threads(); */
        /* cout << "Num threads: " << num_threads << endl; */
        /* chunk = ((*curAlive).size() + num_threads - 1) / num_threads; */

        // Neighbor alive count
        uint8_t num_alive = 0;

        // Iterate through all the alive voxels
        vector<Vec3> neighbors;
        vector<Vec3> n_neighbors;

        /* #pragma omp parallel for shared(to_toggle, curAlive, newAlive, chunk) private(neighbors, n_neighbors, num_alive) schedule(static, chunk) */
        #pragma omp for nowait schedule(guided) private(neighbors, n_neighbors, num_alive)
        for (auto voxel : *curAlive) {
            /* cout << "Hellooo" << endl; */

            num_alive = 0;

            // If current voxel is dead, we want to skip over it
            // We don't want to search cells that are not neighbors of alive cells
            if (!is_alive(voxel, states)) {
                continue;
            }

            // -----------------------------------------------------------------
            // From here, we are working with alive cells and their neighbors
            // -----------------------------------------------------------------
            
            // Check current voxel
            neighbors.clear();
            get_neighbors(voxel, &neighbors);
            
            /* cout << "Voxel: " << voxel << endl; */
            /* cout << "Neighbors: " << endl; */
            for (auto neighbor : neighbors) {
                if (is_alive(neighbor, states)) {
                    /* cout << neighbor << endl; */
                    num_alive++;
                }
            }
            // Check survival rule
            #pragma omp critical
            {
            if ((*rules)[27 + num_alive]) {
                // If it survives
                /* cout << voxel << " survives!" << endl; */
                (*newAlive).push_back(voxel);
            } else {
                /* to_toggle.insert( {voxel, true} ); */
                toggle_state(voxel, states_tmp);
            }
            }

            // -----------------------------------------------------------------
            // From here, we are working with dead neighbors
            // -----------------------------------------------------------------

            // Check neighbors of voxel if they are not yet in newAlive
            /* #pragma omp for */
            for (auto neighbor : neighbors) {
                // If neighbor is alive, skip because we do not want to work
                // with alive neighbors as they will be covered in a later for loop
                if (is_alive(neighbor, states)) continue;

                // If neighbor has already been updated, do not update again
                /* if (find(to_toggle.begin(), to_toggle.end(), neighbor) != to_toggle.end()) continue; */
                if (is_alive(neighbor, states_tmp)) continue;
                
                /* try { */
                /*     to_toggle.at(neighbor); */
                /*     continue; */
                /* } */
                /* catch(const out_of_range &e) { */
                /* } */

                num_alive = 0;
                n_neighbors.clear();

                // Iterate through dead neighbors' neighbors
                get_neighbors(neighbor, &n_neighbors);
                for (auto n_neighbor : n_neighbors) {
                    if (is_alive(n_neighbor, states)) {
                        num_alive++;
                    }
                }

                // Check birth rule
                if ((*rules)[num_alive]) {
                    #pragma omp critical
                    {
                    if (!is_alive(neighbor, states_tmp)) {
                        (*newAlive).push_back(neighbor);
                        /* cout << neighbor << " is born!" << endl; */
                        /* to_toggle.insert( {neighbor, true} ); */
                        toggle_state(neighbor, states_tmp);
                    }
                    }
                }
            }
        }
        #pragma omp barrier
        std::memcpy(states, states_tmp, (n * n * n + 7) / 8);
    }
}


void parse_input(string inputFile, uint64_t sideLength, map<int, bool> *rules, bool *isMoore, int *numStates, vector<Vec3> *voxels, uint8_t *states, uint8_t *states_tmp)
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
    string outputPath = "./output-files/frame";
    string frameOutputFile = outputPath + "0.txt";
    const char* spaceDelim = " ";
    ofstream outputInit; 
    outputInit.open(frameOutputFile);

    // Read lines of input file
    while (getline(input, line)) {
        if (curLine == 0) {
            // Read in initial rule set and store it in variables
            tie((*rules), (*isMoore), (*numStates)) = parseRules(line);
            outputInit << sideLength << endl;
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
            /* uint64_t linearIndex = x + y * n + z * n * n; */
            /* uint8_t stateIndex = linearIndex / 8; */
            /* uint8_t bit = linearIndex % 8; */
            /* uint8_t mask = 1 << (7 - bit); */
            /* states[stateIndex] |= mask; */
            toggle_state(Vec3(x, y, z), states);
            toggle_state(Vec3(x, y, z), states_tmp);

            // Write it to output file
            outputInit << coords[0] << " " << coords[1] << " " << coords[2] << endl;
        }
        curLine++;
    }
    input.close();
    outputInit.close();
}


void write_output(string frameOutputFile, uint64_t sideLength, vector<Vec3> *voxels, uint8_t *states) {
    // Read and write to output file
    ofstream output;
    output.open(frameOutputFile);
    output << sideLength << endl;
    for (auto v : *voxels) {
        // If voxel is alive
        if (is_alive(v, states)) {
            output << v.x << " " << v.y << " " << v.z << endl;
        }
    }
    output.close();
}


int golOpenMP(int argc, char** argv) {
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
    /* fill_n(states, (sideLength * sideLength * sideLength + 7) / 8, 0); */
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
    parse_input(inputFile, sideLength, &rules, &isMoore, &numStates, &voxels, states, states_tmp);

    string outputPath = "./output-files/frame";
    string frameOutputFile;

    double frameTime = 0.0;
    double totalSimulationTime = 0.0;
    Timer frameTimer;

    /* cout << "isMoore: " << isMoore << endl; */
    /* cout << "numStates: " << numStates << endl; */
    /* cout << "numVoxels: " << voxels.size() << endl; */

    for (int f = 0; f < numFrames; f++) {
        frameOutputFile = outputPath + to_string(f + 1) + ".txt";
        frameTimer.reset();
        // Calculate next frame
        
        /* cout << "voxels frame " << f << std::endl; */
        /* for (int i = 0; i < voxels.size(); i++) { */
        /*     cout << voxels[i] << " alive? " << is_alive(voxels[i], states) << endl; */
        /* } */

        /* cout << "Hellooooo!" << endl; */
        if (isMoore) {
            increment_frame(&rules, &voxels, &newVoxels, states, states_tmp, &get_moore_neighbors);
        } else {
            increment_frame(&rules, &voxels, &newVoxels, states, states_tmp, &get_vn_neighbors);
        }

        /* for (int i = 0; i < sizeof(states) / sizeof(uint8_t); i++) { */
        /*     cout << bitset<8>(states[i]) << "_"; */
        /* } */
        /* cout << endl; */

        /* cout << "newVoxels frame " << f + 1 << std::endl; */
        /* for (int i = 0; i < newVoxels.size(); i++) { */
        /*     cout << newVoxels[i] << endl; */
        /* } */

        #pragma omp barrier
        frameTime = frameTimer.elapsed();
        totalSimulationTime += frameTime;
        /* printf("frame %d time: %.6fs\n", f + 1, frameTime); */
        // Write to output
        write_output(frameOutputFile, sideLength, &newVoxels, states);
        newVoxels.swap(voxels);
        newVoxels.clear();
    }
    printf("total simulation time: %.6fs\n", totalSimulationTime);

    free(states);
    return 0;
}
