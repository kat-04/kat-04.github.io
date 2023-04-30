// Sequential version of 3D gol
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
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
#include <iomanip>
#include "timing.h"
#include "parse.h"
#include "vec3.h"
#include "gol-openmp-states.h"

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

// TODO: FIX TO MATCH OTHER OPENMP
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


bool is_alive(Vec3 v, uint8_t numStates, uint8_t *states) {
    // Index in terms of bits (as if the bits were a whole array) 
    uint64_t bit_index;
    // Bit and index in array of neighbor
    uint64_t index;
    uint8_t half;

    bit_index = v.x + v.y * n + v.z * n * n;
    index = bit_index / 2;
    half = bit_index % 2;

    // Get correct half of our state
    uint8_t mask = (uint8_t)0b1111;

    // Zero out other half, and only get the half we are interested in
    //
    // Doesn't matter if it's in the correct half -- aka we only want to
    // check if it's 0 and regardless of if it's in the 1st 4 bits or the 2nd
    // 4 bits, it will always be 0 if 0 and nonzero if nonzero
    uint8_t alive = (states[index] >> (4 * (1 - half))) & mask;

    return (alive == numStates - 1);
}


bool is_dead(Vec3 v, uint8_t *states) {
    // Index in terms of bits (as if the bits were a whole array) 
    uint64_t bit_index;
    // Bit and index in array of neighbor
    uint64_t index;
    uint8_t half;

    bit_index = v.x + v.y * n + v.z * n * n;
    index = bit_index / 2;
    half = bit_index % 2;

    // Get correct half of our state
    uint8_t mask = (uint8_t)0b1111;

    // Zero out other half, and only get the half we are interested in
    //
    // Doesn't matter if it's in the correct half -- aka we only want to
    // check if it's 0 and regardless of if it's in the 1st 4 bits or the 2nd
    // 4 bits, it will always be 0 if 0 and nonzero if nonzero
    uint8_t alive = (states[index] >> (4 * (1 - half))) & mask;

    return (alive == 0);
}


// Set initial state of voxel v
// State contains the voxel's state in the last 4 bits
void set_state(Vec3 v, uint8_t state, uint8_t *states) {
    // Index in terms of 4 bits (as if there are 2 states per uint8_t) 
    /* cout << "Setting voxel " << v << " with state " << unsigned(state) << endl; */
    uint64_t bit_index;
    // Half (first or second) and index in array of neighbor
    uint64_t index;
    uint8_t half;

    bit_index = v.x + v.y * n + v.z * n * n;
    index = bit_index / 2;
    half = bit_index % 2;

    state = state << (4 * (1 - half));
    uint8_t mask = (uint8_t)0b1111 << (4 * half);
    states[index] &= mask;
    states[index] |= state;
}



// Get state at current voxel position
uint8_t get_state(Vec3 v, uint8_t *states) {
    // Index in terms of 4 bits (as if there are 2 states per uint8_t) 
    uint64_t bit_index;
    // Half (first or second) and index in array of neighbor
    uint64_t index;
    uint8_t half;

    bit_index = v.x + v.y * n + v.z * n * n;
    index = bit_index / 2;
    half = bit_index % 2;

    uint8_t state = states[index] >> (4 * (1 - half));
    uint8_t mask = (uint8_t)0b1111;
    return state & mask;
}



// Decrement the state of the voxel (should currently be alive)
void decrement_state(Vec3 v, uint8_t *states) {
    // Index in terms of bits (as if the bits were a whole array) 
    uint64_t bit_index;
    // Bit and index in array of neighbor
    uint64_t index;
    uint8_t half;

    bit_index = v.x + v.y * n + v.z * n * n;
    index = bit_index / 2;
    half = bit_index % 2;

    uint8_t mask = (uint8_t)0b1111;
    uint8_t new_state = ((states[index] >> (4 * (1 - half))) & mask) - 1;
    new_state = new_state << (4 * (1 - half));
    mask = mask << 4 * half;
    states[index] &= mask;
    states[index] |= new_state;
}



// Update an alive cell as dead or dead as alive
void init_state(Vec3 v, int numStates, uint8_t *states) {
    // Index in terms of bits (as if the bits were a whole array) 
    uint64_t bit_index;
    // Bit and index in array of neighbor
    uint64_t index;
    uint8_t half;

    bit_index = v.x + v.y * n + v.z * n * n;
    index = bit_index / 2;
    half = bit_index % 2;

    uint8_t state = (numStates - 1) << 4 * (1 - half);
    uint8_t mask = (uint8_t)0b1111 << 4 * half;
    states[index] &= mask;
    states[index] |= state;
}



void increment_frame(map<int, bool> *rules, vector<Vec3> *curAlive, vector<Vec3> *newAlive, vector<Vec3> *decayVoxels, int numStates, uint8_t *states, uint8_t *states_tmp, get_neighbors_t get_neighbors) {

    /* unordered_map<Vec3, bool, hashVec3> to_toggle; */
    /* int num_threads, chunk; */
    vector<Vec3> temp_decay;

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
            /* #pragma omp critical */
            /* { */
            /* cout << voxel << ": " << hex << get_state(voxel, states) << dec << endl; */
            /* } */

            // If current voxel is dead/decaying, we want to skip over it
            // We don't want to search cells that are not neighbors of alive cells
            if (is_dead(voxel, states)) {
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
                if (is_alive(neighbor, numStates, states)) {
                    /* cout << neighbor << endl; */
                    num_alive++;
                }
            }
            // Check survival rule
            #pragma omp critical
            {
            if ((*rules)[27 + num_alive] && get_state(voxel, states) == numStates - 1) {
                // If it survives
                /* cout << voxel << " survives!" << endl; */
                (*newAlive).push_back(voxel);
            } else {
                /* cout << "Pushing to decay " << voxel << endl; */
                /* to_toggle.insert( {voxel, true} ); */
                (*decayVoxels).push_back(voxel);
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
                if (!is_dead(neighbor, states)) continue;

                // If neighbor has already been updated, do not update again
                /* if (find(to_toggle.begin(), to_toggle.end(), neighbor) != to_toggle.end()) continue; */
                if (!is_dead(neighbor, states_tmp)) continue;

                num_alive = 0;
                n_neighbors.clear();

                // Iterate through dead neighbors' neighbors
                get_neighbors(neighbor, &n_neighbors);
                for (auto n_neighbor : n_neighbors) {
                    if (is_alive(n_neighbor, numStates, states)) {
                        num_alive++;
                    }
                }

                // Check birth rule
                if ((*rules)[num_alive]) {
                    #pragma omp critical
                    {
                    if (is_dead(neighbor, states_tmp)) {
                        (*newAlive).push_back(neighbor);
                        /* cout << neighbor << " is born!" << endl; */
                        /* to_toggle.insert( {neighbor, true} ); */
                        init_state(neighbor, numStates, states_tmp);
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
            decrement_state(voxel, states_tmp);
            if (!is_dead(voxel, states_tmp)) {
                temp_decay.push_back(voxel);
                /* cout << "DECAY: " << voxel << endl; */
            }
            }
        }
    }

    /* (*decayVoxels).clear(); */
    temp_decay.swap(*decayVoxels);
    /* for (auto v : temp_decay) { */
    /*     cout << "Decay: " << v << endl; */
    /*     (*decayVoxels).push_back(v); */
    /* } */
    std::memcpy(states, states_tmp, (n * n * n + 1) / 2);
}


void parse_input(string outputDir, string inputFile, uint64_t sideLength, map<int, bool> *rules, bool *isMoore, int *numStates, vector<Vec3> *voxels, vector<Vec3> *decayVoxels, uint8_t *states, uint8_t *states_tmp)
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

            /* cout << "numStates: " << unsigned(v_state) << endl; */

            // Add voxel to map with "alive" state
            if (v_state == *numStates - 1) {
                (*voxels).push_back(Vec3(x, y, z));
            } else {
                (*decayVoxels).push_back(Vec3(x, y, z));
            }

            // Set state
            set_state(Vec3(x, y, z), v_state, states);
            set_state(Vec3(x, y, z), v_state, states_tmp);

            // Write it to output file
            outputInit << coords[0] << " " << coords[1] << " " << coords[2] << " " << unsigned(v_state) << endl;
        }
        curLine++;
    }
    input.close();
    outputInit.close();
}

void write_init(string frameOutputFile, uint64_t sideLength, int numStates) {
    ofstream output;
    output.open(frameOutputFile);
    output << sideLength << endl;
    /* cout << numStates << endl; */
    output << numStates << endl;
    output.close();
}

void write_output(string frameOutputFile, vector<Vec3> *voxels, vector<Vec3> *decayVoxels, uint8_t *states) {
    // Read and write to output file
    ofstream output;
    output.open(frameOutputFile);
    for (auto v : *decayVoxels) {
        // If voxel is alive
        output << v.x << " " << v.y << " " << v.z << " " << unsigned(get_state(v, states)) << endl;
    }
    for (auto v : *voxels) {
        // If voxel is alive
        output << v.x << " " << v.y << " " << v.z << " " << unsigned(get_state(v, states)) << endl;
    }
    output.close();
}


int golOpenMPStates(int argc, char** argv, string outputDir) {
    string inputFile = argv[1];
    int numFrames = stoi(argv[2]);
    uint64_t sideLength = stoi(argv[3]);

    n = sideLength;

    map<int, bool> rules;
    vector<Vec3> voxels, newVoxels, decayVoxels;
    uint8_t *states;
    uint8_t *states_tmp;
    states = (uint8_t *)calloc(sizeof(uint8_t), ((sideLength * sideLength * sideLength + 1) / 2));
    states_tmp = (uint8_t *)calloc(sizeof(uint8_t), ((sideLength * sideLength * sideLength + 1) / 2));
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
    parse_input(outputDir, inputFile, sideLength, &rules, &isMoore, &numStates, &voxels, &decayVoxels, states, states_tmp);

    // Sanity test
    /* uint8_t half = 0; */
    /* uint8_t state = (uint8_t)(0b10110000); */
    /* cout << "State: " << hex << +state << endl; */
    /* uint8_t mask = (uint8_t)0b1111 << 4 * half; */
    /* cout << "Mask: " << hex << +mask << endl; */
    /* uint8_t new_state = (numStates - 1) << 4 * (1 - half); */
    /* cout << "New state: " << hex << +new_state << endl; */
    /* state &= mask; */
    /* cout << "State: " << hex << +state << endl; */
    /* state |= new_state; */
    /* cout << "Result: " << hex << +state << endl; */

    string outputPath = outputDir + "/frame";
    string frameOutputFile;

    write_init(outputPath + "_init.txt", sideLength, numStates);

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
            increment_frame(&rules, &voxels, &newVoxels, &decayVoxels, numStates, states, states_tmp, &get_moore_neighbors);
        } else {
            increment_frame(&rules, &voxels, &newVoxels, &decayVoxels, numStates, states, states_tmp, &get_vn_neighbors);
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
        /* cout << "Size: " << newVoxels.size() << endl; */
        // Write to output
        write_output(frameOutputFile, &newVoxels, &decayVoxels, states);
        newVoxels.swap(voxels);
        newVoxels.clear();
    }
    printf("total simulation time: %.6fs\n", totalSimulationTime);

    free(states);
    return 0;
}
