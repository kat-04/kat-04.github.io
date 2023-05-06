#include <stdint.h>
#include "bits.h"

// return whether the voxel v is currently alive
bool is_alive(Vec3 v, uint8_t *states, uint64_t n) {
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



// return whether the voxel v is currently alive (not including decaying states)
bool is_alive_states(Vec3 v, uint8_t numStates, uint8_t *states, uint64_t n) {
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


// return whether the voxel v is currently dead (not including decaying states)
bool is_dead(Vec3 v, uint8_t *states, uint64_t n) {
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



// Update an alive cell as dead or dead as alive
void toggle_state(Vec3 v, uint8_t *states, uint64_t n) {
    uint64_t bit_index;
    uint64_t index;
    uint8_t bit;

    bit_index = v.x + v.y * n + v.z * n * n;
    index = bit_index / 8;
    bit = bit_index % 8;

    uint8_t mask = 1 << (7 - bit);
    states[index] ^= mask;
}



// Set initial state of voxel v. State contains the voxel's state in the last 4 bits
void set_state(Vec3 v, uint8_t state, uint8_t *states, uint64_t n) {
    // Index in terms of 4 bits (as if there are 2 states per uint8_t) 
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
uint8_t get_state(Vec3 v, uint8_t *states, uint64_t n) {
    uint64_t bit_index;
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
void decrement_state(Vec3 v, uint8_t *states, uint64_t n) {
    uint64_t bit_index;
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
void init_state(Vec3 v, uint64_t numStates, uint8_t *states, uint64_t n) {
    uint64_t bit_index;
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
