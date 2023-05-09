#include <stdint.h>
#include "bits-ambient.h"

// return whether the voxel v is currently alive
bool is_alive(Vec4 v, uint8_t *states, uint64_t n) {
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
void toggle_state(Vec4 v, uint8_t *states, uint64_t n) {
    uint64_t bit_index;
    uint64_t index;
    uint8_t bit;

    bit_index = v.x + v.y * n + v.z * n * n;
    index = bit_index / 8;
    bit = bit_index % 8;

    uint8_t mask = 1 << (7 - bit);
    states[index] ^= mask;
}


