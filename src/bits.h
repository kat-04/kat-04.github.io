#ifndef __BITS_H
#define __BITS_H

#include <stdint.h>
#include "vec3.h"


// --------------------------------------------------------------------
// 2 states traditional 3D GOL functions
// --------------------------------------------------------------------

// Returns if a given cell is alive
bool is_alive(Vec3 v, uint8_t *states, uint64_t n);

// Toggles bit on or off
void toggle_state(Vec3 v, uint8_t *states, uint64_t n);


// --------------------------------------------------------------------
// Multiple states 3D GOL functions
// --------------------------------------------------------------------

// Returns if a given cell is alive
bool is_alive_states(Vec3 v, uint8_t numStates, uint8_t *states, uint64_t n);

// Returns if a given cell is dead
bool is_dead(Vec3 v, uint8_t *states, uint64_t n);

// Sets state at the input value
void set_state(Vec3 v, uint8_t state, uint8_t *states, uint64_t n);

// Gets the state at a voxel coordinate
uint8_t get_state(Vec3 v, uint8_t *states, uint64_t n);

// Decrements the state by 1 at the given voxel
void decrement_state(Vec3 v, uint8_t *states, uint64_t n);

// Initializes the state to be num_states - 1 (when just born)
void init_state(Vec3 v, uint64_t numStates, uint8_t *states, uint64_t n);

#endif
