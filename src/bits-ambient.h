#ifndef __BITS_AMBIENT_H
#define __BITS_AMBIENT_H

#include <stdint.h>
#include "vec4.h"


// --------------------------------------------------------------------
// 2 states traditional 3D GOL functions
// --------------------------------------------------------------------

// Returns if a given cell is alive
bool is_alive(Vec4 v, uint8_t *states, uint64_t n);

// Toggles bit on or off
void toggle_state(Vec4 v, uint8_t *states, uint64_t n);

#endif