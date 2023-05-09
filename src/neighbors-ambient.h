#ifndef __NEIGHBORS_AMBIENT_H
#define __NEIGHBORS_AMBIENT_H

#include <vector>
#include "vec4.h"

typedef void (*get_neighbors_t) (Vec4, std::vector<Vec4>*, uint64_t n);

// Gets neighbors of cells in the Moore neighborhood
void get_moore_neighbors(Vec4 v, std::vector<Vec4> *neighbors, uint64_t n);

// Gets neighbors of cells in the Von Neumann neighborhood
void get_vn_neighbors(Vec4 v, std::vector<Vec4> *neighbors, uint64_t n);

#endif
