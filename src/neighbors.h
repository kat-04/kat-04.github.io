#ifndef __NEIGHBORS_H
#define __NEIGHBORS_H

#include <vector>
#include "vec3.h"

typedef void (*get_neighbors_t) (Vec3, std::vector<Vec3>*, uint64_t n);

// Gets neighbors of cells in the Moore neighborhood
void get_moore_neighbors(Vec3 v, std::vector<Vec3> *neighbors, uint64_t n);

// Gets neighbors of cells in the Von Neumann neighborhood
void get_vn_neighbors(Vec3 v, std::vector<Vec3> *neighbors, uint64_t n);

#endif
