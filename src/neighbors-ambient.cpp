#include "neighbors-ambient.h"
#include "vec4.h"

using namespace std; 

// get all neighbors according to Moore neighborhood rules
void get_moore_neighbors(Vec4 v, vector<Vec4> *neighbors, uint64_t n) {
    for (uint32_t x_n = (v.x == 0) ? 0 : v.x - 1; x_n < v.x + 2; x_n++) {
        for (uint32_t y_n = (v.y == 0) ? 0 : v.y - 1; y_n < v.y + 2; y_n++) {
            for (uint32_t z_n = (v.z == 0) ? 0 : v.z - 1; z_n < v.z + 2; z_n++) {
                // If neighbor is out of bounds of cube size
                if (x_n >= n || y_n >= n || z_n >= n) {
                    continue;
                }
                // Don't include itself
                if (x_n == v.x && y_n == v.y && z_n == v.z) {
                    continue;
                }
                (*neighbors).push_back(Vec4(x_n, y_n, z_n, 0));
            }
        }
    }
}

// get all neighbors according to Von Neumann neighborhood rules
void get_vn_neighbors(Vec4 v, vector<Vec4> *neighbors, uint64_t n) {
    if (v.x > 0 && v.x < n) (*neighbors).push_back(Vec4(v.x - 1, v.y, v.z, 0));
    if (v.x + 1 < n) (*neighbors).push_back(Vec4(v.x + 1, v.y, v.z, 0));
    if (v.y > 0 && v.y < n) (*neighbors).push_back(Vec4(v.x, v.y - 1, v.z, 0));
    if (v.y + 1 < n) (*neighbors).push_back(Vec4(v.x, v.y + 1, v.z, 0));
    if (v.z > 0 && v.z < n) (*neighbors).push_back(Vec4(v.x, v.y, v.z - 1, 0));
    if (v.z + 1 < n) (*neighbors).push_back(Vec4(v.x , v.y, v.z + 1, 0));
}

