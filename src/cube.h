#ifndef  __CUBE_H__
#define  __CUBE_H__

#include <cmath>

struct Cube {

    Cube(uint64_t n) {
        sideLength = n;
        data = new uint8_t[(sideLength * sideLength * sideLength + 7) / 8];
        if (!data) {
            std::cerr << "Cube data allocation failed" << std::endl;
        }
    }

    ~Cube() {
        if (data) {
            delete [] data;
        }
    }

    void clear() {
        uint64_t numVoxels = ((sideLength * sideLength * sideLength + 7) / 8);
        uint8_t* ptr = data;
        for (uint64_t i=0; i<numVoxels; i++) {
            ptr[0] = 0;
            ptr+=1;
        }
    }

    uint64_t sideLength;
    uint8_t* data;
};


#endif
