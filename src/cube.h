#ifndef  __CUBE_H__
#define  __CUBE_H__

#include <cmath>

struct Cube {

    Cube(uint32_t n) {
        sideLength = n;
        data = new uint8_t[(sideLength * sideLength * sideLength + 7) / 8];
    }

    void clear() {

        uint32_t numVoxels = ((sideLength * sideLength * sideLength + 7) / 8);
        uint8_t* ptr = data;
        for (uint32_t i=0; i<numVoxels; i++) {
            ptr[0] = 0;
            ptr+=1;
        }
    }

    uint32_t sideLength;
    uint8_t* data;
};


#endif
