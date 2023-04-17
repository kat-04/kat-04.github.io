#ifndef  __CUBE_H__
#define  __CUBE_H__


struct Cube {

    Cube(int n) {
        sideLength = n;
        data = new int[sideLength * sideLength * sideLength];
    }

    void clear() {

        int numVoxels = sideLength * sideLength * sideLength;
        int* ptr = data;
        for (int i=0; i<numVoxels; i++) {
            ptr[0] = 0;
            ptr+=1;
        }
    }

    int sideLength;
    int* data;
};


#endif