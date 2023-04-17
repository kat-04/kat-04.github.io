#ifndef __GOL_CUDA_H__
#define __GOL_CUDA_H__

#ifndef uint
#define uint unsigned int
#endif

struct Cube;
class GolCuda {

private:
    Cube* cubeData;

    int* sideLength;
    int* ruleset;

    Cube* cudaDeviceData;
    int* cudaDeviceRuleset;
    int* cudaDeviceSideLength;
public:

    GolCuda();
    virtual ~GolCuda();

    const Cube* getResultCube();

    void setup();

    void loadInput(char* file);

    void allocResultCube(int sideLength);

    void clearResultCube();

    void doIteration();

};


#endif