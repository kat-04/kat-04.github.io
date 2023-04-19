#ifndef __GOL_CUDA_H__
#define __GOL_CUDA_H__

#ifndef uint
#define uint unsigned int
#endif

struct Cube;
class GolCuda {

private:
    int sideLength;
    Cube* cube;
    int* ruleset;
    int* inputData;

    int* cudaDeviceInputData;
    int* cudaDeviceOutputData;
    int* cudaDeviceRuleset;
public:

    GolCuda();
    virtual ~GolCuda();

    const Cube* getCube();

    void setup();

    void loadInput(char* file, int n);

    void allocOutputCube(int sideLength);

    void clearOutputCube();

    void doIteration();

};


#endif