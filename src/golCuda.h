#ifndef __GOL_CUDA_H__
#define __GOL_CUDA_H__

#ifndef uint
#define uint unsigned int
#endif

struct Cube;
class GolCuda {

private:
    int sideLength;
    bool isMoore;
    int numStates;
    
    Cube* cube;
    bool* ruleset;
    int* inputData;

    int* cudaDeviceInputData;
    int* cudaDeviceOutputData;
    bool* cudaDeviceRuleset;
public:

    GolCuda();
    virtual ~GolCuda();

    const Cube* getCube();

    void setup();

    int loadInput(char* file, int n, char* outputDir);

    void allocOutputCube(int sideLength);

    void clearOutputCube();

    void doIteration();

};


#endif