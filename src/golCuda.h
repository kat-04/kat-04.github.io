#ifndef __GOL_CUDA_H__
#define __GOL_CUDA_H__

#ifndef uint
#define uint unsigned int
#endif

struct Cube;
class GolCuda {

private:
    uint64_t sideLength;
    bool isMoore;
    int numStates;
    
    Cube* cube;
    bool* ruleset;
    uint8_t* inputData;

    uint8_t* cudaDeviceInputData;
    uint8_t* cudaDeviceOutputData;
    bool* cudaDeviceRuleset;
public:

    GolCuda();
    virtual ~GolCuda();

    Cube* getCube();

    void setup();

    int loadInput(char* file, uint64_t n, char* outputDir);

    void allocOutputCube(uint64_t sideLength);

    void clearOutputCube();

    void advanceFrame();

    void doIteration();

};


#endif
