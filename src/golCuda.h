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
    uint32_t* minMaxs;
    
    Cube* cube;
    bool* ruleset;
    uint8_t* inputData;

    uint32_t* cudaDeviceMinMaxs;

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

    void updateBounds();

    void advanceFrame();

    void doIteration();

};


#endif
