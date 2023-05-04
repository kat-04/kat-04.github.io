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

    // get the output cube
    Cube* getCube();

    // do Cuda-related setup
    void setup();

    // load input into Cuda memory
    int loadInput(char* file, uint64_t n, char* outputDir);

    // allocate memory for the output cube
    void allocOutputCube(uint64_t sideLength);

    // clear the output cube data
    void clearOutputCube();

    // update the bounding box min/max values
    void updateBounds();

    // move output to input to prepare for the next frame
    void advanceFrame();

    // do one iteration of game of life
    void doIteration(bool doBoundingBox);

};


#endif
