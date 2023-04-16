// TODO: write function to load data into cuda memory
#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include "golCuda.h"

int gol_cuda(int argc, char** argv) {
    GolCuda* golCuda;
    golCuda = new GolCuda();
    
    golCuda->doIteration();
    // TODO: 
    // 1. init golCuda 
    // 2. allocResultCube
    // 3. loadInput
    // 4. setup
    // 5. clearResultCube (make sure existing data reset)
    // 6. doIteration with timing code around it
    // 7. getResultCube 
    // 8. write output to file
    // 9. ???
    // 10. profit
    return 0;
}