// TODO: write function to load data into cuda memory
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <getopt.h>
#include "golCuda.h"

int gol_cuda(int argc, char** argv, std::string fileDir) {
    
    GolCuda* golCuda;
    
    golCuda = new GolCuda();
    std::cout << "TESTING :)" << std::endl;
    //golCuda->allocResultCube(3);
    
    golCuda->setup();
    //golCuda->doIteration();
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

// int main(int argc, char** argv)
// {
    

//     // clean output files
//     system("rm -rf /tmp/output-files/");
//     system("mkdir /tmp/output-files/");

//     // TODO: call golCuda with args (maybe add option flag to input to decide which version to call)
//     return gol_cuda(argc, argv, "/tmp/output-files"); 
//     //return golSequential(argc, argv, "/tmp/output-files");
// }