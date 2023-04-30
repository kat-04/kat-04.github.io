#include <iostream>
#include <string>
#include "gol-sequential.h"
/* #include "gol-cuda.cpp" */
/* #include "gol-openmp.h" */
#include "gol-openmp-states.h"

int main(int argc, char** argv)
{
    // clean output files
    system("rm -rf output-files/");
    system("mkdir output-files/");

    std::string outputDir = "output-files";

    //parse args
    if (argc == 4) {
        return golSequential(argc, argv, outputDir);
    } 
    if (argc == 5) { // version specified 
        string version = argv[4];
        if (version == "seq") {
            return golSequential(argc, argv, outputDir);
        }
        /* if (version == "cuda") { */
        /*     return gol_cuda(argc, argv, outputDir); */
        /* } */
        /* if (version == "omp") { */
        /*     return golOpenMP(argc, argv, outputDir); */
        /* } */
        if (version == "states") {
            return golOpenMPStates(argc, argv, outputDir);
        }
    }
    cerr << "Usage: " << argv[0] << " input_file number_of_frames side_length version:[seq(default)/cuda/omp/states]" << endl;
    return 1;
    
}
