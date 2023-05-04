#include <iostream>
#include <string>
#include "gol-sequential.h"
#include "gol-cuda.cpp"
#include "gol-openmp.h"
#include "timing.h"

// runs game of life for parsed arguments on ghc machines
int main(int argc, char** argv)
{
    std::string outputDir = "/tmp/output-files";
    if (argc == 6) {
        // if running checker change output dir
        string version = argv[4];
        system(("mkdir /tmp/output-files/" + version + "/").c_str());
        outputDir = "/tmp/output-files/" + version;
    } else {
        // clean output files only if not running checker
        system("rm -rf /tmp/output-files/");
        system("mkdir /tmp/output-files/");
    }
    //parse args
    if (argc == 4) {
        return golSequential(argc, argv, outputDir);
    } 
    int result = 0; 
    if (argc == 5 || argc == 6) { // version specified 
        string version = argv[4];
        if (version == "seq") {
            Timer totalTimer;
            result = golSequential(argc, argv, outputDir);
            std::cout << "Total sequential time: " << totalTimer.elapsed() << "s" << std::endl;
        }
        if (version == "cuda") {
            Timer totalTimer;
            result = gol_cuda(argc, argv, outputDir);
            std::cout << "Total CUDA time: " << totalTimer.elapsed() << "s" << std::endl;
        }
        if (version == "omp") {
            Timer totalTimer;
            result = golOpenMP(argc, argv, outputDir);
            std::cout << "Total OpenMP time: " << totalTimer.elapsed() << "s" << std::endl;
        }
        return result;
    }
    cerr << "Usage: " << argv[0] << " input_file number_of_frames side_length version:[seq(default)/cuda/omp]" << endl;
    return 1;
    
}
