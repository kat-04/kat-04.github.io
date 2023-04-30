#include <iostream>
#include <string>
#include "gol-sequential.h"
#include "gol-cuda.cpp"
#include "gol-openmp.h"

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
    if (argc == 5 || argc == 6) { // version specified 
        string version = argv[4];
        if (version == "seq") {
            return golSequential(argc, argv, outputDir);
        }
        if (version == "cuda") {
            return gol_cuda(argc, argv, outputDir);
        }
        if (version == "omp") {
            return golOpenMP(argc, argv, outputDir);
        }
    }
    cerr << "Usage: " << argv[0] << " input_file number_of_frames side_length version:[seq(default)/cuda/omp]" << endl;
    return 1;
    
}
