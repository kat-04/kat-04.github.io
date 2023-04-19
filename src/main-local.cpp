#include "gol-sequential.h"
/* #include "gol-cuda.cpp" */

int main(int argc, char** argv)
{
    // TODO: call golCuda with args (maybe add option flag to input to decide which version to call)
    /* return gol_cuda(argc, argv, "output-files"); */

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
        if (version == "cuda") {
            return gol_cuda(argc, argv, outputDir);
        }
        if (version == "omp") {
            return 0;
        }
    }
    cerr << "Usage: " << argv[0] << " input_file number_of_frames side_length version:[seq(default)/cuda/omp]" << endl;
    return 1;
    
}
