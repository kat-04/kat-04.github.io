// #include "gol-sequential.h"
#include <iostream>
#include <string>
#include "gol-cuda.cpp"

int main(int argc, char** argv)
{
    
    
    // clean output files
    system("rm -rf /tmp/output-files/");
    system("mkdir /tmp/output-files/");

    
    std::string outputDir = "/tmp/output-files";
    // TODO: call golCuda with args (maybe add option flag to input to decide which version to call)
    return gol_cuda(argc, argv, outputDir); 

    //return golSequential(argc, argv, "/tmp/output-files");
}
