#include "gol-sequential.h"
/* #include "gol-cuda.cpp" */

int main(int argc, char** argv)
{
    // TODO: call golCuda with args (maybe add option flag to input to decide which version to call)
    /* return gol_cuda(argc, argv); */

    // clean output files
    system("rm output-files/*.txt");

    return golSequential(argc, argv);
}
