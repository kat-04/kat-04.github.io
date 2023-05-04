#ifndef __GOL_OPENMP_H
#define __GOL_OPENMP_H

#include <string>

// performs the entire OpenMP game of life algorithm for a given number of frames
int golOpenMP(int argc, char** argv, std::string outputDir);

#endif
