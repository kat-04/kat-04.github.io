#ifndef __GOL_OPENMP_AMBIENT_H
#define __GOL_OPENMP_AMBIENT_H

#include <string>

// performs the entire OpenMP game of life algorithm and outputs with # of neighbors for a given number of frames
int golOpenMPAmbient(int argc, char** argv, std::string outputDir);

#endif
