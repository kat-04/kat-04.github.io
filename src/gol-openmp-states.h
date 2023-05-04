#ifndef __GOL_OPENMP_STATES_H
#define __GOL_OPENMP_STATES_H

#include <string>

// performs the entire OpenMP game of life algorithm with states for a given number of frames
int golOpenMPStates(int argc, char** argv, std::string outputDir);

#endif
