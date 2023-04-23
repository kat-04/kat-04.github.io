#ifndef __FILE_LOADER_H__
#define __FILE_LOADER_H__

#include "golCuda.h"

int
loadCubeInput(char* file, int& sideLength, bool* ruleset, int& numStates, bool& isMoore, int* inputData, int n, char* outputDir);

#endif
