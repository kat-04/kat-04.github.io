#ifndef __FILE_LOADER_H__
#define __FILE_LOADER_H__

#include "golCuda.h"

int
loadCubeInput(char* file, uint64_t& sideLength, bool*& ruleset, int& numStates, bool& isMoore, uint8_t*& inputData, uint64_t n, char* outputDir, uint32_t*& minMaxs);

#endif
