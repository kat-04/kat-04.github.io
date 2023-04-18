#ifndef GOL_SEQUENTIAL_H
#define GOL_SEQUENTIAL_H

// Sequential version of 3D gol
#include <string>

//RULE FORMAT: survival/birth/numStates/isMoore
//MAP FORMAT: keys 0-26 -> birth for that num neighbors,
//            keys 27-54 -> survival for that num neighbors

using namespace std; 

int golSequential(int argc, char** argv, string outputDir);

#endif
