#ifndef __PARSE_H__
#define __PARSE_H__

#include <string>
#include <map>
#include <tuple>
#include <vector>



// prints voxel state (either alive or dead) at given position
void printVoxel(uint64_t x, uint64_t y, uint64_t z, bool alive);

// tokenizes a single given line (of the input file)
std::vector<std::string> tokenizeLine(std::string &line, const char* delim);

// parse the ruleset line
std::tuple<std::map<int, bool>, bool, int> parseRules(std::string line);


// parse the rules for Cuda 
std::tuple<bool, int> parseRulesCuda(std::string line, bool *&ruleset);

#endif
