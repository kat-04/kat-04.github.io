#ifndef __PARSE_H__
#define __PARSE_H__

#include <string>
#include <map>
#include <tuple>
#include <vector>


/**
* \brief Prints voxel state (either alive or dead) at given position
*
* \param x The x coordinate of the voxel
* \param y The y coordinate of the voxel
* \param z The x coordinate of the voxel
* \param alive The state of the voxel
*/
void printVoxel(uint32_t x, uint32_t y, uint32_t z, bool alive);

/**
* \brief Tokenizes a single given line (of the input file)
*
* Tokenizes the line with a given delimiter.
*
* \return Vector of strings of the input line separated by the delimiter.
*/
std::vector<std::string> tokenizeLine(std::string &line, const char* delim);

//TODO: write description for this using the following outline
/**
* \brief Checks the dog's energy.
*
* Compares an amount of energy with the dog's current available energy.
* This function can be used to determine if the dog is able to take an
* action that costs energy.
*
* \param energyToExpend An amount of energy to check.
*
* \return True if the dog has enough energy, and false otherwise.
*/
std::tuple<std::map<int, bool>, bool, int> parseRules(std::string line);


//TODO: write description for this using the following outline
/**
* \brief Checks the dog's energy.
*
* Compares an amount of energy with the dog's current available energy.
* This function can be used to determine if the dog is able to take an
* action that costs energy.
*
* \param energyToExpend An amount of energy to check.
*
* \return True if the dog has enough energy, and false otherwise.
*/
std::tuple<bool, int> parseRulesCuda(std::string line, bool *&ruleset);

#endif
