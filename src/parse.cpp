#include <getopt.h>
#include <string>
#include <map>
#include <cstring>
#include <vector>
#include <tuple>
#include <fstream>
#include <iostream>

using namespace std; 


void printVoxel(int x, int y, int z, bool alive) {
    cout << x << ", " << y << ", " << z << ", alive: " << alive << endl;
}



vector<string> tokenizeLine(string &line, const char* delim) {
    vector<string> out;
    char *token = std::strtok(const_cast<char*>(line.c_str()), delim); 
    while (token != nullptr) 
    { 
        out.push_back(string(token)); 
        token = strtok(nullptr, delim); 
    } 
    return out;
}



tuple<map<int, bool>, bool, int> parseRules(string line) {
    const char* slashDelim = "/";
    const char* commaDelim = ",";
    const char* dashDelim = "-";
    map<int, bool> ruleMap;

    vector<string> rules = tokenizeLine(line, slashDelim);
    string survival = rules[0];
    string birth = rules[1];
    int numStates = stoi(rules[2]);
    bool isMoore = (rules[3] == "M");

    //init map
    for (int i = 0; i < 54; i++) {
        ruleMap[i] = false;
    }

    //parse survival and birth rules
    vector<string> survivalSubsets = tokenizeLine(survival, commaDelim);
    vector<string> birthSubsets = tokenizeLine(birth, commaDelim);
    
    for (int i = 0; i < (int)birthSubsets.size(); i++) {
        if (birthSubsets[i].find('-') == string::npos) {    
            if (birthSubsets[i] != "x") {
                ruleMap[stoi(birthSubsets[i])] = true;
            }
            
        } else {
            vector<string> range = tokenizeLine(birthSubsets[i], dashDelim);
            for (int j = stoi(range[0]); j <= stoi(range[1]); j++) {
                ruleMap[j] = true;
            }
        }
    }

    for (int i = 0; i < (int)survivalSubsets.size(); i++) {
        if (survivalSubsets[i].find('-') == string::npos) {  
            if (survivalSubsets[i] != "x") {
                ruleMap[27 + stoi(survivalSubsets[i])] = true;
            }  
        } else {
            vector<string> range = tokenizeLine(survivalSubsets[i], dashDelim);
            for (int j = 27 + stoi(range[0]); j <= 27 + stoi(range[1]); j++) {
                ruleMap[j] = true;
            }
        }
    }  

    return make_tuple(ruleMap, isMoore, numStates);
}



std::tuple<bool, int> parseRulesCuda(std::string line, bool *ruleset) {
    const char* slashDelim = "/";
    const char* commaDelim = ",";
    const char* dashDelim = "-";

    ruleset = new bool[54];

    std::vector<std::string> rules = tokenizeLine(line, slashDelim);
    std::string survival = rules[0];
    std::string birth = rules[1];
    int numStates = stoi(rules[2]);
    bool isMoore = (rules[3] == "M");

    for (int i = 0; i < 54; i++) {
        ruleset[i] = false;
    }

    //parse survival and birth rules
    std::vector<std::string> survivalSubsets = tokenizeLine(survival, commaDelim);
    std::vector<std::string> birthSubsets = tokenizeLine(birth, commaDelim);
    
    for (int i = 0; i < (int)birthSubsets.size(); i++) {
        if (birthSubsets[i].find('-') == std::string::npos) {    
            if (birthSubsets[i] != "x") {
                ruleset[stoi(birthSubsets[i])] = true;
            }
            
        } else {
            std::vector<std::string> range = tokenizeLine(birthSubsets[i], dashDelim);
            for (int j = stoi(range[0]); j <= stoi(range[1]); j++) {
                ruleset[j] = true;
            }
        }
    }

    for (int i = 0; i < (int)survivalSubsets.size(); i++) {
        if (survivalSubsets[i].find('-') == std::string::npos) {  
            if (survivalSubsets[i] != "x") {
                ruleset[27 + stoi(survivalSubsets[i])] = true;
            }  
        } else {
            std::vector<std::string> range = tokenizeLine(survivalSubsets[i], dashDelim);
            for (int j = 27 + stoi(range[0]); j <= 27 + stoi(range[1]); j++) {
                ruleset[j] = true;
            }
        }
    }      
    return std::make_tuple(isMoore, numStates);
}


