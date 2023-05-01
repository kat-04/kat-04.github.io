#include <tuple>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <vector>
#include <map>
#include "raylib.h"
#include "raymath.h"
#include "mesh.h"

using namespace std;


std::tuple<int, int> parse_init(std::string path_to_init) {
    // tokenizes the string to parse integers
    auto tokenize = [&](std::string &s, std::vector<std::string> &out) {
        const char *delim = " ";
        char *token = std::strtok(const_cast<char*>(s.c_str()), delim); 
        while (token != nullptr) { 
            out.push_back(std::string(token)); 
            token = std::strtok(nullptr, delim); 
        }
    };
    std::ifstream frame (path_to_init);
    std::string line;
    std::vector<std::string> out;

    int size = 0;
    int numStates = 0;

    std::getline(frame, line);
    tokenize(line, out);
    size = std::stoi(out[0]);
    out.clear();

    std::getline(frame, line);
    tokenize(line, out);
    numStates = std::stoi(out[0]);

    return std::make_tuple(size, numStates);
}


std::map<int, std::vector<Matrix> > parse_data(std::string path_to_frame, int num_states, int size) {

    // tokenizes the string to parse integers
    auto tokenize = [&](std::string &s, std::vector<std::string> &out) {
        const char *delim = " ";
        char *token = std::strtok(const_cast<char*>(s.c_str()), delim); 
        while (token != nullptr) { 
            out.push_back(std::string(token)); 
            token = std::strtok(nullptr, delim); 
        }
    };

    // open file
    std::ifstream frame (path_to_frame);
    std::map<int, std::vector<Matrix> > vertices;
    std::string line;

    // should be caught by call, but just in case
    if (!frame) {
        std::cout << "File does not exist" << std::endl;
        return vertices;
    }

    // How much to translate each vertex by
    // Centers entire structure at (0, 0, 0)
    Matrix shift = MatrixTranslate(-size / 2.f + 0.5f, -size / 2.f + 0.5f, -size / 2.f + 0.5f);

    char delim = ' ';
    std::vector<std::string> out;
    while (std::getline(frame, line, '\n')) {
        // tokenizes line and places values in out
        tokenize(line, out);

        // Translates each pixel by its coordinates
        // Then translates it down to adjust for origin
        /* std::cout << std::stoi(out[0]) << std::endl; */
        Matrix v = MatrixMultiply(MatrixTranslate(std::stof(out[0]), std::stof(out[1]), std::stof(out[2])), shift);

        // add to vector
        vertices[stoi(out[3])].push_back(v);
        out.clear();
    }

    return vertices;
}
