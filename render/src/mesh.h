#include <tuple>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "raylib.h"
#include "raymath.h"

std::tuple<int, std::vector<Matrix> > parse_data(std::string path_to_frame) {

    // tokensizes the string to parse integers
    auto tokenize = [&](std::string &s, std::vector<std::string> &out) {
        const char *delim = " ";
        char *token = std::strtok(const_cast<char*>(s.c_str()), delim); 
        while (token != nullptr) { 
            out.push_back(std::string(token)); 
            token = strtok(nullptr, delim); 
        }
    };

    // open file
    std::ifstream frame (path_to_frame);
    int size = 0;
    std::vector<Matrix> vertices;
    std::string line;

    // should be caught by call, but just in case
    if (!frame) {
        std::cout << "File does not exist" << std::endl;
        return std::make_tuple(0, vertices);
    }

    // get size separately (always first line of file)
    std::getline(frame, line);
    size = std::stoi(line);

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
        Matrix v = MatrixMultiply(MatrixTranslate(std::stof(out[0]), std::stof(out[1]), std::stof(out[2])), shift);

        // add to vector
        vertices.push_back(v);
        out.clear();
    }

    return std::make_tuple(size, vertices);
}
