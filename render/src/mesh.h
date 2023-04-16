#include <tuple>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "raylib.h"
#include "raymath.h"
/* #include <GL/glew.h> */
/* #include <GLFW/glfw3.h> */
/* #include <glm/glm.hpp> */
/* using namespace glm; */

// Vec3 = vector with (x, y, z)

std::tuple<int, std::vector<Matrix> > parse_data(std::string path_to_frame) {

    auto tokenize = [&](std::string &s, std::vector<std::string> &out) {
        const char *delim = " ";
        char *token = std::strtok(const_cast<char*>(s.c_str()), delim); 
        while (token != nullptr) { 
            out.push_back(std::string(token)); 
            token = strtok(nullptr, delim); 
        }
    };

    // open file
    // TODO: change to take in all frames
    std::ifstream frame (path_to_frame);
    int size = 0;
    std::vector<Matrix> vertices;
    std::string line;

    if (!frame) {
        std::cout << "File does not exist" << std::endl;
        return std::make_tuple(0, vertices);
    }

    std::getline(frame, line);
    size = std::stoi(line);

    char delim = ' ';
    std::vector<std::string> out;
    while (std::getline(frame, line, '\n')) {
        tokenize(line, out);
        Matrix v = MatrixTranslate(std::stof(out[0]), std::stof(out[1]), std::stof(out[2]));
        vertices.push_back(v);
        out.clear();
    }

    /* std::cout << "Side len: " << size << std::endl; */
    /* for (auto &v : vertices) { */
    /*     std::cout << "Vertex: (" << v.x << ", " << v.y << ", " << v.z << ")" << std::endl; */
    /* } */
    return std::make_tuple(size, vertices);
}



/* int main() { */
/*     parse_data("../output-files/frame0.txt"); */
/*     return 0; */
/* } */
