#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
using namespace glm;

// Vec3 = vector with (x, y, z)

std::vector<vec3> parse_data() {

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
    std::ifstream frame ("../output-files/frame0.txt");
    int size = 0;
    std::vector<vec3> vertices;
    std::string line;

    if (!frame) {
        std::cout << "File does not exist" << std::endl;
        return vertices;
    }

    std::getline(frame, line);
    size = std::stoi(line);

    char delim = ' ';
    std::vector<std::string> out;
    while (std::getline(frame, line, '\n')) {
        tokenize(line, out);
        vec3 v = vec3(std::stoi(out[0]), std::stoi(out[1]), std::stoi(out[2]));
        vertices.push_back(v);
        out.clear();
    }

    std::cout << "Side len: " << size << std::endl;
    for (auto &v : vertices) {
        std::cout << "Vertex: (" << v.x << ", " << v.y << ", " << v.z << ")" << std::endl;
    }
    return vertices;
}

int main() {
    parse_data();
    return 0;
}
