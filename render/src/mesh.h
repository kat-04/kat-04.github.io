#ifndef MESH_H
#define MESH_H

#include <tuple>
#include <string>
#include <vector>
#include "raymath.h"

std::tuple<int, std::vector<Matrix> > parse_data(std::string path_to_frame);

#endif
