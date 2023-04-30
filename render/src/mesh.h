#ifndef MESH_H
#define MESH_H

#include <tuple>
#include <string>
#include <vector>
#include <map>
#include "raymath.h"


std::tuple<int, int> parse_init(std::string path_to_init);


std::map<int, std::vector<Matrix> > parse_data(std::string path_to_frame, int num_states, int size);

#endif
