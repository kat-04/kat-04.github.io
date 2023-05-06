#ifndef MESH_H
#define MESH_H

#include <tuple>
#include <string>
#include <vector>
#include <map>
#include "raymath.h"


// Parses the frame_init file to obtain the size of the bounding box and number of states
std::tuple<int, int> parse_init(std::string path_to_init);


// Parses the data of each frame and transforms each voxel coordinate to a transformation matrix
// for a cube centered at the origin
std::map<int, std::vector<Matrix> > parse_data(std::string path_to_frame, int num_states, int size);

#endif
