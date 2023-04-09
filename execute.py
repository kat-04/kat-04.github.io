#!/usr/bin/env python3

import sys
import os
import re

# Usage: ./execute.py FILENAME NUMBER_OF_FRAMES SIDELENGTH
# TODO: can change to take in FILENAME as input and the file contains the name of the scenes we want to run

if len(sys.argv) != 4:
    print("Usage: ./execute.py FILENAME NUMBER_OF_FRAMES SIDELENGTH")
    exit(-1)

filename = sys.argv[1]
num_frames = int(sys.argv[2])
side_len = int(sys.argv[3])

valid = True
valid = valid and os.path.exists(os.path.join(os.getcwd(), filename))
valid = valid and (num_frames >= 0) and (side_len >= 0)

if not valid:
    print("Usage: ./execute.py FILENAME NUMBER_OF_FRAMES SIDELENGTH")
    exit(-1)

# TODO: create starting config names
# scenes = (
#     ('random-50000', 50000, 500.0, 5),
#     ('corner-50000', 50000, 500.0, 5),
#     ('repeat-10000', 10000, 100.0, 50),
#     ('sparse-50000', 50000, 5.0, 50),
#     ('sparse-200000', 200000, 20.0, 50),
# )
# scenes = (('basic-cube', num_frames, side_len))

# TODO: Change if doing parallel
# workers = [1]
# num_scenes = 1

# perfs = [[None] * len(workers) for _ in range(num_scenes)]

# scene_names = ('basic_cube')
# particle_nums = (50000, 50000, 10000, 50000, 200000)
# space_sizes = (num_frames)
# iterations = (side_len)


os.system('mkdir -p output')
os.system('rm -rf output/*')


init_file = f'src/init-files/basic-cube-init.txt'
log_file = f'logs/basic_cube.txt'
cmd = f'./gol-seq -f {init_file} -n {num_frames} -s {side_len} > {log_file}'
ret = os.system(cmd)
assert ret == 0, 'ERROR -- GOL exited with errors'


# for i, (scene_name, iterations, side) in enumerate(scenes):
#     for j, worker in enumerate(workers):
#         print(f'--- running {scene_name} on {worker} workers ---')
#         init_file = f'src/benchmark-files/{scene_name}-init.txt'
#         output_file = f'logs/{scene_name}.txt'
#         log_file = f'logs/{scene_name}.log'
#         cmd = f'g++ -n {worker} {prog} {load_balance} -n {particle_num} -i {iteration} -in {init_file} -s {space_size} -o {output_file} > {log_file}'
#         ret = os.system(cmd)
#         assert ret == 0, 'ERROR -- nbody exited with errors'
#         compare(output_file, f'src/benchmark-files/{scene_name}-ref.txt')
#         t = float(re.findall(
#             r'total simulation time: (.*?)s', open(log_file).read())[0])
#         print(f'total simulation time: {t:.6f}s')
#         perfs[i][j] = t
