import sys
import argparse
import random

parser = argparse.ArgumentParser(
                    prog='fill_box.py',
                    description='Creates input file of vertices with entire cube of side length n filled')

parser.add_argument('filename')
parser.add_argument('-s', '--size', type=int)
parser.add_argument('-r', '--rule')

args = parser.parse_args()

n = args.size

file = args.filename
with open(file, 'w') as f:
    f.write(f"{args.rule}\n")
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if (random.random() < 0.4):
                    f.write(f"{40 + x} {40 + y} { 40 + z} 4\n")
                # if ((x + y + z) % 10 != 0):
                #     f.write(f"{x} {y} {z}\n");
