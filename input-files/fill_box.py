import sys
import argparse
import random

parser = argparse.ArgumentParser(
                    prog='fill_box.py',
                    description='Creates input file of vertices with cube of side length n filled')

parser.add_argument('filename')
parser.add_argument('-s', '--size', type=int)
parser.add_argument('-o', '--offset', default=0, type=int)
parser.add_argument('-r', '--rule')
parser.add_argument('-t', '--state', default=-1, type=int)
parser.add_argument('-d', '--density', default=0.5, type=float)

args = parser.parse_args()

n = args.size
o = args.offset

file = args.filename
with open(file, 'w') as f:
    f.write(f"{args.rule}\n")
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if (random.random() < args.density):
                    if (args.state != -1):
                        f.write(f"{o + x} {o + y} {o + z} {args.state}\n")
                    else:
                        f.write(f"{o + x} {o + y} {o + z}\n")
