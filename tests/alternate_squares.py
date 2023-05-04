import sys
import argparse

parser = argparse.ArgumentParser(
                    prog='alternate_squares.py',
                    description='Creates input file of vertices with cube of side length n with alternating squares')

parser.add_argument('filename')
parser.add_argument('-n', '--size', type=int)
parser.add_argument('-o', '--offset', default=0, type=int)
parser.add_argument('-r', '--rule')
parser.add_argument('-s', '--state', default=-1, type=int)

args = parser.parse_args()

n = args.size
o = args.offset

file = args.filename
with open(file, 'w') as f:
    f.write(f"{args.rule}\n")
    for x in range(n):
        for y in range(n):
            for z in range(n):
                if ((x + y + z) % 2 == 0):
                        if (args.state != -1):
                            f.write(f"{o + x} {o + y} {o + z} {args.state}\n")
                        else:
                            f.write(f"{o + x} {o + y} {o + z}\n")
                # elif (x % 2 == 1 and ((x + y * n + z * n ** 2) % 2) == 0):
                #     if (args.state != -1):
                #         f.write(f"{o + x} {o + y} {o + z} {args.state}\n")
                #     else:
                #         f.write(f"{o + x} {o + y} {o + z}\n")
