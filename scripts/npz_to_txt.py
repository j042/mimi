"""Convert npz history to text files"""
import argparse
import os

import numpy as np

parse = argparse.ArgumentParser(description="npz_to_txt.py")
parse.add_argument("-i", dest="input_file")
parse.add_argument("-o", dest="output_dir")
parse.add_argument("-k", dest="keyword")
args = parse.parse_args()


# load npz
npz = np.load(args.input_file)

# prepare output
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

# loop and convert
for i in range(len(npz.keys())):
    kw = f"{args.keyword}_{i}"

    # assume consecutive timesteps should exist. If not, extend here
    if kw not in npz:
        break

    data = npz[kw]

    with open(os.path.join(args.output_dir, f"{kw}.txt"), "w") as f:
        f.write(
            str(data.ravel().tolist())[1:-1].replace(" ", "").replace(",", " ")
        )
