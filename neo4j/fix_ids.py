#!/usr/bin/env python3
# 
# Fix output from export_*_ids.cypher scripts.
# `fixed` files then can be used as *.ids files lzo2csv scripts
#

import sys
from tqdm import tqdm

if len(sys.argv) < 3:
    print(f"Usage: python {sys.argv[0]} input output")
    exit(1)

input_fp = sys.argv[1]
output_fp = sys.argv[2]
with open(input_fp, "r") as input_f, open(output_fp, "w") as output_f:
    pbar = tqdm()
    input_f.readline()
    for line in input_f.readlines():
        output_f.write(f"{line.strip()[1:-1]}\n")
        pbar.update(1)
pbar.close()
