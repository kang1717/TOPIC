import numpy as np
import os

total = 0
LJ    = 0
NNP   = 0
for i in range(20):
    time  = []
    if not os.path.exists(str(i)):
        continue

    with open(f'{i}/log', 'r') as f:
        lines = f.readlines()

    total += len(lines)

    for j, line in enumerate(lines):
        if j < 1:
            continue
        time.append(float(line.split()[-2]))

        if line.split()[4] == '0':
            LJ += 1
            if line.split()[6] == '0':
                NNP += 1
    print(f"Folder {i}: {np.sum(time):.1f} seconds")

print(f"Total: {total}, LJ success: {LJ} ({LJ/total*100:.2f}%), NNP success: {NNP} ({NNP/total*100:.2f}%)")
