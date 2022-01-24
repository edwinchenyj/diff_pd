import numpy as np
import os
import matplotlib.pyplot 
dir_path = os.path.dirname(os.path.realpath(__file__))
residual_file = dir_path + '/bar_side_siere.txt'
count = 0
residual_before_solve = []
residual_after_solve = []
while True:
    with open(residual_file, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line]
    lines = [line.split() for line in lines]
    for line in lines:
        if line[0].startswith('current'):
            count += 1
            residual_before_solve.append(float(line[2]))    
        if line[0].startswith("Residual"):
            residual_after_solve.append(float(line[1]))
        if count == 10:
            break
    break

matplotlib.pyplot.plot(residual_before_solve, '--', label='before solve')
matplotlib.pyplot.plot(residual_after_solve, label='after solve')
matplotlib.pyplot.legend()
matplotlib.pyplot.title('Residual SIERE')
matplotlib.pyplot.show()