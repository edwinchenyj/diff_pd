import subprocess
import os
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
dir_path = os.path.dirname(os.path.realpath(__file__))

obj_list = ['bar_clamped']
methods = ['bdf2full','siere']
Youngs_modulus = [5e4]
Poissons_ratio = [0.4]
dt = [0.05]
frame_num = [150]
for obj, method, youngs_modulus, poissons_ratio, dt, frame_num in product(obj_list, methods, Youngs_modulus, Poissons_ratio, dt, frame_num):
    os.makedirs(f'{dir_path}/{obj}', exist_ok=True)
    output_file = f'{dir_path}/{obj}/{obj}_{method}_Y_{youngs_modulus}_P_{poissons_ratio}_dt_{dt}_frame_num_{frame_num}.txt'
    with open(output_file, 'w+') as f:
        subprocess.call(['python', f'{dir_path}/{obj}_simulation_script.py', '--dt', f'{dt}', '--frame_num', f'{frame_num}', '--method', f'{method}', '--Y', f'{youngs_modulus}', '--P', f'{poissons_ratio}'], stdout=f)
    residual_before_solve = []
    residual_after_solve = []
    while True:
        with open(output_file, 'r') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if line]
        lines = [line.split() for line in lines]
        for line in lines:
            if line[0].startswith('current'):
                # count += 1
                residual_before_solve.append(float(line[2]))    
            if line[0].startswith("Residual"):
                residual_after_solve.append(float(line[1]))
            # if count == 10:
                # break
        break
    plt.clf()
    plt.plot(residual_before_solve, '--', label='before solve')
    plt.plot(residual_after_solve, label='after solve')
    plt.legend()
    plt.title('frame_num: {}, dt: {}, method: {}, Youngs_modulus: {}, Poissons_ratio: {}'.format(frame_num, dt, method, youngs_modulus, poissons_ratio))
    plt.savefig(f'{dir_path}/{obj}/{obj}_{method}_Y_{youngs_modulus}_P_{poissons_ratio}_dt_{dt}_frame_num_{frame_num}.png')
    # plt.show()
