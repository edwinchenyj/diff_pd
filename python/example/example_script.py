import subprocess
import os
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
dir_path = os.path.dirname(os.path.realpath(__file__))

obj_list = ['spring_vertical','bar_horizontal','bar_clamped']
# methods = ['siere','sibe','sbdf2','sbdf2ere','strbdf2','strbdf2ere','theta6strbdf2','theta6strbdf2ere']
methods = ['sibe','siere','sbdf2','sbdf2ere','strbdf2','strbdf2ere','theta6strbdf2','theta6strbdf2ere']
Youngs_modulus = [5e6,5e7,5e8]
Poissons_ratio = [0.4]
dt = [0.01]
frame_num = [300]
for obj, method, youngs_modulus, poissons_ratio, dt, frame_num in product(obj_list, methods, Youngs_modulus, Poissons_ratio, dt, frame_num):
    os.makedirs(f'{dir_path}/{obj}', exist_ok=True)
    output_file = f'{dir_path}/{obj}/{obj}_{method}_Y_{youngs_modulus}_P_{poissons_ratio}_dt_{dt}_frame_num_{frame_num}.txt'
    with open(output_file, 'w+') as f:
        subprocess.call(['python', f'{dir_path}/{obj}_simulation_script.py', '--dt', f'{dt}', '--frame_num', f'{frame_num}', '--method', f'{method}', '--Y', f'{youngs_modulus}', '--P', f'{poissons_ratio}'], stdout=f)
    residual_before_solve = []
    residual_after_solve = []
    elastic_energy = []
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
            # if line[0].startswith("Residual"):
            #     residual_after_solve.append(float(line[1]))
            # if count == 10:
                # break
            if line[0].startswith("Elastic"):
                elastic_energy.append(float(line[2]))
        break
    print(elastic_energy)
    # plt.clf()
    # plt.plot(residual_before_solve, '--', label='before solve')
    # plt.plot(residual_after_solve, label='after solve')
    # plt.legend()
    # plt.title('frame_num: {}, dt: {}, method: {}, Youngs_modulus: {}, Poissons_ratio: {}'.format(frame_num, dt, method, youngs_modulus, poissons_ratio))
    # plt.savefig(f'{dir_path}/{obj}/{obj}_{method}_Y_{youngs_modulus}_P_{poissons_ratio}_dt_{dt}_frame_num_{frame_num}_residual.png')
    # plt.show()
    plt.clf()
    plt.plot(elastic_energy, '--', label='elastic energy')
    plt.legend()
    plt.title('frame_num: {}, dt: {}, method: {}, Youngs_modulus: {}, Poissons_ratio: {}'.format(frame_num, dt, method, youngs_modulus, poissons_ratio))
    plt.savefig(f'{dir_path}/{obj}/{obj}_{method}_Y_{youngs_modulus}_P_{poissons_ratio}_dt_{dt}_frame_num_{frame_num}_energy.png')
