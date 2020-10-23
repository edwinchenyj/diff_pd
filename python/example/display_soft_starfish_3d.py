import sys
sys.path.append('../')
import os

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error, print_warning
from py_diff_pd.common.project_path import root_path

if __name__ == '__main__':
    from soft_starfish_3d import load_csv_data
    measurement_data, info = load_csv_data('soft_starfish_3d/data_horizontal_cyclic1.csv')

    t = measurement_data['time']
    m1x = measurement_data['M1_rel_x']
    m1z = measurement_data['M1_rel_z']
    m4x = measurement_data['M4_rel_x']
    m4z = measurement_data['M4_rel_z']
    dl = measurement_data['dl']

    # Plot the optimization progress.
    plt.rc('pdf', fonttype=42)
    plt.rc('font', size=14)
    plt.rc('axes', titlesize=14)
    plt.rc('axes', labelsize=14)

    fig = plt.figure(figsize=(12, 8))
    ax_control = fig.add_subplot(311)
    ax_loc_x = fig.add_subplot(312)
    ax_loc_z = fig.add_subplot(313)

    ax_control.set_position((0.1, 0.76, 0.8, 0.23))
    ax_control.plot(t, dl * 1000, color='tab:blue')
    ax_control.set_ylim([-5, 25])
    ax_control.set_xlabel('time (s)')
    ax_control.set_ylabel('contraction (mm)')
    ax_control.grid(True, which='both')

    ax_loc_x.set_position((0.1, 0.43, 0.8, 0.23))
    ax_loc_x.plot(t, m1x * 1000, color='tab:red', label='marker 1')
    ax_loc_x.plot(t, m4x * 1000, color='tab:green', label='marker 4')
    ax_loc_x.set_xlabel('time (s)')
    ax_loc_x.set_ylabel('x loc (mm)')
    ax_loc_x.legend()
    ax_loc_x.grid(True, which='both')

    ax_loc_z.set_position((0.1, 0.1, 0.8, 0.23))
    ax_loc_z.plot(t, m1z * 1000, color='tab:red', label='marker 1')
    ax_loc_z.plot(t, m4z * 1000, color='tab:green', label='marker 4')
    ax_loc_z.set_ylim([-25, 60])
    ax_loc_z.set_xlabel('time (s)')
    ax_loc_z.set_ylabel('y loc (mm)')
    ax_loc_z.legend()
    ax_loc_z.grid(True, which='both')

    plt.show()
    fig.savefig('soft_starfish_3d/cyclic.pdf')