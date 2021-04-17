import sys
sys.path.append('../')

from pathlib import Path

from py_diff_pd.common.tet_mesh import tetrahedralize
from py_diff_pd.common.project_path import root_path
from py_diff_pd.common.common import print_info, ndarray

import shutil
import os
import numpy as np

if __name__ == '__main__':
    obj_file_name = Path(root_path) / 'asset' / 'mesh' / 'armadillo_low_res.obj'
    verts, elements = tetrahedralize(obj_file_name, visualize=True, options={
        'minratio': 1.1 })
    print('Generated {:4d} vertices and {:4d} elements'.format(verts.shape[0], elements.shape[0]))