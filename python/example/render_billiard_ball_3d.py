import sys
sys.path.append('../')

from pathlib import Path
import time
import os
import numpy as np
import scipy.optimize
import pickle
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.tet_mesh import generate_tet_mesh, tet2obj, tetrahedralize
from py_diff_pd.core.py_diff_pd_core import TetMesh3d
from py_diff_pd.common.display import export_mp4
from py_diff_pd.common.common import print_info, print_ok, print_error, print_warning
from py_diff_pd.env.billiard_ball_env_3d import BilliardBallEnv3d
from py_diff_pd.common.renderer import PbrtRenderer
from py_diff_pd.common.project_path import root_path

def load_image(image_file):
    with cbook.get_sample_data(image_file) as f:
        img = plt.imread(f)
    return ndarray(img)

img_height, img_width = 720, 1280
def pxl_to_cal(pxl):
    pxl = ndarray(pxl).copy()
    pxl[:, 1] *= -1
    pxl[:, 1] += img_height
    return pxl
def cal_to_pxl(cal):
    cal = ndarray(cal).copy()
    cal[:, 1] -= img_height
    cal[:, 1] *= -1
    return cal

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    folder = Path('render_billiard_ball_3d')

    # Simulation parameters.
    substeps = 3
    dt = (1 / 60) / substeps

    # Extract the initial information of the balls.
    ball_radius = 0.06858 / 2   # In meters and from measurement/googling the diameter of a tennis ball.
    experiment_data_folder = Path(root_path) / 'python/example/billiard_ball_calibration/experiment_video'
    camera_data = pickle.load(open(Path(root_path) / 'python/example/billiard_ball_calibration/experiment/intrinsic.data', 'rb'))
    optimization_data_folder = Path(root_path) / 'python/example/billiard_ball_3d'
    opt_data = pickle.load(open(optimization_data_folder / 'data_0006_threads.bin', 'rb'))
    R = camera_data['R']
    T = camera_data['T']
    K = camera_data['K']
    alpha = camera_data['alpha']
    cx = camera_data['cx']
    cy = camera_data['cy']
    img_width = cx * 2
    img_height = cy * 2
    # Compute camera_pos, camera_lookat, camera_up, and fov.
    # R.T indicate the camera coordinates.
    camera_pos = -R.T @ T
    camera_x = R[0]
    camera_y = R[1]
    camera_z = R[2]
    # Do we want to flip the directions?
    if camera_x[0] < 0:
        camera_x = -camera_x
        camera_y = -camera_y
    # Now x is left to right and y is bottom to up.
    camera_up = camera_y
    camera_lookat = camera_pos - camera_z
    # Compute fov from alpha.
    # np.tan(half_fov) * alpha = cy
    fov = np.rad2deg(np.arctan(cy / alpha) * 2)

    for name in ['init', 'pd_eigen']:
        sim_data = pickle.load(open(optimization_data_folder / name / 'info.data', 'rb'))
        create_folder(folder / name, exist_ok=True)
        _, info = sim_data
        for i, qi in enumerate(info['q']):
            if i % substeps != 0: continue
            img_name = folder / name / '{:04d}.png'.format(int(i // substeps))

            options = {
                'file_name': img_name,
                'light_map': 'uffizi-large.exr',
                'sample': 4,
                'max_depth': 2,
                'camera_pos': camera_pos,
                'camera_lookat': camera_lookat,
                'camera_up': camera_up,
                'resolution': (img_width, img_height),
                'fov': fov,
            }
            renderer = PbrtRenderer(options)

            # Generate the mesh from qi.
            tmp_bin_file_name = '.tmp.bin'
            obj_file_name = Path(root_path) / 'asset' / 'mesh' / 'sphere.obj'
            _, eles = tetrahedralize(obj_file_name, normalize_input=False)
            all_verts = qi.reshape((2, -1, 3))
            # Shift the z axis by radius.
            all_verts[:, :, 2] += ball_radius
            num_balls = 2
            all_eles = [eles + i * all_verts.shape[1] for i in range(num_balls)]
            all_verts = np.vstack(all_verts)
            all_eles = np.vstack(all_eles)
            generate_tet_mesh(all_verts, all_eles, tmp_bin_file_name)
            mesh = TetMesh3d()
            mesh.Initialize(str(tmp_bin_file_name))
            tet2obj(mesh, obj_file_name=folder / '.tmp.obj')
            renderer.add_tri_mesh(folder / '.tmp.obj', color=[.1, .7, .2], render_tet_edge=True)
            renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj', texture_img='chkbd_24_0.7',
                transforms=[('t', (0.5, 0.5, 0))])

            renderer.render()

            os.remove(folder / '.tmp.obj')