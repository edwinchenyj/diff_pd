import sys
sys.path.append('../')

from pathlib import Path
import time
import numpy as np
import scipy.optimize
import pickle
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.display import export_mp4
from py_diff_pd.common.common import print_info, print_ok, print_error, print_warning
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
    substeps = 4
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
    sim_data = pickle.load(open(optimization_data_folder / 'init/info.data', 'rb'))
    active_frame = int((len(sim_data['q']) - 1) // substeps)
    start_frame = 280
    end_frame = start_frame + active_frame + 1

    for name in ['init', 'pd_eigen']:
        sim_data = pickle.load(open(optimization_data_folder / name / 'info.data', 'rb'))
        create_folder(folder / name, exist_ok=True)
        for i in range(start_frame, end_frame):
            img = load_image(experiment_data_folder / '{:04d}.png'.format(i))
            # Draw sphere centers back on the image.
            centers = np.mean(sim_data['q'][(i - start_frame) * substeps].reshape((2, -1, 3)), axis=1)
            # Shift z because in our video we use the table as the xy plane.
            centers[:, 2] += ball_radius
            centers_in_camera = (centers @ R.T + T) @ K.T
            centers_in_camera = centers_in_camera[:, :2] / centers_in_camera[:, 2][:, None]
            centers_in_camera = cal_to_pxl(centers_in_camera)
            for c in centers_in_camera:
                ci, cj = int(c[0]), int(c[1])
                img[cj - 3 : cj + 4, ci - 3 : ci + 4] = (1, 0, 0)
            plt.imsave(folder / name / '{:04d}.png'.format(i), img)
        export_mp4(folder / name, folder / '{}.mp4'.format(name), fps=60)