import sys
sys.path.append('../')

from pathlib import Path
import time
import numpy as np
import scipy.optimize
import pickle

from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error
from py_diff_pd.common.grad_check import check_gradients
from py_diff_pd.core.py_diff_pd_core import StdRealVector
from py_diff_pd.env.soft_starfish_env_3d import SoftStarfishEnv3d

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    folder = Path('soft_starfish_3d')
    youngs_modulus = 1e6
    poissons_ratio = 0.4
    env = SoftStarfishEnv3d(seed, folder, {
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio })
    deformable = env.deformable()

    # Optimization parameters.
    methods = ('newton_pcg', 'newton_cholesky', 'pd_eigen')
    thread_ct = 6
    opts = (
        { 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct },
        { 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct },
        { 'max_pd_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
            'use_bfgs': 1, 'bfgs_history_size': 10 },
    )

    # Compute the initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position()
    v0 = np.zeros(dofs)
    dt = 1e-2
    frame_num = 50
    a0 = [np.zeros(act_dofs) for _ in range(frame_num)]
    vertex_num = int(dofs // 3)
    f0 = np.zeros((vertex_num, 3))
    f0 = f0.ravel()
    f0 = [f0 for _ in range(frame_num)]
    _, info = env.simulate(dt, frame_num, methods[2], opts[2], q0, v0, a0, f0, require_grad=False,
        vis_folder='random')