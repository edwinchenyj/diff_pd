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
from py_diff_pd.env.bouncing_armadillo_env_3d import BouncingArmaEnv3d
from py_diff_pd.env.bar_side_env_3d import BarSideEnv3d


if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    folder = Path('bar_side_3d')
    youngs_modulus = 6e6
    poissons_ratio = 0.4
    env = BarSideEnv3d(seed, folder, { 'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio, 'spp': 4 })
    deformable = env.deformable()

    # Optimization parameters.
    thread_ct = 12
    newton_opt = { 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-2, 'rel_tol': 1e-4, 'verbose': 2, 'thread_ct': thread_ct }
    pd_opt = { 'max_pd_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-4, 'rel_tol': 1e-2, 'verbose': 2, 'thread_ct': thread_ct,
        'use_bfgs': 1, 'bfgs_history_size': 10 }
    siere_opt = { 'max_pd_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-4, 'rel_tol': 1e-2, 'verbose': 2, 'thread_ct': thread_ct,
        'use_bfgs': 1, 'bfgs_history_size': 10 }
    methods = ('siere','sibefull','newton_pardiso','newton_pcg', 'newton_cholesky','pd_pardiso')
    opts = (siere_opt, newton_opt, newton_opt, pd_opt)

    dt = 10e-3
    frame_num = 300

    # Compute the initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q_gt = ndarray([0.0, 0.0, 0.00])
    v_gt = ndarray([0.0, 0., 0.0])
    q0 = env.default_init_position()
    q0 = (q0.reshape((-1, 3)) + q_gt).ravel()
    v0 = np.zeros(dofs)
    v0 = (v0.reshape((-1, 3)) + v_gt).ravel()
    a0 = [np.zeros(act_dofs) for _ in range(frame_num)]
    f0 = [np.zeros(dofs) for _ in range(frame_num)]

    # Generate groundtruth motion.
    _, info = env.simulate_simple(dt, frame_num, methods[0], opts[0], q0, v0, a0, f0, vis_folder='siere_recomp_1_n_5_dt_10e-3_frame_num_300')
    print("siere")
    print("forward time")
    print(info["forward_time"])
    print("render time")
    print(info["render_time"])

    # Generate groundtruth motion.
    _, info = env.simulate_simple(dt, frame_num, methods[1], opts[0], q0, v0, a0, f0, vis_folder='sibefull_dt_10e-3_frame_num_300')
    print("sibefull")
    print("forward time")
    print(info["forward_time"])
    print("render time")
    print(info["render_time"])


    dt = 50e-3
    frame_num = 100

    # Compute the initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q_gt = ndarray([0.0, 0.0, 0.00])
    v_gt = ndarray([0.0, 0., 0.0])
    q0 = env.default_init_position()
    q0 = (q0.reshape((-1, 3)) + q_gt).ravel()
    v0 = np.zeros(dofs)
    v0 = (v0.reshape((-1, 3)) + v_gt).ravel()
    a0 = [np.zeros(act_dofs) for _ in range(frame_num)]
    f0 = [np.zeros(dofs) for _ in range(frame_num)]

    # Generate groundtruth motion.
    _, info = env.simulate_simple(dt, frame_num, methods[0], opts[0], q0, v0, a0, f0, vis_folder='siere_recomp_1_n_5_dt_50e-3_frame_num_100')
    print("siere")
    print("forward time")
    print(info["forward_time"])
    print("render time")
    print(info["render_time"])

    # Generate groundtruth motion.
    _, info = env.simulate_simple(dt, frame_num, methods[1], opts[0], q0, v0, a0, f0, vis_folder='sibefull_dt_50e-3_frame_num_100')
    print("sibefull")
    print("forward time")
    print(info["forward_time"])
    print("render time")
    print(info["render_time"])
