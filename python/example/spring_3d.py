import sys
sys.path.append('../')
from contextlib import redirect_stdout

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
from py_diff_pd.env.bouncing_tet_env_3d import BouncingTetEnv3d
from py_diff_pd.env.bouncing_spring_env_3d import BouncingSpringEnv3d


if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    folder = Path('spring_3d')
    youngs_modulus = 6e6
    poissons_ratio = 0.4
    env = BouncingSpringEnv3d(seed, folder, { 'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio, 'spp': 4 })
    deformable = env.deformable()

    # Optimization parameters.
    thread_ct = 12
    newton_opt = { 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-2, 'rel_tol': 1e-4, 'verbose': 1, 'thread_ct': thread_ct }
    pd_opt = { 'max_pd_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-2, 'verbose': 1, 'thread_ct': thread_ct,
        'use_bfgs': 1, 'bfgs_history_size': 10 }
    siere_opt = { 'max_pd_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-4, 'rel_tol': 1e-2, 'verbose': 1, 'thread_ct': thread_ct,
        'use_bfgs': 1, 'bfgs_history_size': 10 , 'recompute_eigen_decomp_each_step': 1, 'num_modes': 5 }
    methods = ('siere','sibefull','sibe','newton_pardiso','newton_pcg', 'newton_cholesky','pd_pardiso')
    opts = (siere_opt, newton_opt, newton_opt, pd_opt)

    dt = 10e-3
    frame_num = 1 

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
    method = methods[0]
    opt = opts[0]
    simulation_name = f'{method}_recomp_{opt["recompute_eigen_decomp_each_step"]}_nmode_{opt["num_modes"]}_dt_{dt}_Y_{youngs_modulus}_frame_num_{frame_num}'
    print(simulation_name)
    with open( 'output.txt', 'w') as f:
        with redirect_stdout(f):
            _, info = env.simulate_simple(dt, frame_num, method, opt, q0, v0, a0, f0, vis_folder=simulation_name)
            print('info')
    print("siere")
    print("forward time")
    print(info["forward_time"])
    print("render time")
    print(info["render_time"])

    # Generate siere data. 

    # Generate sibe_full data.
    # method = methods[1]
    # opt = opts[0]
    # simulation_name = f'{method}_dt_{dt}_frame_num_{frame_num}'
    # print(simulation_name)
    # _, info = env.simulate_simple(dt, frame_num, methods[1], opts[0], q0, v0, a0, f0, vis_folder=simulation_name)
    # print("sibefull")
    # print("forward time")
    # print(info["forward_time"])
    # print("render time")
    # print(info["render_time"])

    # # Generate groundtruth motion.
    # _, info = env.simulate_simple(dt, frame_num, methods[1], opts[0], q0, v0, a0, f0, vis_folder='sibefull')
    # print("sibefull")
    # print("forward time")
    # print(info["forward_time"])
    # print("render time")
    # print(info["render_time"])
