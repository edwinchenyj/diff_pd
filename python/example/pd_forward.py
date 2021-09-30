import sys
sys.path.append('../')

import os
from pathlib import Path
import time
import scipy.optimize
import numpy as np

from py_diff_pd.core.py_diff_pd_core import QuadMesh2d, QuadDeformable, StdRealVector
from py_diff_pd.common.common import create_folder, ndarray, print_info, print_error, print_ok
from py_diff_pd.env.benchmark_env_2d import BenchmarkEnv2d
from py_diff_pd.env.circle_env_2d import CircleEnv2d
from py_diff_pd.env.spring_env_3d import SpringEnv3d
from py_diff_pd.env.armadillo_env_3d import ArmadilloEnv3d
from py_diff_pd.env.benchmark_env_3d import BenchmarkEnv3d

def test_pd_forward(verbose):
    seed = 42
    folder = Path('pd_forward')

    def test_env(env_class_name):
        # env = env_class_name(seed, folder, { 'refinement': 6 })
        env =  SpringEnv3d(seed, folder, {
        'youngs_modulus': 2e5,
        'init_rotate_angle': 0,
        'state_force_parameters': [0, 0, -9],    # No gravity.
        'spp': 4
    })
        # env = BenchmarkEnv3d(seed, folder, { 'refinement': 8 })
        methods = ['pd_pardiso']
        opts = [{ 'max_pd_iter': 1000, 'max_ls_iter': 1, 'abs_tol': 1e-4, 'rel_tol': 1e-6, 'verbose': 0, 'thread_ct': 4,
                'use_bfgs': 1, 'bfgs_history_size': 10 }]
        # Check if Pardiso is available
        pardiso_available = 'PARDISO_LIC_PATH' in os.environ
        # if pardiso_available:
        #     methods.append('pd_pardiso')
        #     opts.append(opts[-1])

        # Forward simulation.
        dt = 0.01
        frame_num = 20
        deformable = env.deformable()
        dofs = deformable.dofs()
        act_dofs = deformable.act_dofs()
        a0 = np.random.uniform(size=(frame_num, act_dofs))
        q = {}
        for method, opt in zip(methods, opts):
            loss, info = env.simulate(dt, frame_num, method, opt, act=a0, vis_folder=method if verbose else None)
            # if verbose:
            print_info('{} finishes in {:3.3f} seconds. Loss: {:3.3f}'.format(method, info['forward_time'], loss))
            q[method] = info['q']
            # print(f'forward time: {info['forward_time']}')
        
        # Compare results.
    #     atol = 1e-4
    #     rtol = 5e-3
    #     for qn, qp in zip(q['newton_pcg'], q['pd_eigen']):
    #         state_equal = np.linalg.norm(qn - qp) < rtol * np.linalg.norm(qn) + atol
    #         if not state_equal:
    #             if verbose:
    #                 print_error(np.linalg.norm(qn - qp), np.linalg.norm(qn))
    #             return False

    #     if pardiso_available:
    #         for qn, qp in zip(q['newton_pcg'], q['pd_pardiso']):
    #             state_equal = np.linalg.norm(qn - qp) < rtol * np.linalg.norm(qn) + atol
    #             if not state_equal:
    #                 if verbose:
    #                     print_error(np.linalg.norm(qn - qp), np.linalg.norm(qn))
    #                 return False

    #     # Visualize results.
        if verbose:
            print_info('PD and Newton solutions are the same.')
            for method in methods:
                print_info('Showing {} gif...'.format(method))
                os.system('eog {}/{}.gif'.format(folder, method))

        return True

    # if not test_env(BenchmarkEnv2d): return False
    if not test_env(SpringEnv3d): return False
    # return True

if __name__ == '__main__':
    verbose = True
    test_pd_forward(verbose)