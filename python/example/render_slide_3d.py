import sys
sys.path.append('../')

from pathlib import Path
import numpy as np
import pickle

from py_diff_pd.common.renderer import PbrtRenderer
from py_diff_pd.common.project_path import root_path
from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error
from py_diff_pd.env.slide_env_3d import SlideEnv3d

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)

    folder = Path('render_slide_3d')
    center = ndarray([-0.6, 0, 7.7])
    start_deg = 60
    end_deg = 100
    init_deg = 70
    radius = 7.5
    init_speed = 10
    target = ndarray([-7.85, 0, 0.315])
    env = SlideEnv3d(seed, folder, {
        'state_force_parameters': [0, 0, -9.81, 1e5, 0.025, 1e4],
        'center': center,
        'start_degree': start_deg,
        'end_degree': end_deg,
        'initial_degree': init_deg,
        'radius': radius,
        'target': target })
    deformable = env.deformable()


    # Optimization parameters.
    thread_ct = 8
    newton_opt = { 'max_newton_iter': 4000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct }
    pd_opt = { 'max_pd_iter': 4000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
        'use_bfgs': 1, 'bfgs_history_size': 10 }
    methods = ('pd_eigen', 'newton_pcg', 'newton_cholesky')
    opts = (pd_opt, newton_opt, newton_opt)

    dt = 5e-3
    frame_num = 200

    # Initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position()
    v0 = np.zeros(dofs).reshape((-1, 3))
    v0 += ndarray([-np.sin(np.deg2rad(init_deg)), 0, -np.cos(np.deg2rad(init_deg))]) * init_speed
    v0 = v0.ravel()
    a0 = [np.zeros(act_dofs) for _ in range(frame_num)]
    f0 = [np.zeros(dofs) for _ in range(frame_num)]

    def variable_to_env(x):
        x = ndarray(x).copy().ravel()
        assert len(x) == 3
        kn = 10 ** x[0]
        kf = x[1]
        mu = 10 ** x[2]
        env = SlideEnv3d(seed, folder, {
            'state_force_parameters': [0, 0, -9.81, kn, kf, mu],
            'center': center,
            'start_degree': start_deg,
            'end_degree': end_deg,
            'initial_degree': init_deg,
            'radius': radius,
            'target': target })
        return env

    # Load data and render.
    data = pickle.load(open('slide_3d/data_{:04d}_threads.bin'.format(thread_ct), 'rb'))
    for method, opt in zip(methods, opts):
        create_folder(folder / '{}/init'.format(method), exist_ok=True)
        x_init = data[method][0]['x']
        init_env = variable_to_env(x_init)
        init_env.simulate(dt, frame_num, method, opt, q0, v0, a0, f0, require_grad=False, vis_folder='{}/init'.format(method))

        create_folder(folder / '{}/final'.format(method), exist_ok=True)
        x_final = data[method][-1]['x']
        final_env = variable_to_env(x_final)
        final_env.simulate(dt, frame_num, method, opt, q0, v0, a0, f0, require_grad=False, vis_folder='{}/final'.format(method))