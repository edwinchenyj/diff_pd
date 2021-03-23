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
from py_diff_pd.env.sim_to_real_env_3d import SimToRealEnv3d
from py_diff_pd.common.project_path import root_path

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    folder = Path('sim_to_real_experiment')

    motion_data_folder = Path(root_path) / 'python/example/sim_to_real_calibration/experiment'
    substeps = 4
    start_frame = 160   # This is the frame we estimate the object starts rolling. Could be outside the range of the camera.
    end_frame = 260     # This is typically the last frame in the `python/example/sim_to_real_calibration/experiment/` folder.

    env = SimToRealEnv3d(seed, folder, {
        'camera_pos': [0, -0.3, 0],
        'camera_yaw': 0,
        'camera_pitch': 0,
        'camera_alpha': 1000,
        'experiment_folder': motion_data_folder,
        'img_height': 720,
        'img_width': 1280,
        'substeps': substeps,
        'start_frame': start_frame
    })
    deformable = env.deformable()

    # Optimization parameters.
    thread_ct = 6
    opt = { 'max_pd_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
        'use_bfgs': 1, 'bfgs_history_size': 10 }
    method = 'pd_eigen'

    dt = (1 / 60) / substeps
    frame_num = (end_frame - start_frame) * substeps

    # Initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position()
    q0_offset = q0.reshape((-1, 3)) - np.mean(q0.reshape((-1, 3)), axis=0)
    v0 = np.zeros(dofs)
    a0 = [np.zeros(act_dofs) for _ in range(frame_num)]
    f0 = [np.zeros(dofs) for _ in range(frame_num)]

    # Variables to be optimized:
    # - Initial com: 3D. The initial location of the center of mass.
    # - Initial yaw: only yaw is needed because the object was originally in contact with the table with its black face.
    # - Initial omega: 3D. The initial angular velocity.
    def variable_to_initial_states(x):
        x = ndarray(x).copy().ravel()
        assert x.size == 7
        init_com = x[:3]
        init_yaw = x[3]
        init_omega = x[4:]
        # Compute init q and init v.
        c_yaw, s_yaw = np.cos(init_yaw), np.sin(init_yaw)
        init_R = ndarray([
            [c_yaw, -s_yaw, 0],
            [s_yaw, c_yaw, 0],
            [0, 0, 1]
        ])
        init_q = (q0_offset @ init_R.T + init_com).ravel()
        wx, wy, wz = init_omega
        W = ndarray([
            [0, -wz, wy],
            [wz, 0, -wx],
            [-wy, wx, 0],
        ])  # Skew(init_omega)
        # v = W (init_q - init_com).
        init_v = (((q0_offset @ init_R.T) + ndarray([0.015, 0, 0.015])) @ W.T).ravel()
        return ndarray(init_q).copy(), ndarray(init_v).copy()

    def variable_to_initial_states_gradient(x, grad_init_q, grad_init_v):
        x = ndarray(x).copy().ravel()
        grad = np.zeros(x.size)
        # Init com.
        grad[:3] = np.sum(grad_init_q.reshape((-1, 3)), axis=0)
        # Init yaw.
        init_yaw = x[3]
        c_yaw, s_yaw = np.cos(init_yaw), np.sin(init_yaw)
        R = ndarray([
            [c_yaw, -s_yaw, 0],
            [s_yaw, c_yaw, 0],
            [0, 0, 1]
        ])
        dR = ndarray([
            [-s_yaw, -c_yaw, 0],
            [c_yaw, -s_yaw, 0],
            [0, 0, 0]
        ])
        wx, wy, wz = x[4:]
        W = ndarray([
            [0, -wz, wy],
            [wz, 0, -wx],
            [-wy, wx, 0],
        ])
        grad[3] = (q0_offset @ dR.T).ravel().dot(grad_init_q) + (q0_offset @ dR.T @ W.T).ravel().dot(grad_init_v)
        # Init omega.
        dWx = ndarray([
            [0, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        dWy = ndarray([
            [0, 0, 1],
            [0, 0, 0],
            [-1, 0, 0]
        ])
        dWz = ndarray([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ])
        grad[4] = (q0_offset @ R.T @ dWx.T).ravel().dot(grad_init_v)
        grad[5] = (q0_offset @ R.T @ dWy.T).ravel().dot(grad_init_v)
        grad[6] = (q0_offset @ R.T @ dWz.T).ravel().dot(grad_init_v)
        return grad

    # Optimization.
    # Variables to be optimized:
    # Init_com (3D), init_yaw, init_omega (3D).
    # The estimated value: 0.3, 0, 0, np.pi / 2, 0, 3pi, 0.
    x_lb = ndarray([0.0, -0.2, 0.001, np.pi / 2 - 0.2, -5, -6 * np.pi, -5])
    x_ub = ndarray([0.2, -0.1, 0.001, np.pi / 2 + 0.2, 5, -2 * np.pi, 5])
    x_init = np.random.uniform(x_lb, x_ub)
    data = {}
    data[method] = []
    def loss_and_grad(x):
        init_q, init_v = variable_to_initial_states(x)
        loss, grad, info = env.simulate(dt, frame_num, method, opt, init_q, init_v, a0, f0, require_grad=True, vis_folder=None)
        # Assemble the gradients.
        grad_init_q = grad[0]
        grad_init_v = grad[1]
        grad_x = variable_to_initial_states_gradient(x, grad_init_q, grad_init_v)
        print('loss: {:8.3f}, |grad|: {:8.3f}, forward time: {:6.3f}s, backward time: {:6.3f}s'.format(
            loss, np.linalg.norm(grad_x), info['forward_time'], info['backward_time']))
        print(grad_x)
        single_data = {}
        single_data['loss'] = loss
        single_data['grad'] = np.copy(grad_x)
        single_data['x'] = np.copy(x)
        single_data['forward_time'] = info['forward_time']
        single_data['backward_time'] = info['backward_time']
        data[method].append(single_data)
        return loss, np.copy(grad_x)
    # Use the two lines below to sanity check the gradients.
    # Note that you might need to fine tune the rel_tol in opt to make it work.
    # We have observed our gradients are correct when applied to few frames without contact.
    '''
    from py_diff_pd.common.grad_check import check_gradients
    check_gradients(loss_and_grad, x_init, eps=1e-4)
    data[method] = []
    '''

    # Normalize the loss.
    rand_state = np.random.get_state()
    random_guess_num = 1
    random_loss = []
    best_loss = np.inf
    best_x_init = None
    for _ in range(random_guess_num):
        x_rand = np.random.uniform(low=x_lb, high=x_ub)
        init_q, init_v = variable_to_initial_states(x_rand)
        loss, _ = env.simulate(dt, frame_num, method, opt, init_q, init_v, a0, f0, require_grad=False, vis_folder=None)
        print('loss: {:3f}'.format(loss))
        random_loss.append(loss)
        if loss < best_loss:
            best_loss = loss
            best_x_init = x_rand
    loss_range = ndarray([0, np.mean(random_loss)])
    data['loss_range'] = loss_range
    print_info('Loss range: {:3f}, {:3f}'.format(loss_range[0], loss_range[1]))
    np.random.set_state(rand_state)
    x_init = best_x_init
    print('x_init:', x_init)

    # Visualize initial guess.
    init_q, init_v = variable_to_initial_states(x_init)
    env.simulate(dt, frame_num, method, opt, init_q, init_v, a0, f0, require_grad=False, vis_folder='init',
        render_frame_skip=substeps * 10)
    bounds = scipy.optimize.Bounds(x_lb, x_ub)

    # Optimization.
    t0 = time.time()
    result = scipy.optimize.minimize(loss_and_grad, np.copy(x_init),
        method='L-BFGS-B', jac=True, bounds=bounds, options={ 'ftol': 1e-3 })
    t1 = time.time()
    assert result.success
    x_final = result.x
    print_info('Optimizing with {} finished in {:6.3f} seconds'.format(method, t1 - t0))
    pickle.dump(data, open(folder / 'data_{:04d}_threads.bin'.format(thread_ct), 'wb'))

    # Visualize results.
    final_q, final_v = variable_to_initial_states(x_final)
    env.simulate(dt, frame_num, method, opt, final_q, final_v, a0, f0, require_grad=False, vis_folder=method,
        render_frame_skip=substeps * 10)