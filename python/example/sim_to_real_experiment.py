import sys
sys.path.append('../')

from pathlib import Path
import time
import numpy as np
import scipy.optimize
import pickle

from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error, print_warning
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
    # This is the frame we estimate the object starts rolling. Could be outside the range of the camera.
    start_frame = 160
    # This is typically the last frame in the `python/example/sim_to_real_calibration/experiment/` folder.
    end_frame = 190
    # The video was taken at 60 fps.
    dt = (1 / 60) / substeps
    frame_num = (end_frame - start_frame) * substeps
    # We estimate from the video that the initial force is applied for roughly 10 frames.
    init_force_frame_num = 10 * substeps
    # Optimization parameters.
    thread_ct = 6
    opt = { 'max_pd_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
        'use_bfgs': 1, 'bfgs_history_size': 10 }
    method = 'pd_eigen'

    # Build an environment.
    def get_env(camera_parameters):
        camera_parameters = ndarray(camera_parameters).copy().ravel()
        camera_pos = camera_parameters[:3]
        camera_yaw, camera_pitch, camera_alpha = camera_parameters[3:6]
        env = SimToRealEnv3d(folder, {
            'camera_pos': camera_pos,
            'camera_yaw': camera_yaw,
            'camera_pitch': camera_pitch,
            'camera_alpha': camera_alpha,
            'experiment_folder': motion_data_folder,
            'img_height': 720,
            'img_width': 1280,
            'substeps': substeps,
            'start_frame': start_frame
        })
        return env

    # Obtain some educated guess of the camera parameters.
    camera_data_files = Path(motion_data_folder).glob('*.data')
    all_alpha = []
    for f in camera_data_files:
        info = pickle.load(open(f, 'rb'))
        all_alpha.append(info['alpha'])
        all_alpha.append(info['beta'])
    init_camera_alpha = np.mean(all_alpha)
    init_camera_pos = ndarray([0, 0, 0])
    init_camera_yaw = 0
    init_camera_pitch = 0
    init_env = get_env(np.concatenate([init_camera_pos, [init_camera_yaw, init_camera_pitch, init_camera_alpha]]))
    deformable = init_env.deformable()

    # Build the initial state of the object.
    # Initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    v0 = np.zeros(dofs)
    a0 = [np.zeros(act_dofs) for _ in range(frame_num)]

    # Determine initial q --- this is based on the decision variables.
    # Our decision variables include the following:
    # - init_com: 3D. The center of mass.
    # - init_yaw: 1D.
    # - init_f: 3D. The initial force applied to force_node_idx below.
    # - camera_pos: 3D. The location of the camera.
    # - camera_yaw: 1D.
    # - camera_pitch: 1D.
    # - camera_alpha: 1D.
    q0 = init_env.default_init_position()
    q0_offset = q0.reshape((-1, 3)) - np.mean(q0.reshape((-1, 3)), axis=0)
    def get_init_q(x):
        x = ndarray(x).copy().ravel()
        init_com = x[:3]
        init_yaw = x[3]
        c_yaw, s_yaw = np.cos(init_yaw), np.sin(init_yaw)
        init_R = ndarray([
            [c_yaw, -s_yaw, 0],
            [s_yaw, c_yaw, 0],
            [0, 0, 1]
        ])
        init_q = (q0_offset @ init_R.T + init_com).ravel()
        return ndarray(init_q).copy()

    def get_init_q_gradient(x, grad_init_q):
        x = ndarray(x).copy().ravel()
        grad = np.zeros(x.size)
        # Init com.
        grad[:3] = np.sum(grad_init_q.reshape((-1, 3)), axis=0)
        # Init yaw.
        init_yaw = x[3]
        c_yaw, s_yaw = np.cos(init_yaw), np.sin(init_yaw)
        dR = ndarray([
            [-s_yaw, -c_yaw, 0],
            [c_yaw, -s_yaw, 0],
            [0, 0, 0]
        ])
        grad[3] = (q0_offset @ dR.T).ravel().dot(grad_init_q)
        return ndarray(grad).copy()

    # Determine external force f0 --- this is also based on the decision variables.
    # Find the top q locations to apply external forces.
    q_max_z = np.max(q0_offset[:, 2])
    force_node_idx = []
    for i in range(q0_offset.shape[0]):
        if np.abs(q0_offset[i, 2] - q_max_z) < 1e-5:
            force_node_idx.append(i)

    def get_external_f(x):
        x = ndarray(x).copy().ravel()
        init_f = x[4:7] / 1000  # Convert from Newton to mN.
        f0 = [np.zeros(dofs) for _ in range(frame_num)]
        for i in range(init_force_frame_num):
            fi = f0[i].reshape((-1, 3))
            fi[force_node_idx] = init_f
            f0[i] = ndarray(fi).copy().ravel()
        return f0

    def get_external_f_gradient(x, grad_f):
        x = ndarray(x).copy().ravel()
        grad = np.zeros(x.size)
        for i in range(init_force_frame_num):
            dfi = grad_f[i].reshape((-1, 3))
            grad[4:7] += np.sum(dfi[force_node_idx], axis=0) / 1000
        return ndarray(grad).copy()

    # Optimization.
    # Variables to be optimized:
    # - init_com: 3D. The center of mass.
    # - init_yaw: 1D.
    # - init_f: 3D. The initial force applied to force_node_idx below. Note that we use mN instead of Newton as the unit.
    # - camera_pos: 3D. The location of the camera.
    # - camera_yaw: 1D.
    # - camera_pitch: 1D.
    # - camera_alpha: 1D.
    x_ref = ndarray(np.concatenate([
        [0.1, 0.2, 0.0, np.pi / 2, -3, 0, 0],
        init_camera_pos,
        [init_camera_yaw, init_camera_pitch, init_camera_alpha]
    ]))
    x_lb = ndarray(np.concatenate([
        [0.05, 0.15, 0.0, np.pi / 2 - 0.2, -3.5, -0.1, 0],
        [init_camera_pos[0], init_camera_pos[1] - 0.05, init_camera_pos[2] - 0.05],
        [init_camera_yaw - 0.2, init_camera_pitch - 0.2, init_camera_alpha - 300]
    ]))
    x_ub = ndarray(np.concatenate([
        [0.15, 0.25, 0.0, np.pi / 2 + 0.2, -2.5, 0.1, 0],
        [init_camera_pos[0], init_camera_pos[1] + 0.05, init_camera_pos[2] + 0.05],
        [init_camera_yaw + 0.2, init_camera_pitch + 0.2, init_camera_alpha + 300]
    ]))
    x_fixed = np.array([False, False, True, False, False, False, True, True, False, False, False, False, False])

    # Define the loss and grad function.
    data = {}
    data[method] = []
    def loss_and_grad(x_reduced):
        x_full = ndarray(x_ref).copy().ravel()
        x_full[~x_fixed] = x_reduced
        env = get_env(x_full[7:13])
        init_q = get_init_q(x_full)
        init_f = get_external_f(x_full)
        loss, grad, info = env.simulate(dt, frame_num, method, opt, init_q, v0, a0, init_f, require_grad=True, vis_folder=None)
        # Assemble the gradients.
        grad_init_q = grad[0]
        grad_f = grad[3]
        grad_x_full = np.zeros(x_full.size)
        grad_x_full += get_init_q_gradient(x_full, grad_init_q)
        grad_x_full += get_external_f_gradient(x_full, grad_f)
        grad_x_full[7:10] = info['grad_custom']['camera_loc']
        grad_x_full[10] = info['grad_custom']['camera_yaw']
        grad_x_full[11] = info['grad_custom']['camera_pitch']
        grad_x_full[12] = info['grad_custom']['camera_alpha']
        grad_x_reduced = grad_x_full[~x_fixed]
        print('loss: {:8.3f}, |grad|: {:8.3f}, forward time: {:6.3f}s, backward time: {:6.3f}s'.format(
            loss, np.linalg.norm(grad_x_reduced), info['forward_time'], info['backward_time']))
        single_data = {}
        single_data['loss'] = loss
        single_data['grad'] = np.copy(grad_x_reduced)
        single_data['x'] = np.copy(x_reduced)
        single_data['forward_time'] = info['forward_time']
        single_data['backward_time'] = info['backward_time']
        data[method].append(single_data)
        return loss, np.copy(grad_x_reduced)

    # Sanity check the gradients.
    #check_gradients(loss_and_grad, x_ref[~x_fixed], eps=1e-6, skip_var=lambda i: i <= 4)

    # Optimization starts here.
    x_lb_reduced = x_lb[~x_fixed]
    x_ub_reduced = x_ub[~x_fixed]
    bounds = scipy.optimize.Bounds(x_lb_reduced, x_ub_reduced)

    # Normalize the loss.
    rand_state = np.random.get_state()
    random_guess_num = 16
    random_loss = []
    best_loss = np.inf
    best_x_init = None
    for _ in range(random_guess_num):
        x_rand = np.random.uniform(low=x_lb_reduced, high=x_ub_reduced)
        x_full = ndarray(x_ref).copy().ravel()
        x_full[~x_fixed] = x_rand
        env = get_env(x_full[7:13])
        init_q = get_init_q(x_full)
        init_f = get_external_f(x_full)
        loss, _ = env.simulate(dt, frame_num, method, opt, init_q, v0, a0, init_f, require_grad=False, vis_folder=None)
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
    x_full = ndarray(x_ref).copy().ravel()
    x_full[~x_fixed] = best_x_init
    env = get_env(x_full[7:13])
    init_q = get_init_q(x_full)
    init_f = get_external_f(x_full)
    env.simulate(dt, frame_num, method, opt, init_q, v0, a0, init_f, require_grad=False, vis_folder='init',
        render_frame_skip=substeps)

    # Optimization.
    t0 = time.time()
    result = scipy.optimize.minimize(loss_and_grad, np.copy(x_init),
        method='L-BFGS-B', jac=True, bounds=bounds, options={ 'ftol': 1e-3 })
    t1 = time.time()
    if not result.success:
        print_warning('Optimization is not successful. Using the last iteration results.')
    x_final = result.x
    print_info('Optimizing with {} finished in {:6.3f} seconds'.format(method, t1 - t0))
    pickle.dump(data, open(folder / 'data_{:04d}_threads.bin'.format(thread_ct), 'wb'))

    # Visualize results.
    x_full = ndarray(x_ref).copy().ravel()
    x_full[~x_fixed] = x_final
    env = get_env(x_full[7:13])
    init_q = get_init_q(x_full)
    init_f = get_external_f(x_full)
    env.simulate(dt, frame_num, method, opt, init_q, v0, a0, init_f, require_grad=False, vis_folder=method,
        render_frame_skip=substeps)

    '''
    data = pickle.load(open(folder / 'data_{:04d}_threads.bin'.format(thread_ct), 'rb'))
    single_data = data[method][-1]
    def loss(x):
        x_final = single_data['x']
        # Visualize results.
        x_full = ndarray(x_ref).copy().ravel()
        x_full[~x_fixed] = x
        env = get_env(x_full[7:13])
        init_q = get_init_q(x_full)
        init_f = get_external_f(x_full)
        l, _ = env.simulate(dt, frame_num, method, opt, init_q, v0, a0, init_f, require_grad=False, vis_folder=None)
        return l
    print(loss(single_data['x']))
    print(loss(data[method][0]['x']))
    '''