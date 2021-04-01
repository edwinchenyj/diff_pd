import sys
sys.path.append('../')

from pathlib import Path
import time
import numpy as np
import scipy.optimize
import pickle
import matplotlib.pyplot as plt

from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error, print_warning
from py_diff_pd.common.grad_check import check_gradients
from py_diff_pd.core.py_diff_pd_core import StdRealVector
from py_diff_pd.env.billiard_ball_env_3d import BilliardBallEnv3d
from py_diff_pd.common.project_path import root_path

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    folder = Path('billiard_ball_3d')

    # Simulation parameters.
    substeps = 4
    dt = (1 / 60) / substeps
    newton_method = 'newton_pcg'
    pd_method = 'pd_eigen'
    thread_ct = 6
    newton_opt = { 'max_newton_iter': 1000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-9,
        'verbose': 0, 'thread_ct': thread_ct }
    pd_opt = { 'max_pd_iter': 1000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-9,
        'verbose': 0, 'thread_ct': thread_ct, 'use_bfgs': 1, 'bfgs_history_size': 10 }

    # Extract the initial information of the balls.
    ball_radius = 0.06858 / 2   # In meters and from measurement/googling the diameter of a tennis ball.
    experiment_data_folder = Path(root_path) / 'python/example/billiard_ball_calibration/experiment'
    ball_xy_positions = pickle.load(open(experiment_data_folder / 'ball_xy_positions.data', 'rb'))
    active_frame = np.min([(len(ball_xy_positions) - 1), 50])
    ball_xy_positions = ball_xy_positions[:active_frame + 1]
    frame_num = active_frame * substeps
    # Unlike in calibration, the height is set to be 0 here.
    ball_0_positions = [(pos[0, 0], pos[0, 1], 0) for _, pos in ball_xy_positions]
    ball_1_positions = [(pos[1, 0], pos[1, 1], 0) for _, pos in ball_xy_positions]
    ball_0_positions = ndarray(ball_0_positions).copy()
    ball_1_positions = ndarray(ball_1_positions).copy()
    ball_positions = [ball_0_positions, ball_1_positions]
    # Extract the initial position and velocity of each ball.
    init_positions = []
    init_angular_velocities = []
    for b in ball_positions:
        init_positions.append(b[0])
        # Estimate the angular velocity using the first 5 frames.
        # Dist = omega * time * radius.
        steps = 5
        offset = b[steps] - b[0]
        dist = np.linalg.norm(offset)
        omega_mag = dist / ball_radius / (dt * substeps * steps)
        if dist < 5e-3:
            w = [0, 0, 0]
        else:
            w = ndarray([-offset[1], offset[0], 0]) / dist * omega_mag
        init_angular_velocities.append(w)
    init_positions = ndarray(init_positions)
    init_angular_velocities = ndarray(init_angular_velocities)

    # Build the environment.
    env = BilliardBallEnv3d(folder, {
        'init_positions': init_positions,
        'init_angular_velocities': init_angular_velocities,
        'radius': ball_radius,
        'reference_positions': ball_positions,
        'substeps': substeps,
    })
    deformable = env.deformable()
    # Initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position()
    v0 = env.default_init_velocity()
    a0 = [np.zeros(act_dofs) for _ in range(frame_num)]
    f0 = [np.zeros(dofs) for _ in range(frame_num)]

    # Uncomment the code below to sanity check the gradients.
    '''
    q_perturb = np.random.normal(size=q0.size, scale=1e-3)
    v_perturb = np.random.normal(size=v0.size, scale=1e-2)
    def loss_and_grad_sanity_check(x):
        x = ndarray(x).copy().ravel()
        q = q0 + q_perturb * x[0]
        v = v0 + v_perturb * x[1]
        loss, grad, info = env.simulate(dt, 30, (pd_method, newton_method),
            (pd_opt, newton_opt), q, v, a0[:30], f0[:30], require_grad=True)
        g = np.zeros(x.size)
        g[0] = grad[0].dot(q_perturb)
        g[1] = grad[1].dot(v_perturb)
        return loss, ndarray(g).copy().ravel()
    check_gradients(loss_and_grad_sanity_check, ndarray([0.1, 0.2]), eps=1e-3)
    '''

    # Decision variables to optimize:
    # - theta and scale of the initial angular velocity of the balls.
    # - Initial angular velocity of the ball.
    def get_init_state(x):
        x = ndarray(x).copy().ravel()
        assert x.size == 4
        theta0, scale0, theta1, scale1 = x
        c0, s0 = np.cos(theta0), np.sin(theta0)
        c1, s1 = np.cos(theta1), np.sin(theta1)
        w = ndarray([[c0 * scale0, s0 * scale0, 0],
            [c1 * scale1, s1 * scale1, 0]])
        e = BilliardBallEnv3d(folder, {
            'init_positions': init_positions,
            'init_angular_velocities': w,
            'radius': ball_radius,
            'reference_positions': ball_positions,
            'substeps': substeps,
        })
        dw_dx = ndarray([
            [scale0 * -s0, c0, 0, 0],
            [scale0 * c0, s0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, scale1 * -s1, c1],
            [0, 0, scale1 * c1, s1],
            [0, 0, 0, 0],
        ])
        info = { 'env': e, 'v0': e.default_init_velocity(), 'dw_dx': dw_dx }
        return info

    # Optimization.
    init_theta0 = np.arctan2(init_angular_velocities[0, 1], init_angular_velocities[0, 0])
    init_scale0 = np.linalg.norm(init_angular_velocities[0])
    init_theta1 = np.arctan2(init_angular_velocities[1, 1], init_angular_velocities[1, 0])
    init_scale1 = np.linalg.norm(init_angular_velocities[1])
    x_lower = ndarray([
        init_theta0 - 0.05, init_scale0 * 0.9, init_theta1 - 0.05, init_scale1 * 0.9
    ])
    x_upper = ndarray([
        init_theta0 + 0.05, init_scale0 * 3.0, init_theta1 + 0.05, init_scale1 * 3.0
    ])
    bounds = scipy.optimize.Bounds(x_lower, x_upper)
    x_init = np.random.uniform(low=x_lower, high=x_upper)

    data = []
    def loss_and_grad(x):
        init_info = get_init_state(x)
        e = init_info['env']
        v = init_info['v0']
        loss, grad, info = e.simulate(dt, frame_num, (pd_method, newton_method), (pd_opt, newton_opt), q0, v, a0, f0, require_grad=True)
        g = e.backprop_init_velocities(grad[1]) @ init_info['dw_dx']
        print('loss: {:8.3f}, |grad|: {:8.3f}, forward time: {:6.3f}s, backward time: {:6.3f}s'.format(
            loss, np.linalg.norm(g), info['forward_time'], info['backward_time']))
        single_data = {}
        single_data['loss'] = loss
        single_data['grad'] = np.copy(g)
        single_data['x'] = np.copy(x)
        single_data['forward_time'] = info['forward_time']
        single_data['backward_time'] = info['backward_time']
        data.append(single_data)
        return loss, ndarray(g).copy().ravel()

    # Sanity check the gradients.
    # check_gradients(loss_and_grad, x_init, eps=1e-3)

    # Visualize the initial solution.
    if not (folder / 'init/0000.png').is_file():
        info = get_init_state(x_init)
        e_init = info['env']
        v_init = info['v0']
        _, info = e_init.simulate(dt, frame_num, pd_method, pd_opt, q0, v_init, a0, f0, require_grad=False, vis_folder='init',
            render_frame_skip=substeps)
        pickle.dump(info, open(folder / 'init/info.data', 'wb'))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        q = info['q']
        traj = ndarray([np.mean(qi.reshape((2, -1, 3)), axis=1) for qi in q])
        ax.plot(traj[:, 0, 0], traj[:, 0, 1], 'r+', label='ball_0_sim')
        ax.plot(traj[:, 1, 0], traj[:, 1, 1], 'b+', label='ball_1_sim')
        ax.plot(ball_0_positions[:, 0], ball_0_positions[:, 1], 'tab:red', label='ball_0_real')
        ax.plot(ball_1_positions[:, 0], ball_1_positions[:, 1], 'tab:blue', label='ball_1_real')
        ax.legend()
        plt.show()
        fig.savefig(folder / 'init/compare.png')

    # Optimization.
    t0 = time.time()
    def callback(xk):
        print_info('Another iteration is finished.')
    result = scipy.optimize.minimize(loss_and_grad, np.copy(x_init),
        method='L-BFGS-B', jac=True, bounds=bounds, options={ 'ftol': 1e-3, 'maxfun': 40, 'maxiter': 10 }, callback=callback)
    t1 = time.time()
    if not result.success:
        print_warning('Optimization is not successful. Using the last iteration results.')
        idx = np.argmin([d['loss'] for d in data])
        print_warning('Using loss =', data[idx]['loss'])
        x_final = data[idx]['x']
    else:
        x_final = result.x
    print_info('Optimizing with {} finished in {:6.3f} seconds'.format(pd_method, t1 - t0))
    pickle.dump(data, open(folder / 'data_{:04d}_threads.bin'.format(thread_ct), 'wb'))

    # Visualize the final results.
    if not (folder / pd_method / '0000.png').is_file():
        info = get_init_state(x_final)
        e_init = info['env']
        v_init = info['v0']
        _, info = e_init.simulate(dt, frame_num, pd_method, pd_opt, q0, v_init, a0, f0, require_grad=False, vis_folder=pd_method,
            render_frame_skip=substeps)
        pickle.dump(info, open(folder / pd_method / 'info.data', 'wb'))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        q = info['q']
        traj = ndarray([np.mean(qi.reshape((2, -1, 3)), axis=1) for qi in q])
        ax.plot(traj[:, 0, 0], traj[:, 0, 1], 'r+', label='ball_0_sim')
        ax.plot(traj[:, 1, 0], traj[:, 1, 1], 'b+', label='ball_1_sim')
        ax.plot(ball_0_positions[:, 0], ball_0_positions[:, 1], 'tab:red', label='ball_0_real')
        ax.plot(ball_1_positions[:, 0], ball_1_positions[:, 1], 'tab:blue', label='ball_1_real')
        ax.legend()
        plt.show()
        fig.savefig(folder / pd_method / 'compare.png')