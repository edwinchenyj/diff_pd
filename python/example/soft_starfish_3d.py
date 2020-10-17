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

def check_hydro_gradients(deformable, x0):
    # Sanity check the gradients of the hydrodynamic forces.
    water_force_weight = np.random.normal(size=dofs)
    def loss_and_grad(x):
        q = x[:dofs]
        v = x[dofs:]
        water_force = ndarray(env._deformable.PyForwardStateForce(q, v))
        loss = water_force.dot(water_force_weight)
        grad_q = StdRealVector(dofs)
        grad_v = StdRealVector(dofs)
        deformable.PyBackwardStateForce(q, v, water_force, water_force_weight, grad_q, grad_v)
        grad = np.zeros(2 * dofs)
        grad[:dofs] = ndarray(grad_q)
        grad[dofs:] = ndarray(grad_v)
        return loss, grad

    eps = 1e-8
    atol = 1e-4
    rtol = 1e-2
    grads_equal = check_gradients(loss_and_grad, x0, eps, rtol=rtol, atol=atol, verbose=True)
    if not grads_equal:
        print_error('ForwardStateForce and BackwardStateForce do not match.')

def visualize_hydro(deformable, bin_mesh_file, img_file, q, v):
    from py_diff_pd.common.renderer import PbrtRenderer
    from py_diff_pd.common.project_path import root_path
    from py_diff_pd.common.tet_mesh import TetMesh3d
    options = {
        'file_name': img_file,
        'light_map': 'uffizi-large.exr',
        'sample': 64,
        'max_depth': 2,
        'camera_pos': (0.15, -0.75, 1.4),
        'camera_lookat': (0, .15, .4)
    }
    renderer = PbrtRenderer(options)

    mesh = TetMesh3d()
    mesh.Initialize(str(bin_mesh_file))
    transforms = [
        ('t', (-0.075, -0.075, 0)),
        ('s', 3),
        ('t', (0.2, 0.4, 0.2))
    ]
    #renderer.add_tri_mesh(mesh, color=(.6, .3, .2),
    #    transforms=transforms, render_tet_edge=False,
    #)

    # Render water force.
    hydro_force = deformable.PyForwardStateForce(q, v)
    f = np.reshape(ndarray(hydro_force), (-1, 3))
    q = np.reshape(ndarray(mesh.py_vertices()), (-1, 3))
    for fi, qi in zip(f, q):
        scale = 1.0
        v0 = qi
        v3 = scale * fi + qi
        v1 = (2 * v0 + v3) / 3
        v2 = (v0 + 2 * v3) / 3
        if np.linalg.norm(fi) == 0: continue
        renderer.add_shape_mesh({
                'name': 'curve',
                'point': ndarray([v0, v1, v2, v3]),
                'width': 0.001
            },
            color=(.2, .6, .3),
            transforms=transforms
        )

    renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj',
        texture_img='chkbd_24_0.7', transforms=[('s', 2)])

    renderer.render()

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    folder = Path('soft_starfish_3d')
    youngs_modulus = 5e5
    poissons_ratio = 0.4
    act_stiffness = 2e6
    env = SoftStarfishEnv3d(seed, folder, {
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
        'act_stiffness': act_stiffness })
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
    frame_num = 4
    a0 = [np.random.uniform(low=1.99, high=2.01, size=act_dofs) for _ in range(frame_num)]
    vertex_num = int(dofs // 3)
    f0 = np.zeros((vertex_num, 3))
    f0 = f0.ravel()
    f0 = [f0 for _ in range(frame_num)]
    _, info = env.simulate(dt, frame_num, methods[2], opts[2], q0, v0, a0, f0, require_grad=False,
        vis_folder='random')

    # Uncomment the following lines to check gradients.
    # q_final = ndarray(info['q'][-1])
    # v_final = ndarray(info['v'][-1])
    # x0 = np.concatenate([q_final, v_final])
    # check_hydro_gradients(deformable, x0)

    # Visualize hydrodynamic forces.
    create_folder(folder / 'vis')
    for i in range(frame_num + 1):
        q = ndarray(info['q'][i])
        v = ndarray(info['v'][i])
        visualize_hydro(deformable, folder / 'random/{:04d}.bin'.format(i),
            folder / 'vis/{:04d}.png'.format(i), q, v)