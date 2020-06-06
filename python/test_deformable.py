import numpy as np
from pathlib import Path
from py_diff_pd.core.py_diff_pd_core import Deformable, QuadMesh, StdRealVector
from py_diff_pd.common.common import ndarray, create_folder, to_std_real_vector, to_std_map
from py_diff_pd.common.common import print_info
from py_diff_pd.common.display import display_quad_mesh, export_gif
from py_diff_pd.common.mesh import generate_rectangle_obj

if __name__ == '__main__':
    # Uncomment the following line to try random seeds.
    #seed = np.random.randint(1e5)
    seed = 42
    print_info('seed: {}'.format(seed))
    np.random.seed(seed)

    # Hyperparameters.
    youngs_modulus = 1e6
    poissons_ratio = 0.45
    density = 1e4
    method = 'newton'
    cell_nums = (20, 10)
    dx = 0.1
    opt = { 'max_newton_iter': 10, 'max_ls_iter': 10, 'rel_tol': 1e-2, 'verbose': 0 }
    
    # Initialization.
    folder = Path('test_deformable')
    create_folder(folder)
    obj_file_name = folder / 'rectangle.obj'
    generate_rectangle_obj(cell_nums, dx, (0, 0), obj_file_name)

    mesh = QuadMesh()
    mesh.Initialize(str(obj_file_name))

    deformable = Deformable()
    deformable.Initialize(str(obj_file_name), density, 'corotated', youngs_modulus, poissons_ratio)
    # Boundary conditions.
    deformable.SetDirichletBoundaryCondition(0, 0)
    deformable.SetDirichletBoundaryCondition(1, 0)

    dofs = deformable.dofs()
    vertex_num = int(dofs / 2)
    q0 = ndarray(mesh.py_vertices())
    v0 = np.zeros(dofs)

    # Simulation.
    dt = 0.02
    num_frames = 1000
    q = [q0,]
    v = [v0,]
    f_ext = np.zeros((vertex_num, 2))
    f_ext[:, 1] = np.random.random(vertex_num) * density * deformable.cell_volume()
    f_ext = ndarray(f_ext)
    for i in range(num_frames):
        q_cur = np.copy(q[-1])
        deformable.PySaveToMeshFile(to_std_real_vector(q_cur), str(folder / '{:04d}.obj'.format(i)))

        v_cur = np.copy(v[-1])
        f_ext = ndarray(f_ext).ravel()
        q_next_array = StdRealVector(dofs)
        v_next_array = StdRealVector(dofs)
        deformable.PyForward(method, to_std_real_vector(q_cur), to_std_real_vector(v_cur),
            to_std_real_vector(f_ext), dt, to_std_map(opt), q_next_array, v_next_array)

        q_next = ndarray(q_next_array)
        v_next = ndarray(v_next_array)
        q.append(q_next)
        v.append(v_next)

    # Display the results.
    frame_cnt = 0
    frame_skip = 20
    for i in range(0, num_frames, frame_skip):
        mesh = QuadMesh()
        mesh.Initialize(str(folder / '{:04d}.obj'.format(frame_cnt)))
        display_quad_mesh(mesh, xlim=[-0.5, 3], ylim=[-0.5, 2], title='Frame {:04d}'.format(i),
            file_name=folder / '{:04d}.png'.format(i), show=False)
        frame_cnt += 1

    export_gif(folder, '{}.gif'.format(str(folder)), 50)