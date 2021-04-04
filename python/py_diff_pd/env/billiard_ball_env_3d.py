import time
from pathlib import Path
import os

import numpy as np

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.common import create_folder, ndarray
from py_diff_pd.common.tet_mesh import generate_tet_mesh, tet2obj, tetrahedralize
from py_diff_pd.common.tet_mesh import get_contact_vertex as get_tet_contact_vertex
from py_diff_pd.core.py_diff_pd_core import TetMesh3d, TetDeformable, StdRealVector
from py_diff_pd.common.renderer import PbrtRenderer
from py_diff_pd.common.project_path import root_path

class BilliardBallEnv3d(EnvBase):
    def __init__(self, folder, options):
        EnvBase.__init__(self, folder)

        create_folder(folder, exist_ok=True)

        youngs_modulus = options['youngs_modulus'] if 'youngs_modulus' in options else 1e6
        poissons_ratio = options['poissons_ratio'] if 'poissons_ratio' in options else 0.49
        state_force_parameters = ndarray(options['state_force_parameters'])

        # Obtain the initial billiard ball centers.
        init_ball_positions = ndarray(options['init_positions']).copy()
        init_angular_velocities = ndarray(options['init_angular_velocities']).copy()
        # The sphere has a basic radius of 0.03m (or 3cm).
        radius = float(options['radius']) if 'radius' in options else 0.03
        self.__radius = radius
        # Mesh parameters.
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        density = 1e3

        tmp_bin_file_name = '.tmp.bin'
        obj_file_name = Path(root_path) / 'asset' / 'mesh' / 'sphere.obj'
        verts, eles = tetrahedralize(obj_file_name, normalize_input=False)
        # Rescale the verts --- the sphere, by default, is centered at the origin with a radius of 3 cm.
        verts *= radius / 0.03
        # Find the central location of the sphere -- the sphere has a radius of 0.03 m (3 cm).
        vert_norm = np.sum(verts ** 2, axis=1)
        center_idx = np.argmin(vert_norm)
        verts_corrected = np.copy(verts)
        verts_corrected[center_idx] = 0
        # Check out all the adjacent tet meshes.
        for ei in eles:
            if center_idx in ei:
                assert np.linalg.det(verts[ei[1:]] - verts[ei[0]]) * np.linalg.det(verts_corrected[ei[1:]] - verts_corrected[ei[0]]) > 0
        verts = verts_corrected
        # Now we can safely use (verts, eles).
        generate_tet_mesh(verts, eles, tmp_bin_file_name)
        single_ball_mesh = TetMesh3d()
        single_ball_mesh.Initialize(str(tmp_bin_file_name))
        os.remove(tmp_bin_file_name)
        # Collisions.
        friction_node_idx = get_tet_contact_vertex(single_ball_mesh)
        # Uncomment the code below if you would like to display the contact set for a sanity check:
        '''
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        v = ndarray([ndarray(mesh.py_vertex(idx)) for idx in friction_node_idx])
        ax.scatter(v[:, 0], v[:, 1], v[:, 2])
        plt.show()
        '''
        # Depending on the number of balls, we will dumplicate the vertices.
        num_balls = init_ball_positions.shape[0]
        self.__num_balls = num_balls
        num_ball_vertices = verts.shape[0]
        self.__num_ball_vertices = num_ball_vertices
        all_verts = []
        all_eles = []
        all_velocities = []
        all_center_indices = []
        all_friction_node_indices = []
        for i, (c, w) in enumerate(zip(init_ball_positions, init_angular_velocities)):
            all_eles.append(eles + num_ball_vertices * i)
            all_verts.append(ndarray(verts + c))
            wx, wy, wz = w
            W = ndarray([
                [0, -wz, wy],
                [wz, 0, -wx],
                [-wy, wx, 0]
            ])
            all_velocities.append(verts @ W.T)
            all_center_indices.append(num_ball_vertices * i + center_idx)
            all_friction_node_indices += [idx + num_ball_vertices * i for idx in friction_node_idx]
        # Compute gradients.
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
        self.__grad_v_grad_w = ndarray([(verts @ dW.T).ravel() for dW in [dWx, dWy, dWz]]).T

        all_verts = np.vstack(all_verts)
        all_eles = np.vstack(all_eles)
        all_velocities = np.vstack(all_velocities)
        generate_tet_mesh(all_verts, all_eles, tmp_bin_file_name)
        mesh = TetMesh3d()
        mesh.Initialize(str(tmp_bin_file_name))
        deformable = TetDeformable()
        deformable.Initialize(tmp_bin_file_name, density, 'none', youngs_modulus, poissons_ratio)
        os.remove(tmp_bin_file_name)

        # Elasticity.
        deformable.AddPdEnergy('corotated', [2 * mu,], [])
        deformable.AddPdEnergy('volume', [la,], [])
        # State-based forces.
        deformable.AddStateForce('gravity', [0, 0, -9.81])
        deformable.AddStateForce('billiard_ball', np.concatenate([[radius, num_ball_vertices],
            state_force_parameters[:num_balls], state_force_parameters[num_balls:]]))

        # Friction_node_idx = all vertices on the edge.
        deformable.SetFrictionalBoundary('planar', [0.0, 0.0, 1.0, radius], all_friction_node_indices)

        # Uncomment it if you prefer having Dirichlet boundary conditions.
        # Dirichlet boundary conditions: the z axis for the central point should be fixed.
        # for c in all_center_indices:
        #     deformable.SetDirichletBoundaryCondition(3 * int(c) + 2, 0.0)

        # Initial states.
        dofs = deformable.dofs()
        act_dofs = deformable.act_dofs()
        q0 = ndarray(all_verts).ravel()
        v0 = ndarray(all_velocities).ravel()
        f_ext = np.zeros(dofs)

        # Data members.
        self._deformable = deformable
        self._q0 = q0
        self._v0 = v0
        self._f_ext = f_ext
        self._youngs_modulus = youngs_modulus
        self._poissons_ratio = poissons_ratio
        self._state_force_parameters = state_force_parameters
        self._stepwise_loss = True

        self.__spp = options['spp'] if 'spp' in options else 4
        self.__center_idx = center_idx
        self.__reference_positions = ndarray(options['reference_positions']).copy()
        self.__substeps = int(options['substeps'])

    def backprop_init_velocities(self, dl_dv):
        # Input: dl_dv.
        # Output: dl_domega.
        dl_dv_reshaped = dl_dv.reshape((self.__num_balls, -1))
        return ndarray(dl_dv_reshaped @ self.__grad_v_grad_w).ravel()

    def material_stiffness_differential(self, youngs_modulus, poissons_ratio):
        jac = self._material_jacobian(youngs_modulus, poissons_ratio)
        jac_total = np.zeros((2, 2))
        jac_total[0] = 2 * jac[1]
        jac_total[1] = jac[0]
        return jac_total

    def is_dirichlet_dof(self, dof):
        return dof == 3 * self.__center_idx + 2

    def _display_mesh(self, mesh_file, file_name):
        options = {
            'file_name': file_name,
            'light_map': 'uffizi-large.exr',
            'sample': self._spp,
            'max_depth': 2,
            'camera_pos': ndarray([0.55, 0.34, 1]),
            'camera_lookat': ndarray([0.55, 0.33, 0]),
            'camera_up': ndarray([0, 1, 0]),
            'resolution': (800, 600),
            'fov': 30,
            'sample': self.__spp
        }
        renderer = PbrtRenderer(options)

        mesh = TetMesh3d()
        mesh.Initialize(mesh_file)
        tet2obj(mesh, obj_file_name=self._folder / '.tmp.obj')
        renderer.add_tri_mesh(self._folder / '.tmp.obj', color=[.1, .7, .2], render_tet_edge=True)
        renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj',
            texture_img='chkbd_24_0.7', transforms=[('t', (0.55, 0.33, -self.__radius))])

        renderer.render()

        os.remove(self._folder / '.tmp.obj')

    def _stepwise_loss_and_grad(self, q, v, i):
        if i % self.__substeps != 0:
            return 0, ndarray(np.zeros(q.size)), ndarray(np.zeros(v.size))
        reference_positions = ndarray([p[int(i // self.__substeps)] for p in self.__reference_positions]).copy()
        offset = np.mean(q.reshape((self.__num_balls, self.__num_ball_vertices, 3)), axis=1) - reference_positions
        loss = 0.5 * np.sum(offset ** 2)
        grad_q = np.zeros((self.__num_balls, self.__num_ball_vertices, 3))
        for b in range(self.__num_balls):
            contribution = offset[b] / self.__num_ball_vertices
            grad_q[b] += contribution
        grad_q = ndarray(grad_q).ravel()
        grad_v = ndarray(np.zeros(v.size))
        return loss, grad_q, grad_v