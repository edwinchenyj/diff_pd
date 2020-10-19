import time
from pathlib import Path
import os

import numpy as np

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.common import create_folder, ndarray
from py_diff_pd.common.tet_mesh import generate_tet_mesh, tet2obj, tetrahedralize, get_boundary_face
from py_diff_pd.common.tet_mesh import get_contact_vertex as get_tet_contact_vertex
from py_diff_pd.core.py_diff_pd_core import TetMesh3d, TetDeformable
from py_diff_pd.common.renderer import PbrtRenderer
from py_diff_pd.common.project_path import root_path

class SoftStarfishEnv3d(EnvBase):
    def __init__(self, seed, folder, options):
        EnvBase.__init__(self, folder)

        np.random.seed(seed)
        create_folder(folder, exist_ok=True)

        youngs_modulus = options['youngs_modulus'] if 'youngs_modulus' in options else 1e6
        poissons_ratio = options['poissons_ratio'] if 'poissons_ratio' in options else 0.49
        act_stiffness = options['act_stiffness'] if 'act_stiffness' in options else 5e5

        # Mesh parameters.
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        density = 1e3
        tmp_bin_file_name = '.tmp.bin'
        obj_file_name = Path(root_path) / 'asset' / 'mesh' / 'starfish_simplified.obj'
        verts, eles = tetrahedralize(obj_file_name, normalize_input=False)
        generate_tet_mesh(verts, eles, tmp_bin_file_name)
        mesh = TetMesh3d()
        mesh.Initialize(str(tmp_bin_file_name))
        deformable = TetDeformable()

        # Rescale the mesh.
        deformable.Initialize(tmp_bin_file_name, density, 'none', youngs_modulus, poissons_ratio)
        os.remove(tmp_bin_file_name)

        # Elasticity.
        deformable.AddPdEnergy('corotated', [2 * mu,], [])
        deformable.AddPdEnergy('volume', [la,], [])

        # Hydrodynamic forces.
        v_rho = 1e3
        v_water = ndarray([0, 0, 0])    # Velocity of the water.
        # Cd_points = (angle, coeff) pairs where angle is normalized to [0, 1].
        Cd_points = ndarray([[0.0, 0.05], [0.4, 0.05], [0.7, 1.85], [1.0, 2.05]])
        # Ct_points = (angle, coeff) pairs where angle is normalized to [-1, 1].
        Ct_points = ndarray([[-1, -0.8], [-0.3, -0.5], [0.3, 0.1], [1, 2.5]])
        # The current Cd and Ct are similar to Figure 2 in SoftCon.
        # surface_faces is a list of (v0, v1, v2) where v0, v1, v2 are the vertex indices of the corners of a boundary face.
        # The order of v0, v1, v2 follows the right-hand rule: if your right hand follows v0 -> v1 -> v2, your thumb will
        # point to the outward normal.
        surface_faces = get_boundary_face(mesh)
        deformable.AddStateForce('hydrodynamics', np.concatenate([[v_rho,], v_water, Cd_points.ravel(), Ct_points.ravel(),
            ndarray(surface_faces).ravel()]))

        # Dirichlet boundary conditions: TODO.

        # Actuators.
        # Range of the center of the starfish: -18mm to +18mm.
        # Width of the limb: -7.5mm to +7.5mm.
        half_limb_width = 7.5 / 1000
        half_center_size = 18 / 1000
        ele_num = mesh.NumOfElements()
        x_pos_act_eles = []
        x_neg_act_eles = []
        y_pos_act_eles = []
        y_neg_act_eles = []
        for ei in range(ele_num):
            v_idx = list(mesh.py_element(ei))
            v = [ndarray(mesh.py_vertex(i)) for i in v_idx]
            v = ndarray(v)
            # Compute the average location.
            v_com = np.mean(v, axis=0)
            x, y, z = v_com
            if np.abs(x) > half_limb_width and np.abs(y) > half_limb_width: continue
            if z > 0: continue
            if x < -half_center_size:
                x_neg_act_eles.append(ei)
            elif x > half_center_size:
                x_pos_act_eles.append(ei)
            elif y < -half_center_size:
                y_neg_act_eles.append(ei)
            elif y > half_center_size:
                y_pos_act_eles.append(ei)

        deformable.AddActuation(act_stiffness, ndarray([1, 0, 0]), x_neg_act_eles)
        deformable.AddActuation(act_stiffness, ndarray([1, 0, 0]), x_pos_act_eles)
        deformable.AddActuation(act_stiffness, ndarray([0, 1, 0]), y_neg_act_eles)
        deformable.AddActuation(act_stiffness, ndarray([0, 1, 0]), y_pos_act_eles)

        # Initial states.
        dofs = deformable.dofs()
        act_dofs = deformable.act_dofs()
        q0 = ndarray(mesh.py_vertices())
        v0 = np.zeros(dofs)
        f_ext = np.zeros(dofs)

        # Data members.
        self._deformable = deformable
        self._q0 = q0
        self._v0 = v0
        self._f_ext = f_ext
        self._youngs_modulus = youngs_modulus
        self._poissons_ratio = poissons_ratio
        self._stepwise_loss = False

        self.__spp = options['spp'] if 'spp' in options else 4

    def material_stiffness_differential(self, youngs_modulus, poissons_ratio):
        jac = self._material_jacobian(youngs_modulus, poissons_ratio)
        jac_total = np.zeros((2, 2))
        jac_total[0] = 2 * jac[1]
        jac_total[1] = jac[0]
        return jac_total

    def is_dirichlet_dof(self, dof):
        # TODO.
        return False

    def _display_mesh(self, mesh_file, file_name):
        options = {
            'file_name': file_name,
            'light_map': 'uffizi-large.exr',
            'sample': self.__spp,
            'max_depth': 2,
            'camera_pos': (0.15, -0.75, 1.4),
            'camera_lookat': (0, .15, .4)
        }
        renderer = PbrtRenderer(options)

        mesh = TetMesh3d()
        mesh.Initialize(mesh_file)
        renderer.add_tri_mesh(mesh, color=(.6, .3, .2),
            transforms=[
                ('t', (-0.075, -0.075, 0)),
                ('s', 3),
                ('t', (0.2, 0.4, 0.2))
            ],
            render_tet_edge=False,
        )

        renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj',
            texture_img='chkbd_24_0.7', transforms=[('s', 2)])

        renderer.render()

    def _loss_and_grad(self, q, v):
        # Compute the center of mass.
        com = np.mean(q.reshape((-1, 3)), axis=0)
        loss = -com[2]
        # Compute grad.
        grad_q = np.zeros(q.size)
        vertex_num = int(q.size // 3)
        grad_q[2::3] = -1.0 / vertex_num
        grad_v = np.zeros(v.size)
        return loss, grad_q, grad_v