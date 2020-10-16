import time
from pathlib import Path
import os

import numpy as np

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.common import create_folder, ndarray
from py_diff_pd.common.tet_mesh import generate_tet_mesh, tet2obj, tetrahedralize
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

        # Mesh parameters.
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        density = 1e3
        tmp_bin_file_name = '.tmp.bin'
        obj_file_name = Path(root_path) / 'asset' / 'mesh' / 'starfish.obj'
        verts, eles = tetrahedralize(obj_file_name)
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

        # Dirichlet boundary conditions: TODO.

        # Actuators: TODO.

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
        self._stepwise_loss = True

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
            'camera_pos': (0.15, -1.75, 0.6),
            'camera_lookat': (0, .15, .4)
        }
        renderer = PbrtRenderer(options)

        mesh = TetMesh3d()
        mesh.Initialize(mesh_file)
        renderer.add_tri_mesh(mesh, color=(.6, .3, .2),
            transforms=[
                ('s', 0.75),
                ('t', (-0.3, 0, 0.2))
            ],
            render_tet_edge=True,
        )

        renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj',
            texture_img='chkbd_24_0.7', transforms=[('s', 2)])

        renderer.render()

    def _stepwise_loss_and_grad(self, q, v, i):
        # TODO.
        loss = 0
        grad_q = np.zeros(q.size)
        grad_v = np.zeros(v.size)
        return loss, grad_q, grad_v