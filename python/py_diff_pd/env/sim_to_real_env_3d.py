import time
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import pickle
import numpy as np

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.common import create_folder, ndarray
from py_diff_pd.common.hex_mesh import generate_hex_mesh, get_contact_vertex
from py_diff_pd.common.display import render_hex_mesh, export_gif
from py_diff_pd.core.py_diff_pd_core import HexMesh3d, HexDeformable, StdRealVector
from py_diff_pd.common.renderer import PbrtRenderer
from py_diff_pd.common.project_path import root_path

class SimToRealEnv3d(EnvBase):
    # Refinement is an integer controlling the resolution of the mesh.
    def __init__(self, seed, folder, options):
        EnvBase.__init__(self, folder)

        np.random.seed(seed)
        create_folder(folder, exist_ok=True)

        # A 30mm x 30mm x 30mm cube discretized into 10 x 10 x 10 voxels.
        substeps = int(options['substeps'])
        refinement = 10
        youngs_modulus = 0.68e6  # Ground truth value. Don't change it.
        poissons_ratio = 0.45   # This is an educated guess. The true value is probably between 0.4 and 0.45.
        density = 31e-3 / ((30e-3) ** 3)    # Ground truth value. Don't change it.

        # Mesh parameters.
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))

        # World coordinates:
        # - x: left to right.
        # - y: near to far.
        # - z: bottom to top.
        # Body coordinates:
        # - Origin is at the center of the object.
        # - Axes do not matter because the shape is symmetric.

        # Shape of the rolling jelly.
        cell_nums = (refinement, refinement, refinement)
        node_nums = [n + 1 for n in cell_nums]
        radius = 0.015
        dx = radius * 2 / refinement
        origin = -ndarray([refinement / 2, refinement / 2, refinement / 2]) * dx
        bin_file_name = folder / 'mesh.bin'
        voxels = np.ones(cell_nums)
        for i in range(cell_nums[0]):
            for j in range(cell_nums[1]):
                for k in range(cell_nums[2]):
                    cell_center = ndarray([(i - refinement / 2 + 0.5) * dx,
                        (j - refinement / 2 + 0.5) * dx, (k - refinement / 2 + 0.5) * dx])
                    if np.linalg.norm(cell_center) > radius:
                        voxels[i][j][k] = 0
        generate_hex_mesh(voxels, dx, origin, bin_file_name)
        mesh = HexMesh3d()
        mesh.Initialize(str(bin_file_name))

        deformable = HexDeformable()
        deformable.Initialize(str(bin_file_name), density, 'none', youngs_modulus, poissons_ratio)
        # State-based forces.
        deformable.AddStateForce('gravity', [0, 0, -9.81])
        # Elasticity.
        deformable.AddPdEnergy('corotated', [2 * mu,], [])
        deformable.AddPdEnergy('volume', [la,], [])

        # Collisions.
        friction_node_idx = get_contact_vertex(mesh)
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

        # Friction_node_idx = all vertices on the edge.
        deformable.SetFrictionalBoundary('planar', [0.0, 0.0, 1.0, radius], friction_node_idx)

        # Initial states.
        dofs = deformable.dofs()
        act_dofs = deformable.act_dofs()
        q0 = ndarray(mesh.py_vertices())
        q0_reshaped = q0.reshape((-1, 3))
        v0 = np.zeros(dofs)
        f_ext = np.zeros(dofs)

        # Intrinsic camera parameters.
        # The camera has the following parameters (6D in total):
        # - t: Location in the world coordinates.
        # - (yaw, pitch): indicate the camera orientation.
        #   Here is how we obtain the orientation:
        #   - Initially, we place the camera's x, y, and z axes so that they are parallel to the world axes.
        #   - Next, we rotate pitch along the x axis.
        #   - Then, we rotate yaw along the *world*'s z axis.
        # - Now the x and z axes are used for the image space. Ojects should have +y values after they are
        #   translated into the camera space. We follow the method below to compute pixel locations:
        #   - First, convert the world coordinates (x, y, z) to the camera space using (t, yaw, pitch).
        #   - Next, divide x and z by y (y should be positive) to obtain x' and z'.
        #   - Finally, multiple x' and z' by alpha (This is the last parameter in the camera model) to obtain the pixel location.
        self.__camera_pos = ndarray(options['camera_pos'])
        self.__camera_yaw = options['camera_yaw']
        self.__camera_pitch = options['camera_pitch']
        self.__camera_alpha = options['camera_alpha']

        # Motion video data folder.
        experiment_folder = options['experiment_folder']
        start_frame = int(options['start_frame'])
        # Reconstruct the image meta data and the motion sequence data.
        data_names = Path(experiment_folder).glob('*.data')
        data_names = sorted([str(n) for n in data_names])
        # Extract image id.
        self.__data = {}
        for n in data_names:
            info = pickle.load(open(n, 'rb'))
            obj_pixel_coordinates = ndarray(info['obj_pxl'])
            obj_body_coordinates = ndarray(info['obj_bd'])
            data_idx = int(n.split('/')[-1].split('.')[0])
            # Find the indices of obj_body_coordinates.
            obj_body_indices = [np.argmin(np.sum((q0_reshaped - p) ** 2, axis=1)) for p in obj_body_coordinates]
            # Just to be safe...
            assert np.linalg.norm(q0_reshaped[obj_body_indices] - obj_body_coordinates) < 1e-6
            self.__data[substeps * (data_idx - start_frame)] = (obj_pixel_coordinates, obj_body_indices)

        # Fetch image height and width.
        self.__img_height = int(options['img_height'])
        self.__img_width = int(options['img_width'])
        # tan(fov / 2) * alpha * 2 == min([height, width])
        self.__fov = np.arctan(np.min([self.__img_height, self.__img_width]) / 2 / self.__camera_alpha) * 2

        # Data members.
        self._deformable = deformable
        self._q0 = q0
        self._v0 = v0
        self._f_ext = f_ext
        self._youngs_modulus = youngs_modulus
        self._poissons_ratio = poissons_ratio
        self._stepwise_loss = True
        self.__radius = radius
        self.__friction_node_idx = friction_node_idx
        self.__contact_dofs = len(friction_node_idx) * 3

        self.__spp = options['spp'] if 'spp' in options else 4
        self.__radius = radius

    def radius(self):
        return self.__radius

    def contact_dofs(self):
        return self.__contact_dofs

    def material_stiffness_differential(self, youngs_modulus, poissons_ratio):
        jac = self._material_jacobian(youngs_modulus, poissons_ratio)
        jac_total = np.zeros((2, 2))
        jac_total[0] = 2 * jac[1]
        jac_total[1] = jac[0]
        return jac_total

    def is_dirichlet_dof(self, dof):
        return int(dof // 3) in self.__friction_node_idx

    def _display_mesh(self, mesh_file, file_name):
        # TODO: write a function to compute R (and potentially its gradients?)
        pitch = self.__camera_pitch
        c_pitch, s_pitch = np.cos(pitch), np.sin(pitch)
        R_pitch = ndarray([
            [1, 0, 0],
            [0, c_pitch, -s_pitch],
            [0, s_pitch, c_pitch]
        ])
        yaw = self.__camera_yaw
        c_yaw, s_yaw = np.cos(yaw), np.sin(yaw)
        R_yaw = ndarray([
            [c_yaw, -s_yaw, 0],
            [s_yaw, c_yaw, 0],
            [0, 0, 1]
        ])
        R_camera = R_yaw @ R_pitch
        options = {
            'file_name': file_name,
            'light_map': 'uffizi-large.exr',
            'sample': self._spp,
            'max_depth': 2,
            'camera_pos': ndarray(self.__camera_pos),
            'camera_lookat': R_camera[:, 1] + ndarray(self.__camera_pos),
            'camera_up': R_camera[:, 2],
            'resolution': (self.__img_width, self.__img_height),
            'fov': np.rad2deg(self.__fov)
        }
        renderer = PbrtRenderer(options)

        mesh = HexMesh3d()
        mesh.Initialize(mesh_file)
        renderer.add_hex_mesh(mesh, render_voxel_edge=True, color=[.1, .7, .3])
        renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj',
            texture_img='chkbd_24_0.7', transforms=[('t', (0, 0, -self.__radius))])

        renderer.render()

    def _stepwise_loss_and_grad(self, q, v, i):
        q = ndarray(q).copy().ravel()
        v = ndarray(v).copy().ravel()
        if i not in self.__data:
            return 0, ndarray(np.zeros(q.size)), ndarray(np.zeros(v.size))

        # Now compute the loss.
        pixel_loc, body_idx = self.__data[i]
        pixel_num = pixel_loc.shape[0]
        assert pixel_loc.shape == (pixel_num, 2) and len(body_idx) == pixel_num
        world_loc = q.reshape((-1, 3))[body_idx]

        # Convert world_loc to camera_loc.
        pitch = self.__camera_pitch
        c_pitch, s_pitch = np.cos(pitch), np.sin(pitch)
        R_pitch = ndarray([
            [1, 0, 0],
            [0, c_pitch, -s_pitch],
            [0, s_pitch, c_pitch]
        ])
        yaw = self.__camera_yaw
        c_yaw, s_yaw = np.cos(yaw), np.sin(yaw)
        R_yaw = ndarray([
            [c_yaw, -s_yaw, 0],
            [s_yaw, c_yaw, 0],
            [0, 0, 1]
        ])
        R_camera = R_yaw @ R_pitch
        t_camera = ndarray(self.__camera_pos)
        # camera_loc = R.T * (world_loc - t_camera).
        camera_loc = (world_loc - t_camera) @ R_camera
        # Dimension of camera_loc: M x 2.

        # Now convert camera_loc to pixel_loc.
        camera_loc_x = camera_loc[:, 0]
        camera_loc_y = camera_loc[:, 1]
        camera_loc_z = camera_loc[:, 2]

        alpha = self.__camera_alpha
        predicted_pixel_loc = ndarray(np.vstack([camera_loc_x / camera_loc_y, camera_loc_z / camera_loc_y])).T * alpha
        loss = 0.5 * np.sum((predicted_pixel_loc - pixel_loc) ** 2)

        # Compute gradients w.r.t q and v.
        grad_q = np.zeros(q.size).reshape((-1, 3))
        for row_idx, node_idx in enumerate(body_idx):
            # camera_loc[row_idx] depends on q[node_idx] only.
            d_camera_loc = R_camera.T
            d_camera_x = R_camera[:, 0]
            d_camera_y = R_camera[:, 1]
            d_camera_z = R_camera[:, 2]
            # d_camera_x = partial camera_loc[row_idx, 0] / partial q[node_idx].
            # d_camera_y = partial camera_loc[row_idx, 1] / partial q[node_idx].
            # d_camera_z = partial camera_loc[row_idx, 2] / partial q[node_idx].
            # (f/g)' = f'/g - fg'/g^2 = (f'g - fg') / g2.
            cx, cy, cz = camera_loc[row_idx]
            d_predicted_xy = (d_camera_x * cy - cx * d_camera_y) / (cy ** 2) * alpha
            d_predicted_zy = (d_camera_z * cy - cz * d_camera_y) / (cy ** 2) * alpha
            d_pixel = predicted_pixel_loc[row_idx] - pixel_loc[row_idx]
            grad_q[node_idx] = d_pixel[0] * d_predicted_xy + d_pixel[1] * d_predicted_zy
        grad_q = ndarray(grad_q).ravel()
        grad_v = ndarray(np.zeros(v.size))

        scale = np.min([self.__img_height, self.__img_width]) ** 2 * pixel_num
        loss /= scale
        grad_q /= scale
        grad_v /= scale
        return loss, grad_q, grad_v