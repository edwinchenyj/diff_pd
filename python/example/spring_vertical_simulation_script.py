import sys
sys.path.append('../')

from pathlib import Path
import numpy as np

from py_diff_pd.common.common import ndarray 
from py_diff_pd.env.spring_vertical_env_3d import SpringVerticalEnv3d
from script_util import parse_args

args = parse_args()
seed = 42
np.random.seed(seed)
folder = Path('spring_vertical')
youngs_modulus = args.Y 
poissons_ratio = args.P 
env = SpringVerticalEnv3d(seed, folder, { 'youngs_modulus': youngs_modulus,
    'poissons_ratio': poissons_ratio, 'spp': 4 })
deformable = env.deformable()

# Optimization parameters.
thread_ct = 4

dt = args.dt
frame_num = args.frame_num 

# Compute the initial state.
dofs = deformable.dofs()
act_dofs = deformable.act_dofs()
q_gt = ndarray([0.0, 0.0, 0.00])
v_gt = ndarray([0.0, 0., 0.0])
q0 = env.default_init_position()
q0 = (q0.reshape((-1, 3)) + q_gt).ravel()
v0 = np.zeros(dofs)
v0 = (v0.reshape((-1, 3)) + v_gt).ravel()
a0 = [np.zeros(act_dofs) for _ in range(frame_num)]
f0 = [np.zeros(dofs) for _ in range(frame_num)]

# Generate groundtruth motion.
method = args.method
opt = { 'max_pd_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 1, 'thread_ct': thread_ct,
     'recompute_eigen_decomp_each_step': 1, 'num_modes': 5 }
if method != 'siere':
    opt['num_modes'] = 0 
    opt['recompute_eigen_decomp_each_step'] = 0
simulation_name = f'{method}_recomp_{opt["recompute_eigen_decomp_each_step"]}_nmode_{opt["num_modes"]}_dt_{dt}_Y_{youngs_modulus}_frame_num_{frame_num}'
print(simulation_name)
_, info = env.simulate_simple(dt, frame_num, method, opt, q0, v0, a0, f0, vis_folder=simulation_name)
print("forward time")
print(info["forward_time"])
print("render time")
print(info["render_time"])
