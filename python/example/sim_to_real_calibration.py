import sys
sys.path.append('../')

import os
from pathlib import Path
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib.backend_bases import MouseButton
import pickle
from re import split

from py_diff_pd.common.project_path import root_path
from py_diff_pd.common.common import ndarray, create_folder, print_info, print_warning

def extract_intrinsic_parameters(K):
    K = ndarray(K).copy()
    cx = K[0, 2]
    cy = K[1, 2]
    alpha = K[0, 0]
    cot_theta = K[0, 1] / -alpha
    tan_theta = 1 / cot_theta
    theta = np.arctan(tan_theta)
    if theta < 0:
        theta += np.pi
    beta = K[1, 1] * np.sin(theta)
    return { 'alpha': alpha, 'beta': beta, 'theta': theta, 'cx': cx, 'cy': cy }

def solve_camera(points_in_pixel, points_in_world):
    # This is a better reference: https://web.stanford.edu/class/cs231a/course_notes/01-camera-models.pdf
    #
    # The pixel space is:
    # - Origin: lower left.
    # - x: left to right.
    # - y: bottom to top.
    # Let p and P be points_in_pixel (2D) and points_in_world (3D), respectively.
    # Let R and t be the orientation and location of the world frame in the camera frame.
    # T = [R, t]
    #     [0, 1]
    # K = [alpha, -alpha * cot theta, cx, 0]
    #     [0, beta / sin theta, cy, 0]
    #     [0, 0, 1, 0]
    # Pixels: alpha * (x - cot theta * y) / z + cx
    #         beta / sin theta * y / z + cy
    # which makes sense if the image is skewed to its right.

    # [p, 1] = Homogenous(KT[P, 1]).
    # Let M = KT \in R^{3 x 4} = [m1, m2, m3]
    # p.x = <m1, [P, 1]> / <m3, [P, 1]>.
    # p.y = <m2, [P, 1]> / <m3, [P, 1]>.
    # p.x * <m3, [P, 1]> - <m1, [P, 1]> = 0.
    # p.y * <m3, [P, 1]> - <m2, [P, 1]> = 0.
    # Let's flatten them into a linear system.
    points_in_pixel = ndarray(points_in_pixel).copy()
    points_in_world = ndarray(points_in_world).copy()
    num_points = points_in_pixel.shape[0]
    assert (num_points, 2) == points_in_pixel.shape
    assert (num_points, 3) == points_in_world.shape
    P = ndarray(np.zeros((2 * num_points, 12)))
    for i in range(num_points):
        # Assemble the x equation.
        # m1:
        P[2 * i, :3] = -points_in_world[i]
        P[2 * i, 3] = -1
        # m3:
        P[2 * i, 8:11] = points_in_world[i] * points_in_pixel[i, 0]
        P[2 * i, 11] = points_in_pixel[i, 0]
        # Assemble the y equation.
        # m2:
        P[2 * i + 1, 4:7] = -points_in_world[i]
        P[2 * i + 1, 7] = -1
        # m3:
        P[2 * i + 1, 8:11] = points_in_world[i] * points_in_pixel[i, 1]
        P[2 * i + 1, 11] = points_in_pixel[i, 1]
    # Now m can be obtained from P * m = 0.
    # We solve this by minimizing \|P * m\|^2 s.t. \|m\|^2 = 1.
    # Consider SVD of P: P = U * Sigma * V.T.
    U, Sigma, Vt = np.linalg.svd(P)
    # U @ np.diag(Sigma) @ Vt = P.
    # So, Vt * m = [0, 0, 0, ..., 1], or m = V * [0, 0, 0, ..., 1].
    m = Vt[-1]
    M = ndarray(np.reshape(m, (3, 4)))
    # Now we know M = 1 / rho * KT. Let's extract camera parameters.
    a1 = M[0, :3]
    a2 = M[1, :3]
    a3 = M[2, :3]
    rho = 1 / np.linalg.norm(a3)
    cx = rho * rho * (a1.dot(a3))
    cy = rho * rho * (a2.dot(a3))
    a1_cross_a3 = np.cross(a1, a3)
    a2_cross_a3 = np.cross(a2, a3)
    cos_theta = -a1_cross_a3.dot(a2_cross_a3) / (np.linalg.norm(a1_cross_a3) * np.linalg.norm(a2_cross_a3))
    theta = np.arccos(cos_theta)
    alpha = rho * rho * np.linalg.norm(a1_cross_a3) * np.sin(theta)
    beta = rho * rho * np.linalg.norm(a2_cross_a3) * np.sin(theta)
    K = ndarray([[alpha, -alpha / np.tan(theta), cx],
        [0, beta / np.sin(theta), cy],
        [0, 0, 1]])

    # Extrinsic camera info:
    r1 = a2_cross_a3 / np.linalg.norm(a2_cross_a3)
    # r3 has two possibilities. We need to figure out which one is better.
    r3_pos = rho * a3
    r2_pos = np.cross(r3_pos, r1)
    R_pos = np.vstack([r1, r2_pos, r3_pos])
    r3_neg = -rho * a3
    r2_neg = np.cross(r3_neg, r1)
    R_neg = np.vstack([r1, r2_neg, r3_neg])
    # Compare K @ R and rho M[:, :3].
    if np.linalg.norm(K @ R_pos - rho * M[:, :3]) < np.linalg.norm(K @ R_neg + rho * M[:, :3]):
        R = R_pos
    else:
        R = R_neg
        rho = -rho
    T = rho * np.linalg.inv(K) @ M[:, 3]
    info = {
        'K': ndarray(K).copy(),
        'R': ndarray(R).copy(),
        'T': ndarray(T).copy(),
        'alpha': alpha,
        'beta': beta,
        'theta': theta,
        'cx': cx,
        'cy': cy,
    }
    return info

# Input:
# - image_data: H x W x 3 ndarray.
# Output:
# - M x 2 pixel coordinates and M x 3 3D coordinates in the object coordinates.
#   The object is a 30mm x 30mm x 30mm cube discretized into 10 x 10 x 10 voxels.
# The body frame is defined as follows:
# - origin: center of the object.
# - x: red (positive) and blue (negative).
# - y: orange (positive) and green (negative).
# - z: white (positive) and black (negative).
object_corners_in_pixel = []
object_corners_in_body = []
grid_corners_in_pixel = []
grid_corners_in_body = []
last_img_x = None
last_img_y = None
visible_corners = None
selected_corners = None
def select_corners(image_data):
    global object_corners_in_pixel
    global object_corners_in_body
    global grid_corners_in_pixel
    global grid_corners_in_body
    global last_img_x
    global last_img_y
    global selected_corners
    selected_corners = []
    object_corners_in_pixel = []
    object_corners_in_body = []
    grid_corners_in_pixel = []
    grid_corners_in_body = []
    last_img_x = -1
    last_img_y = -1

    fig = plt.figure()
    ax_img = fig.add_subplot(221)
    ax_img.imshow(image_data)
    # Plot the object.
    ax_object = fig.add_subplot(222, projection='3d', proj_type='ortho')
    contour1 = [
        [5, -2, -2],
        [5, -2, 2],
        [5, 2, 2],
        [5, 2, -2],
    ]
    contour2 = [
        [4, -2, -2],
        [4, -2, 2],
        [4, 2, 2],
        [4, 2, -2],
    ]
    contour3 = [
        [4, -3, -3], [4, -1, -3], [4, -1, -4], [4, 1, -4], [4, 1, -3],
        [4, 3, -3], [4, 3, -1], [4, 4, -1], [4, 4, 1], [4, 3, 1],
        [4, 3, 3], [4, 1, 3], [4, 1, 4], [4, -1, 4], [4, -1, 3],
        [4, -3, 3], [4, -3, 1], [4, -4, 1], [4, -4, -1], [4, -3, -1],
    ]
    colors = [{ -1: 'tab:blue', 1: 'tab:red'}, { -1: 'tab:green', 1: 'tab:orange' }, { -1: 'black', 1: 'tab:gray' }]

    # The flat sheet.
    ax_sheet = fig.add_subplot(223)
    ax_sheet.set_xlabel('x')
    ax_sheet.set_ylabel('y')
    # We also know the 3D coordinates of the four sheet corners accordingly (in meters).
    # Note that the x offset does not matter. It is the y offset that we choose to use.
    sheet_corners = ndarray([
        [0, 0, 0],
        [1.0, 0, 0],
        [1.0, 0.67, 0],
        [0, 0.67, 0],
    ])
    sheet_corners_aug = np.vstack([sheet_corners, sheet_corners[0]])
    ax_sheet.plot(sheet_corners_aug[:, 0], sheet_corners_aug[:, 1], 'k')
    ax_sheet.set_aspect('equal')

    def on_click(event):
        global object_corners_in_pixel
        global object_corners_in_body
        global grid_corners_in_pixel
        global grid_corners_in_body
        global last_img_x
        global last_img_y
        if event.button == MouseButton.LEFT:
            return
        if event.inaxes == ax_img:
            last_img_x, last_img_y = event.xdata, event.ydata
            # Plot the selected corner.
            ax_img.plot(last_img_x, last_img_y, 'y+')
        elif event.inaxes == ax_object:
            # store the current mousebutton
            b = ax_object.button_pressed
            # set current mousebutton to something unreasonable
            ax_object.button_pressed = -1
            # get the coordinate string out
            coords = ax_object.format_coord(event.xdata, event.ydata)
            # set the mousebutton back to its previous state
            ax_object.button_pressed = b
            xyz = list(map(lambda x: float(x), split(',|=', coords.replace(' ', ''))[1::2]))
            xyz = ndarray(xyz)
            # Now we need to select from all 3d points the one that is closest to the ray from xyz along the camera angle.
            azim = np.deg2rad(ax_object.azim)
            elev = np.deg2rad(ax_object.elev)
            d = ndarray([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
            # Now the line is d * t + xyz.
            min_dist = np.inf
            min_ci = None
            min_color = None
            for axis_dir in [0, 1, 2]:
                for axis_sign in [-1, 1]:
                    c = visible_corners[axis_dir][axis_sign]
                    for ci in c:
                        # <d * t + xyz - ci, d> = 0
                        # t + <xyz, d> = <ci, d>
                        t = (ci - xyz).dot(d)
                        dist = np.linalg.norm(d * t + xyz - ci)
                        if dist < min_dist:
                            min_dist = dist
                            min_ci = np.copy(ci)
                            min_color = colors[axis_dir][axis_sign]
            global selected_corners
            selected_corners.append((min_ci, min_color))
            # Convert it to meters.
            # The dimension of the cube is 30mm x 30mm x 30mm in the resolution of 10 x 10 x 10 voxels.
            object_corners_in_body.append(min_ci * 3 / 1000)
            object_corners_in_pixel.append([last_img_x, last_img_y])
        elif event.inaxes == ax_sheet:
            ix, iy = event.xdata, event.ydata
            xy = ndarray([ix, iy])
            selected = sheet_corners[np.argmin(np.sum((sheet_corners[:, :2] - xy) ** 2, axis=1))]
            # Plot the selected corner.
            ax_sheet.plot(selected[0], selected[1], 'y+')
            grid_corners_in_body.append(selected)
            grid_corners_in_pixel.append([last_img_x, last_img_y])

        if len(grid_corners_in_body) == 4:
            fig.canvas.mpl_disconnect(cid)
        plt.gcf().canvas.draw_idle()

    def on_move(event):
        if event.inaxes != ax_object: return
        ax_object.clear()
        global visible_corners
        visible_corners = [{ -1: [], 1: [] }, { -1: [], 1: [] }, { -1: [], 1: [] }]
        azim = np.deg2rad(ax_object.azim)
        elev = np.deg2rad(ax_object.elev)
        d = ndarray([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
        for axis_dir in range(3):
            for axis_sign in [-1, 1]:
                face_dir = [0, 0, 0]
                face_dir[axis_dir] = axis_sign
                if d.dot(face_dir) <= 0: continue
                # Plot the corners in three contours.
                corners = []
                for c in [contour1, contour2, contour3]:
                    c_aug = np.hstack([c, c])
                    c_aug = c_aug[:, 3 - axis_dir:6 - axis_dir]
                    c_aug[:, axis_dir] *= axis_sign
                    corners.append(c_aug)
                    c_aug = np.vstack([c_aug, c_aug[0]])
                    ax_object.plot(c_aug[:, 0], c_aug[:, 1], c_aug[:, 2], color=colors[axis_dir][axis_sign])
                visible_corners[axis_dir][axis_sign] = ndarray(np.vstack(corners))
            ax_object.set_xlabel('x')
            ax_object.set_ylabel('y')
            ax_object.set_zlabel('z')
        global select_corners
        for ci, clr in selected_corners:
            ax_object.scatter3D(ci[0], ci[1], ci[2], c=clr)

        global grid_corners_in_body
        if len(grid_corners_in_body) == 4:
            fig.canvas.mpl_disconnect(mid)
        plt.gcf().canvas.draw_idle()

    mid = fig.canvas.mpl_connect('motion_notify_event', on_move)
    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
    return ndarray(object_corners_in_pixel).copy(), ndarray(object_corners_in_body).copy(), \
        ndarray(grid_corners_in_pixel).copy(), ndarray(grid_corners_in_body).copy()

def convert_video_to_images(video_name, folder):
    os.system('ffmpeg -i {} -f image2 "{}/%04d.png"'.format(str(video_name), str(folder)))
    # Get the frame rate.
    os.system('ffprobe -v quiet -show_streams -select_streams v:0 {} | grep "r_frame_rate" > {}/frame_rate.txt'.format(video_name, folder))
    with open(folder / 'frame_rate.txt', 'r') as f:
        line = f.readline()
    os.remove(folder / 'frame_rate.txt')
    _, fps_str = line.strip().split('=')
    a, b = fps_str.split('/')
    fps = float(a) / float(b)
    dt = 1 / fps
    print_info('Loading video {}...'.format(video_name))
    print('fps: {:2.2f}'.format(fps))
    print('dt: {:2.4f}s'.format(dt))

img_height, img_width = 720, 1280
def pxl_to_cal(pxl):
    pxl = ndarray(pxl).copy()
    pxl[:, 1] *= -1
    pxl[:, 1] += img_height
    return pxl
def cal_to_pxl(cal):
    cal = ndarray(cal).copy()
    cal[:, 1] -= img_height
    cal[:, 1] *= -1
    return cal

def load_image(image_file):
    with cbook.get_sample_data(image_file) as f:
        img = plt.imread(f)
    return ndarray(img)

if __name__ == '__main__':
    # This is a script for calibrating the intrinsic camera parameters as well as the states of cube.
    np.random.seed(42)

    folder = Path(root_path) / 'python/example/sim_to_real_calibration'
    create_folder(folder, exist_ok=True)

    # Step 1: extract video information.
    calibration_video_name = Path(root_path) / 'asset/video/sim_to_real_calibration.mov'
    calibration_video_data_folder = folder / 'intrinsic_calibration_video'
    create_folder(calibration_video_data_folder, exist_ok=True)
    if not (calibration_video_data_folder / '0001.png').is_file():
        convert_video_to_images(calibration_video_name, calibration_video_data_folder)

    experiment_video_name = Path(root_path) / 'asset/video/sim_to_real_experiment.mov'
    experiment_video_data_folder = folder / 'experiment_video'
    create_folder(experiment_video_data_folder, exist_ok=True)
    if not (experiment_video_data_folder / '0001.png').is_file():
        convert_video_to_images(experiment_video_name, experiment_video_data_folder)

    # Step 2: calibration.
    calibration_folder = folder / 'intrinsic_calibration'
    create_folder(calibration_folder, exist_ok=True)
    experiment_folder = folder / 'experiment'
    create_folder(experiment_folder, exist_ok=True)

    calibration_tasks = [
        (calibration_video_data_folder, calibration_folder, [800, 1200, 1600, 1900, 2200, 2700]),
        (experiment_video_data_folder, experiment_folder, [180, 190, 200, 210, 220, 230, 240, 250, 260])
    ]
    for data_folder, output_folder, frames in calibration_tasks:
        for i in frames:
            img = load_image(data_folder / '{:04d}.png'.format(i))
            # Dimension of img: (height, width, channel).
            # img_height, img_width, num_channels = img.shape
            assert img.shape[0] == img_height and img.shape[1] == img_width and img.shape[2] == 3
            # Call the label program.
            f = output_folder / '{:04d}.data'.format(i)
            if not f.is_file():
                print('Labeling image {:04d}.png...'.format(i))
                samples = select_corners(img)
                # Calibrate the camera system.
                obj_pxl, obj_bd, grid_pxl, grid_bd = samples
                info = {}
                info['obj_pxl'] = ndarray(obj_pxl).copy()
                info['obj_bd'] = ndarray(obj_bd).copy()
                info['grid_pxl'] = ndarray(grid_pxl).copy()
                info['grid_bd'] = ndarray(grid_bd).copy()
                # Save data.
                pickle.dump(info, open(f, 'wb'))

                # The pixel space in matplotlib is different from the pixel space in the calibration algorithm.
                camera_info = solve_camera(pxl_to_cal(obj_pxl), obj_bd)
                K = camera_info['K']
                R = camera_info['R']
                T = camera_info['T']
                print('Camera information: alpha: {:2.2f}, beta: {:2.2f}, theta: {:2.2f}, cx: {:4.1f}, cy: {:4.1f}'.format(
                    camera_info['alpha'], camera_info['beta'], np.rad2deg(camera_info['theta']), camera_info['cx'], camera_info['cy']
                ))
                # Now R and t are the orientation and location of the object in the camera space.
                # Verification:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.imshow(img)
                # Now plot the predicted object location.
                obj_predicted_camera = (obj_bd @ R.T + T) @ K.T
                obj_predicted_calib = obj_predicted_camera[:, :2] / obj_predicted_camera[:, 2][:, None]
                obj_predicted_pixl = cal_to_pxl(obj_predicted_calib)
                ax.plot(obj_predicted_pixl[:, 0], obj_predicted_pixl[:, 1], 'y+')
                plt.show()
                fig.savefig(output_folder / '{:04d}.png'.format(i))
                plt.close('all')
                for k, v in camera_info.items():
                    info[k] = v
                # Save data.
                pickle.dump(info, open(f, 'wb'))
                print('Data saved to', f)