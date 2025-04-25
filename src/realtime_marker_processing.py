#!/usr/bin/env python3

import numpy as np
import cv2
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.animation import FuncAnimation

import rospy
import sensor_msgs
import std_msgs
import ros_numpy
import std_msgs.msg

from std_msgs.msg import Float32MultiArray

from scipy.spatial.transform import Rotation as R


from marker_processing_utils import (
    cpd_lle,
    get_init_marker_location,
    get_tactile_mask,
    indices_arr_pixel_coord,
    process_marker_location,
    plot_vector_field_subplot,
    helmholtz_hodge_decomposition,
    compute_tactile_dipole_moment,
    normalize_error_S,
    normalize_error_P,
    normalize_error_H,
    calc_error_direction_curl, 
    calc_error_direction_div,
    calc_error_direction_har
)




def get_single_frame_tactile_movement (img, pts_prev, pts_init):
    
    mask = get_tactile_mask(img, (0, 0, 0), (100, 110, 140))


    # ===== perform registration to get the displacement of markers =====

    # use the white pixels as current point cloud
    X = indices_arr_pixel_coord(mask.shape[0], mask.shape[1])[mask.astype(bool)]
    X = X[::3]

    # perform registration
    # tracking params
    beta = 1                   # beta   : a constant representing the strength of interaction between points
    alpha = 10                 # alpha  : a constant regulating the strength of MCT smoothing
    gamma = 1                  # gamma  : a constant regulating the strength of LLE
    mu = 0.15                  # mu     : a constant representing how noisy the input point cloud is (0 - 1)
    max_iter = 30
    tol = 0.0000001
    kernel = 3                 # Gaussian kernel
    include_lle = True
    use_prev_sigma2 = True
    sigma2_0 = 1e-5

    dimension_scaling = 1000.0 / 3

    pts_registered, sigma2 = cpd_lle(
        X = X/dimension_scaling, 
        Y_0 = pts_prev/dimension_scaling, 
        beta = beta,
        alpha = alpha, 
        gamma = gamma,
        mu = mu,
        max_iter = max_iter, 
        tol = tol, 
        kernel = kernel,
        include_lle = include_lle, 
        use_prev_sigma2 = use_prev_sigma2, 
        sigma2_0 = sigma2_0
    )


    # ===== visualization =====

    # draw registration result on mask
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for marker in pts_registered*dimension_scaling:
        cv2.circle(mask, (int(marker[0]), int(marker[1])), 2, (0, 0, 255), -1)

    # draw marker displacement arrows
    visualization = img.copy()

    for pt_init, pt_registered in zip(pts_init, pts_registered*dimension_scaling):

        start_coord = pt_init
        end_coord = pt_registered

        magnifying_scale = 3.0
        # vector_length = pt2pt_dis(start_coord, end_coord)

        new_end_coord = (end_coord - start_coord) * magnifying_scale + end_coord

        visualization = cv2.arrowedLine(
            visualization, 
            (int(end_coord[0]), int(end_coord[1])), 
            (int(new_end_coord[0]), int(new_end_coord[1])), 
            (0, 255, 255), 
            thickness=2, 
            tipLength=0.3
        )

    return pts_registered * dimension_scaling, mask, visualization


def decomposition_per_frame (frame_vectors, frame_width, frame_height):
    # currently only returns the visualization image

    start_coords = []
    end_coords = []

    for start, end in frame_vectors:
        start_coords.append(start)
        end_coords.append(end)

    start_coords = np.array(start_coords)
    end_coords = np.array(end_coords)
    
    # Calculate vector field components
    U = end_coords[:, 0] - start_coords[:, 0]
    V = end_coords[:, 1] - start_coords[:, 1]
    U = U.reshape(7, 9)
    V = V.reshape(7, 9)

    # Compute the tactile dipole moment for this vector field
    p_tilt = compute_tactile_dipole_moment(np.stack((U, V), axis=-1))
    
    # Perform Helmholtz-Hodge Decomposition
    P, S, H = helmholtz_hodge_decomposition(U, V)
    # cprint("finish decomposition", "green")

    # Interpolate curl_S to the entire plane
    x_grid = np.linspace(start_coords[:, 0].min(), start_coords[:, 0].max(), 9)
    y_grid = np.linspace(start_coords[:, 1].min(), start_coords[:, 1].max(), 7)
    X, Y = np.meshgrid(x_grid, y_grid)
                    
    # Plotting
    plt.figure(figsize=(15, 5))
    # plot_vector_field_subplot(X, Y, U, V, 'Original Vector Field', 1)
    plot_vector_field_subplot(X, Y, P[0], P[1], 'Potential Component', 1)
    plot_vector_field_subplot(X, Y, S[0], S[1], 'Solenoidal Component', 2)
    plot_vector_field_subplot(X, Y, H[0], H[1], 'Harmonic Component', 3)
    
    plt.tight_layout()  
    canvas = FigureCanvas(plt.gcf())
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(canvas.get_width_height()[::-1] + (3,))
    frame_size = (frame_width, int(frame_width / buf.shape[1] * buf.shape[0]))
    buf_resized = cv2.resize(buf, frame_size)
    plt.close()

    return buf_resized, P, S, H


def initialize (tactile_image):
    pts_init = get_init_marker_location(get_tactile_mask(tactile_image, (0, 0, 0), (100, 110, 140)))
    return pts_init

def plot_magnitude_error(S, P):
    plt.figure(figsize = (10,5))
    plt.plot(normalize_error_S(S), color = 'r', label = 'S')
    plt.plot(normalize_error_P(P), color = 'g', label = 'P')
    plt.legend()
    plt.tight_layout()  
    canvas = FigureCanvas(plt.gcf())
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return buf

def create_rotation_matrix(angle_correction, axis):
    R = np.eye(3)
    R[0][0] = np.power(axis[0],2) * (1 - np.cos(angle_correction)) + np.cos(angle_correction)
    R[0][1] = axis[0]*axis[1]*(1 - np.cos(angle_correction)) - axis[2]*np.sin(angle_correction)
    R[0][2] = axis[0]*axis[2]*(1 - np.cos(angle_correction)) + axis[1] * np.sin(angle_correction)
    R[1][0] = axis[0]*axis[1]*(1 - np.cos(angle_correction)) + axis[2] * np.sin(angle_correction)
    R[1][1] = np.power(axis[1],2) * (1 - np.cos(angle_correction)) + np.cos(angle_correction)
    R[1][2] = axis[1]*axis[2]*(1 - np.cos(angle_correction)) - axis[0]*np.sin(angle_correction)
    R[2][0] = axis[0]*axis[2]*(1 - np.cos(angle_correction)) - axis[1]*np.sin(angle_correction)
    R[2][1] = axis[1]*axis[2]*(1 - np.cos(angle_correction)) + axis[0]*np.sin(angle_correction)
    R[2][2] = np.power(axis[2],2) * (1 - np.cos(angle_correction)) + np.cos(angle_correction)
    return R

    


pts_init = None
pts_prev = None

def callback (tactile_image):
    global pts_init, pts_prev

    cur_tactile_image = ros_numpy.numpify(tactile_image)  # shape (240, 320, 3)

    if pts_init is None:
        pts_init = initialize(cur_tactile_image)
        pts_prev = pts_init.copy()
    else:
        # marker tracking
        pts_cur, mask, vis = get_single_frame_tactile_movement(cur_tactile_image, pts_prev, pts_init)
        pts_prev = pts_cur.copy()
        marker_movement_vis = cv2.hconcat((cur_tactile_image, mask, vis))

        # decomposition
        movement_vectors_per_frame = process_marker_location(pts_cur, pts_init, False)
        decomposition_vis, P, S, H = decomposition_per_frame(movement_vectors_per_frame, marker_movement_vis.shape[1], int(marker_movement_vis.shape[1]/8*3))

        
        cv2.imshow('frame', cv2.vconcat((marker_movement_vis, decomposition_vis)))
        cv2.waitKey(1)

        # ===== work with decomposed marker field below =====

        #try and normalize the error, scale based on the magnitude of the normalized error

        error_marker = plot_magnitude_error(S,P)
        cv2.imshow('frame_2', error_marker)
        cv2.waitKey(1)


        msg = Float32MultiArray()
        msg.data = [0,0,0]

        # Normal Correction, one at a time, discrete correction in both the x and y direction
        #----------------------------------------------------------------------------------------
        # if normalize_error_P(P)[-1] > 1 or normalize_error_S(S)[-1] > 2.5:
        #     if normalize_error_S(S)[-1] > normalize_error_P(P)[-1] and normalize_error_S(S)[-1] > 3:
        #         if calc_error_direction_curl(S) == 1:
        #             msg.data = [0,0,-0.4 * np.power(normalize_error_S(S)[-1],2)]
        #         elif calc_error_direction_curl(S) == -1:
        #             msg.data = [0,0,0.4 * np.power(normalize_error_S(S)[-1],2)]
        #     elif normalize_error_P(P)[-1] > 1:
        #         if calc_error_direction_div(H) == 1 and 2*normalize_error_P(P)[-1] > normalize_error_S(S)[-1]:
        #             msg.data = [0, 3 * np.power(normalize_error_P(P)[-1],2),0]
        #         elif calc_error_direction_div(H) == -1 and 2*normalize_error_P(P)[-1] > normalize_error_S(S)[-1]:
        #             msg.data = [0, -3 *np.power(normalize_error_P(P)[-1],2),0]
        #         else:
        #             msg.data = [0,0,0]
        # euler_angle_pub.publish(msg)

        # Correction where we correct for both direction at once
        #------------------------------------------------------------------------------------------
        # if normalize_error_P(P)[-1] > 1 or normalize_error_S(S)[-1] > 2.5:
        #     if calc_error_direction_curl(S) == 1 and calc_error_direction_div(H) == 1:
        #         msg.data = [0,3* np.power(normalize_error_P(P)[-1],2), -0.4 * np.power(normalize_error_S(S)[-1], 2)]
        #     elif calc_error_direction_curl(S) == 1 and calc_error_direction_div(H) == -1:
        #         msg.data = [0,-3* np.power(normalize_error_P(P)[-1],2), -0.4 * np.power(normalize_error_S(S)[-1], 2)]
        #     elif calc_error_direction_curl(S) == -1 and calc_error_direction_div(H) == 1:
        #         msg.data = [0,3* np.power(normalize_error_P(P)[-1],2), 0.4 * np.power(normalize_error_S(S)[-1], 2)]
        #     elif calc_error_direction_curl(S) == -1 and calc_error_direction_div(H) == -1:
        #         msg.data = [0,-3* np.power(normalize_error_P(P)[-1],2), 0.4 * np.power(normalize_error_S(S)[-1], 2)]
        #     elif calc_error_direction_curl(S) == 1 and calc_error_direction_div(H) == 0:
        #         msg.data = [0,0, -0.4 * np.power(normalize_error_S(S)[-1], 2)]
        #     elif calc_error_direction_curl(S) == -1 and calc_error_direction_div(H) == 0:
        #         msg.data = [0,0, 0.4 * np.power(normalize_error_S(S)[-1], 2)]
        # euler_angle_pub.publish(msg)


        # Correction Using The angle of Contact and rotating about some axis     
        if normalize_error_P(P)[-1] > 1 or normalize_error_S(S)[-1] > 2.5:
            contact_force_vector = [-np.cos(calc_error_direction_har(H)), np.sin(calc_error_direction_har(H)),0]
            lever_arm_vector = [0,0,1]
            axis = np.cross(contact_force_vector, lever_arm_vector)
            axis = axis/np.linalg.norm(axis)
            T = create_rotation_matrix(0.3*np.power(normalize_error_P(P)[-1] + normalize_error_S(S)[-1],2) * np.pi/180, axis)
            r = R.from_dcm(T)
            correction = r.as_euler('ZYX', degrees = True)
            print(correction)
            msg.data = [correction[0], correction[1], correction[2]]
        euler_angle_pub.publish(msg)








if __name__ == '__main__':

    rospy.init_node('marker_processing')

    tactile_sub = rospy.Subscriber('/gelsight_0', sensor_msgs.msg.Image, callback, queue_size=1, buff_size=52428800*200)

    euler_angle_pub = rospy.Publisher('/target_euler_angles', std_msgs.msg.Float32MultiArray, queue_size = 1)
    
    rospy.spin()