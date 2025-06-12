#!/usr/bin/env python3

from easyUR import UR
import numpy as np
import rospy
from scipy.spatial.transform import Rotation as R
import std_msgs.msg


def pose_msg_to_arrays (pose_stamped):
                                      
    pos = np.array([pose_stamped.pose.position.x, pose_stamped.pose.position.y, pose_stamped.pose.position.z])
    quat = np.array([pose_stamped.pose.orientation.x, pose_stamped.pose.orientation.y, pose_stamped.pose.orientation.z, pose_stamped.pose.orientation.w])
    
    return pos, quat

def world_eef_init(R,t):
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = t 
    return T

def world_peg_init(R, t):
    T = np.eye(4)
    T[:3, :3] = R 
    T[:3, 3] = t 
    return T

def peg_old_peg_new(R,t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T

def end_effector_to_peg(t):
    T = np.eye(4)
    T[:3, 3] = t
    return T

def inverse_end_effector_to_peg(t):
    T = np.linalg.inv(end_effector_to_peg(t))
    return T

def gelsight_to_end_effector():
    T = np.eye(3)
    T[:3, 0] = [0,0,-1]
    T[:3, 1] = [0,-1,0]
    T[:3, 2] = [-1,0,0]
    return T

def callback(msg):
    global correction_euler
    correction_euler = np.array(msg.data)


if __name__ == '__main__':

    robot = UR()

    starting_pos,starting_quat = pose_msg_to_arrays(robot.cur_pos)

    robot.set_ee_speed(0.01)
    robot.set_ee_acceleration(0.05)
    t = np.array([0,0,0.25])

    correction_euler = np.array([0,0,0])
    rospy.Subscriber('/target_euler_angles', std_msgs.msg.Float32MultiArray, callback, queue_size = 1)

    while not rospy.is_shutdown():

        init_pos, init_quat = pose_msg_to_arrays(robot.cur_pos)
        #quaterion to euler angles 
        r = R.from_quat(init_quat)
        init_euler_eef = r.as_euler('ZYX', degrees = True)
        init_rotMatrix = r.as_dcm()

        #Compute Old End Effector Matrix, and Transformation Matrix 
        T_world_eef_init = world_eef_init(init_rotMatrix, init_pos)
        T_eef_init_to_peg = end_effector_to_peg(t)
        T_eef_init_to_peg_inverse = inverse_end_effector_to_peg(t)


        init_peg_pos = np.matmul(T_world_eef_init, T_eef_init_to_peg)[:3,3] # world_peg_old
        init_peg_rotmatrix = np.matmul(T_world_eef_init, T_eef_init_to_peg)[:3,:3] #world_peg_old

        T_world_peg_old = world_peg_init(init_peg_rotmatrix, init_peg_pos)#world_peg_old

        #rotation matrix to euler angles
        r = R.from_dcm(init_peg_rotmatrix)
        init_euler_world_peg_old = r.as_euler('ZYX', degrees = True) #world_peg_old

        if(np.linalg.norm(correction_euler) == 0 and init_pos[2] > 0.25):
            # target_peg_pos_difference = np.array([0.005*np.cos(60*np.pi/180),0.005*np.sin(60*np.pi/180),0]) #peg_old_peg_new
            target_peg_pos_difference = np.array([0,0,0.005])
        else:
            target_peg_pos_difference = np.array([0,0,0]) # peg_old_peg_new

       
        #transform from peg new in peg old frame to peg in world frame
        r = R.from_euler('ZYX', correction_euler, degrees = True)
        peg_correction_rotMatrix = r.as_dcm()

        T_peg_old_peg_new = peg_old_peg_new(peg_correction_rotMatrix, target_peg_pos_difference) #peg_old_peg_new

        T_world_peg_new = np.matmul(T_world_peg_old, T_peg_old_peg_new) #world_peg_new

        #Coordinate Change + Matrix Manipulation

        T_world_eef_new = np.matmul( T_world_peg_new, T_eef_init_to_peg_inverse) #world_eef_new

        r = R.from_dcm(T_world_eef_new[:3,:3])

        target_quat = r.as_quat()

        target_pos = T_world_eef_new[:3, 3]    

        robot.set_pose(target_pos, target_quat)
        rospy.sleep(0.5)

    # move the robot back
    # target_pos = starting_pos
    # target_quat = starting_quat
    # robot.set_pose(target_pos, target_quat)