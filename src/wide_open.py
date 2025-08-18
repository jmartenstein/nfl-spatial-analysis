#!/usr/bin/python

# wide_open.py - implementation of Bornn and Fernandez's Pitch Control Model

import numpy as np
import pandas as pd

import scipy.stats as st

def get_influence_radius(distance):

    # compute the influence radius, based on the distance from the player
    # to the ball

    radius = np.minimum( ((3 / 180) * (distance**2)) + 4, 10)

    return radius

def get_covariance_matrix(direction, speed, distance_from_ball):

    # first compute the rotation matrix for direction (see Appendix 1, Equation 16 
    # in Bornn and Fernandez)

    # R = [ [ cos(theta), -sin(theta) ], [ sin(theta), cos(theta) ] ]
    th = np.radians(direction)
    rotation = np.array( [[np.cos(th), -np.sin(th)],
                          [np.sin(th), np.cos(th)]]
                       )

    # next compute the scaling matrix for speed (Appendix 1, Eq. 17)

    max_speed = 18
    inf_radius = get_influence_radius(distance_from_ball)

    # Appendix 1, Eq. 18)
    speed_ratio = (speed**2) / (max_speed**2)

    # Appendix 1, Equation 17 and Equation 19
    # S = [ [scale_x, 0], [0, scale_y]]
    scale_x = (inf_radius + (inf_radius*speed_ratio) / 2)
    scale_y = (inf_radius - (inf_radius*speed_ratio) / 2)
    scaling = np.array( [[scale_x, 0], [0, scale_y]] )

    # multiply the rotation and scaling matrices, per equation 15 

    # COV = RSSR^-1
    covariance = rotation @ scaling @ scaling @ rotation.T

    return covariance

def get_player_influence_func(position, direction, speed, distance_from_ball):

    cov = get_covariance_matrix(direction, speed, distance_from_ball)

    radian_dir = np.pi * (direction / 180)
    speed_x = speed * np.cos(radian_dir)
    speed_y = speed * np.sin(radian_dir)

    # per Appendix 1, Equation 21 calculate the gaussian
    # distribution mean (mu) as the X and Y position plus half
    # of the current speed

    # mean = position + speed * 0.5

    dist_mean = np.array([
        position[0] + (speed_x * 0.5),
        position[1] + (speed_y * 0.5)
    ])

    dist = st.multivariate_normal(dist_mean, cov)

    func = lambda p: dist.pdf(p) / dist.pdf(dist_mean)

    return func
