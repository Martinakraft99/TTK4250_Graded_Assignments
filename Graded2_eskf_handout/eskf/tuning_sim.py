import numpy as np
from quaternion import RotationQuaterion
from datatypes.eskf_params import ESKFTuningParams
from datatypes.eskf_states import NominalState, ErrorStateGauss

deg2rad = 3.1415/180

vel_rand_walk   = 0.07
acc_rand_walk   = 0.04
ang_rand_walk   = 0.15
rate_rand_walk  = 0.9
accm_bias_insta = 0 #0.05
gyro_bias_insta = 0 #0.5

tuning_params_sim = ESKFTuningParams(
    accm_std =      (vel_rand_walk * 60)*1 ,                 
    accm_bias_std=  (acc_rand_walk * np.sqrt(3/5000) + accm_bias_insta) * 9810,    
    accm_bias_p=    0.5/10000.,

    gyro_std =      (ang_rand_walk * deg2rad*3600)*0.5,        
    gyro_bias_std = rate_rand_walk * deg2rad/216000 + gyro_bias_insta * deg2rad/3600, 
    gyro_bias_p =   0.5/10000.,

    gnss_std_ne =   1,
    gnss_std_d =    5)

x_nom_init_sim = NominalState(
    np.array([0, 0., 0.]),  # position
    np.array([0., 0., 0.]),  # velocity
    RotationQuaterion.from_euler([0., 0., 0.]),  # orientation
    np.ones(3),  # accelerometer bias
    np.zeros(3),  # gyro bias
    ts=0.)

init_std_sim = np.repeat(repeats=3,  # repeat each element 3 times
                         a=[1.,  # position
                            1.,  # velocity
                            np.deg2rad(1),  # angle vector
                            1.,  # accelerometer bias
                            1.])  # gyro bias
x_err_init_sim = ErrorStateGauss(np.zeros(15), np.diag(init_std_sim**2), 0.)
