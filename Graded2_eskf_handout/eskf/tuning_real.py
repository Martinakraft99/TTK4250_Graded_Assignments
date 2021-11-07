import numpy as np
from quaternion import RotationQuaterion
from datatypes.eskf_params import ESKFTuningParams
from datatypes.eskf_states import NominalState, ErrorStateGauss

deg2rad = 3.1415/180

vel_rand_walk   = 0.07                
acc_rand_walk   = 0.04
ang_rand_walk   = 0.15
rate_rand_walk  = 0.9
accm_bias_insta = 0.05
gyro_bias_insta = 0.03 #0.5
'''
tuning_params_real = ESKFTuningParams(
    accm_std=1.,
    accm_bias_std=1.,
    accm_bias_p=1.,

    gyro_std=1.,
    gyro_bias_std=1.,
    gyro_bias_p=1.,

    gnss_std_ne=1.,
    gnss_std_d=1.,

    use_gnss_accuracy=False)

x_nom_init_real = NominalState(
    np.array([0., 0., 0.]),  # position
    np.array([0., 0., 0.]),  # velocity
    RotationQuaterion.from_euler([0., 0., 0.]),  # orientation
    np.zeros(3),  # accelerometer bias
    np.zeros(3),  # gyro bias
    ts=0.)

init_std_real = np.repeat(repeats=3,  # repeat each element 3 times
                          a=[1.,  # position
                             1.,  # velocity
                             np.deg2rad(1),  # angle vector
                             1.,  # accelerometer bias
                             1.])  # gyro bias

x_err_init_real = ErrorStateGauss(np.zeros(15), np.diag(init_std_real**2), 0.)
'''
tuning_params_real = ESKFTuningParams(
    accm_std =      2 * (vel_rand_walk / 60) ,                 
    accm_bias_std=  5 * (acc_rand_walk * np.sqrt(3/5000) + accm_bias_insta) / 100,    
    accm_bias_p=    0.5/10000.,

    gyro_std =      2 * (ang_rand_walk * deg2rad / 60),        
    gyro_bias_std = 30 * ((rate_rand_walk * deg2rad / (60 ** 3)) 
                        + gyro_bias_insta * deg2rad / (60 ** 2)), 
    gyro_bias_p =   0.5/10000.,

    gnss_std_ne =   0.8 * 0.45,
    gnss_std_d =    0.8 * 0.7,
    
    use_gnss_accuracy=False)


x_nom_init_real = NominalState(
    np.array([0, 0., 0.]),  # position
    np.array([0., 0., 0.]),  # velocity
    RotationQuaterion.from_euler([0., 0., 0.]),  # orientation
    np.zeros(3),  # accelerometer bias
    np.zeros(3),  # gyro bias
    ts=0.)

init_std_real = np.repeat(repeats=3,  # repeat each element 3 times
                         a=[.5,  # position
                            .05,  # velocity
                            np.deg2rad(1),  # angle vector
                            0.1,  # accelerometer bias
                            0.01])  # gyro bias
x_err_init_real = ErrorStateGauss(np.zeros(15), np.diag(init_std_real**2), 0.)