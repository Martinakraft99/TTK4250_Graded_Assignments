import numpy as np
import math
from quaternion import RotationQuaterion
from datatypes.eskf_params import ESKFTuningParams
from datatypes.eskf_states import NominalState, ErrorStateGauss

deg2rad = math.pi/180

vel_rand_walk   = 0.07    # datasheet            
acc_rand_walk   = 0.001   # Allan deviations
ang_rand_walk   = 0.15    # datasheet
rate_rand_walk  = 0.9     # Allan deviations
accm_bias_insta = 0.05    # datasheet
gyro_bias_insta = 0.5     # datasheet 

p_common = 0.5/10000

tuning_params_sim = ESKFTuningParams(
    accm_std =      2 * (vel_rand_walk / 60),
    accm_bias_std=  5 * (acc_rand_walk + accm_bias_insta) / 100,
    accm_bias_p=    p_common,

    gyro_std =      2 * (ang_rand_walk * deg2rad / 60),
    gyro_bias_std = 3 * ((rate_rand_walk * deg2rad / (60 ** 3))
                        + gyro_bias_insta * deg2rad / (60 ** 2)),
    gyro_bias_p =   p_common,

    gnss_std_ne =   0.4,
    gnss_std_d =    0.45)

x_nom_init_sim = NominalState(
    np.array([0.2, 0., -5.]),  # position
    np.array([20., 0., 0.]),  # velocity
    RotationQuaterion.from_euler([0., 0., 0]),  # orientation
    np.zeros(3),  # accelerometer bias
    np.zeros(3),  # gyro bias
    ts=0.)

init_std_sim = np.repeat(repeats=3,  # repeat each element 3 times
                         a=[1.,  # position
                            1.,  # velocity
                            np.deg2rad(1),  # angle vector
                            0.1,  # accelerometer bias
                            0.01])  # gyro bias
x_err_init_sim = ErrorStateGauss(np.zeros(15), np.diag(init_std_sim**2), 0.)
