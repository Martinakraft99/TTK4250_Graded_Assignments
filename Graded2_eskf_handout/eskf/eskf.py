from copy import deepcopy
from math import sqrt
import numpy as np
from numpy import errstate, ndarray, zeros
from numpy.lib.function_base import average
import scipy
from dataclasses import dataclass, field
from typing import Tuple
from functools import cache

from datatypes.multivargaussian import MultiVarGaussStamped
from datatypes.measurements import (ImuMeasurement,
                                    CorrectedImuMeasurement,
                                    GnssMeasurement)
from datatypes.eskf_states import NominalState, ErrorStateGauss
from utils.indexing import block_3x3
from utils.indexing import block_kxk


from quaternion import RotationQuaterion
from cross_matrix import get_cross_matrix

import solution
import math




@dataclass
class ESKF():

    accm_std: float
    accm_bias_std: float
    accm_bias_p: float

    gyro_std: float
    gyro_bias_std: float
    gyro_bias_p: float

    gnss_std_ne: float
    gnss_std_d: float

    accm_correction: 'ndarray[3,3]'
    gyro_correction: 'ndarray[3,3]'
    lever_arm: 'ndarray[3]'

    do_approximations: bool
    use_gnss_accuracy: bool = False

    Q_err: 'ndarray[12,12]' = field(init=False, repr=False)
    g: 'ndarray[3]' = np.array([0, 0, 9.82])

    def __post_init__(self):

        self.Q_err = scipy.linalg.block_diag(
            self.accm_std ** 2 * self.accm_correction @ self.accm_correction.T,
            self.gyro_std ** 2 * self.gyro_correction @ self.gyro_correction.T,
            self.accm_bias_std ** 2 * np.eye(3),
            self.gyro_bias_std ** 2 * np.eye(3),
        )
        self.gnss_cov = np.diag([self.gnss_std_ne]*2 + [self.gnss_std_d])**2

    def correct_z_imu(self,
                      x_nom_prev: NominalState,
                      z_imu: ImuMeasurement,
                      ) -> CorrectedImuMeasurement:
        """Correct IMU measurement so it gives a measurmenet of acceleration 
        and angular velocity in body.

        Hint: self.accm_correction and self.gyro_correction translates 
        measurements from IMU frame (probably not correct name) to body frame

        Args:
            x_nom_prev (NominalState): previous nominal state
            z_imu (ImuMeasurement): raw IMU measurement

        Returns:
            CorrectedImuMeasurement: corrected IMU measurement
        """
        
        # Found by trial and error
        a_corr = self.accm_correction @ (z_imu.acc - x_nom_prev.accm_bias)
        v_corr = self.gyro_correction @ (z_imu.avel - x_nom_prev.gyro_bias)
        ts = z_imu.ts
        z_corr = CorrectedImuMeasurement(ts, a_corr, v_corr)

        #z_corr = solution.eskf.ESKF.correct_z_imu(self, x_nom_prev, z_imu)

        return z_corr

    def predict_nominal(self,
                        x_nom_prev: NominalState,
                        z_corr: CorrectedImuMeasurement,
                        ) -> NominalState:
        """Predict the nominal state, given a corrected IMU measurement

        Hint: Discrete time prediction of equation (10.58)
        See the assignment description for more hints 

        Args:
            x_nom_prev (NominalState): previous nominal state
            z_corr (CorrectedImuMeasurement): corrected IMU measuremnt

        Returns:
            x_nom_pred (NominalState): predicted nominal state
        """
        
        # Fetch the acceleration and velocity from exisiting data
        # Uses assumptions from problem description
        ori_prev = x_nom_prev.ori
        R = ori_prev.as_rotmat()
        acc = R @ z_corr.acc + self.g
        w = z_corr.avel
        
        # Compute the predicted prosition and velocity according to hints
        Ts = z_corr.ts - x_nom_prev.ts
        pos_pred = x_nom_prev.pos + Ts * x_nom_prev.vel + (Ts**2 / 2) * acc
        vel_pred = x_nom_prev.vel + Ts * acc

        # Compute the predicted quaternion according to hints
        kappa = Ts * w
        kappa_2norm = sqrt(sum(kappa**2))
        
        if kappa_2norm == 0:
            ori_pred = ori_prev
        else:
            ori_other = RotationQuaterion(
                math.cos(kappa_2norm / 2), math.sin(kappa_2norm / 2) * kappa.T / kappa_2norm
            )
            ori_pred = ori_prev @ ori_other
        
        # Compute the biases from equation (10.50) This i
        #accm_bias = x_nom_prev.accm_bias - np.eye(3) @ x_nom_prev.accm_bias * self.accm_bias_p * Ts
        #gyro_bias = x_nom_prev.gyro_bias - np.eye(3) @ x_nom_prev.gyro_bias * self.gyro_bias_p * Ts

        accm_bias = x_nom_prev.accm_bias * np.exp(-self.accm_bias_p*Ts)
        gyro_bias = x_nom_prev.gyro_bias * np.exp(-self.gyro_bias_p*Ts)
        
        x_nom_pred = NominalState(pos_pred, vel_pred, ori_pred, accm_bias, gyro_bias, z_corr.ts)

        # TODO replace this with your own code
        #x_nom_pred = solution.eskf.ESKF.predict_nominal(
        #    self, x_nom_prev, z_corr)

        return x_nom_pred

    def get_error_A_continous(self,
                              x_nom_prev: NominalState,
                              z_corr: CorrectedImuMeasurement,
                              ) -> 'ndarray[15,15]':
        """Get the transition matrix, A, in (10.68)

        Hint: The S matrices can be created using get_cross_matrix. In the book
        a perfect IMU is expected (thus many I matrices). Here we have 
        to use the correction matrices, self.accm_correction and 
        self.gyro_correction, instead of som of the I matrices.  

        You can use block_3x3 to simplify indexing if you want to.
        The first I element in A can be set as A[block_3x3(0, 1)] = np.eye(3)

        Args:
            x_nom_prev (NominalState): previous nominal state
            z_corr (CorrectedImuMeasurement): corrected IMU measurement
        Returns:
            A (ndarray[15,15]): A
        """

        R = x_nom_prev.ori.as_rotmat() 
        skew_accl = get_cross_matrix(z_corr.acc)
        skew_gyro = get_cross_matrix(z_corr.avel)

        A = np.zeros([15,15])
        A[block_3x3(0, 1)] = np.eye(3)                              #vel -> pos
        A[block_3x3(1, 2)] = -R @ skew_accl                         #att -> vel
        A[block_3x3(2, 2)] = -skew_gyro                             #att -> att
        A[block_3x3(1, 3)] = -R @ self.accm_correction              #acm -> vel, with bias
        A[block_3x3(2, 4)] = -np.eye(3) @ self.gyro_correction      #gyr -> att, with bias
        A[block_3x3(3, 3)] = -np.eye(3) * self.accm_bias_p          #acm -> acm
        A[block_3x3(4, 4)] = -np.eye(3) * self.gyro_bias_p          #gyr -> gyr
        
        # TODO replace this with your own code
        #A = solution.eskf.ESKF.get_error_A_continous(self, x_nom_prev, z_corr)

        return A

    def get_error_GQGT_continous(self,
                                 x_nom_prev: NominalState
                                 ) -> 'ndarray[15, 12]':
        """The noise covariance matrix, GQGT, in (10.68)

        From (Theorem 3.2.2) we can see that (10.68) can be written as 
        d/dt x_err = A@x_err + G@n == A@x_err + m
        where m is gaussian with mean 0 and covariance G @ Q @ G.T. Thats why
        we need GQGT.

        Hint: you can use block_3x3 to simplify indexing if you want to.
        The first I element in G can be set as G[block_3x3(2, 1)] = -np.eye(3)

        Args:
            x_nom_prev (NominalState): previous nominal state
        Returns:
            GQGT (ndarray[15, 12]): G @ Q @ G.T
        """
        
        R = x_nom_prev.ori.as_rotmat()
        G = np.zeros([15, 12])
        G[block_3x3(1, 0)] = -R
        G[block_3x3(2, 1)] = -np.eye(3)   
        G[block_3x3(3, 2)] = np.eye(3)  
        G[block_3x3(4, 3)] = np.eye(3)       

        GQGT = G @ self.Q_err @ G.T
        #GQGT = solution.eskf.ESKF.get_error_GQGT_continous(self, x_nom_prev)

        return GQGT

    def get_van_loan_matrix(self, V: 'ndarray[30, 30]'):
        """Use this funciton in get_discrete_error_diff to get the van loan 
        matrix. See (4.63)

        All the tests are ran with do_approximations=False

        Args:
            V (ndarray[30, 30]): [description]

        Returns:
            VanLoanMatrix (ndarray[30, 30]): VanLoanMatrix
        """
        if self.do_approximations:
            # second order approcimation of matrix exponential which is faster
            VanLoanMatrix = np.eye(*V.shape) + V + (V@V) / 2
        else:
            VanLoanMatrix = scipy.linalg.expm(V)
        return VanLoanMatrix

    def get_discrete_error_diff(self,
                                x_nom_prev: NominalState,
                                z_corr: CorrectedImuMeasurement,
                                ) -> Tuple['ndarray[15, 15]',
                                           'ndarray[15, 15]']:
        """Get the discrete equivalents of A and GQGT in (4.63)

        Hint: you should use get_van_loan_matrix to get the van loan matrix

        See (4.5 Discretization) and (4.63) for more information. 
        Or see "Discretization of process noise" in 
        https://en.wikipedia.org/wiki/Discretization

        Args:
            x_nom_prev (NominalState): previous nominal state
            z_corr (CorrectedImuMeasurement): corrected IMU measurement

        Returns:
            Ad (ndarray[15, 15]): discrede transition matrix
            GQGTd (ndarray[15, 15]): discrete noise covariance matrix
        """

        A = self.get_error_A_continous(x_nom_prev, z_corr)
        GQGT = self.get_error_GQGT_continous(x_nom_prev)
        Ts = z_corr.ts - x_nom_prev.ts


        V_10 = np.zeros(GQGT.shape)

        V = np.block([[-A,   GQGT],[V_10, A.T]])*Ts

        VanLoan = self.get_van_loan_matrix(V)

        Ad = VanLoan[block_kxk(1, 1, 15)].T
        GQGTd = Ad @ VanLoan[block_kxk(0, 1,15)]


        # TODO replace this with your own code
        #Ad, GQGTd = solution.eskf.ESKF.get_discrete_error_diff(
        #    self, x_nom_prev, z_corr)

        return Ad, GQGTd

    def predict_x_err(self,
                      x_nom_prev: NominalState,
                      x_err_prev_gauss: ErrorStateGauss,
                      z_corr: CorrectedImuMeasurement,
                      ) -> ErrorStateGauss:
        """Predict the error state

        Hint: This is doing a discrete step of (10.68) where x_err 
        is a multivariate gaussian.

        Args:
            x_nom_prev (NominalState): previous nominal state
            x_err_prev_gauss (ErrorStateGauss): previous error state gaussian
            z_corr (CorrectedImuMeasurement): corrected IMU measuremnt

        Returns:
            x_err_pred (ErrorStateGauss): predicted error state
        """

        Ad, GQGTd = self.get_discrete_error_diff(x_nom_prev, z_corr)

        #Bit unsure
        Ts = z_corr.ts

        x_err_pred_mean = Ad @ x_err_prev_gauss.mean

        x_err_pred_cov = Ad @ x_err_prev_gauss.cov @ Ad.T  + GQGTd

        #dx_err_prev_mean = Ad @ x_err_prev_gauss.mean + GQGT @ n_mean

        #x_err_pred_mean = x_err_prev_gauss.mean + dx_err_prev_mean * Ts

        x_err_pred = ErrorStateGauss(x_err_pred_mean, x_err_pred_cov, Ts)
        # TODO replace this with your own code
        #x_err_pred = solution.eskf.ESKF.predict_x_err(
        #    self, x_nom_prev, x_err_prev_gauss, z_corr)

        return x_err_pred

    def predict_from_imu(self,
                         x_nom_prev: NominalState,
                         x_err_gauss: ErrorStateGauss,
                         z_imu: ImuMeasurement,
                         ) -> Tuple[NominalState, ErrorStateGauss]:
        """Method called every time an IMU measurement is received

        Args:
            x_nom_prev (NominalState): previous nominal state
            x_err_gauss (ErrorStateGauss): previous error state gaussian
            z_imu (ImuMeasurement): raw IMU measurement

        Returns:
            x_nom_pred (NominalState): predicted nominal state
            x_err_pred (ErrorStateGauss): predicted error state
        """


        # TODO replace this with your own code
        #x_nom_pred, x_err_pred = solution.eskf.ESKF.predict_from_imu(
        #    self, x_nom_prev, x_err_gauss, z_imu)

        #return x_nom_pred, x_err_pred

        z_cor = self.correct_z_imu(x_nom_prev, z_imu)

        return self.predict_nominal(x_nom_prev,z_cor), self.predict_x_err(x_nom_prev, x_err_gauss, z_cor)

    def get_gnss_measurment_jac(self, x_nom: NominalState) -> 'ndarray[3,15]':
        """Get the measurement jacobian, H.

        Hint: the gnss antenna has a relative position to the center given by
        self.lever_arm. How will the gnss measurement change if the drone is 
        rotated differently? Use get_cross_matrix and some other stuff :) 

        Returns:
            H (ndarray[3, 15]): [description]
        """
        antenna = x_nom.ori.as_rotmat() @ get_cross_matrix(self.lever_arm)
        Hx = np.block([np.eye(3), np.zeros((3,13))])

        Qd1 = -1 * x_nom.ori.vec_part.T
        Qd2 = get_cross_matrix(x_nom.ori.vec_part) + np.eye(3)*x_nom.ori.real_part
        Qd = np.block([[Qd1] ,[Qd2]])

        X  = scipy.linalg.block_diag(np.eye(6), Qd, np.eye(6))

        H = Hx @ X
        H[block_3x3(0,2)] = -antenna
        # TODO replace this with your own code
        #H = solution.eskf.ESKF.get_gnss_measurment_jac(self, x_nom)

        return H

    def get_gnss_cov(self, z_gnss: GnssMeasurement) -> 'ndarray[3,3]':
        """Use this function in predict_gnss_measurement to get R. 
        Get gnss covariance estimate based on gnss estimated accuracy. 

        All the test data has self.use_gnss_accuracy=False, so this does not 
        affect the tests.

        There is no given solution to this function, feel free to play around!

        Returns:
            gnss_cov (ndarray[3,3]): the estimated gnss covariance
        """
        if self.use_gnss_accuracy and z_gnss.accuracy is not None:
            # play around with this part, the suggested way is not optimal
            gnss_cov = (z_gnss.accuracy/3)**2 * self.gnss_cov

        else:
            # dont change this part
            gnss_cov = self.gnss_cov
        return gnss_cov

    def predict_gnss_measurement(self,
                                 x_nom: NominalState,
                                 x_err: ErrorStateGauss,
                                 z_gnss: GnssMeasurement,
                                 ) -> MultiVarGaussStamped:
        """Predict the gnss measurement

        Hint: z_gnss is only used in get_gnss_cov and to get timestamp for 
        the predicted measurement

        Args:
            x_nom (NominalState): previous nominal state
            x_err (ErrorStateGauss): previous error state gaussian
            z_gnss (GnssMeasurement): gnss measurement

        Returns:
            z_gnss_pred_gauss (MultiVarGaussStamped): gnss prediction gaussian
        """

        H = self.get_gnss_measurment_jac(x_nom)
        R = self.get_gnss_cov(z_gnss) 

        z_gnss_mean = x_nom.pos + x_nom.ori.as_rotmat() @ self.lever_arm
        z_gnss_cov = R + H @ x_err.cov @ H.T
        
        z_gnss_pred_gauss = MultiVarGaussStamped(z_gnss_mean, z_gnss_cov, z_gnss.ts)
        # TODO replace this with your own code
        #z_gnss_pred_gauss = solution.eskf.ESKF.predict_gnss_measurement(    
        #    self, x_nom, x_err, z_gnss)

        return z_gnss_pred_gauss

    def get_x_err_upd(self,
                      x_nom: NominalState,
                      x_err: ErrorStateGauss,
                      z_gnss_pred_gauss: MultiVarGaussStamped,
                      z_gnss: GnssMeasurement
                      ) -> ErrorStateGauss:
        """Update the error state from a gnss measurement

        Hint: see (10.75)
        Due to numerical error its recomended use the robust calculation of 
        posterior covariance.

        I_WH = np.eye(*P.shape) - W @ H
        P_upd = (I_WH @ P @ I_WH.T + W @ R @ W.T)

        Args:
            x_nom (NominalState): previous nominal state
            x_err (ErrorStateGauss): previous error state gaussian
            z_gnss_pred_gauss (MultiVarGaussStamped): gnss prediction gaussian
            z_gnss (GnssMeasurement): gnss measurement

        Returns:
            x_err_upd_gauss (ErrorStateGauss): updated error state gaussian
        """
        H = self.get_gnss_measurment_jac(x_nom)
        R = self.get_gnss_cov(z_gnss)
        S = R + H @ x_err.cov @ H.T
        P = x_err.cov
        W = x_err.cov @ H.T @ (np.linalg.inv(S))

        I_WH = np.eye(*P.shape) - W @ H
        P_upd = (I_WH @ P @ I_WH.T + W @ R @ W.T)
        
        x_err_upd_gauss_cov  = P_upd

        x_err_upd_gauss_mean = W@((z_gnss.pos-x_nom.pos -x_nom.ori.as_rotmat() @ self.lever_arm) 
            - H @ x_err.mean)
        
        x_err_upd_gauss = ErrorStateGauss(x_err_upd_gauss_mean,x_err_upd_gauss_cov,z_gnss.ts)

        # TODO replace this with your own code
        #x_err_upd_gauss = solution.eskf.ESKF.get_x_err_upd(
        #    self, x_nom, x_err, z_gnss_pred_gauss, z_gnss)
        return x_err_upd_gauss

    def inject(self,
               x_nom_prev: NominalState,
               x_err_upd: ErrorStateGauss
               ) -> Tuple[NominalState, ErrorStateGauss]:
        """Perform the injection step

        Hint: see (10.85) and (10.72) on how to inject into nominal state.
        See (10.86) on how to find error state after injection

        Args:
            x_nom_prev (NominalState): previous nominal state
            x_err_upd (ErrorStateGauss): updated error state gaussian

        Returns:
            x_nom_inj (NominalState): nominal state after injection
            x_err_inj (ErrorStateGauss): error state gaussian after injection
        """
        x_nom_inj = deepcopy(x_nom_prev)
        x_nom_inj.pos = x_nom_prev.pos + x_err_upd.pos
        x_nom_inj.vel = x_nom_prev.vel + x_err_upd.vel
        t3 = RotationQuaterion(1,x_err_upd.avec/2)
        x_nom_inj.ori = x_nom_prev.ori.multiply(t3)
        norm = sqrt(x_nom_inj.ori.real_part**2 + x_nom_inj.ori.vec_part.T @ x_nom_inj.ori.vec_part )
        x_nom_inj.ori = RotationQuaterion.from_euler((x_nom_inj.ori.as_euler())/norm)
        x_nom_inj.accm_bias = x_nom_prev.accm_bias + x_err_upd.accm_bias
        x_nom_inj.gyro_bias = x_nom_prev.gyro_bias + x_err_upd.gyro_bias


        G = np.eye(15)
        G[block_kxk(2,2,3)] = np.eye(3) - get_cross_matrix(0.5*x_err_upd.avec)
        x_err_inj = deepcopy(x_err_upd)
        x_err_inj.mean = np.zeros(15,)
        x_err_inj.cov = G @ x_err_inj.cov @ G.T


        # TODO replace this with your own code
        #x_nom_inj, x_err_inj = solution.eskf.ESKF.inject(
        #    self, x_nom_prev, x_err_upd)

        return x_nom_inj, x_err_inj

    def update_from_gnss(self,
                         x_nom_prev: NominalState,
                         x_err_prev: NominalState,
                         z_gnss: GnssMeasurement,
                         ) -> Tuple[NominalState,
                                    ErrorStateGauss,
                                    MultiVarGaussStamped]:
        """Method called every time an gnss measurement is received.


        Args:
            x_nom_prev (NominalState): previous nominal state 
            x_err_prev (ErrorStateGauss): previous error state gaussian
            z_gnss (GnssMeasurement): gnss measurement

        Returns:
            x_nom_inj (NominalState): nominal state after injection
            x_err_inj (ErrorStateGauss): error state gauss after injection
            z_gnss_pred_gauss (MultiVarGaussStamped): predicted gnss 
                measurement, used for NIS calculations.
        """
        z_gnss_pred_gauss = self.predict_gnss_measurement(x_nom_prev, x_err_prev, z_gnss)
        x_err_upd = self.get_x_err_upd(x_nom_prev, x_err_prev, z_gnss_pred_gauss, z_gnss)
        x_nom_inj, x_err_inj = self.inject(x_nom_prev, x_err_upd)

        # TODO replace this with your own code
        #x_nom_inj, x_err_inj, z_gnss_pred_gauss = solution.eskf.ESKF.update_from_gnss(
        #    self, x_nom_prev, x_err_prev, z_gnss)

        return x_nom_inj, x_err_inj, z_gnss_pred_gauss
