import numpy as np
from numpy import SHIFT_OVERFLOW, float128, float64, ndarray, zeros
from typing import Sequence, Optional
#from Graded2_eskf_handout.eskf.eskf import ESKF
from utils.indexing import block_kxk

from datatypes.measurements import GnssMeasurement
from datatypes.eskf_states import NominalState, ErrorStateGauss
from datatypes.multivargaussian import MultiVarGaussStamped

import solution


def get_NIS(z_gnss: GnssMeasurement,
            z_gnss_pred_gauss: MultiVarGaussStamped,
            marginal_idxs: Optional[Sequence[int]] = None
            ) -> float:
    """Calculate NIS

    Args:
        z_gnss (GnssMeasurement): gnss measurement
        z_gnss_pred_gauss (MultiVarGaussStamped): predicted gnss measurement
        marginal_idxs (Optional[Sequence[int]]): Sequence of marginal indexes.
            For example used for calculating NIS in only xy direction.  

    Returns:
        NIS (float): NIS value
    """

    v = z_gnss.pos - z_gnss_pred_gauss.mean

    S = z_gnss_pred_gauss.cov

    if(marginal_idxs != None):
        S_new = np.zeros((len(v),))
        for k in marginal_idxs:
            S_new[k] = 1/S[k,k]
        NIS = v.T @ (S_new * v)

    else:
        NIS = v.T @ np.linalg.inv(S) @ v

    # TODO replace this with your own code
    #NIS = solution.nis_nees.get_NIS(z_gnss, z_gnss_pred_gauss, marginal_idxs)

    return NIS


def get_error(x_true: NominalState,
              x_nom: NominalState,
              ) -> 'ndarray[15]':
    """Finds the error (difference) between True state and 
    nominal state. See (Table 10.1).


    Returns:
        error (ndarray[15]): difference between x_true and x_nom. 
    """
    error = np.zeros(15,)
    error[0:3] = x_true.pos - x_nom.pos
    error[3:6] = x_true.vel - x_nom.vel
    err_quat = (x_nom.ori.conjugate()).multiply(x_true.ori)
    error[6:9] = err_quat.as_avec()
    error[9:12] = x_true.accm_bias - x_nom.accm_bias
    error[12:16] = x_true.gyro_bias - x_nom.gyro_bias

    
    # TODO replace this with your own code
    #error = solution.nis_nees.get_error(x_true, x_nom)

    return error


def get_NEES(error: 'ndarray[15]',
             x_err: ErrorStateGauss,
             marginal_idxs: Optional[Sequence[int]] = None
             ) -> float:
    """Calculate NEES

    Args:
        error (ndarray[15]): errors between x_true and x_nom (from get_error)
        x_err (ErrorStateGauss): estimated error
        marginal_idxs (Optional[Sequence[int]]): Sequence of marginal indexes.
            For example used for calculating NEES for only the position. 

    Returns:
        NEES (float): NEES value
    """
    err_vec = x_err.mean - error
    P = x_err.cov

    if(marginal_idxs != None):
        P_masked = P[block_kxk(int(marginal_idxs[0]/len(marginal_idxs)) ,int(marginal_idxs[0]/len(marginal_idxs)), len(marginal_idxs))]
        err_vec_masked = err_vec[marginal_idxs[0]:marginal_idxs[-1]+1]
        
        NEES = err_vec_masked.T @ np.linalg.inv(P_masked) @ err_vec_masked
        
    else:

        NEES = (err_vec).T @ np.linalg.inv(P) @ (err_vec) 

    # TODO replace this with your own code
    #NEES = solution.nis_nees.get_NEES(error, x_err, marginal_idxs)

    return NEES


def get_time_pairs(unique_data, data):
    """match data from two different time series based on timestamps"""
    gt_dict = dict(([x.ts, x] for x in unique_data))
    pairs = [(gt_dict[x.ts], x) for x in data if x.ts in gt_dict]
    times = [pair[0].ts for pair in pairs]
    return times, pairs
