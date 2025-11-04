from src import (
    TransferMatrixCompletion,
    TransferMatrixCompletionNaive,
    USVTCompletion,
    BhattacharyaUSVT, 
    TransferMatrixCompletionRegression, 
    MatrixCompletionDataGenerator, 
    GivenMatrixCompletionData
)

from matrix_completion_data import (
    MatrixCompletionDataGenerator, 
    GivenMatrixCompletionData
)
from matrix_completion_levin_jmlr import *

from matrix_completion_data import (
    MatrixCompletionDataGenerator, 
    GivenMatrixCompletionData
)


from matrix_completion_algorithms_loop import transfer_test

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy

def main(): 
    P = np.load('data/gds_4971_rnaseq/gds4971_day1_rnaseq.npy').T
    Q = np.load('data/gds_4971_rnaseq/gds4971_day2_rnaseq.npy').T

    mean_expr_P = P.mean(axis=0)
    first_300_indices = np.argsort(mean_expr_P)[-300:]

    P_most_expressed = P[:, first_300_indices].copy()
    Q_most_expressed = Q[:, first_300_indices].copy()
    
    # Manually checked beforehand. If not known, compute effective rank
    # using SVD. 
    d = 4

    transfer_test_results = transfer_test(data_gen_most_expressed.P, data_gen_most_expressed.Q, 
                                matrix_rank=d, 
                                verbose=True, 
                                num_repetitions=10, 
                                sigmaP=0.0, sigmaQ=0.1, 
                                p_values = p_values_range)

    transfer_test_results.to_csv('demos/rna_transfer_demo.csv')

    # TODO finish 
    plot_error_frobenius(rna_results_most_expressed_short, title='RNA Sequencing Transfer', x_logscale=False, 
                     savepath=None)


def transfer_test(P, Q, matrix_rank, 
                  num_repetitions=5, 
                  p_values=None, # Sampling rates 
                  sigmaP=0.0, # additive noise standard deviation for P 
                  sigmaQ=0.0, # additive noise standard deviation for Q 
                  verbose=False,
                  normalize_Q=False, 
                  Q_nonnegative=False):
    '''
    Runs multiple experiments for both our method and USVT
    at different sampling rates. 
    '''
    if p_values is None:
        p_values = np.linspace(0.1, 1.0, 10)
    
    results = []
    data_gen = GivenMatrixCompletionData(P, Q, d=matrix_rank, 
                                         sigmaP=sigmaP, sigmaQ=sigmaQ)

    for p in p_values:
        for rep in range(num_repetitions):
            seed = int(p * 10000) + rep
            np.random.seed(seed)
            
            # Generate observed matrices
            _, AQ = data_gen._generate_AP_AQ(
                p_row_p=1.0,
                p_col_p=1.0,
                p_row_q=p,
                p_col_q=p
            )
            
            # USVT baseline (only uses AQ)
            usvt = USVT(USVTConfig(rank=matrix_rank))
            usvt.fit(AQ)
            Q_hat_usvt = usvt.predict()
            if Q_nonnegative:
                Q_hat_usvt = np.abs(Q_hat_usvt)

            # Error metrics
            usvt_error = frob_error(Q_hat_usvt, data_gen.Q)
            usvt_max_error = max_error(Q_hat_usvt, data_gen.Q)
            usvt_mae = mae_error(Q_hat_usvt, data_gen.Q)
            usvt_rmse = rmse_error(Q_hat_usvt, data_gen.Q)
            
            # -----------------------
            # Transfer Regression
            # -----------------------
            AP, _ = data_gen._generate_AP_AQ(
                p_row_p=1.0,
                p_col_p=1.0,
                p_row_q=1.0,
                p_col_q=1.0
            )

            transfer_reg = TransferMatrixCompletionRegression(
                TransferRegConfig(rank=matrix_rank))
            transfer_reg.fit(AP, AQ)
            Q_hat_transfer = transfer_reg.predict()
            if Q_nonnegative:
                Q_hat_transfer = np.abs(Q_hat_transfer)

            # Error metrics
            transfer_error = frob_error(Q_hat_transfer, data_gen.Q)
            transfer_max_error = max_error(Q_hat_transfer, data_gen.Q)
            transfer_mae = mae_error(Q_hat_transfer, data_gen.Q)
            transfer_rmse = rmse_error(Q_hat_transfer, data_gen.Q)

            # -----------------------
            # Store results
            # -----------------------
            results.append({
                'method': 'usvt',
                'error': usvt_error,
                'max_error': usvt_max_error,
                'mae': usvt_mae,
                'rmse': usvt_rmse,
                'repetition': rep,
                'seed': seed,
                'p': p,
            })
            results.append({
                'method': 'transfer_regression_ours',
                'error': transfer_error,
                'max_error': transfer_max_error,
                'mae': transfer_mae,
                'rmse': transfer_rmse,
                'repetition': rep,
                'seed': seed,
                'p': p,
            })

            if verbose and rep % 10 == 0:
                print(f'Finished p={p}, rep={rep}')
    
    return pd.DataFrame(results)
