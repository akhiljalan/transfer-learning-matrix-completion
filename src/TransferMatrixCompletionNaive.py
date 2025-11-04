from MatrixCompletionSolver import MatrixCompletionSolver

class TransferMatrixCompletionNaive(MatrixCompletionSolver):
    '''
    Implements "naive" matrix completion with P, Q 
    '''
    
    def __init__(self, 
        rank: int = None):
        assert rank >= 1, f'Bad rank: {rank}. Must be positive integer.'
        super().__init__(rank)
        
    def fit(self, AP, AQ):
        '''
        AP, AQ are observed P, Q matrices. 
        '''
        self.AP = AP
        
        # Scale AQ to [-1, 1]
        self.scale = np.max(np.abs(AQ))
        if self.scale > 1:
            self.AQ = AQ / self.scale
        else:
            self.scale = 1
            self.AQ = AQ
        
        # Estimate sampling rate
        self.p_hat = np.mean(AP != 0)
        
        # Scale AP by sampling rate before SVD
        AP_scaled = AP / self.p_hat
        
        # Get truncated SVDs
        self.U, self.V, _ = self._get_svd_truncated(AP_scaled)
        self.U_q, self.V_q, _ = self._get_svd_truncated(self.AQ)
        
        self.fitted = True
        return self
    
    def predict(self):
        '''
        Estimate Q matrix. 
        '''
        # Predict must be called after fit()
        self._check_fitted()

        # Q estimate is a matrix product
        Q_hat = self.U @ self.U.T @ self.AQ @ self.V @ self.V.T
        return Q_hat * self.scale  # Unscale the result