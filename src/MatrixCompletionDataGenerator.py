import numpy as np
from scipy.linalg import expm

class MatrixCompletionDataGenerator:
    '''
    Generator for matrix completion transfer learning experiments.
    
    This class handles generation of source and target matrices with controlled
    rotations between their singular vectors, along with sampling masks.
    '''
    def __init__(self, n, d, theta=0.1, seed=0, sv_distribution='wigner', sv_params=None, 
                 sigmaP=0.0, sigmaQ=0.0):
        """Initialize generator with matrix dimensions and rotation parameters.
        
        Args:
            n (int): Matrix dimension (n x n matrices)
            d (int): Rank of matrices
            theta (float): Maximum rotation angle between source/target (default: 0.1)
            seed (int): Random seed (default: 0)
            sv_distribution (str): Type of singular value distribution 
                                 ('wigner', 'uniform', 'exponential')
            sv_params (dict): Parameters for singular value generation
            sigmaP (float): Standard deviation of Gaussian noise for P matrix (default: 0.0)
            sigmaQ (float): Standard deviation of Gaussian noise for Q matrix (default: 0.0)
        """
        self.n = n
        self.d = d
        self.theta = theta
        self.seed = seed
        self.sv_distribution = sv_distribution
        self.sv_params = sv_params or {}
        np.random.seed(seed)
        
        # Generate singular values according to specified distribution
        self.s_p = self._gen_singular_values()
        self.s_q = self._gen_singular_values()
        
        # Initialize source matrices
        UP_full, VP_full = self._gen_random_rotation_matrix(n), self._gen_random_rotation_matrix(n)
        self.UP = UP_full[:, :self.d].copy()
        self.VP = VP_full[:, :self.d].copy()
        
        # Generate rotation between source and target
        self.rot1, self.rot2 = self._gen_small_rotations()
        
        # Compute target matrices
        self.UQ = self.UP @ self.rot1
        self.VQ = self.VP @ self.rot2
        
        # Generate full matrices
        self.P = self._construct_matrix(self.UP, self.VP, self.s_p)
        self.Q = self._construct_matrix(self.UQ, self.VQ, self.s_q)
        
        # Normalize matrices to have max absolute value of 1
        self.P = self.P / np.max(np.abs(self.P))
        self.Q = self.Q / np.max(np.abs(self.Q))
        
        self.sigmaP = sigmaP
        self.sigmaQ = sigmaQ
    
    def _gen_random_rotation_matrix(self, d):
        """Generate random rotation matrix using QR decomposition."""
        Q, _ = np.linalg.qr(np.random.normal(size=(d, d)))
        return Q
    
    def _gen_small_rotations(self):
        """Use SpecialOrthogonalGroup""" 
        so_group = SpecialOrthogonalGroup(self.d, k=2, retraction='qr', theta=self.theta)
        return so_group.random_point(seed=self.seed)

        # """Generate small random rotation using skew-symmetric matrix."""
        # A = np.random.randn(self.n, self.n)
        # A = A - A.T  # Make skew-symmetric
        # A = A / np.linalg.norm(A, 'fro') * self.theta
        # return expm(A)
    
    def _construct_matrix(self, U, V, s):
        """Construct matrix from its SVD components."""
        s_len_d = s[:self.d].copy()
        return U @ np.diag(s_len_d) @ V.T
    
    def generate_masks(self, p_row_p, p_col_p, p_row_q, p_col_q):
        """Generate sampling masks for source and target matrices.
        
        Args:
            p_row_p (float): Row sampling probability for source
            p_col_p (float): Column sampling probability for source
            p_row_q (float): Row sampling probability for target
            p_col_q (float): Column sampling probability for target
            
        Returns:
            tuple: (mask_p, mask_q) Binary mask matrices
        """
        mask_p = self._generate_row_col_mask(p_row_p, p_col_p)
        mask_q = self._generate_row_col_mask(p_row_q, p_col_q)
        return mask_p, mask_q
    
    def _generate_row_col_mask(self, p_row, p_col):
        """Generate single mask matrix with row/column sampling."""
        row_mask = np.random.uniform(size=(self.n,)) < p_row
        col_mask = np.random.uniform(size=(self.n,)) < p_col
        return (row_mask[:, np.newaxis] * col_mask).astype(float)
    
    def generate_sample(self, p_row_p=0.1, p_col_p=0.1, p_row_q=0.1, p_col_q=0.1):
        """Generate complete and masked matrices for an experiment, with noise.
        
        Args:
            p_row_p (float): Row sampling probability for source (default: 0.1)
            p_col_p (float): Column sampling probability for source (default: 0.1)
            p_row_q (float): Row sampling probability for target (default: 0.1)
            p_col_q (float): Column sampling probability for target (default: 0.1)
            
        Returns:
            tuple: (P, Q, AP, AQ) Complete and masked matrices, where AP and AQ include noise
        """
        mask_p, mask_q = self.generate_masks(p_row_p, p_col_p, p_row_q, p_col_q)
        
        # Add noise only to the observed entries
        noise_p = np.random.normal(0, self.sigmaP, size=self.P.shape)
        noise_q = np.random.normal(0, self.sigmaQ, size=self.Q.shape)
        
        AP = np.multiply(mask_p, self.P + noise_p)
        AQ = np.multiply(mask_q, self.Q + noise_q)
        
        return self.P, self.Q, AP, AQ
    
    @property
    def source_singular_vectors(self):
        """Get singular vectors of source matrix."""
        return self.UP, self.VP
    
    @property
    def target_singular_vectors(self):
        """Get singular vectors of target matrix."""
        return self.UQ, self.VQ
    
    @property
    def singular_values(self):
        """Get singular values of source and target matrices."""
        return self.s_p, self.s_q 

    
    def _gen_singular_values(self):
        """Generate singular values according to specified distribution."""
        if self.sv_distribution == 'wigner':
            return self._gen_wigner_singular_values()
        elif self.sv_distribution == 'uniform':
            return self._gen_uniform_singular_values()
        elif self.sv_distribution == 'exponential':
            return self._gen_exponential_singular_values()
        elif self.sv_distribution == 'constant':
            return self._gen_constant_singular_values()
        else:
            raise ValueError(f"Unknown distribution: {self.sv_distribution}")
    
    def _gen_wigner_singular_values(self):
        """Generate singular values according to Wigner's semicircle law."""
        # Parameters for the semicircle distribution
        R = 2 * np.sqrt(self.n)  # radius (using σ=1)
        
        # Generate points from semicircle distribution
        x = np.linspace(-R, R, self.n)
        pdf = np.sqrt(R**2 - x**2) / (np.pi * R**2)
        pdf = pdf / np.sum(pdf)  # normalize
        
        # Sample singular values from this distribution
        s = np.random.choice(x, size=self.d, p=pdf)
        s = np.abs(s)  # ensure positive
        s = np.sort(s)[::-1]  # sort in descending order
        
        # Pad with zeros to full dimension
        return np.concatenate([s, np.zeros(self.n - self.d)])
    
    def _gen_uniform_singular_values(self, low=5.0, high=10.0):
        """Generate uniformly distributed singular values."""
        # low = self.sv_params.get('low', 5.0)
        # high = self.sv_params.get('high', 10.0)
        
        s = np.random.uniform(low=low, high=high, size=self.d)
        s = np.sort(s)[::-1]  # sort in descending order
        return np.concatenate([s, np.zeros(self.n - self.d)])
    
    def _gen_exponential_singular_values(self):
        """Generate exponentially decaying singular values."""
        scale = self.sv_params.get('scale', 1.0)
        base = self.sv_params.get('base', 0.9)
        
        s = scale * np.power(base, np.arange(self.d))
        return np.concatenate([s, np.zeros(self.n - self.d)])
    
    def _gen_constant_singular_values(self):
        """Generate constant singular values."""
        s = np.ones(self.d)
        return np.concatenate([s, np.zeros(self.n - self.d)])


# from pymanopt.tools.multi import (
#     multiexpm,
#     multihconj,
#     multiherm,
#     multilogm,
#     multiqr,
#     multiskew,
#     multiskewh,
#     multitransp,
# )
from pymanopt.manifolds.group import _UnitaryBase

def random_skew_symmetric_matrix(n: int, k: int = 1):
    """Generate random skew-symmetric matrices.
    
    Args:
        n: Size of each matrix
        k: Number of matrices to generate
        
    Returns:
        Array of shape (k, n, n) containing k random skew-symmetric matrices
        (or just (n, n) if k=1)
    """
    # Generate random upper triangular entries
    triu_indices = np.triu_indices(n, k=1)
    if k == 1:
        skew = np.zeros((n, n))
        skew[triu_indices] = np.random.randn(len(triu_indices[0]))
        # Make it skew-symmetric
        skew = skew - skew.T
        return skew
    else:
        skew = np.zeros((k, n, n))
        entries = np.random.randn(k, len(triu_indices[0]))
        # Fill upper triangle for each matrix
        for i in range(k):
            skew[i][triu_indices] = entries[i]
        # Make them skew-symmetric
        skew = skew - np.transpose(skew, axes=(0, 2, 1))
        return skew

import scipy 
class SpecialOrthogonalGroup(_UnitaryBase):
    r"""The (product) manifold of rotation matrices.
    
    Additional Parameters:
    theta (float, optional): Maximum rotation angle in radians. If provided, 
           random points will be constrained to rotations of at most theta radians.
    """

    def __init__(self, n: int, *, k: int = 1, retraction: str = "qr", theta: float = None):
        self._n = n
        self._k = k
        self._theta = theta  # Store the maximum rotation angle

        if k == 1:
            name = f"Special orthogonal group SO({n})"
        elif k > 1:
            name = f"Special orthogonal group SO({n})^{k}"
        else:
            raise ValueError("k must be an integer no less than 1.")
        dimension = int(k * scipy.special.comb(n, 2))
        super().__init__(name, dimension, retraction)

    def random_point(self, theta: float = None, seed: int = None):
        n, k = self._n, self._k
        if seed is not None:
            np.random.seed(seed)
        theta = theta if theta is not None else self._theta

        if n == 1:
            return np.ones((k, 1, 1)) if k > 1 else np.ones((1, 1))

        if theta is None:
            # Original behavior - full random rotation
            point, z = multiqr(np.random.normal(size=(k, n, n)))
        else:
            # Generate restricted rotation
            # First generate skew-symmetric matrices with bounded norm
            skew = np.random.uniform(-1, 1, size=(k, n, n))
            skew = (skew - skew.transpose((0, 2, 1))) / 2
            # Scale the matrices to bound the rotation angle
            norms = np.linalg.norm(skew, axis=(1, 2), keepdims=True)
            skew = skew * np.minimum(1, theta / (norms + 1e-10))
            # Convert to rotation matrices using matrix exponential
            point = np.array([scipy.linalg.expm(s) for s in skew])

        # Handle determinant signs
        # Fix the unpacking issue
        negative_det = np.where(np.linalg.det(point) < 0)[0]  # Just take first array from np.where
        if len(negative_det) > 0:
            negative_det = np.expand_dims(negative_det, (-2, -1))
            point[negative_det, :, [0, 1]] = point[negative_det, :, [1, 0]]


        return point[0] if k == 1 else point
    def identity(self):
        """Return the identity element of SO(n).
        
        Returns:
            numpy.ndarray: Identity matrix of shape (n,n) if k=1,
                        or shape (k,n,n) if k>1
        """
        if self._k == 1:
            return np.eye(self._n)
        else:
            return np.tile(np.eye(self._n), (self._k, 1, 1))
    
    def random_tangent_vector(self, point, theta: float = None):
        theta = theta if theta is not None else self._theta
        vector = random_skew_symmetric_matrix(self._n, self._k)
        
        if theta is not None:
            # Scale the tangent vector to respect the maximum rotation
            norms = np.linalg.norm(vector, axis=(-2, -1), keepdims=True)
            vector = vector * np.minimum(1, theta / (norms + 1e-10))

        if self._k == 1:
            vector = vector[0]
        return vector / self.norm(point, vector)

class GivenMatrixCompletionData(MatrixCompletionDataGenerator):
    """Handler for matrix completion transfer learning experiments with given matrices.
    
    This class handles source and target matrices that are provided as inputs,
    along with sampling mask generation. Inherits sampling functionality from 
    MatrixCompletionDataGenerator.
    """
    
    def __init__(self, P, Q, d, seed=0, sigmaP=0.0, sigmaQ=0.0):
        """Initialize handler with source and target matrices.
        
        Args:
            P (np.ndarray): Source matrix (n x n)
            Q (np.ndarray): Target matrix (n x n)
            d (int, optional): Rank to use for SVD truncation. If None, no truncation.
            seed (int): Random seed (default: 0)
            sigmaP (float): Standard deviation of Gaussian noise for P matrix (default: 0.0)
            sigmaQ (float): Standard deviation of Gaussian noise for Q matrix (default: 0.0)
        """
        assert P.shape == Q.shape, "P and Q must have the same shape"
        assert len(P.shape) == 2, "Must be a 2D matrix"
        # and P.shape[0] == P.shape[1], "Matrices must be square"
        
        # Initialize parent class with minimal parameters
        super().__init__(n=P.shape[0], d=d, seed=seed, 
                        sigmaP=sigmaP, sigmaQ=sigmaQ)
        self.m = P.shape[0]
        self.n = P.shape[1]
        
        # Store original matrices
        self.P_original = P.copy()
        self.Q_original = Q.copy()
        
        # Compute SVD and potentially truncate
        UP, sP, VhP = np.linalg.svd(P, full_matrices=False)
        UQ, sQ, VhQ = np.linalg.svd(Q, full_matrices=False)
        
        # if d is not None:
        self.d = min(min(d, self.n), self.m)
        self.UP = UP[:, :self.d]
        self.VP = VhP.T[:, :self.d]
        self.UQ = UQ[:, :self.d]
        self.VQ = VhQ.T[:, :self.d]
        self.s_p = sP[:self.d]
        self.s_q = sQ[:self.d]
            
    
        # Normalize matrices to have max absolute value of 1
        self.P = self.P_original / np.max(np.abs(self.P_original))
        self.Q = self.Q_original / np.max(np.abs(self.Q_original))

    def _generate_row_col_mask(self, p_row, p_col):
        """Generate single mask matrix with row/column sampling."""
        row_mask = np.random.uniform(size=(self.m,)) < p_row
        col_mask = np.random.uniform(size=(self.n,)) < p_col
        return np.outer(row_mask, col_mask).astype(float)

    def _generate_AP_AQ(self, p_row_p, p_col_p, p_row_q, p_col_q):
        mask_p, mask_q = self.generate_masks(p_row_p, p_col_p, p_row_q, p_col_q)
        if self.sigmaP > 0:
            noise_p = np.random.normal(0, self.sigmaP, size=self.P.shape)
        else:
            noise_p = np.zeros(self.P.shape)
        if self.sigmaQ > 0:
            noise_q = np.random.normal(0, self.sigmaQ, size=self.Q.shape)
        else:
            noise_q = np.zeros(self.Q.shape)
        AP = np.multiply(mask_p, self.P + noise_p)
        AQ = np.multiply(mask_q, self.Q + noise_q)
        return AP, AQ

    def _generate_mask_MCAR(self, p_entry): 
        mask = np.random.uniform(size=(self.m, self.n)) < p_entry
        return mask
    
    def _generate_AP_MCAR(self, p_entry): 
        mask_p = self._generate_mask_MCAR(p_entry)
        if self.sigmaP > 0:
            noise_p = np.random.normal(0, self.sigmaP, size=self.P.shape)
            AP = np.multiply(mask_p, self.P + noise_p)
        else:
            AP = np.multiply(mask_p, self.P)
        return AP