import numpy as np
from typing import Optional

class RBFN():
    def __init__(self, rbf_func, activation_func, target_pts: np.ndarray, num_linear_units: int, epsilon: float, weights: Optional[np.ndarray]=None) -> None:
        """Initializes single hidden layer RBF network 

        rbf_func (function): Function to use for radial basis function
            Must accept epsilon, input point, and target point
            as inputs
            Returns float
        activation_func (function): Nonlinear activation function to apply
            to the output of the linear layer
            Must accept array as input point
            Returns 
        target_pts (np.ndarray): NxM numpy array
            N (rows) is the target points
                One target point for each RBF node in the hidden layer
            M (cols) is the features of each point
                One feature in target points for each feature in input points
        num_linear_units (int): number of linear units to use in output layer
        epsilon (float): epsilon to use for RBF functions
        weights (Optional[np.ndarray]): (num_linear_units)xN numpy array (for debugging)
        """
        self.rbf_func = rbf_func
        self.activation_func = activation_func
        self.target_pts = target_pts
        self.epsilon = epsilon
        # In self.weights, each row is a linear unit, each column is a weight of that unit
        if weights is None:
            self.weights = np.random.rand( num_linear_units, target_pts.shape[0])
        else:
            self.weights = weights

    def forward(self, input_pts: np.ndarray)->np.ndarray:
        """Run input points through the network

        input_pts (np.ndarray): NxM numpy array
            N (rows) is the input points
            M (cols) is the features for each point
        """
        # Ouput of rbf layer is (number of input points) X (number of target points)
        out_rbf_layer = np.zeros((input_pts.shape[0], self.target_pts.shape[0]))

        # Apply rbfs to input point and save output
        for pt_num in range(input_pts.shape[0]):
            for rbf_num in range(self.target_pts.shape[0]):
                out_rbf_layer[pt_num, rbf_num] = guassian_rbf_func(input_pts[pt_num], self.target_pts[rbf_num], self.epsilon)

        # Output of linear layer is (number of input points) X (number of linear units)
        out_linear_layer = np.zeros((input_pts.shape[0], self.weights.shape[0]))

        # Apply weights to output of rbf hidden layer
        for pt_num in range(input_pts.shape[0]):
            for unit_num in range(self.weights.shape[0]):
                # Apply weights and take sum
                out_linear_layer[pt_num, unit_num] = np.sum(out_rbf_layer[pt_num] * self.weights[unit_num])

        # Apply activation function
        out_activation_func = self.activation_func(out_linear_layer)

        return out_activation_func

def guassian_rbf_func(input_pt: np.ndarray, target_pt: np.ndarray, epsilon: float)->float:
    r = np.linalg.norm(input_pt-target_pt)
    out = np.exp(- (epsilon * r)**2 )
    return out

def dummy_activation_func(input_pt: np.ndarray)->np.ndarray:
    return input_pt