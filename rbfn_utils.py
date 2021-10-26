import numpy as np
from typing import Optional

class RBFN():
    def __init__(self, rbf_func, target_pts: np.ndarray, num_linear_units: int, epsilon: float, weights: Optional[np.ndarray]=None) -> None:
        """Initializes single hidden layer RBF network 

        rbf_func (function): Function to use for radial basis function
            Must accept epsilon, input point, and target point
            as inputs
            Returns float
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
        self.target_pts = target_pts
        self.epsilon = epsilon
        # In self.weights, each row is a linear unit, each column is a weight of that unit
        if weights is None:
            self.weights = np.random.rand( num_linear_units, target_pts.shape[0]+1)
        else:
            self.weights = weights
        
        self.out_rbf_layer = None
        self.out_linear_layer = None

    def forward(self, input_pts: np.ndarray)->np.ndarray:
        """Run input points through the network

        input_pts (np.ndarray): NxM numpy array
            N (rows) is the input points
            M (cols) is the features for each point
        """

        # Ouput of rbf layer is (number of input points) X (number of target points)
        self.out_rbf_layer = np.zeros((input_pts.shape[0], self.target_pts.shape[0]))

        # Apply rbfs to input point and save output
        for pt_num in range(input_pts.shape[0]):
            for rbf_num in range(self.target_pts.shape[0]):
                self.out_rbf_layer[pt_num, rbf_num] = guassian_rbf_func(input_pts[pt_num], self.target_pts[rbf_num], self.epsilon)

        # Add bias term to rbf layer output before putting output through the linear layer
        out_rbf_layer_b = add_bias(self.out_rbf_layer)

        # Output of linear layer is (number of input points) X (number of linear units)
        self.out_linear_layer = np.zeros((input_pts.shape[0], self.weights.shape[0]))

        # Apply weights to output of rbf hidden layer
        for pt_num in range(input_pts.shape[0]):
            for unit_num in range(self.weights.shape[0]):
                # Apply weights and take sum
                self.out_linear_layer[pt_num, unit_num] = np.sum(out_rbf_layer_b[pt_num] * self.weights[unit_num])

        return self.out_linear_layer
    
    def update_target(self, num_target: int, new_target_pt: np.ndarray)->None:
        """Updates a target point for an RBF function by replacing the current 
        target point with a new target point 

        num_target (int): number of the target point you want to update
        target_pt (np.ndarray): N numpy array
            Each element represents a feature of the new target point
        """
        if new_target_pt.shape[0] != self.target_pts.shape[1]:
            raise Exception("New target point does not have the same number of features as existing target point")

        self.target_pts[num_target] = new_target_pt

        return None

def add_bias(arr: np.ndarray):
    return np.hstack((arr, np.ones((arr.shape[0], 1))))

def guassian_rbf_func(input_pt: np.ndarray, target_pt: np.ndarray, epsilon: float)->float:
    r = np.linalg.norm(input_pt-target_pt)
    out = np.exp(- (epsilon * r)**2 )
    return out

def multiquadric_rbf_func(input_pt: np.ndarray, target_pt: np.ndarray, epsilon: float)->float:
    r = np.linalg.norm(input_pt-target_pt)
    out = np.sqrt(1+(epsilon*r)**2)
    return out

def inverse_quadric_rbf_func(input_pt: np.ndarray, target_pt: np.ndarray, epsilon: float)->float:
    r = np.linalg.norm(input_pt-target_pt)
    out = 1/(1+(epsilon*r)**2)
    return out

def inverse_multiquadric_func(input_pt: np.ndarray, target_pt: np.ndarray, epsilon: float)->float:
    r = np.linalg.norm(input_pt-target_pt)
    out = 1/np.sqrt(1+(epsilon*r)**2)
    return out
