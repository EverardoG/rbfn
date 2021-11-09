from rbfn import RBFN, guassian_rbf_func
import numpy as np
import unittest

class TestRBFN(unittest.TestCase):

    def test_forward_out(self):
        """Test if we get the output we expect when we do a forward pass on the network

        target_pts is (num_targets) X (num_features)
        Each row is a target point for an RBF
        Each column is a feature of that target point

        First target is all 0s, second is all 1s, etc

        If we send forward points with features that are all 0,
        we expect the first target will be activated the most, and so on

        We should just see the output always sum to 2 no matter
        which point we put in. 1 for the sum of all the outputs of the RB functions
        for a particular input point PLUS 1 for the bias term

        """
        # Set up network with arbitrary parameters
        num_targets = 5
        num_features = 20
        target_pts = np.full((num_features, num_targets), np.arange(5)).T
        num_linear_units = 2
        epsilon = 0.99
        weights = np.ones((num_linear_units, num_targets+1))

        # One target is all 0s, next is all 1s, etc
        # Activation function will just return linear layer output without modification
        rbfn = RBFN(rbf_func=guassian_rbf_func, target_pts=target_pts, 
                    num_linear_units=num_linear_units, epsilon=epsilon, weights=weights)
        
        # Pass input pts forward
        num_input_pts = 1
        input_vals = np.arange(num_targets)
        outs = []
        for input_val in input_vals:
            input_pts = np.ones((num_input_pts, num_features)) * input_val
            outs.append(rbfn.forward(input_pts=input_pts))

        # Compare to expected output
        exp = np.ones((num_input_pts, num_linear_units))*2
        for out in outs:
            self.assertTrue(np.allclose(out, exp))
    
    def test_update_targets(self):
        """Test if we can update the target points the RBFN uses
        """
        # Set up network with arbitrary parameters
        num_targets = 5
        num_features = 20
        target_pts = np.zeros((num_targets, num_features))
        num_linear_units = 2
        epsilon = 0.99
        rbfn = RBFN(rbf_func=guassian_rbf_func, target_pts=target_pts,
                    num_linear_units=num_linear_units, epsilon=epsilon)

        # Update target point 2
        new_target_pt = np.ones(num_features)
        rbfn.update_target(2, new_target_pt)

        # Make sure targets were updated properly
        self.assertTrue(np.allclose(rbfn.target_pts[2,:], new_target_pt))

        # Check if the update fails with an improperly sized target point
        new_target_pt = np.ones(num_features-1)
        with self.assertRaises(Exception):
            rbfn.update_target(2, new_target_pt)
        
    def test_backprop(self):
        """Test if the network backpropogates error properly

        1) Test if linear layer weights are updated properly 
        Create 4 center points at RBFs with known locations
        Create 4 linear output units with all weights set to 0
        Learning rate for centers is 0
        Learning rate for linear weights is low

        Create 4 input points for training that are copies of the RBF center points
        Each point belongs to a distinct class: 0,1,2,3

        The network should update weights such that each point is easily classified
        with training
        """

        """
        2) Test if rbf centers are updated properly
        """
        pass


if __name__ == "__main__":
    unittest.main()