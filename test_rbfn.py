from rbfn import RBFN, RBFUNC
import numpy as np
import unittest

class TestRBFN(unittest.TestCase):

    def test_forward_out(self)->None:
        """Test if we get the output we expect when we do a forward pass on the network

        center_pts is (num_centers) X (num_features)
        Each row is a center point for an RBF
        Each column is a feature of that center point

        First center is all 0s, second is all 1s, etc

        If we send forward points with features that are all 0,
        we expect the first center will be activated the most, and so on

        We should just see the output always sum to 2 no matter
        which point we put in. 1 for the sum of all the outputs of the RB functions
        for a particular input point PLUS 1 for the bias term

        """
        # Set up network with arbitrary parameters
        num_centers = 5
        num_features = 20
        center_pts = np.full((num_features, num_centers), np.arange(5)).T
        num_linear_units = 2
        epsilon = 0.99
        weights = np.ones((num_linear_units, num_centers+1))

        # One center is all 0s, next is all 1s, etc
        # Activation function will just return linear layer output without modification
        rbfn = RBFN(rbfunc=RBFUNC.GUASSIAN, center_pts=center_pts, 
                    num_linear_units=num_linear_units, epsilon=epsilon, 
                    linear_lr=None, rbf_lr=None, weights=weights)
        
        # Pass input pts forward
        num_input_pts = 1
        input_vals = np.arange(num_centers)
        outs = []
        for input_val in input_vals:
            input_pts = np.ones((num_input_pts, num_features)) * input_val
            outs.append(rbfn.forward(input_pts=input_pts))

        # Compare to expected output
        exp = np.ones((num_input_pts, num_linear_units))*2
        for out in outs:
            self.assertTrue(np.allclose(out, exp))
    
    def test_update_centers(self)->None:
        """Test if we can manually update the center points the RBFN uses
        """
        # Set up network with arbitrary parameters
        num_centers = 5
        num_features = 20
        center_pts = np.zeros((num_centers, num_features))
        num_linear_units = 2
        epsilon = 0.99
        rbfn = RBFN(rbfunc=RBFUNC.GUASSIAN, center_pts=center_pts,
                    num_linear_units=num_linear_units, epsilon=epsilon,
                    linear_lr=None, rbf_lr=None)

        # Update center point 2
        new_center_pt = np.ones(num_features)
        rbfn.update_center(2, new_center_pt)

        # Make sure centers were updated properly
        self.assertTrue(np.allclose(rbfn.center_pts[2,:], new_center_pt))

        # Check if the update fails with an improperly sized center point
        new_center_pt = np.ones(num_features-1)
        with self.assertRaises(Exception):
            rbfn.update_center(2, new_center_pt)
        
    def test_backprop(self)->None:
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
        # Setup network and test data
        center_pts = np.array([
            [-1, 1],
            [1,1],
            [1,-1],
            [-1,-1]
        ])
        target_out = np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]
        ])
        input_pts = np.copy(center_pts)
        weights = np.zeros((target_out.shape[1], center_pts.shape[0]+1))
        epsilon = 0.99
        linear_lr = 0.05
        rbf_lr = 0
        rbfn_c = RBFN(rbfunc=RBFUNC.GUASSIAN, center_pts=center_pts, num_linear_units=weights.shape[0], epsilon=epsilon, linear_lr=linear_lr, rbf_lr=rbf_lr, weights=weights)
        # Run input data through RBFN and run backprop one step
        _ = rbfn_c.forward(input_pts)
        rbfn_c.backprop(target_out)
        print(rbfn_c.weights)
        """
        2) Test if rbf centers are updated properly
        """


if __name__ == "__main__":
    unittest.main()