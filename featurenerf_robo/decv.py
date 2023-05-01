def estimate_2D_projective_transformation_linear(x1, x2, normalize=True):
    # Inputs:
    #    x1 - inhomogeneous inlier correspondences in image 1
    #    x2 - inhomogeneous inlier correspondences in image 1
    #    normalize - if True, apply data normalization to x1 and x2
    #
    # Outputs:
    #    H - the DLT estimate of the planar projective transformation   
    #    cost - Sampson cost for the above DLT Estimate H. Assume points in image 1 as scene points.
    
    """your code here"""
    if normalize:
        x1, T1 = normalize_points(x1)
        x2, T2 = normalize_points(x2)
    else:
        T1 = np.eye(3)
        T2 = np.eye(3)  