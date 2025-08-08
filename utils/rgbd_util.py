# --*-- coding:utf-8 --*--
import numpy as np
from utils.util import *
np.seterr(divide='ignore', invalid='ignore')

'''
z: depth image in 'centimetres'
missingMask: a mask
C: camera matrix
'''
def processDepthImage(z, missingMask, C):
    yDirParam_angleThresh = np.array([45, 15]) # threshold to estimate the direction of the gravity
    yDirParam_iter = np.array([5, 5])
    yDirParam_y0 = np.array([0, 1, 0])

    normalParam_patchSize = np.array([3, 10])

    X, Y, Z = getPointCloudFromZ(z, C, 1)

    # with open('pd.txt', 'w', encoding='utf-8') as f:
    #     for i in range(X.shape[0]):
    #         for j in range(X.shape[1]):
    #             f.write('{} {} {}\n'.format(str(X[i,j]), str(Y[i,j]), str(Z[i,j])))

    # restore x-y-z position
    pc = np.zeros([z.shape[0], z.shape[1], 3])
    pc[:,:,0] = X
    pc[:,:,1] = Y
    pc[:,:,2] = Z

    N1, b1 = computeNormalsSquareSupport(z/100, missingMask, normalParam_patchSize[0],
    1, C, np.ones(z.shape))
    N2, b2 = computeNormalsSquareSupport(z/100, missingMask, normalParam_patchSize[1],
    1, C, np.ones(z.shape))

    N = N1

    # Compute the direction of gravity
    yDir = getYDir(N2, yDirParam_angleThresh, yDirParam_iter, yDirParam_y0)
    y0 = np.array([[0, 1, 0]]).T
    R = getRMatrix(y0, yDir)

    # rotate the pc and N
    NRot = rotatePC(N, R.T)
    pcRot = rotatePC(pc, R.T)

    # --- RANSAC-based floor plane estimation (all units are in centimetres) ---
    def _fit_plane_ransac(points, max_iters=800, thresh_cm=2.0, min_inlier_ratio=0.1, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        n_pts = points.shape[0]
        if n_pts < 3:
            return None
        best_inliers = None
        best_model = None
        for _ in range(max_iters):
            idx = rng.choice(n_pts, size=3, replace=False)
            p1, p2, p3 = points[idx]
            v1 = p2 - p1
            v2 = p3 - p1
            n = np.cross(v1, v2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-6:
                continue
            n = n / norm_n
            d = -np.dot(n, p1)
            # distances
            dist = np.abs(points @ n + d)
            inliers = dist <= thresh_cm
            if best_inliers is None or inliers.sum() > best_inliers.sum():
                best_inliers = inliers
                best_model = (n, d)
        if best_inliers is None or best_inliers.sum() < max(int(min_inlier_ratio * n_pts), 50):
            return None
        # Refit using inliers (least squares)
        P = points[best_inliers]
        # plane through SVD: minimize ||P*n + d|| with ||n||=1
        # Augment to solve for [n_x, n_y, n_z, d]
        A = np.hstack([P, np.ones((P.shape[0], 1))])
        # Solve in least squares sense using SVD of A[:, :3]
        # Constrain ||n||=1 by normalizing after solution
        # Use normal from PCA of points (smallest singular vector of covariance)
        P_center = P.mean(axis=0)
        U, S, Vt = np.linalg.svd(P - P_center, full_matrices=False)
        n_ls = Vt[-1, :]
        n_ls = n_ls / (np.linalg.norm(n_ls) + 1e-12)
        d_ls = -np.dot(n_ls, P_center)
        return (n_ls, d_ls)

    # Build candidate set: valid points in lower ROI
    H, W = pcRot.shape[:2]
    roi_start = int((1.0 - 0.35) * H)  # lower 35% of image
    valid = (~missingMask.astype(bool)) & np.isfinite(pcRot[:, :, 2])
    valid[:roi_start, :] = False
    yy, xx = np.where(valid)
    if yy.size > 0:
        P = np.stack([pcRot[yy, xx, 0], pcRot[yy, xx, 1], pcRot[yy, xx, 2]], axis=1)
    else:
        P = np.empty((0, 3), dtype=np.float64)

    model = _fit_plane_ransac(P, max_iters=800, thresh_cm=2.0, min_inlier_ratio=0.1)

    if model is not None:
        n_plane, d_plane = model
        # Orient normal upwards relative to +Y
        if np.dot(n_plane, np.array([0.0, 1.0, 0.0])) < 0:
            n_plane = -n_plane
            d_plane = -d_plane
        # Height = signed distance to plane (>=0)
        h = pcRot[:, :, 0] * n_plane[0] + pcRot[:, :, 1] * n_plane[1] + pcRot[:, :, 2] * n_plane[2] + d_plane
        h = np.maximum(h, 0.0)
        # Replace gravity direction with plane normal
        yDir = n_plane.reshape(3, 1)
    else:
        # Fallback to legacy heuristic
        h = -pcRot[:, :, 1]
        yMin = np.percentile(h, 0)
        if (yMin > -90):
            yMin = -130
        h = h - yMin

    return pc, N, yDir, h,  pcRot, NRot

'''
getPointCloudFromZ: use depth image and camera matrix to get pointcloud
Z is in 'centimetres'
C: camera matrix
s: is the factor by which Z has been upsampled
'''
def getPointCloudFromZ(Z, C, s=1):
    h, w= Z.shape
    xx, yy = np.meshgrid(np.array(range(w))+1, np.array(range(h))+1)
    # color camera parameters
    cc_rgb = C[0:2,2] * s       # the first two lines of colomn-3, x0 and the y0
    fc_rgb = np.diag(C[0:2,0:2]) * s    # number on the diagonal line
    x3 = np.multiply((xx - cc_rgb[0]), Z) / fc_rgb[0]
    y3 = np.multiply((yy - cc_rgb[1]), Z) / fc_rgb[1]
    z3 = Z
    return x3, y3, z3

'''
  Clip out a 2R+1 x 2R+1 window at each point and estimate 
  the normal from points within this window. In case the window 
  straddles more than a single superpixel, only take points in the 
  same superpixel as the centre pixel. 
  
Input:
    depthImage: in meters
    missingMask:  boolean mask of what data was missing
    R: radius of clipping
    sc: to upsample or not
    superpixels:  superpixel map to define bounadaries that should
                    not be straddled
'''
def computeNormalsSquareSupport(depthImage, missingMask, R, sc, cameraMatrix, superpixels):
    depthImage = depthImage*100     # convert to centi metres
    X, Y, Z = getPointCloudFromZ(depthImage, cameraMatrix, sc)
    Xf = X
    Yf = Y
    Zf = Z
    pc = np.zeros([depthImage.shape[0], depthImage.shape[1], 3])
    pc[:,:,0] = Xf
    pc[:,:,1] = Yf
    pc[:,:,2] = Zf
    XYZf = np.copy(pc)

    # find missing value
    ind = np.where(missingMask == 1)
    X[ind] = np.nan
    Y[ind] = np.nan
    Z[ind] = np.nan

    one_Z = np.expand_dims(1 / Z, axis=2)
    X_Z = np.divide(X, Z)
    Y_Z = np.divide(Y, Z)
    one = np.copy(Z)
    one[np.invert(np.isnan(one[:, :]))] = 1
    ZZ = np.multiply(Z, Z)
    X_ZZ = np.expand_dims(np.divide(X, ZZ), axis=2)
    Y_ZZ = np.expand_dims(np.divide(Y, ZZ), axis=2)

    X_Z_2 = np.expand_dims(np.multiply(X_Z, X_Z), axis=2)
    XY_Z = np.expand_dims(np.multiply(X_Z, Y_Z), axis=2)
    Y_Z_2 = np.expand_dims(np.multiply(Y_Z, Y_Z), axis=2)

    AtARaw = np.concatenate((X_Z_2, XY_Z, np.expand_dims(X_Z, axis=2), Y_Z_2,
                             np.expand_dims(Y_Z, axis=2), np.expand_dims(one, axis=2)), axis=2)

    AtbRaw = np.concatenate((X_ZZ, Y_ZZ, one_Z), axis=2)

    # with clipping
    AtA = filterItChopOff(np.concatenate((AtARaw, AtbRaw), axis=2), R, superpixels)
    Atb = AtA[:, :, AtARaw.shape[2]:]
    AtA = AtA[:, :, :AtARaw.shape[2]]

    AtA_1, detAtA = invertIt(AtA)
    N = mutiplyIt(AtA_1, Atb)

    divide_fac = np.sqrt(np.sum(np.multiply(N, N), axis=2))
    # with np.errstate(divide='ignore'):
    b = np.divide(-detAtA, divide_fac)
    for i in range(3):
        N[:, :, i] = np.divide(N[:, :, i], divide_fac)

    # Reorient the normals to point out from the scene.
    # with np.errstate(invalid='ignore'):
    SN = np.sign(N[:, :, 2])
    SN[SN == 0] = 1
    extend_SN = np.expand_dims(SN, axis=2)
    extend_SN = np.concatenate((extend_SN, extend_SN, extend_SN), axis=2)
    N = np.multiply(N, extend_SN)
    b = np.multiply(b, SN)
    sn = np.sign(np.sum(np.multiply(N, XYZf), axis=2))
    sn[np.isnan(sn)] = 1
    sn[sn == 0] = 1
    extend_sn = np.expand_dims(sn, axis=2)
    N = np.multiply(extend_sn, N)
    b = np.multiply(b, sn)
    return N, b