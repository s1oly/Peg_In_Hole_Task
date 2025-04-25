import numpy as np
import cv2
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from scipy.stats import zscore
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


error_calc_S = []
error_calc_P = []
error_calc_H = []

error_direction_S = []
error_direction_P = []
error_direction_H = []

def pt2pt_dis(pt1, pt2):
    return np.sqrt(np.sum(np.square(pt1 - pt2)))


def pt2pt_dis_sq(pt1, pt2):
    return np.sum(np.square(pt1 - pt2))


# assuming Y is sorted
# k -- going left for k indices, going right for k indices. a total of 2k neighbors.
def get_nearest_indices (k, Y, idx):
    if idx - k < 0:
        # use more neighbors from the other side?
        indices_arr = np.append(np.arange(0, idx, 1), np.arange(idx+1, idx+k+1+np.abs(idx-k)))
        # indices_arr = np.append(np.arange(0, idx, 1), np.arange(idx+1, idx+k+1))
        return indices_arr
    elif idx + k >= len(Y):
        last_index = len(Y) - 1
        # use more neighbots from the other side?
        indices_arr = np.append(np.arange(idx-k-(idx+k-last_index), idx, 1), np.arange(idx+1, last_index+1, 1))
        # indices_arr = np.append(np.arange(idx-k, idx, 1), np.arange(idx+1, last_index+1, 1))
        return indices_arr
    else:
        indices_arr = np.append(np.arange(idx-k, idx, 1), np.arange(idx+1, idx+k+1, 1))
        return indices_arr


def calc_LLE_weights (k, X):
    W = np.zeros((len(X), len(X)))
    for i in range (0, len(X)):
        indices = get_nearest_indices(int(k/2), X, i)
        xi, Xi = X[i], X[indices, :]
        component = np.full((len(Xi), len(xi)), xi).T - Xi.T
        Gi = np.matmul(component.T, component)
        # Gi might be singular when k is large
        try:
            Gi_inv = np.linalg.inv(Gi)
        except:
            epsilon = 0.00001
            Gi_inv = np.linalg.inv(Gi + epsilon*np.identity(len(Gi)))
        wi = np.matmul(Gi_inv, np.ones((len(Xi), 1))) / np.matmul(np.matmul(np.ones(len(Xi),), Gi_inv), np.ones((len(Xi), 1)))
        W[i, indices] = np.squeeze(wi.T)

    return W


def cpd_lle (X, Y_0, beta, alpha, gamma, mu, max_iter=50, tol=0.00001, kernel=3, include_lle=True, use_prev_sigma2=False, sigma2_0=None):

    # define params
    M = len(Y_0)
    N = len(X)
    D = len(X[0])

    # initialization
    # faster G calculation
    diff = Y_0[:, None, :] - Y_0[None, :,  :]
    diff = np.square(diff)
    diff = np.sum(diff, 2)

    if kernel == 3:
        # Gaussian Kernel
        G = np.exp(-diff / (2 * beta**2))
    elif kernel == 2:
        # 2nd order
        # 3/(16 * pow(beta, 3)) * (-sqrt(6)*diff_yy_sqrt/beta).array().exp() * (2*sqrt(6)*diff_yy.array() + 6*beta*diff_yy_sqrt.array() + sqrt(6) * pow(beta, 2))
        G = 3/(16 * beta**3) * np.exp(-np.sqrt(6) * np.sqrt(diff) / beta) * (2*np.sqrt(6)*diff + 6*beta*np.sqrt(diff) + np.sqrt(6)*beta**2)
    elif kernel == 1:
        # 1st order
        # 1/(2 * pow(beta, 2)) * (-2 *diff_yy_sqrt/beta).array().exp() * (2*diff_yy_sqrt.array() + sqrt(2)*beta)
        G = 1/(2 * beta**2) * np.exp(-2 * np.sqrt(diff)) * (2*np.sqrt(diff) + np.sqrt(2)*beta)
    else:
        # default to gaussian
        G = np.exp(-diff / (2 * beta**2))
    
    Y = Y_0.copy()

    # initialize sigma2
    if not use_prev_sigma2:
        (N, D) = X.shape
        (M, _) = Y.shape
        diff = X[None, :, :] - Y[:, None, :]
        err = diff ** 2
        sigma2 = np.sum(err) / (D * M * N)
    else:
        sigma2 = sigma2_0

    # get the LLE matrix
    L = calc_LLE_weights(6, Y_0)
    H = np.matmul((np.identity(M) - L).T, np.identity(M) - L)
    
    # loop until convergence or max_iter reached
    for it in range (0, max_iter):

        # ----- E step: compute posteriori probability matrix P -----
        # faster P computation
        pts_dis_sq = np.sum((X[None, :, :] - Y[:, None, :]) ** 2, axis=2)
        c = (2 * np.pi * sigma2) ** (D / 2)
        c = c * mu / (1 - mu)
        c = c * M / N
        P = np.exp(-pts_dis_sq / (2 * sigma2))
        den = np.sum(P, axis=0)
        den = np.tile(den, (M, 1))
        den[den == 0] = np.finfo(float).eps
        den += c
        P = np.divide(P, den)

        Pt1 = np.sum(P, axis=0)
        P1 = np.sum(P, axis=1)
        Np = np.sum(P1)
        PX = np.matmul(P, X)
    
        # ----- M step: solve for new weights and variance -----
        if include_lle:
            A_matrix = np.matmul(np.diag(P1), G) + alpha * sigma2 * np.identity(M) + sigma2 * gamma * np.matmul(H, G)
            B_matrix = PX - np.matmul(np.diag(P1) + sigma2*gamma*H, Y_0)
        else:
            A_matrix = np.matmul(np.diag(P1), G) + alpha * sigma2 * np.identity(M)
            B_matrix = PX - np.matmul(np.diag(P1), Y_0)

        # solve for W
        W = np.linalg.solve(A_matrix, B_matrix)

        T = Y_0 + np.matmul(G, W)
        trXtdPt1X = np.trace(np.matmul(np.matmul(X.T, np.diag(Pt1)), X))
        trPXtT = np.trace(np.matmul(PX.T, T))
        trTtdP1T = np.trace(np.matmul(np.matmul(T.T, np.diag(P1)), T))

        # solve for sigma^2
        sigma2 = (trXtdPt1X - 2*trPXtT + trTtdP1T) / (Np * D)

        # update Y
        if pt2pt_dis_sq(Y, Y_0 + np.matmul(G, W)) < tol:
            # if converged, break loop
            Y = Y_0 + np.matmul(G, W)
            # print("iteration until convergence:", it)
            break
        else:
            # keep going until max iteration is reached
            Y = Y_0 + np.matmul(G, W)

            if it == max_iter - 1:
                print("Registration did not converge!")
    
    return Y, sigma2


def get_init_marker_location (tactile_mask_init):

    # blob detector definition
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 5
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    detector = cv2.SimpleBlobDetector_create(params)
    
    keypoints = detector.detect(255 - tactile_mask_init)
    pts = []  # (x, y) in image coordinate
    for keypoint in keypoints:
        pts.append(keypoint.pt)

    pts = process_marker_location(pts, np.array(pts), True)
    pts = np.array(pts)
    # after processing:
    # marker_location_init = marker_locations[0, :, 0]
    # marker_locations = marker_locations[:, :, 1]
    pts = pts[:, 1]

    canvas = cv2.cvtColor(tactile_mask_init, cv2.COLOR_GRAY2BGR)
    for pt in pts:
        cv2.circle(canvas, (int(pt[0]), int(pt[1])), radius=2, color=(0, 0, 255), thickness=-1)

    #     cv2.imshow('mask with keypoints', canvas)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # exit()

    if len(keypoints) != 63:
        print('Invalid init mask!!')
        exit()

    return pts


# get the image coordinate of every pixel in an image
def indices_arr_pixel_coord (m, n):
    rows = np.arange(m)
    cols = np.arange(n)
    ret = np.empty((m, n, 2),dtype=int)
    ret[:, :, 1] = rows[:,None]
    ret[:, :, 0] = cols
    return ret


def get_tactile_mask (img, lower, upper):

    # Gaussian blur
    img_blurred = cv2.GaussianBlur(img, (105, 105), 15)

    # compute diff
    diff = img.astype(np.float64) / 255. - img_blurred.astype(np.float64) / 255.
    diff = 255 * (diff + 0.5)
    diff = diff.astype(np.uint8)

    # color thresholding
    mask = cv2.inRange(diff, lower, upper)
    
    # corners should be zeros
    mask[0:30, :] = 0
    mask[-30:, :] = 0
    mask[:, 0:30] = 0
    mask[:, -30:] = 0

    # smooth image
    im = Image.fromarray(mask)
    smoothed_mask = im.filter(ImageFilter.ModeFilter(size=5))
    mask = np.array(smoothed_mask)

    return mask


def get_sorting_indices(vectors):
    """
    Returns the sorting indices based on the start coordinates of the vectors,
    sorted from top to bottom and left to right.
    This is needed because the detected markers are not necessarily in the correct order,
    but constructing the matirx requires them to be sorted.
    
    Parameters:
        vectors (list of tuples): List of tuples where each tuple contains a start coordinate and an end coordinate.
    
    Returns:
        sorted_indices (np.ndarray): Indices that can be used to sort the vectors.
    """
    start_coords = np.array([v[0] for v in vectors])
    y_sorted = np.sort(start_coords[:, 1])
    y_max_9_avg = np.mean(y_sorted[-9:])
    y_min_9_avg = np.mean(y_sorted[:9])
    row_height = (y_max_9_avg - y_min_9_avg) / 6.0
    row_indices = np.rint((start_coords[:, 1] - y_min_9_avg) / row_height).astype(int)
    sortable_array = np.array([(row, start_coords[i, 0], i) for i, row in enumerate(row_indices)])
    sorted_indices = sortable_array[np.lexsort((sortable_array[:, 1], sortable_array[:, 0]))][:, 2].astype(int)
    return sorted_indices


def process_marker_location (cur_marker_locations, marker_location_init, sort=False):

    pts_init = marker_location_init.copy()
    movement_vectors_per_frame = []

    for i, pt in enumerate(cur_marker_locations):
        init_start_coord = np.array([int(pts_init[i, 0]), int(pts_init[i, 1])])
        init_end_coord = np.array([int(pt[0]), int(pt[1])])
        new_end_coord = (init_end_coord - init_start_coord) * 4.0 + init_end_coord

        movement_vectors_per_frame.append((init_start_coord, new_end_coord))

    if sort:
        sorted_indices = get_sorting_indices(movement_vectors_per_frame)
        sorted_vectors_per_frame = [movement_vectors_per_frame[i] for i in sorted_indices]
    else:
        sorted_vectors_per_frame = movement_vectors_per_frame.copy()

    return sorted_vectors_per_frame


# utils for plotting
def plot_vector_field_subplot(X, Y, U, V, title, subplot_position):
    plt.subplot(1, 3, subplot_position)
    plt.quiver(X, Y, U, V, scale=0.2, scale_units='xy') # need to change back to 1
    plt.axis('scaled')
    plt.xlim(25, 300)
    plt.ylim(20, 225)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
# utils for plotting
def plot_scalar_field(X, Y, scalar_field, title, subplot_position):
    plt.subplot(1, 3, subplot_position)
    plt.imshow(scalar_field, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap='jet')
    plt.colorbar(label='Scalar field magnitude')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')


def helmholtz_hodge_decomposition(U, V):
    """
    Perform Helmholtz-Hodge Decomposition on a 2D vector field (U, V).
    Returns:
    P - Potential (curl-free) components (P_U, P_V)
    S - Solenoidal (divergence-free) components (S_U, S_V)
    H - Harmonic components (H_U, H_V)
    """

    # Fourier transform of the vector field components
    F_U = fft2(U)
    F_V = fft2(V)
    
    # Grid of wave numbers
    nx, ny = U.shape
    kx = np.fft.fftfreq(nx).reshape(-1, 1)
    ky = np.fft.fftfreq(ny).reshape(1, -1)
    
    # Avoid division by zero at the origin
    kx[0, 0] = ky[0, 0] = 1
    
    # Compute the divergence
    div_F = kx * F_U + ky * F_V
    
    # Compute the curl-free component (potential)
    P_U = (kx * div_F) / (kx**2 + ky**2)
    P_V = (ky * div_F) / (kx**2 + ky**2)
    
    # Compute the curl (z-component of curl in 2D)
    curl_F = ky * F_U - kx * F_V
    
    # Compute the divergence-free component (solenoidal)
    S_U = (ky * curl_F) / (kx**2 + ky**2)
    S_V = -(kx * curl_F) / (kx**2 + ky**2)
    
    # Inverse Fourier transform to get back to spatial domain
    P_U = np.real(ifft2(P_U))
    P_V = np.real(ifft2(P_V))
    S_U = np.real(ifft2(S_U))
    S_V = np.real(ifft2(S_V))

    # Set mean of curl-free and divergence-free components to zero in spatial domain
    P_U -= np.mean(P_U)
    P_V -= np.mean(P_V)
    S_U -= np.mean(S_U)
    S_V -= np.mean(S_V)
    
    # Compute the harmonic component as the residual in spatial domain
    H_U = U - (P_U + S_U)
    H_V = V - (P_V + S_V)
    
    return (P_U, P_V), (S_U, S_V), (H_U, H_V)


def compute_tactile_dipole_moment(vector_field):
    """
    Compute the tactile dipole moment for a given 2D discrete vector field.

    Parameters:
    vector_field (numpy.ndarray): A 3D numpy array of shape (m, n, 2) representing the 2D vector field.

    Returns:
    numpy.ndarray: The tactile dipole moment as a 2-element array.
    """
    m, n, _ = vector_field.shape
    N = m * n  # Total number of vectors
    grad_x = np.gradient(vector_field[:, :, 0], axis=1)
    grad_y = np.gradient(vector_field[:, :, 1], axis=0)
    r = np.array([(i, j) for i in range(m) for j in range(n)])
    p_tilt = np.zeros(2)

    for i in range(m):
        for j in range(n):
            div_v = grad_x[i, j] + grad_y[i, j]  # Divergence of v
            r_i = r[i * n + j]
            p_tilt += r_i * div_v

    p_tilt /= N
    return p_tilt

def normalize_error_S(U):
    magnitude = []
    for i in range(7):
        for j in range(9):
            a = np.sqrt(np.power(U[0][i][j], 2) + np.power(U[1][i][j],2))
            magnitude.append(a)
    total = np.mean(magnitude)
    error_calc_S.append(total)
    return error_calc_S

def normalize_error_P(U):
    magnitude = []
    for i in range(7):
        for j in range(9):
            a = np.sqrt(np.power(U[0][i][j], 2) + np.power(U[1][i][j],2))
            magnitude.append(a)
    total = np.mean(magnitude)
    error_calc_P.append(total)
    return error_calc_P

def normalize_error_H(U):
    magnitude = []
    for i in range(7):
        for j in range(9):
            a = np.sqrt(np.power(U[0][i][j], 2) + np.power(U[1][i][j],2))
            magnitude.append(a)
    total = np.mean(magnitude)
    error_calc_H.append(total)
    return error_calc_H

def calc_error_direction_curl(U):
    magnitude_1 = []
    magnitude_2 = []
    for i in range(7):
        magnitude_1.append(np.sqrt(np.power(U[0][i][0], 2) + np.power(U[1][i][0], 2)))
        magnitude_2.append(np.sqrt(np.power(U[0][i][-1], 2) + np.power(U[1][i][-1], 2)))
    if(np.sum(magnitude_1) > np.sum(magnitude_2)):
        direction = []
        for i in range(7):
            angle_result = U[1][i][0]/magnitude_1[i]
            if(np.abs(-1 - angle_result)) < 0.2:
                direction.append(1)
            elif(np.abs(1 - angle_result)) < 0.2:
                direction.append(-1)
        if np.mean(direction) > 0.2:
            total = 1
            error_direction_S.append(total)
        elif np.mean(direction) < -0.2:
            total = -1
            error_direction_S.append(total)
    elif(np.sum(magnitude_2) > np.sum(magnitude_1)):
        direction = []
        for i in range(7):
            angle_result = U[1][i][-1]/magnitude_2[i]
            if(np.abs(-1 - angle_result)) < 0.2:
                direction.append(-1)
            elif(np.abs(1 - angle_result)) < 0.2:
                direction.append(1)
        if np.mean(direction) > 0.2:
            total = 1
            error_direction_S.append(total)
        elif np.mean(direction) < -0.2:
            total = -1
            error_direction_S.append(total)

    return error_direction_S[-1]

def calc_error_direction_div(U):
    direction = []
    for i in range(7):
        for j in range(9):
            magnitude = np.sqrt(np.power(U[0][i][j], 2) + np.power(U[1][i][j], 2))
            angle_result = U[0][i][j]/magnitude
            if magnitude > 0.5:
                if np.abs(-1 - angle_result) < 0.9:
                    direction.append(-1)
                elif np.abs(1 - angle_result) < 0.9:
                    direction.append(1)
    if np.mean(direction) > 0.2:
        total = 1
        error_direction_P.append(total)
    elif np.mean(direction) < -0.2:
        total = -1
        error_direction_P.append(total)
    else:
        error_direction_P.append(0)
    return error_direction_P[-1]

#Need to add filters in here as well
def calc_error_direction_har(U):
    direction = []
    for i in range(7):
        for j in range(9):
            magnitude = np.sqrt(np.power(U[0][i][j], 2) + np.power(U[1][i][j], 2))
            if magnitude > 0.5:
                theta = np.arccos(1*U[0][i][j]/magnitude)
                if(U[1][i][j] < 0):
                    theta = 2*np.pi - theta
                direction.append(theta)
    error_direction_H.append(np.mean(direction))
    return error_direction_H[-1]
    
    
   # for potential component, if the magnitude of the error is present, then the harmonmic will point all to the left mainly,
   # if certain type of contact is happening


