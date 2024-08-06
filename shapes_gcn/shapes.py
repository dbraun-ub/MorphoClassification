from io import StringIO
from typing import Text

import numpy as np
import pandas as pd
from joblib import load
from matplotlib import pyplot as plt
from scipy.linalg import orthogonal_procrustes
from sklearn.neighbors import NearestNeighbors

"""
Copy pasted from shape-gcn repo from Vinicius. 
Not all functions are required for our program. There are still in this file for a matter of consistency in the code.
We will mostly use the confmat2D creation and the OPA method.
"""


def centroid(x: np.array) -> np.array:
    """
    The function calculates the centroid (center of mass) of a
    configuration matrix, (k,m,n)
    """

    n = x.shape[0]
    m = x.shape[1]

    center = np.zeros([m])

    for i in range(m):
        center[i] = x[:, i].sum() / n

    return center


def translation(data: np.array, center=np.array([0, 0])):
    """
    The function apply translation of a configuration matrix (k x m)
    to an specific position. Center is pre-defined as the origin for
    a configuration matrix of 2 dimensions, m=2.
    """

    trans = data - np.mean(data, 0) + center

    return trans


def confmat2D_to_df(x, orig=None):
    """
    orig is a numpy array (k,m,n) with the original configuration matrices
    to calculate centroid sizes

    x is a numpy array (k,m,n) for the configuration matrices to be represented
    in a pandas DataFrame

    OBS: For 2D configuration matrices, (x,y) coordinates
    """
    index = [str(i) for i in range(x.shape[2])]
    colx = ["x" + str(i) for i in range(1, x.shape[0] + 1)]
    coly = ["y" + str(i) for i in range(1, x.shape[0] + 1)]

    return pd.concat(
        [
            pd.DataFrame(x[:, 0, :].T, index=index, columns=colx),
            pd.DataFrame(x[:, 1, :].T, index=index, columns=coly),
        ],
        axis=1,
    )


def create_confmat_array(data: Text, view: Text, col_id="assessment_id",) -> np.array:
    """
    The function create the numpy array (k,m,n) with the configuration matrices
    registered in the coordinate table. Mathematical markers and the
    calibration markers are eliminated. The order of the markers follows
    the original numerical order.

    The function accepts coordinates tables with uncomplete sets of evaluations,
    i.e. evaluations may contain a subset of the 3 views set - FA, FP and SD.
    The configuration matrices will have the specific n dimensions for
    each view.
    """
    df = pd.read_csv(data, usecols=[col_id, "view", "marker", "x_pix", "y_pix"])

    df = df[df["view"] == view]
    """
    Remove mathematical markers
    """
    df = df[~df["marker"].str.contains("C|FA10|FA17|FP08|FP15|FP18|SD07")]

    markers = list(df[df.marker.str.contains(view)]["marker"].unique())
    dfp = df.pivot(index=col_id, columns="marker", values=["x_pix", "y_pix"])

    conf_mat = np.zeros([len(markers), 2, dfp.shape[0]])

    for i, j in enumerate(markers):
        conf_mat[i, 0, :] = dfp["x_pix"].loc[:, j].values
        conf_mat[i, 1, :] = dfp["y_pix"].loc[:, j].values

    return conf_mat, np.array(dfp.index)


class OPA(object):
    """
    Ordinary Procustes analysis : the matching of one configuration to another
    using translation, rotation and (possibly) scale. The function matches
    configuration B onto A by least squares.

    Returns:
    R: The estimated rotation matrix (may be an orthogonal matrix if
    reflection is allowed)

    s: The estimated scale matrix

    Ahat: The centred configuration A

    Bhat: The Procrustes registered configuration B

    OSS: The ordinary Procrustes sum of squares, which is $|Ahat-Bhat|^2$

    rmsd: rmsd = sqrt(OSS/(km))

    riemd: Riemannian distance in shape space (for scale=True)

    ssriemd: Riemannian distance in size-and-shape space (for scale=False)
    """

    def __init__(self, a, b, scale=True, trans=True):

        if trans:
            # Translation the configuration matrices to the origin
            center = np.zeros([a.shape[0], a.shape[1]])
            self.Ahat = translation(a, center)
            btrans = translation(b, center)
        else:
            self.Ahat = a
            btrans = b

        if scale:
            # Scaling the centred configuration matrices
            """
            centroid sizes
            """
            norm1 = np.linalg.norm(self.Ahat)
            norm2 = np.linalg.norm(btrans)

            # Zc
            atrans_scale = self.Ahat / norm1
            btrans_scale = btrans / norm2

            # transform b to minimize disparity
            self.R, scale = rotation(atrans_scale, btrans_scale)
            self.s = scale * norm1 / norm2
            btrans_scale_rot = np.dot(btrans_scale, self.R) * scale
            self.Bhat = btrans_scale_rot * norm1

            """
            Not included in the original procOPA() function in R
            Full Procrustes distance and the
            Riemannian distance in shape space
            """
            proc_dist = np.sqrt(np.sum(np.square(atrans_scale - btrans_scale_rot)))
            self.riemd = np.arcsin(proc_dist)

        else:
            self.R, scale = rotation(self.Ahat, btrans)
            self.s = 1
            btrans_rot = np.dot(btrans, self.R)
            self.Bhat = btrans_rot

            """
            Not included in the original procOPA() function in R
            The Riemannian distance in size-and-shape space
            equals Procrustes distance or intrinsic size-and-shape distance
            (pag. 101 Dryden & Mardia)
            """
            proc_dist = np.sqrt(np.sum(np.square(self.Ahat - btrans_rot)))
            self.ssriemd = proc_dist

        # The ordinary Procrustes sum of squares
        self.OSS = np.sum(np.square(self.Ahat - self.Bhat))

        # Root mean square deviation measures
        self.rmsd = np.sqrt(self.OSS / a.shape[0])


def outlier_detection_gmm(ans_pre, view, config, outliers):
    """
    The GMM-based outlier detection function
    """

    """
    Load Gaussian Mixture Model
    """
    gmm = load(config.get("gmm").get(view))

    """
    MARKERS' ORDER CHANGES WITH THE GMM MODEL (model-dependent)
    Define marker_order using predict() method on the mean shape is
    more secure.
    """
    marker_order = gmm.predict(ans_pre.Ahat)
    # marker_order = config.get("marker_order").get(view)

    log_prob = []

    for i, j in enumerate(marker_order):
        log_prob.append(gmm._estimate_log_prob(ans_pre.Bhat)[i][j])
    print("Log GMM probabilities:")
    print(log_prob)
    print("")

    # ordered log_prob as pandas Series
    log_prob = pd.Series(log_prob).sort_values()

    """
    Exclude previously detected outliers
    """
    for i in outliers:
        log_prob = log_prob.drop(index=i)

    for i in log_prob.index:
        tol = config.get("dic_tols").get(view).get(i)[1]

        """
        Adjusted tolerance (hip markers)
        TO DO: These index seem to be model-dependent (GMM)
        Search for a definitive solution
        """
        if view == "FA":
            if (6 in outliers) & (i == 7) | (7 in outliers) & (i == 6):
                tol = tol * 0.88

        if view == "FP":
            if (4 in outliers) & (i == 5) | (5 in outliers) & (i == 4):
                tol = tol * 0.88

        print(tol, config.get("dic_tols").get(view).get(i)[0])

        if log_prob[i] < tol:
            return i
            break

    return 999


def NearestNeighborsShapes(ans, k_outlier, view, config):
    """
    Load the view training set
    """
    X_train = np.load(config.get("proc_shapes").get(view))

    """
    Create the training set
    """
    k = ans.Bhat.shape[0]
    y = np.arange(k * 2)
    mask = (y != k_outlier) & (y != k_outlier + k)
    X_train_ = X_train[:, mask]
    print("Training set data:")
    print("")

    """
    Instantiate and fit the model
    """
    neigh = NearestNeighbors(algorithm="brute", n_neighbors=10)
    neigh.fit(X_train_)

    """
    Nearest Neighbors Search
    """
    conf_mat_post = ans.Bhat.copy()
    X_test_df = confmat2D_to_df(
        conf_mat_post.reshape((conf_mat_post.shape[0], conf_mat_post.shape[1], 1))
    )
    X_test = X_test_df.to_numpy()
    X_test_ = X_test[:, mask]
    print("Test set data:")
    print("")

    dist, idx = neigh.kneighbors(X_test_)
    print("Finds the K-neighbors of a point (inference): distances and indices:")
    print(dist, idx)
    print("")

    x_inf = X_train[idx, k_outlier].mean().round(2)
    y_inf = X_train[idx, k_outlier + k].mean().round(2)

    conf_mat_post[k_outlier, 0] = x_inf
    conf_mat_post[k_outlier, 1] = y_inf

    return conf_mat_post


def inference_regulation(conf_mat, ans_pre, mshape, k_outlier, view, config, stab):
    """
    Phase 1 - Match the SSD shape to the Mean shape and replace the
    outlier marker by its mean shape version close to its correct position.
    """

    print("Matching the SSD shape to the Mean Shape iteratively:")

    if not stab:
        conf_mat_pre = ans_pre.Bhat.copy()  # aligned shape
        conf_mat_pre[k_outlier, :] = mshape[k_outlier, :]
    else:
        conf_mat_pre = conf_mat.copy()

    tol = 1
    rmsd = ans_pre.rmsd
    print("Riemannian distance 0: {}".format(rmsd.round(3)))
    kk = 1
    while True:
        # Apply OPA
        ans_post = OPA(mshape, conf_mat_pre)
        print("Riemannian distance {}: {}".format(kk, ans_post.rmsd.round(3)))
        current_rmsd = ans_post.rmsd

        if abs(rmsd - current_rmsd) < tol:
            break
        kk += 1
        rmsd = current_rmsd
        conf_mat_pre = ans_post.Bhat.copy()
        conf_mat_pre[k_outlier, :] = mshape[k_outlier, :]
    print("")

    """
    Phase 2 - Inference by Nearest Neighbors Search
    """
    # transform (k, m) -> (1, mk)
    # TWO OPTIONS HERE SLIGHTLY DIFFERENT:
    # 1) ans_post.Bhat (last configuration fitted)
    # 2) conf_mat_pre (last configuration that pass through the while loop)

    conf_mat_post = NearestNeighborsShapes(ans_post, k_outlier, view, config)

    """
    Phase 3 - Match the inference to the SSD shape
    """
    print("Align the inference to the SSD shape:")
    ans_pre = OPA(conf_mat, conf_mat_post)
    center_conf_mat = centroid(conf_mat)
    # print('centroid 0: {}'.format(center_conf_mat))
    conf_mat_post_trans = translation(ans_pre.Bhat, center_conf_mat)
    conf_mat_pre = conf_mat.copy()
    conf_mat_pre[k_outlier, :] = conf_mat_post_trans[k_outlier, :]

    tol = 1
    rmsd = ans_pre.rmsd
    print("Riemannian distance 0: {}".format(rmsd.round(3)))
    kk = 1
    # cc = 1
    while True:
        # Apply OPA
        ans_post = OPA(conf_mat_pre, conf_mat_post)
        print("Riemannian distance {}: {}".format(kk, ans_post.rmsd.round(3)))
        current_rmsd = ans_post.rmsd

        if abs(rmsd - current_rmsd) < tol:
            print("stop")
            break

        rmsd = current_rmsd

        center_conf_mat_pre = centroid(conf_mat_pre)
        # print('centroid iter {}: {}'.format(cc,center_conf_mat_pre))
        kk += 1
        # cc += 1
        conf_mat_post_trans = translation(ans_post.Bhat, center_conf_mat_pre)
        conf_mat_pre[k_outlier, :] = conf_mat_post_trans[k_outlier, :]
    print("")

    conf_mat = conf_mat_pre.copy()

    return conf_mat


def shape_correction(string_coords, view, config, currentAxis=None):
    """
    Load the SSD detections into a numpy array (k, m)
    """
    df = pd.read_csv(StringIO(string_coords), header=None, usecols=[8, 9, 10, 11])
    df = df.rename(columns={8: "marker", 9: "x_pix", 10: "y_pix", 11: "score"})
    df = df[~df["marker"].str.contains("C")]
    df = df.sort_values(by="marker").reset_index(drop=True)
    df_ = df.drop(["marker", "score"], axis=1)
    conf_mat = df_.to_numpy()

    """
    Load the view mean shape
    """
    mshape = np.load(config.get("mshapes").get(view))

    outliers = []
    while True:
        """
        Apply Procrustes matching (full OPA) of the SSD shape onto the mean shape
        """
        ans_pre = OPA(mshape, conf_mat)

        """
        Outlier detection
        """
        k_outlier = outlier_detection_gmm(ans_pre, view, config, outliers)

        """
        If k_outlier = 999 break the while loop
        and process the stabilization
        """

        if k_outlier == 999:
            print("NO OUTLIER")

            """
            If at least 2 outliers are detected, process the stabilization
            """
            if len(outliers) >= 2:
                print("Stabilizing corrections...")
                print(outliers)
                for i in outliers:
                    print("Stabilizing {}...".format(i))
                    conf_mat = inference_regulation(
                        conf_mat, ans_pre, mshape, i, view, config, True,
                    )
            break

        outliers.append(k_outlier)
        print("File and the assigned outlier number:")
        print("Outlier: {}".format(k_outlier))
        print("")

        conf_mat = inference_regulation(
            conf_mat, ans_pre, mshape, k_outlier, view, config, False,
        )

    """
    Save results into the string_coords
    """
    if len(outliers) >= 1:
        for k_outlier in outliers:
            string_to_be_replaced = (
                df["marker"][k_outlier]
                + ","
                + "{:.2f}".format(df["x_pix"][k_outlier])
                + ","
                + "{:.2f}".format(df["y_pix"][k_outlier])
                + ","
                + "{:.2f}".format(df["score"][k_outlier])
            )

            new_string = (
                df["marker"][k_outlier]
                + ","
                + "{:.2f}".format(conf_mat[k_outlier, 0])
                + ","
                + "{:.2f}".format(conf_mat[k_outlier, 1])
                + ","
                + "-2.00"
            )

            print(string_to_be_replaced + " --> " + new_string)

            """
            Plot regulations if currentAxis, i.e. plot == True
            """
            if currentAxis:
                currentAxis.add_patch(
                    plt.Circle(
                        (conf_mat[k_outlier, 0], conf_mat[k_outlier, 1]),
                        radius=abs(
                            400 + (conf_mat[conf_mat.shape[0] - 1, 1] - conf_mat[0, 1])
                        )
                        * 0.005,
                        # facecolor=None,
                        edgecolor="yellow",
                        fill=False,
                        alpha=1,
                    )
                )
            string_coords = string_coords.replace(string_to_be_replaced, new_string)

    return string_coords


def OPA_anchors(conf1, anchor1, conf2, anchor2):
    """
    Aligns two configurations using Ordinary Procrustes Analysis (OPA) based on anchor points.

    Parameters:
    - conf1: The reference configuration matrix.
    - anchor1: Indices of anchor points for conf1.
    - conf2: The configuration matrix to be aligned to conf1.
    - anchor2: Indices of anchor points for conf2.

    Returns:
    - conf1_aligned: Aligned version of conf1.
    - conf2_aligned: conf2 aligned to conf1.
    """

    # Center anchor configurations at the origin.
    atrans = translate(conf1[anchor1, :])
    btrans = translate(conf2[anchor2, :])

    # Calculate the norms to scale configurations so their size is equal to 1.
    norm1 = np.linalg.norm(atrans)
    norm2 = np.linalg.norm(btrans)
    atrans_scale = atrans / norm1
    btrans_scale = btrans / norm2

    # Determine the rotation matrix using OPA.
    R, scale = rotation(atrans_scale, btrans_scale)

    # Translate the entire configurations based on anchor points.
    conf1_aligned = translate(conf1, conf1[anchor1, :])
    btrans_ = translate(conf2, conf2[anchor2, :])

    # Apply the determined rotation and scale to align conf2 with conf1.
    btrans_scale_ = btrans_ / norm2
    btrans_scale_rot_ = np.dot(btrans_scale_, R) * scale
    conf2_aligned = btrans_scale_rot_ * norm1

    return conf1_aligned, conf2_aligned


def translate(data, reference=None, center=np.array([0, 0])):
    """
    Translates a configuration matrix to a specific position based on a reference.

    Parameters:
    - data: Configuration matrix to be translated.
    - reference: Reference configuration matrix for determining the translation
    (if None, data is used).
    - center: Desired position to which the configuration should be translated.

    Returns:
    - Translated configuration matrix.
    """
    if reference is None:
        reference = data
    return data - np.mean(reference, axis=0) + center


def rotation(a, b):
    """
    Calculates the rotation matrix to align two configurations.

    Parameters:
    - a: Reference configuration matrix.
    - b: Configuration matrix to be aligned.

    Returns:
    - R: Rotation matrix.
    - scale: Scaling factor.
    """
    R, scale = orthogonal_procrustes(a, b)
    R = R.T

    return R, scale


def rotate(vector, theta, rotation_around=None) -> np.ndarray:
    """
    reference: https://en.wikipedia.org/wiki/Rotation_matrix#In_two_dimensions
    :param vector: list of length 2 OR
                   list of list where inner list has size 2 OR
                   1D numpy array of length 2 OR
                   2D numpy array of size (number of points, 2)
    :param theta: rotation angle in degree (+ve value of anti-clockwise rotation)
    :param rotation_around: "vector" will be rotated around this point,
                    otherwise [0, 0] will be considered as rotation axis
    :return: rotated "vector" about "theta" degree around rotation
             axis "rotation_around" numpy array
    """
    vector = np.array(vector)

    if vector.ndim == 1:
        vector = vector[np.newaxis, :]

    if rotation_around is not None:
        vector = vector - rotation_around

    vector = vector.T

    theta = np.radians(theta)

    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )

    output: np.ndarray = (rotation_matrix @ vector).T

    if rotation_around is not None:
        output = output + rotation_around

    return output.squeeze()
