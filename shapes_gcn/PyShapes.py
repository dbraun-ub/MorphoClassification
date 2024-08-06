from typing import Dict, Text, TypeVar

import numpy as np
import pandas as pd
import rdata
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import data, importr
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

"""
Here we import the R shapes library as a python object to be
manipulated using methods from rpy2
"""
print('shapes = importr("shapes")')
shapes = importr("shapes")
print('stats = importr("stats")')
stats = importr("stats")
print('numpy2ri.activate()')
numpy2ri.activate()
print("Le problème ne vient pas d'ici.")


def R_dataset(dataset: Text) -> Dict:
    """
    This function reads a specific R dataset from shapes library,
    that is stored locally and return it as a python dictionary
    """

    path = (
        "/home/vbordalo/R/x86_64-pc-linux-gnu-library/3.4/shapes/data/"
        + dataset
        + ".rda"
    )

    parsed = rdata.parser.parse_file(rdata.TESTDATA_PATH / path)
    converted = rdata.conversion.convert(parsed)

    return converted


class ShapesData(object):
    """
    Read shapes datasets as numpy arrays
    """

    def __init__(self):
        self.datasets = data(shapes).names()

    def read_data(self, dataset):
        return np.array(data(shapes).fetch(dataset)[dataset])


class OPA(object):
    """
    Ordinary Procrustes analysis
    Description
    Ordinary Procustes analysis : the matching of one configuration
    to another using translation, rotation and (possibly) scale.
    Reflections can also be included if desired. The function matches
    configuration B onto A by least squares.
    (see R documentation, help(procOPA))
    """

    def __init__(self, a, b, scale=True, reflect=False):

        output = shapes.procOPA(a, b, scale, reflect)

        self.R = np.array(output.rx2("R"))
        self.s = np.array(output.rx2("s"))[0]
        self.Ahat = np.array(output.rx2("Ahat"))
        self.Bhat = np.array(output.rx2("Bhat"))
        self.OSS = np.array(output.rx2("OSS"))[0]
        self.rmsd = np.array(output.rx2("rmsd"))[0]


class GPA(object):
    """
    Generalised Procrustes analysis
    Description
    Generalised Procrustes analysis to register landmark configurations
    into optimal registration using translation, rotation and scaling.
    Reflection invariance can also be chosen, and registration without
    scaling is also an option. Also, obtains principal components,
    and some summary statistics.
    (see R documentation, help(procGPA))
    """

    def __init__(
        self,
        x,
        scale=True,
        reflect=False,
        eigen2d=False,
        tol1=1e-05,
        tol2=1e-05,
        tangentcoords="residual",
        verbose=False,
        distances=True,
        pcaoutput=True,
        alpha=0,
        affine=False,
    ):

        output = shapes.procGPA(
            x,
            scale,
            reflect,
            eigen2d,
            tol1,
            tol2,
            tangentcoords,
            verbose,
            distances,
            pcaoutput,
            alpha,
            affine,
        )

        self.k = np.array(output.rx2("k"))[0]
        self.m = np.array(output.rx2("m"))[0]
        self.n = np.array(output.rx2("n"))[0]
        self.mshape = np.array(output.rx2("mshape"))
        self.tan = np.array(output.rx2("tan"))
        self.rotated = np.array(output.rx2("rotated"))
        self.pcar = np.array(output.rx2("pcar"))
        self.pcasd = np.array(output.rx2("pcasd"))
        self.percent = np.array(output.rx2("percent"))
        self.size = np.array(output.rx2("size"))
        self.stdscores = np.array(output.rx2("stdscores"))
        self.rawscores = np.array(output.rx2("rawscores"))
        self.rho = np.array(output.rx2("rho"))
        self.rmsrho = np.array(output.rx2("rmsrho"))[0]
        self.rmsd1 = np.array(output.rx2("rmsd1"))[0]
        self.GSS = np.array(output.rx2("GSS"))[0]


class Frechet(object):
    """
    Mean shape estimators
    Description
    Calculation of different types of Frechet mean shapes,
    or the isotropic offset Gaussian MLE mean shape
    (see R documentation, help(frechet))
    """

    def __init__(self, x, mean="intrinsic"):
        """
        frechet function in R only accepts floats when
        the mean is set as h value. page 115,
        Eq. 6.12 Dryden & Mardia (2016). This will force the argument
        to be a float python object even if the the user enters an int.
        """
        if type(mean) is int:
            mean = float(mean)
        output = shapes.frechet(x, mean)

        self.mshape = np.array(output.rx2("mshape"))
        if mean != "mle":
            self.var = np.array(output.rx2("var"))[0]
        if mean == "mle":
            self.kappa = np.array(output.rx2("kappa"))[0]
            self.loglike = np.array(output.rx2("loglike"))[0]
        self.code = np.array(output.rx2("code"))[0]
        self.gradient = np.array(output.rx2("gradient"))


def riemd(x, y, reflect=False):
    """
    Riemannian shape distance
    Description
    Calculates the Riemannian shape distance rho between two configurations
    (see R documentation, help(riemdist))
    """

    return np.array(shapes.riemdist(x, y, reflect))[0]


def ssriemd(x, y, reflect=False):
    """
    Riemannian size-and-shape distance
    Description
    Calculates the Riemannian size-and-shape distance d_S
    between two configurations
    (see R documentation, help(ssriemdist))
    """

    return np.array(shapes.ssriemdist(x, y, reflect))[0]


def booksteinshpv(x):
    """
    Bookstein shape variables
    """

    return np.array(shapes.bookstein_shpv(x))


class Bookstein2D(object):
    """
    Bookstein's baseline registration for 2D data
    Description
    Carries out Bookstein's baseline registration and calculates a mean shape
    (see R documentation, help(bookstein2d))
    """

    def __init__(self, A, l1=1, l2=2):

        output = shapes.bookstein2d(A, l1, l2)

        self.k = np.array(output.rx2("k"))[0]
        self.n = np.array(output.rx2("n"))[0]
        self.mshape = np.array(output.rx2("mshape"))
        self.bshpv = np.array(output.rx2("bshpv"))


def cent_size(x):
    """
    Centroid size
    Description
    Calculate cetroid size from a configuration or a sample of configurations.
    (see R documentation, help(centroid.size))
    """

    return np.array(shapes.centroid_size(x))


def def_h(n):
    """
    Return the Helmert submatrix where n = k - 1 and k is the number of
    landmarks (page 50, Dryden & Mardia 2016)
    """
    return np.array(shapes.defh(n))


def cmd_scale(d, k=2):
    """
    Classical (Metric) Multidimensional Scaling
    Description
    Classical multidimensional scaling (MDS) of a data matrix. Also known as
    principal coordinates analysis (Gower, 1966).
    (see R documentation, help(cmdscale))
    """
    return np.array(stats.cmdscale(d, k))


def translation(data: np.array, center=np.array([0, 0])) -> np.array:
    """
    The function apply translation of a configuration matrix (k x m)
    to an specific position. Center is pre-defined as the origin for
    a configuration matrix of 2 dimensions, m=2.
    """

    trans = data - np.mean(data, 0) + center

    return trans


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


def confmat2D_to_df(x, add_cent_size=False, orig=None):
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

    if add_cent_size:
        return pd.concat(
            [
                pd.DataFrame(cent_size(orig), index=index, columns=["s"]),
                pd.DataFrame(x[:, 0, :].T, index=index, columns=colx),
                pd.DataFrame(x[:, 1, :].T, index=index, columns=coly),
            ],
            axis=1,
        )
    else:
        return pd.concat(
            [
                pd.DataFrame(x[:, 0, :].T, index=index, columns=colx),
                pd.DataFrame(x[:, 1, :].T, index=index, columns=coly),
            ],
            axis=1,
        )


class ShapesCva2D(object):
    """
    The function adapted to Python from shapes.cva in Shapes.R receives n
    configuration matrices, X (k x m), as numpy array (k x m x n) and the
    vector, groups (n,), containing categorical variables associated to each X.

    The function returns LDA predictions (x) for the canonical variables.
    TO DO: implement for m > 2
    (see shapes.cva in Shapes.R)
    """

    def __init__(self, X: np.array, groups: np.array, scale=True):
        le = preprocessing.LabelEncoder()
        y = le.fit(groups).transform(groups)

        g = np.unique(groups).shape[0]
        print("g = {}".format(g))
        ans = GPA(X, scale=scale)

        if scale:
            pp = int((ans.k - 1) * ans.m - (ans.m * (ans.m - 1) / 2) - 1)
        else:
            pp = int((ans.k - 1) * ans.m - (ans.m * (ans.m - 1) / 2))

        print("pp = {}".format(pp))

        pracdim = int(min(pp, ans.n - g))

        print("pracdim = {}".format(pracdim))
        lda = LinearDiscriminantAnalysis()
        self.x = lda.fit(ans.rawscores[:, :pracdim], y).transform(
            ans.rawscores[:, :pracdim]
        )


def create_confmat_array(
    data: str, view: str, col_id="assessment_id", group_array=False, pat_ref=False,
) -> np.array:
    """
    The function create the numpy array (k,m,n) with the configuration matrices
    registered in the coordinate table. Mathematical markers and the
    calibration markers are eliminated. The order of the markers follows
    the original numerical order.

    The function accepts coordinates tables with uncomplete sets of evaluations,
    i.e. evaluations may contain a subset of the 3 views set - FA, FP and SD.
    The configuration matrices will have the specific n dimensions for
    each view.

    groups are returned based on fields of the coordinate table.

    group array returns demographic data with the col_id as a primary key.
    """

    cols = ["patient_ref"] * bool(pat_ref) + [
        col_id,
        "view",
        "marker",
        "x_pix",
        "y_pix",
        "gender",
        "age",
        "height_cm",
        "weight_kg",
    ]

    df = pd.read_csv(data, usecols=cols,)
    df = df[df["view"] == view]
    """
    Remove mathematical markers
    """
    df = df[~df["marker"].str.contains("C|FA10|FA17|FP08|FP15|FP18|SD07")]

    markers = list(df[df.marker.str.contains(view)]["marker"].unique())
    # To force the correct ordering of markers in the array
    markers.sort()
    dfp = df.pivot(index=col_id, columns="marker", values=["x_pix", "y_pix"])

    conf_mat = np.zeros([len(markers), 2, dfp.shape[0]])
    # Inserts values into the array
    for i, j in enumerate(markers):
        conf_mat[i, 0, :] = dfp["x_pix"].loc[:, j].values
        conf_mat[i, 1, :] = dfp["y_pix"].loc[:, j].values

    if group_array:
        group = df.drop_duplicates(subset=col_id).reset_index(drop=True)
        group_ = group.loc[
            :,
            ["patient_ref"] * bool(pat_ref)
            + [col_id, "gender", "age", "height_cm", "weight_kg"],
        ].to_numpy()

        if group[col_id].equals(pd.Series(dfp.index)):
            print(
                "Good!! The order of configurations in conf_mat corresponds to \
                that in the group table."
            )
        else:
            print(
                "WARNIG!! The order of configurations in conf_mat does not\
                 match that in group table!!"
            )

        return group_, conf_mat
    else:
        return np.array(dfp.index), conf_mat


def calc_t(pcasd: np.array, p=0.95) -> TypeVar:
    """
    Calculates the t largest eigenvalues such that the sum of these eigenvalues
    captures a certain proportion (p) of the total shape variance.
    In Statistical Shape Models (SSM), t represents the number of shape modes
    to be included in the model. The input vector is the square roots of
    eigenvalues of Sv using tan (s.d.'s of PCs) that is provided by GPA() class

    See Lindner 2017, CHAPTER 1, Automated Image Interpretation Using
    Statistical Shape Models, page 7.
    """

    cum_var = 0
    t = 0
    i = 0
    total_sh_var = (pcasd ** 2).sum()

    while True:

        cum_var = cum_var + (pcasd[i] ** 2) / total_sh_var

        i += 1

        if cum_var > p:
            break

        t += 1

    return t


def load_string_csv_into_conf_mat(data, view):

    # Load the infered data
    df = pd.read_csv(data, usecols=["marker", "x_pix", "y_pix", "score"])
    df = df.sort_values(by="marker").reset_index(drop=True)

    # Load the scores
    df_scores = df.iloc[:, [0, 3]]

    df = df.drop("score", 1)

    df["assessment_id"] = pd.Series([0 for i in range(df.shape[0])])
    df = df[~df["marker"].str.contains("C|FA10|FA17|FP08|FP15|FP18|SD07")]
    markers = list(df[df.marker.str.contains(view)]["marker"].unique())

    # Pivot forces the marker order along the columns
    dfp = df.pivot(
        index=["assessment_id"], columns=["marker"], values=["x_pix", "y_pix"]
    )

    conf_mat = np.zeros([len(markers), 2])
    for i, j in enumerate(markers):
        conf_mat[i, 0] = dfp["x_pix"].loc[:, j].values
        conf_mat[i, 1] = dfp["y_pix"].loc[:, j].values

    return conf_mat, df_scores, markers
