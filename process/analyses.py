"""
Complex analyses transforming data into other data.
"""
import numpy as np
import pandas as pd
import xarray as xr
import statsmodels.api as sm
import matplotlib.pyplot as plt
import gsw
from skimage.measure import marching_cubes
from statsmodels.multivariate.pca import PCA
from statsmodels.stats.outliers_influence import OLSInfluence
from scipy.interpolate import griddata, interp1d, LinearNDInterpolator
from scipy.optimize import leastsq
from scipy.signal import find_peaks
from scipy.spatial import cKDTree as KDTree
from .math_ import *
from .signal_ import *
from .convert import dt2epoch, binc2edge


__all__ = ['idw_scatter_interp',
           'pca',
           'pd_pca',
           'sm_pca',
           'principal_modes',
           'gaussian_smoothing',
           'lsq_curve_fit',
           'pd_regression_statistics',
           'sm_lin_fit',
           'sm_lin_fit_diagnostics',
           'successive_interp_1D',
           'transect_bins',
           'xr_2D_to_1D_interp',
           'xr_2D_to_2D_interp',
           'xr_array_to_transect',
           'xr_cross_correlate',
           'xr_definite_integral',
           'xr_irregular_grid_subset',
           'xr_marching_cubes',
           'xr_nsquared_1D_depth',
           'xr_stratification_parameter',
           'xr_time_aht']


def idw_scatter_interp(x,
                       y,
                       v,
                       x_vec,
                       y_vec,
                       neighboors=8,
                       power=1,
                       x_name='x',
                       y_name='y',
                       v_name='z',
                       lonlat=False,
                       n_par=None,
                       radius=None):
    """
    Inverse distance weighted interpolation of 2D scattered data

    Parameters
    ----------
    x, y : 1D array
        the horizontal and vertical coordinates of the scatter
    v : 1D array
        the values of the field to interpolate at coordinates `x`,`y`
    x_vec, y_vec : 1D array
        the horizontal and vertical axes of the regular interpolation grid
    neighboors : int
        the number of neighboors the calculate the IDW values from
    power : float
        weights = 1 / distance ** `power`
    x_name, y_name, v_name : str
        names of the coordinates and field in the output dataset
    lonlat : bool
        scatter and grid coordinates are supplied in lon,lat coordinates. In this
        case, the distances are calculated in a 3D cartesian space, converted from
        the spherical earth coordinates.
    n_par : int or None
        Large scatters (>>1_000_000 pts) could result in calculating a very big
        matrix. Set n_par to indicate how many grid points for which to calculate the
        IDW values at once.
    radius : int, float or None
        require all neighboord to be within this radius, otherwise return NaN

    Returns
    -------
    v_interp : xarray.Dataset
        containing the gridded observations and their coordinates
    """

    
    def lonlat2threeD_(lon, lat):
        """ Convert spherical lon, lat -> cartesian x, y, z """
        lon_rad, lat_rad = np.radians(lon), np.radians(lat)
        x_3D = np.cos(lat_rad) * np.cos(lon_rad) * 6_371_000
        y_3D = np.cos(lat_rad) * np.sin(lon_rad) * 6_371_000
        z_3D = np.sin(lat_rad) * 6_371_000
        return x_3D, y_3D, z_3D


    # Build interpolation grid
    x_grid, y_grid = np.meshgrid(x_vec, y_vec)
    v_grid_flat = x_grid.ravel() * np.nan

    # Convert to 3D cartesian coordinates if given in latitude, longitude horizontal coordinates
    if lonlat == True:
        x_3D, y_3D, z_3D = lonlat2threeD_(x, y)
        tree_coords = np.array([x_3D, y_3D, z_3D]).T
        x_grid_3D, y_grid_3D, z_grid_3D = lonlat2threeD_(x_grid, y_grid)
        query_coords = np.array([x_grid_3D.ravel(), y_grid_3D.ravel(), z_grid_3D.ravel()]).T
    else:
        tree_coords = np.array([x, y]).T
        query_coords = np.array([x_grid.ravel(), y_grid.ravel()]).T

    # Build KDTree
    tree = KDTree(tree_coords)

    # Nearest neighboor lookup and distance calculation
    distance, idx_nearest = tree.query(query_coords, k=neighboors)

    if (n_par is not None) and (v_grid_flat.size > n_par):
        """ If the scatter contains >> 1_000_000 """
        idx = np.arange(0, v_grid_flat.size, n_par)

        # Loop over sets of points to calculate in parallel
        for i_start, i_stop in zip(idx[:-1], idx[1:]):
            # Weights and norm of weights
            w_ = (1 / distance[i_start:i_stop, :]) ** power
            w_norm_ = np.sum(w_, axis=-1)

            # Calcuate IDW value(s)
            v_grid_flat[i_start:i_stop] = np.sum(w_ * v[idx_nearest[i_start:i_stop]], axis=-1) / w_norm_

        # Weights and norm of weights
        w_ = (1 / distance[i_stop:, :]) ** power
        w_norm_ = np.sum(w_, axis=-1)

        # Calculate the last slice
        v_grid_flat[i_stop:] = np.sum(w_ * v[idx_nearest[i_stop:]], axis=-1) / w_norm_
    else:
        """ If the scatter is in the tested size range """
        weights = (1 / distance) ** power
        v_grid_flat = np.sum(weights * v[idx_nearest], axis=-1) / np.sum(weights, axis=-1)

    # Remove clear points excluded by the radius
    if isinstance(radius, (int, float)):
        v_grid_flat[(distance > radius).all(axis=-1)] = np.nan

    # Reshape to gridded form
    v_grid = v_grid_flat.reshape(x_grid.shape)

    # Format as xarray
    v_interp = xr.Dataset({v_name: ([y_name, x_name], v_grid)}, coords={y_name: y_vec, x_name: x_vec})

    # Calculate the data and grid spatial resolutions
    distance_nearest, _ = tree.query(query_coords, k=1)
    v_interp.attrs['scatter_resolution_mean'] = np.mean(distance_nearest)
    v_interp.attrs['scatter_resolution_std'] = np.std(distance_nearest)
    v_interp.attrs['scatter_resolution_min'] = np.min(distance_nearest)
    v_interp.attrs['scatter_resolution_max'] = np.max(distance_nearest)
    v_interp.attrs['scatter_resolution_percentile_10'] = np.percentile(distance_nearest, 10)
    v_interp.attrs['scatter_resolution_percentile_25'] = np.percentile(distance_nearest, 25)
    v_interp.attrs['scatter_resolution_percentile_50'] = np.percentile(distance_nearest, 50)
    v_interp.attrs['scatter_resolution_percentile_75'] = np.percentile(distance_nearest, 75)
    v_interp.attrs['scatter_resolution_percentile_90'] = np.percentile(distance_nearest, 90)

    return v_interp


def pca(input_):
    """
    Perform principal component analysis on numpy array.

    Parameters
    ----------
    input_ : 2D array
        Columns (n) are variables, rows (m) are observations.

    Returns
    -------
    U : 2D array (n by n)
        Columns are the eigenvectors.
    lbda : 1D array
        Eigenvalues.
    F : 2D array
        Matrix of component scores.
    UL_sr : 2D array
        Projection of the descriptors to PC space.

    Note
    ----

       Implemented following,

       Legendre and Legendre (1998), Numerical Ecology, Elsevier, 852 pp.
    """
    Y_h = input_ - input_.mean(axis=0)

    # Calculate the dispersion matrix
    S = (1 / (1 - Y_h.shape[0])) * Y_h.T @ Y_h

    # Find eigenvalues and eigenvectors
    vals, vecs = np.linalg.eig(S)
    I = np.argsort(abs(vals))[::-1]
    U = vecs[:, I]
    lbda = vals[I]

    # Normalize eigenvectors to unit length
    U /= np.sqrt((U ** 2).sum(axis=0))

    # Calculate the matrix of component scores
    F = Y_h @ U

    # Calculate projection of descriptors in reduced space
    UL_sr = U @ np.sqrt(np.diag(np.abs(lbda)))

    return U, lbda, F, UL_sr


# Function definition
def pd_pca(dataframe, features, target=None, plot=False, **plt_kwargs):
    """
    Add principal component vectors to dataframe.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Data to analyse.
    features : list of str
        Descriptors to use.
    target : str
        Name of target column.
    plot : bool
        Visualize results.
    plt_kwargs : dict
        Keyword arguments passed to pandas.plot.scatter.

    Returns
    -------
    pcdf : pandas.Dataframe
        Input dataframe with added PC columns.
    lbda : 1D array
        Eigenvalues of principal modes.
    UL_sr : 2D array
        Projection of features to reduced space.

    """
    # Standardize dataset
    Y = dataframe.loc[:, features].values

    # Perform PCA
    U, lbda, F, UL_sr = pca(Y)

    # Format results
    pcdf = pd.DataFrame(data=F,
                        columns=['PC%d' % (i + 1) for i in range(len(features))])
    if target:
        cols = [*features, target]
    else:
        cols = features
    pcdf = pd.concat([pcdf, dataframe.loc[:, cols]], axis=1)

    # Visualize
    if plot:
        # Scattered data in reduced space
        ax = pcdf.plot.scatter(x='PC1',
                               y='PC2',
                               c='k',
                               **plt_kwargs)

        # Feature axes projected to reduced space
        for i in range(len(features)):
            ax.plot([0, UL_sr[i, 0]], [0, UL_sr[i, 1]], 'k')
            ax.text(UL_sr[i, 0], UL_sr[i, 1], features[i])

        # Label axes with explained variance
        ax.set(xlabel='PC1 (%.2f%%)' % (100 * lbda[0] / lbda.sum()),
               ylabel='PC2 (%.2f%%)' % (100 * lbda[1] / lbda.sum()))

        # Set plot aspect ratio
        high_lim = np.max(ax.get_ylim() + ax.get_xlim())
        low_lim = np.min(ax.get_ylim() + ax.get_xlim())
        ax.set(xlim=(low_lim, high_lim),
               ylim=(low_lim, high_lim))
        ax.figure.set_figheight(5)
        ax.figure.set_figwidth(5)
        plt.show()

    return pcdf, lbda, UL_sr


def sm_pca(u, v, *args, **kwargs):
    """
    Compute principal directions of variation

    Parameters
    ----------
    u, v : 1D arrays
        observations in the 1st and 2nd dimension
    *args, **kwargs :
        passed to `statsmodels.multivariate.pca.PCA`. Defaults are
        `demean=True` and `standardize=False`.

    Returns
    -------
    u1, v1 : float
        1st and 2nd component of the most powerful eigenvector.
    u2, v2 : float
        1st and 2nd component of the second most powerful eigenvector.
    l_1, l_2 : float
        eigenvalues of the first 2 eigenvectors
    theta : float
        orientation of the first eigenvector

    """
    # Form input into dataframe
    data = pd.DataFrame({'u': u, 'v': v})

    # Clean data
    data = data.query('~u.isnull() & ~v.isnull()', engine='python')

    # Perform PCA
    kwargs = {'demean': True, 'standardize': False, **kwargs}
    pca_model = PCA(data, *args, **kwargs)

    # Component vectors
    u_1, v_1 = pca_model.eigenvecs.iloc[:, 0]
    u_2, v_2 = pca_model.eigenvecs.iloc[:, 1]
    l_1, l_2 = pca_model.eigenvals

    # Compute angle of eigenvector 1
    theta = 180 * np.arctan2(v_1, u_1) / np.pi

    return u_1, v_1, u_2, v_2, l_1, l_2, theta


def principal_modes(args):
    """
    Perform eigenvalue decomposition using numpy.

    Given a dataset described by `n` vectors of length
    `m`, the principal modes of variation can be extracted
    using eigenvalue decomposition. The `n` vectors are
    first detrended by subtracting the mean. A covariance
    matrix `R` is then calculated as,

    .. math::

       \\mathbf{R} = \\mathbf{F}\\cdot\\mathbf{F'}

    where :math:`F` is a matrix whose columns are formed from
    the `n` detrended input vectors. This matrix is then
    passed to `numpy.linalg.eig` whose outputs `w` and `v`
    are returned.

    In addition, the matrix :math:`\\mathbf{v}` is
    inverted and returned as :math:`\\mathbf{B}` to provide
    a basis change matrix into principal mode space. The
    orginal coordinates may then be rotated into principal
    mode space by arranging them into the `n` by `m` matrix
    :math:`O` by multiplication,

    .. math::

       \\mathbf{C} = \\mathbf{B}\\cdot\\mathbf{O}

    we get the matrix `n` by `m` matrix :math:`c` whose rows
    are the rotated vectors.

    Parameters
    ----------
    args : tuple
        Vectors of length m.

    Returns
    -------
    vals : array (n)
        Eigenvalues of the covariance matrix.
    vecs : ndarray (n by n)
        Eigenvectors (columns).
    B : ndarray (n by n)
        Basis change matrix.

    """
    F = np.array(args).T

    # Detrend
    F -= np.nanmean(F, axis=0)

    # Covariance matrix
    R = F.T @ F

    # Eigenvalues/vectors
    vals, vecs = np.linalg.eig(R)

    # Basis change matrix
    B = np.linalg.inv(vecs)

    return vals, vecs, B


def gaussian_smoothing(xs, ys, zs, xg, yg, XR=1., YR=1.):
    """
    Interpolates scattered data to grid via Gaussian smoothing.

    Simplified version of the Barnes smoothing algorithm. This
    routine computes a value for each grid point by averaging
    all scattered data are weighted by their distance to the grid
    point. The weighing function for scatter point `i` is,

    .. math::

       w_i = \\text{exp}\\left[-\\left(\\frac{xg-xs_i}{XR}\\right)^2 +
                        \\left(\\frac{yg-ys_i}{YR}\\right)^2\\right]

    Parameters
    ----------
    xs, ys : 1D array
        Coordinates of the scattered data.
    zs : 1D array
        Values of the scattered data.
    xg, yg : 2D array
        Grid coordinates.
    XR, YR : float
        Horizontal and vertical search radius.

    Returns
    -------
    zg : 2D array
        Scattered data interpolated to grid.

    """
    zg = np.zeros_like(xg)

    # Interpolate on grid
    if xg.size < xs.size:
        # If xg.size << than xs.size this is faster
        for i in range(xv.size):
            for j in range(yv.size):
                dx = (xg[j, i] - xs) / XR
                dy = (yg[j, i] - ys) / YR
                w = np.exp(-(dx ** 2 + dy ** 2))
                zg[j, i] = np.matmul(w, zs) / w.sum()
    else:
        # If xg.size >> than xs.size this is faster
        z_sum = np.zeros_like(zg)
        w_sum = np.zeros_like(zg)
        for i in range(xs.size):
            dx = (xg - xs[i]) / XR
            dy = (yg - ys[i]) / YR
            w = np.exp(-(dx ** 2 + dy ** 2))
            z_sum += w * zs[i]
            w_sum += w
        zg = z_sum / w_sum

    return zg


def lsq_curve_fit(model, x, y, *parameters):
    """
    Least squares fitting using scipy.optimize.

    Formatting input for scipy.optimize.leastsq can
    be confusing. This wrapper serves the purpose of
    not having to remember how it works.

    Let :math:`y=f(x)` and `g` be a model for `f` 
    dependent on a variable number of parameters
    :math:`\\mathbf{a}`. Then this routine requires
    requires `model` to be a function object taking
    as arguments `x`, :math:`a_1, a_2, ..., a_n` in this
    order, and `*parameters` to be ordered initial guesses
    for :math:`\\mathbf{a}`.

    The following predefined models can serve as examples
    and can be called by setting `model` to the function
    name.

    * mxtoolbox.process.math.f_gaussian
    * mxtoolbox.process.math.f_sine


    Parameters
    ----------
    model : function object or str
        Must be of form `y = model(x, a_1, a_2, ...)`
    x, y : 1D array
        Dependent and independent variables.
    parameters : tuple of floats
        Ordered initial guesses for :math:`\\mathbf{a}`

    Returns
    -------
    fit_values : 1D array
        model(`x`, a_1, a_2, ...) using optimized parameters.
    fit_parameters : 1D array
        Optimized model parameters.
    """
    # Switch between predefined models
    if model == 'gaussian':
        _model = f_gaussian
    elif model == 'sine':
        _model = f_sine
    else:
        _model = model

    # Formatted for leastsq
    def residuals(parameters, y, x):
        return y - _model(x, *parameters)

    # Perform fit
    fit_parameters, _ = leastsq(residuals, parameters, args=(y, x))

    # Calculate fit y values
    fit_values = _model(x, *fit_parameters)

    return fit_values, fit_parameters


def pd_regression_statistics(dataframe, xname, yname):
    """
    Print analysis of variance on linear regression.

    Print analysis of variance (ANOVA) table for a linear regression
    drawn for yname from xname, two columns of dataframe. The independent
    variable is considered to be xname.

    Parameters
    ----------
    dataframe : pandas.Dataframe
        The dataframe to operate on.
    xname : str
        Column name of the independent variable.
    yname : str
        Column name of the dependent variable.

    References
    ----------

    Van Emden, Helmut (2008)
    Statistics for terrified biologists,
    Blackwell publishing
    p. 262
    """
    x = dataframe.get([xname])
    y = dataframe.get([yname])
    n = x.size

    # Regression coefficient
    b = np.polyfit(x.values.flatten(), y.values.flatten(), 1)[0]

    # Deviation from regression line
    sum_of_sq_mean_x = b ** 2 * ((x - x.mean()) ** 2).sum()

    # Deviation from mean y
    sum_of_sq_mean_y = (y ** 2).sum() - (y.sum()) ** 2 / n

    # Residual variation
    sum_of_sq_mean_residual = sum_of_sq_mean_y.values - sum_of_sq_mean_x.values

    # Degrees of freedom
    df_b = 1
    df_r = n - df_b - 1

    # Mean squares
    mean_sq_b = sum_of_sq_mean_x / df_b
    mean_sq_res = sum_of_sq_mean_residual / df_r

    # Variation ratio
    f_statistic = mean_sq_b / mean_sq_res

    # Readout
    print('\nAnalysis of variance table\n')
    print('Independent variable: %s' % xname)
    print('Dependent   variable: %s\n' % yname)
    print("%15s\t%10s\t%10s\t%10s\t%10s" % ('',
                                            'd.f.',
                                            'sum. sq.',
                                            'mean sq.',
                                            'var. ratio'))
    print("%s\t%10d\t%10.2f\t%10.2f\t%10.3f" % ('Regression (b):'.ljust(15),
                                                df_b,
                                                sum_of_sq_mean_x,
                                                sum_of_sq_mean_x / df_b,
                                                f_statistic))
    print("%s\t%10d\t%10.2f\t%10.2f\t%10s" % ('Residual:'.ljust(15),
                                              df_r,
                                              sum_of_sq_mean_residual,
                                              mean_sq_res,
                                              'xx'))
    print("%s\t%10d\t%10.2f\t%10s\t%10s" % ('Total:'.ljust(15),
                                            n - 1,
                                            sum_of_sq_mean_y,
                                            'xx',
                                            'xx'))


def sm_lin_fit(dataframe, x, y, xfit=None):
    """
    Fit x,y data linearly with statsmodels.

    For more info on the model results type

       model.summary()

    Other fit parameters can be accessed via the model output
    variable, for example:

       model.pvalues
       model.params

    Parameters
    ----------
    dataframe: pandas.DataFrame
        Where to look for x, y variables.
    x, y: str
        Names of independent and dependent variables.

    Returns
    -------
    model
        Result of statsmodels.api.OLS.fit().
    fit: pandas.Series or 1D array
        Least squares fit of dependent variable.
    rsquared: float
        Variance in y explained by x.
    """
    # Clean
    dataframe = dataframe.query('~%s.isnull() & ~%s.isnull()' % (x, y), engine='python')

    # Convert time to julian day if time series
    if isinstance(dataframe[x].values[0], np.datetime64):
        x_data = [t_.to_julian_date() for t_ in dataframe[x].to_list()]
    elif isinstance(dataframe[x].values[0], pd.Timestamp):
        x_data = [t_.to_julian_date() for t_ in dataframe[x].to_list()]
    else:
        x_data = dataframe[x]

    # Set up variables for OLS
    dependent = dataframe[y]
    independent = sm.add_constant(x_data)

    # Fit
    model = sm.OLS(dependent, independent).fit()

    # Manage fit coordinates
    if xfit is None:
        xfit = dataframe[x].values

    # Convert to julian date if time series
    if isinstance(xfit[0], np.datetime64):
        fit_labels = xfit
        xfit = [t_.to_julian_date() for t_ in pd.Series(xfit).to_list()]
    elif isinstance(xfit[0], pd.Timestamp):
        fit_labels = xfit
        xfit = [t_.to_julian_date() for t_ in pd.Series(xfit).to_list()]
    else:
        fit_labels = xfit

    # Calculate fit line
    yfit = model.params.iloc[0] + model.params.iloc[1] * np.array(xfit)
    fit = pd.DataFrame({'x': fit_labels, 'y': yfit}).sort_values(by='x')

    return model, fit, model.rsquared


def sm_lin_fit_diagnostics(model, max_leverage=0.3, cook_th=0.25, cook_labels=None):
    """
    Make diagnostic plots of linear regression.

    Parameters
    ----------
    model: statsmodels.RegressionResult
        E.g. returned by `sm_lin_fit`.

    Returns
    -------
    pyplot.Axes:
        On which diagnostics are drawn.

    """
    def cook_distance(residuals, leverage, k_vars):
        return residuals ** 2 * leverage / (1 - leverage) / k_vars

    influence = OLSInfluence(model)
    k_vars = influence.k_vars
    standardized_resid = influence.resid_studentized_internal

    # Init plot
    gs_kw = {'hspace': 0.35, 'wspace': 0.35, 'right': 0.95, 'left':0.1}
    _, ax = plt.subplots(2, 2, figsize=(8, 8), gridspec_kw=gs_kw)

    # Residuals vs. predicted panel
    ax[0, 0].plot(model.fittedvalues, model.resid, 'ko')
    ax[0, 0].set(xlabel='Fitted values', ylabel='Residuals')

    # Q-Q plot panel
    sm.qqplot(model.resid, ax=ax[0, 1], line='r')

    # Scale-location panel
    ax[1, 0].plot(model.fittedvalues, np.sqrt(standardized_resid ** 2), 'ko')
    ax[1, 0].set(xlabel='Fitted values', ylabel=r'$|$Standardized residuals$|^{\frac{1}{2}}$')

    # Leverage panel
    leverage = model.get_influence().hat_matrix_diag
    ax[1, 1].plot(leverage, standardized_resid, 'ko')
    ax[1, 1].set(xlabel='Leverage', ylabel='Standradized residuals')
    if cook_labels is not None:
        for n_, l_, r_ in zip(cook_labels, leverage, standardized_resid):
            if cook_distance(r_, l_, k_vars) > cook_th:
                ax[1, 1].text(l_, r_, n_)

    # --- Cook distance lines
    x = np.linspace(0, max_leverage, 100)
    y = np.linspace(-6, 6, 100)
    X, Y = np.meshgrid(x, y)
    C = cook_distance(Y, X, k_vars)
    ctr = ax[1, 1].contour(X, Y, C, [0.5, 1], colors='r', linestyles='--')
    plt.clabel(ctr, fmt='%.1f')

    return ax, cook_distance(standardized_resid, leverage, k_vars)


def successive_interp_1D(dataset,
                         xgrid,
                         ygrid,
                         var,
                         xvar,
                         yvar,
                         max_xgap=None):
    """
    Make 2D interpolated array from jagged 1D data.

    Given a 1D dataset varying along dimensions `x` and `y` (e.g., an
    autonomous profiling instrument) interpolate successively along
    `x` and `y` to fill the grid defined by `xgrid` and `grid` for each
    variable listed in `var`.

    Parameters
    ==========
    dataset: xarray.Dataset
        From which to read 1D data.
    xgrid, ygrid: 1D array
        Definition of the interpolation grid.
    var: list of str
        Names of the variables to interpolate.
    xvar, yvar: str
        Names of the dimensions of all variables in `var`.
    max_xgap: int, float, pandas.Timestamp or np.datetime64
        Limit `x` interpolation to points closer to each other than
        `max_xgaq`. Must be in same units as `xgrid`.

    Returns
    =======
    xarray.Dataset:
        with `var` interpolated to the input grid.

    """

    # Init Gridded array
    vgrid = np.zeros((ygrid.size, xgrid.size, len(var))) * np.nan

    # Calculate bin width
    dy = np.median(np.diff(ygrid))

    # Loop over y levels
    for i_, y_ in enumerate(ygrid):
        at_level = (dataset[yvar] < y_ + dy / 2) & (dataset[yvar] >= y_ - dy / 2)

        # Require two points for this level
        if at_level.sum() > 2:
            # Subset data at this y level
            ss_ = dataset.where(at_level, drop=True)

            # Bin average for this level along x
            for j_, v_ in enumerate(var):
                vgrid[i_, :, j_] = xr_bin(ss_[v_], xvar, xgrid)

    # Form new 2D dataset
    crds_ = {xvar : xgrid, yvar: ygrid}
    vars_ = {v_: ([yvar, xvar], vgrid[:, :, i_]) for i_, v_ in enumerate(var)}
    gridded = xr.Dataset(vars_, coords=crds_)

    # Interpolate missing values
    gridded = gridded.interpolate_na(dim=xvar, max_gap=max_xgap)
    gridded = gridded.interpolate_na(dim=yvar)

    return gridded


def transect_bins(profile_pos, depth_x, depth_y, z):
    """
    Divide a transect into bins centered on profiles.

    Parameters
    ----------
    profile_pos: float or iterable of float
        Positions of the profiles along the transect (km).
    depth_x, depth_y: 1D array
        Bathymetric curve along the transect, (km) and (m).
    z: 1D array
        Requested profile depths (m). Must be the same for
        all profiles.

    Returns
    -------
    areas: pandas.DataFrame
        Surface area of each bin (m^2).
    vertices_x: list of 5-list
        Distance along transect of each bin's vertices (km).
    vertices_y: list of 5-list
        Depth of each bin's vertices (m).

    """
    # If the transect is open, close it at the sides
    if depth_y[0] > 0:
        depth_x = np.hstack((depth_x[0], depth_x))
        depth_y = np.hstack((-1, depth_y))
    if depth_y[-1] > 0:
        depth_x = np.hstack((depth_x, depth_x[-1]))
        depth_y = np.hstack((depth_y, -1))

    # Make an x grid enclosing the whole transect
    x_inf_m = depth_x.min() - 1
    x_inf_p = depth_x.max() + 1
    x_grid = np.hstack((x_inf_m, profile_pos, x_inf_p))

    # Get vertical bin widths
    _ = np.diff(z)
    z_widths = np.hstack((_[0], _))
    bin_top = binc2edge(z)[:-1]
    bin_bot = binc2edge(z)[1:]

    # Ensure the surface bin does not extend beyond the surface
    if bin_top[0] < 0:
        z_widths[0] += bin_top[0]
        bin_top[0] = 0

    # Ensure input positions are iterable
    if isinstance(profile_pos, (int, float)):
        profile_pos = tuple(profile_pos)

    # Init list of box vertices
    vertices_x = list()
    vertices_y = list()

    # Build the empty output dataframe
    areas = pd.DataFrame({'z' : z,
                          'z_widths': z_widths,
                          'bin_top': bin_top,
                          'bin_bottom': bin_bot})
    for pp_ in profile_pos:
        areas.loc[:, pp_] = np.nan

    # Loop over depths to fill the dataframe
    for i_ in areas.index:
        # Make a z grid for this depth
        zp_ = areas.loc[i_, 'z']
        z_grid = zp_ * np.ones_like(x_grid)

        # Find intersections between this level and the bathy
        xc, yc = intersections(depth_x, depth_y, x_grid, z_grid)

        # Check that this level intersects with the bathy
        if (xc is not None) and (yc is not None):

            # Assemble the relevant positions
            grid = np.sort([*xc, *profile_pos])

            # Loop over relevant positions
            for j_, gp_ in enumerate(grid):

                # Check that position is requested and in water
                cond_1 = gp_ in profile_pos
                cond_2 = in_polygon(gp_, zp_, depth_x, depth_y)[0]

                # Calculate bin width and vertices
                if cond_1 and cond_2:

                    # Get neighboor positions
                    ln_ = grid[j_ - 1]
                    rn_ = grid[j_ + 1]

                    # Both positions are land boundaries
                    if (rn_ in xc) and (ln_ in xc):
                        lb_ = ln_
                        rb_ = rn_
                    # Right land boundary, left profile boundary
                    elif (rn_ in xc) and (ln_ in profile_pos):
                        lb_ = (gp_ - (gp_ - ln_) / 2)
                        rb_ = rn_
                    # Right profile boundary, left land boundary
                    elif (ln_ in xc) and (rn_ in profile_pos):
                        lb_ = ln_
                        rb_ = (gp_ + (rn_ - gp_) / 2)
                    # Right profile boundary, left profile boundary
                    elif (ln_ in profile_pos) and (rn_ in profile_pos):
                        lb_ = (gp_ - (gp_ - ln_) / 2)
                        rb_ = (gp_ + (rn_ - gp_) / 2)
                    else:
                        raise ValueError('Grid pt neither neighboor or profile')

                    # Set width in output
                    areas.loc[i_, gp_] = (rb_ - lb_) * areas.loc[i_, 'z_widths']
                    areas.loc[i_, gp_] *= 1000

                    # Append to lists of vertices
                    zt_ = areas.loc[i_, 'bin_top']
                    zb_ = areas.loc[i_, 'bin_bottom']
                    xv_ = [lb_, rb_, rb_, lb_, lb_]
                    yv_ = [zt_, zt_, zb_, zb_, zt_]
                    vertices_x.append(xv_)
                    vertices_y.append(yv_)

    return areas, vertices_x, vertices_y


def xr_2D_to_1D_interp(dataset, x, y, xcoord, ycoord, z, **kwargs):
    """
    Interpolate 2D field along arbitrary track.

    Parameters
    ----------
    dataset : xarray.Dataset or xarray.DataArray
        Data from which to interpolate.
    x, y : 1D array
        Coordinates of interpolation track.
    xcoord, ycoord : str
        Coordinate names of the 2D field.
    z : str or dict
        Name of the track coordinate or dictionnary with two fields:
        `name` a string to be the name of the track and `values` to
        be the coordinate of the track.
    **kwargs : dict
        passed to `xarray.Dataset.interp`

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Field interpolated to track.

    """
    if isinstance(z, str):
        da_args = dict(dims=z)
    elif isinstance(z, dict):
        da_args = dict(dims=z['name'], coords={z['name']: z['values']})
    else:
        raise TypeError('z must be str or dict')

    da_x = xr.DataArray(x, **da_args)
    da_y = xr.DataArray(y, **da_args)
    return dataset.interp({xcoord: da_x, ycoord: da_y})


def xr_2D_to_2D_interp(dataarray, xx, yy, dims):
    """
    Interpolate regular 2D data to irregular grid.

    Parameters
    ----------
    dataarray: xarray.DataArray
        Original data to interpolate from.
    xx, yy: 2D array
        New horizontal and vertical coordinate grids.
    dims: 2-iterable
        Ordered names of original dimensions.

    Returns
    -------
    2D array
        Data interpolated to new coordinates.

    """
    # Set up new coordinates as DataArrays
    x_i = xr.DataArray(xx,
                       dims=['x', 'y'],
                       coords={'x': np.arange(xx.shape[0]),
                               'y': np.arange(xx.shape[1])})
    y_i = xr.DataArray(yy,
                       dims=['x', 'y'],
                       coords={'x': np.arange(yy.shape[0]),
                               'y': np.arange(yy.shape[1])})

    # Interpolate
    return dataarray.interp(**{dims[0]: x_i, dims[1]: y_i}).values


def xr_array_to_transect(grid,
                         transect,
                         x_name='x',
                         y_name='y',
                         lon_name='lon',
                         lat_name='lat',
                         transect_dim='distance'):
    """
    Linear interpolation of irregularly gridded data to a 1D axis in 2D

    Parameters
    ----------
    grid : xarray.Dataset
        containing the irregularly gridded data
    transect : xarray.Dataset
        containing the coordinates of the track on which to interpolate
    x_name, y_name : str
        names of the regular coordinates in `grid`
    lon_name, lat_name : str
        names of the irregular coordinates in `grid` and `transect`
    transect_dim : str
        name of the dimension along which transect[lon_name;lat_name] are aligned

    Returns
    -------
    subset : xarray.Dataset
        containing the data interpolated on the `transect` path.

    """
    # Make the grids, and points arrays
    lon_array = grid[lon_name].values
    lat_array = grid[lat_name].values
    x_array, y_array = np.meshgrid(grid[x_name].values, grid[y_name].values)
    points = np.array([lon_array.ravel(), lat_array.ravel()]).T

    # Generate the scattered interpolators going from irregular -> regular coords
    x_interpolator = LinearNDInterpolator(points, x_array.ravel())
    y_interpolator = LinearNDInterpolator(points, y_array.ravel())

    # Generate regular coordinates for the transect
    transect_x = x_interpolator(transect[lon_name].values, transect[lat_name].values)
    transect_y = y_interpolator(transect[lon_name].values, transect[lat_name].values)

    # Format these as DataArray s along the transect dimension
    transect = transect.assign_coords({x_name: (transect_dim, transect_x),
                                       y_name: (transect_dim, transect_y)})

    # Interpolate model values to the transect coordinates
    return grid.interp({x_name: transect[x_name], y_name: transect[y_name]})


def xr_cross_correlate(dataarray_a, dataarray_b, coord='time'):
    """
    Compute cross correlation between DataArrays dataarray_a and dataarray_b.

    The two input arrays can contain NaN values and be of different
    lengths but they must have the same coordinate resolution
    and coordinates must match on overlap. Resample prior to
    using if necessary.

    Parameters
    ----------
    dataarray_a, dataarray_b : xarray.DataArray
        Arrays to correlate.
    coord : str
        Name of dimension along which to operate.

    Returns
    -------
    tau : float
        dataarray_a shifted forward by `tau` matches dataarray_b
    r2 : float
        Variance explained at maximum cross correlation.
    dataarray_c : xarray.DataArray
        dataarray_b shifted to maximum correlation.
    cc : 1D array
        Cross correlation vector.

    """
    a = dataarray_a.values.copy()
    b = dataarray_b.values.copy()
    t = dataarray_b[coord].values.copy()

    # Missing values to zero
    amask, bmask = np.isnan(a), np.isnan(b)
    a[amask] = 0
    b[bmask] = 0

    # Calculate cross-correlation
    cc = np.correlate(a, b, 'full')
    dn = (b.size - cc.argmax() - 1)
    tau = dn * np.diff(t).mean()

    # Pad with zeros to same size and calculate correlation after lag
    a_pad = np.hstack((np.zeros(b.size - 1), a, np.zeros(b.size - 1)))
    b_pad = np.roll(np.hstack((b, np.zeros(a.size + b.size - 2))), cc.argmax())
    r2 = np.corrcoef(a_pad, b_pad)[0, 1] ** 2

    # Mask values with missing values in either input
    a_pad_mask = np.hstack((np.zeros(b.size - 1, dtype='bool'),
                            amask,
                            np.zeros(b.size - 1, dtype='bool')))
    b_pad_mask = np.hstack((bmask,
                            np.zeros(a.size + b.size - 2, dtype='bool')))
    b_pad_mask = np.roll(b_pad_mask, cc.argmax())
    b_pad[a_pad_mask | b_pad_mask] = np.nan

    # Remove pads
    c = np.roll(b_pad, -cc.argmax())[:b.size]

    # Format shifted array to dataarray
    dataarray_c = xr.DataArray(c, dims=[coord], coords={coord: t - tau})

    return tau, r2, dataarray_c, cc


def xr_definite_integral(dataset,
                         variable,
                         low_bounds,
                         high_bounds,
                         dimension):
    """
    Integrate ND DataArray along dimension with specified integration bounds

    Parameters
    ----------
    dataset : xarray.Dataset
       containing the variable to integrate
    variable : str
       the name of the variable to integrate
    low_bounds, high_bounds : int, float or (N-1)D array
       bottom and top integration bounds. Numeric values are used for all
       integrated vectors.
    dimension : str
       coordinate along which to integrate

    Returns
    -------
    integrated : xarray.DataArray
       the integrated variable of N-1 dimensions

    """
    # The 1D case is standard xarray functionnality
    if dataset[variable].ndim == 1:
        integrated = (dataset[variable]
                      .sel({dimension: slice(low_bounds, high_bounds)})
                      .integrate(coord=dimension))

    # The ND case needs user implementation
    else:
        # Assess output dimension
        output_dims = tuple([d for d in dataset[variable].dims if d != dimension])
        output_shape = tuple([dataset[d].size for d in output_dims])

        # Init output
        output = np.zeros(output_shape) * np.nan

        # Manage input bounds provided as int
        if np.array(low_bounds).size == 1:
            low_bounds = np.tile(low_bounds, output_shape)
        if np.array(high_bounds).size == 1:
            high_bounds = np.tile(high_bounds, output_shape)

        # Loop over dimension
        for indices in np.ndindex(output_shape):
            sel_dims = {k: v for k, v in zip(output_dims, indices)}
            output[*indices] = (dataset[variable]
                                .isel(sel_dims)
                                .sel({dimension: slice(low_bounds[*indices],
                                                       high_bounds[*indices])})
                                .integrate(coord=dimension))

        # Format to xarray
        integrated = xr.DataArray(output,
                                  dims=output_dims,
                                  coords={d: dataset[d] for d in output_dims})

    return integrated


def xr_irregular_grid_subset(dataset,
                             bounds,
                             x_name='x',
                             y_name='y',
                             lon_name='lon',
                             lat_name='lat'):
    """
    Subset an irregularly gridded, n-dim xarray.Dataset by bounds on the irregular grid

    Parameters
    ----------
    dataset : xarray.Dataset
        to subset according to input parameters
    bounds : 4-tuple
        limits of the subset in irregular grid coordinates (lon_min, lon_max, lat_min, lat_max)
    x_name, y_name : str
        names of the regular grid coordinates in `dataset`
    lon_name, lat_name : str
        names of the irregular grid coordinates in `dataset`

    Returns
    -------
    subset : xarray.Dataset
        smallest rectangle of regular coordinates containing the area defined in `bounds`

    """
    # unpack the input bounds
    lon_min, lon_max, lat_min, lat_max = bounds

    # Identify the irregular grid points in bounds
    in_lon = (dataset[lon_name] <= lon_max) & (dataset[lon_name] >= lon_min)
    in_lat = (dataset[lat_name] <= lat_max) & (dataset[lat_name] >= lat_min)

    # Identify the corresponding regular grid indices
    X, Y = np.meshgrid(dataset[x_name].values, dataset[y_name].values)
    X = np.where(in_lon & in_lat, X, np.nan)
    Y = np.where(in_lon & in_lat, Y, np.nan)

    # Subset to the smallest regular grid which includes the provided bounds
    x_min, x_max = np.nanmin(X).astype(int), np.nanmax(X).astype(int)
    y_min, y_max = np.nanmin(Y).astype(int), np.nanmax(Y).astype(int)

    # Perform subsetting
    dataset = dataset.sel({x_name: slice(x_min, x_max), y_name: slice(y_min, y_max)})

    # Save subsetting indices
    dataset.attrs['subset_x_0'] = x_min
    dataset.attrs['subset_x_f'] = x_max
    dataset.attrs['subset_y_0'] = y_min
    dataset.attrs['subset_y_f'] = y_max

    return dataset


def xr_marching_cubes(dataset,
                      variable,
                      level,
                      crd_1=None,
                      crd_2=None,
                      crd_3=None):
    """
    Find x,y,z coordinates of a `value` layer inside a 3D volume

    Parameters
    ----------
    dataset : xarray.Dataset
        containing the 3D volume and coordinates
    variable : str
        name of the 3D volume variable
    level : float
        value identifying the layer of interest
    crd_1 : str
        name of the 3D volume first dimension
    crd_2 : str
        name of the 3D volume second dimension
    crd_3 : str
        name of the 3D volume third dimension, i.e., whose value to return
        for the `level` layer.

    Returns
    -------
    output : xr.Dataset
        containing the coordinates of `level` layer in the 3D volume

    """
    if (crd_1 == None) or (crd_2 == None) or (crd_3 == None):
        crd_1, crd_2, crd_3 = dataset[variable].dims

    # Get real vertices of the surface using MC algorithm
    verts, _, _, _ = marching_cubes(dataset[variable].values, level)
    verts_real = np.isfinite(verts).all(axis=1)

    # Convert back to original coordinates
    dataframe = pd.DataFrame()
    for i_, d_ in enumerate(dataset[variable].dims):
        # Make interpolant from the vertices
        y_ = dataset[d_].values
        x_ = np.arange(y_.size)

        # Add interpolated coordinates to the dataframe
        dataframe[d_] = interp1d(x_, y_).__call__(verts[verts_real, i_])

    # Construct a 2D linear interpolant
    xg, yg = np.meshgrid(dataset[crd_2].values, dataset[crd_1].values)
    layer_coord = griddata((dataframe[crd_2], dataframe[crd_1]), dataframe[crd_3], (xg, yg))

    # Construct the 2D mask for input coordinates
    crd_1_bins = binc2edge(dataset[crd_1])
    crd_2_bins = binc2edge(dataset[crd_2])
    data_count, _, _ = np.histogram2d(dataframe[crd_1],
                                      dataframe[crd_2],
                                      bins=[crd_1_bins, crd_2_bins],
                                      density=False)
    mask = data_count > 0

    # Mask values outside surface
    layer_coord = np.where(mask, layer_coord, np.nan)

    # Form as Dataset
    output = xr.Dataset({crd_3: ([crd_1, crd_2], layer_coord)},
                        coords={crd_1: dataset[crd_1], crd_2: dataset[crd_2]})

    return output


def xr_nsquared_1D_depth(dataset, s, t, z):
    """
    Calculate buoyancy frequency from ocean data in xarray

    Paramaters:
    -----------
    dataset : xarray.Dataset
        containing the ocean properties
    s, t, z : str
        names of the salinity, temperature and depth objects

    Returns:
    --------
    nsquared : xr.DataArray
        interpolated back to the dimensions and coordinates of
        the input salinity. The output values are in (s^-2)

    Notes:
    ------
    Mostly useful for 3+ dimension arrays without pressure data, i.e.,
    depth is used as pressure. The array broadcasting rules inside
    gsw.nsquared can complicate this or silently use the wrong dimension
    to calculate the vertical derivative.
    """
    # Work on copy to avoid affecting input axes and dimensions
    ds = dataset.copy()

    # Get input arrays with depth on the last axis
    salinity = ds[s].transpose(..., z)
    temperature = ds[t].transpose(..., z)
    pressure = ds[z]

    # Calculate
    nsquared, pmid = gsw.Nsquared(salinity, temperature, pressure, axis=-1)
    other_dims = [0 for i in range(pmid.ndim - 1)]

    # Insert pressure at the depth coordinate
    coords = dict()
    for c_ in salinity.dims:
        if c_ in [z]:
            coords[c_] = pmid[*other_dims, :]
        else:    
            coords[c_] = salinity[c_].values

    # Form xarray, interpolated to original depths and transposed to input dims
    da_nsquared = (xr
                   .DataArray(nsquared, dims=salinity.dims, coords=coords)
                   .interp({z: pressure})
                   .transpose(*dataset[s].dims))

    # Return in units of (1 / s ** 2)
    return  da_nsquared / 2 / np.pi


def xr_stratification_parameter(dataset,
                                density,
                                depth,
                                depth_up=False,
                                ):
    """
    Potential density anomaly (PHI): Simpson and Sharples (2012)

    Parameters
    ----------
    dataset : xarray.Dataset
       containing the density values and depth coordinate
    density : str
       name of the density field variable (should work with ND data).
       Units of kg / m^3 are assumed.
    depth : str
       name of the vertical coordinate. Units of meters are assumed.
    depth_up : bool
       set `True` if coordinate `depth` is positive upwards

    Returns
    -------
    phi : xarray.DataArray
       same shape as `density`, flattened along `depth`. The potential
       density anomaly per meter; units of J m^-3

    Notes
    -----
    Reference: Simpson, J. and Sharples, J. (2012).
               Introduction to the Physical and Biological Oceanography of Shelf Seas
               Chapter 6: Seasonal stratification and the spring bloom.
               Cambridge University Press.
               
    """

    # Ensure not modifying input
    dataset = dataset.copy()

    # Make the depth axis positive upwards
    if depth_up == False:
        dataset = dataset.assign_coords({depth: -1 * dataset[depth]}).sortby(depth)

    # Find the water depth at which the profile ends
    index = dataset[density].notnull().sum(dim=depth) - 1
    water_depth = (dataset[depth].isel({depth: -index}) ** 2) ** (1/2)

    # Calculate the average density
    rho_hat = dataset[density].mean(dim=depth)

    # Calculate the integrand for PHI
    dataset['integrand'] = (rho_hat - dataset[density]) * dataset[depth] * 9.81 / water_depth

    # Calculate the stratification parameter
    phi = xr_definite_integral(dataset, 'integrand', -water_depth, 0, depth)

    return phi


def xr_time_aht(dataset, field='h', period=12.4166):
    """
    Add time after high tide variable to dataset.

    The cleaner the input time series, the cleaner the
    ouptut values will be. Before using consider,

       * Sorting by times.
       * Removing duplicate times.
       * Replacing missing values with harmonic analysis.
       * Interpolating to regular time.

    Parameters
    ----------
    dataset : xarray.Dataset
        Data structure to operate on.
    field : str
        Name of sea level variable.
    period : float
        Number of hours in one tidal cycle.

    Returns
    -------
    xarray.Dataset
        Data structure with the added `aht` variable (h) and the
        `tide_cycle_label` variable.
    1D array
        Index values of high tides in input time array.

    """
    # Get median time step
    time = dt2epoch(dataset.time.values, div=3600)
    step = np.median(np.diff(time))
    period_steps = int(period / step)

    # Peak finding
    locs, _ = find_peaks(dataset[field].values, distance=period_steps/2)

    # Remove peaks within 1/2 period of dataset borders
    locs = locs[(locs > period_steps / 4) &
                (locs < time.size - period_steps / 4)]

    # Numerical values of time after first high tide
    aht = time - time[0]
    labels = np.zeros(aht.shape, dtype=int)

    # Subtract time of the previous high tide
    for label, (start, end) in enumerate(zip(locs[:-1], locs[1:])):
        aht[start: end] = aht[start: end] - aht[start]
        labels[start: end] = label + 1  # start at 1

    # Manage data before first high tide
    aht[:locs[0]] = period - time[locs[0]] + time[:locs[0]]
    aht[aht < 0] += period

    # Manage data after last high tide
    aht[locs[-1]:] = aht[locs[-1]:] - aht[locs[-1]]
    labels[locs[-1]:] = labels.max() + 1

    # Add to dataset and return
    dataset['aht'] = (['time'], aht)
    dataset['tide_cycle_label'] = (['time'], labels)

    return dataset, locs
