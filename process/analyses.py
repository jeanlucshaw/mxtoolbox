"""
Complex analyses transforming data into other data.
"""
import numpy as np
import pandas as pd
import xarray as xr
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.multivariate.pca import PCA
from statsmodels.stats.outliers_influence import OLSInfluence
from scipy.optimize import leastsq
from scipy.signal import find_peaks
from .math_ import *
from .signal_ import *
from .convert import dt2epoch, binc2edge


__all__ = ['pca',
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
           'xr_cross_correlate',
           'xr_time_aht']


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


def sm_pca(u, v):
    """
    Compute principal directions of variation
    """
    # Form input into dataframe
    data = pd.DataFrame({'u': u, 'v': v})

    # Clean data
    data = data.query('~u.isnull() & ~v.isnull()', engine='python')

    # Perform PCA
    pca_model = PCA(data, demean=True, standardize=False)

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
    yfit = model.params[0] + model.params[1] * np.array(xfit)
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


def xr_2D_to_1D_interp(dataset, x, y, xcoord, ycoord, z):
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
        Data structure with the added `aht` variable (h).
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

    # Subtract time of the previous high tide
    for (start, end) in zip(locs[:-1], locs[1:]):
        aht[start: end] = aht[start: end] - aht[start]

    # Manage data before first high tide
    aht[:locs[0]] = period - time[locs[0]] + time[:locs[0]]
    aht[aht < 0] += period

    # Manage data after last high tide
    aht[locs[-1]:] = aht[locs[-1]:] - aht[locs[-1]]

    # Add to dataset and return
    dataset['aht'] = (['time'], aht)

    return dataset, locs
