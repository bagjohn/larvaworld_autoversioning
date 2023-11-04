"""
Methods for managing spatial metrics (2D x-y arrays)
"""

import random
from shapely import geometry, ops
import numpy as np
import pandas as pd
import scipy as sp
from typing import Optional

from . import nam

__all__ = [
    'sense_food',
    'generate_seg_shapes',
    'Collision',
    'rearrange_contour',
    'comp_bearing',
    'compute_dispersal_solo',
    'compute_dispersal_multi',
    'compute_component_velocity',
    'compute_velocity_threshold',
    'get_display_dims',
    'get_window_dims',
    'get_arena_bounds',
    'screen2space_pos',
    'space2screen_pos',
    'circle_to_polygon',
    'apply_per_level',
    'moving_average',
    'boolean_indexing',
    'concat_datasets',
    'body_contour',
    'unwrap_deg',
    'unwrap_rad',
    'rate',
    'eudist',
    'eudi5x',
    'eudiNxN',
    'compute_dst',
    'comp_extrema',

]

def sense_food(pos, sources=None, grid=None, radius=None):
    if grid:
        cell = grid.get_grid_cell(pos)
        if grid.grid[cell] > 0:
            return cell
    elif sources and radius is not None:
        valid = sources.select(eudi5x(np.array(sources.pos), pos) <= radius)
        valid.select(valid.amount > 0)

        if len(valid) > 0:
            return random.choice(valid)
    return None


def generate_seg_shapes(Nsegs: int, points: np.ndarray, seg_ratio: Optional[np.ndarray] = None,
                        centered: bool = True, closed: bool = False) -> np.ndarray:
    """
    Segments a body into equal-length or given-length segments via vertical lines.

    Args:
    - Nsegs: Number of segments to divide the body into.
    - points: Array with shape (M,2) representing the contour of the body to be segmented.
    - seg_ratio: List of N floats specifying the ratio of the length of each segment to the length of the body.
                Defaults to None, in which case equal-length segments will be generated.
    - centered: If True, centers the segments around the origin. Defaults to True.
    - closed: If True, the last point of each segment is connected to the first point. Defaults to False.

    Returns:
    - ps: Numpy array with shape (Nsegs,L,2), where L is the number of vertices of each segment.
          The first segment in the list is the front-most segment.
    """

    # If segment ratio is not provided, generate equal-length segments
    if seg_ratio is None:
        seg_ratio = np.array([1 / Nsegs] * Nsegs)

    # Create a polygon from the given body contour
    p = geometry.Polygon(points)
    # Get maximum y value of contour
    y0 = np.max(p.exterior.coords.xy[1])

    # Segment body via vertical lines
    ps = [p]
    for cum_r in np.cumsum(seg_ratio):
        l = geometry.LineString([(1 - cum_r, y0), (1 - cum_r, -y0)])
        new_ps = []
        for p in ps:
            new_ps += list(ops.split(p, l).geoms)
        ps = new_ps

    # Sort segments so that front segments come first
    ps.sort(key=lambda x: x.exterior.xy[0], reverse=True)

    # Transform to 2D array of coords
    ps = [p.exterior.coords.xy for p in ps]
    ps = [np.array([[x, y] for x, y in zip(xs, ys)]) for xs, ys in ps]

    # Center segments around 0,0
    if centered:
        for i, (r, cum_r) in enumerate(zip(seg_ratio, np.cumsum(seg_ratio))):
            ps[i] -= [(1 - cum_r) + r / 2, 0]

    # Put front point at the start of segment vertices. Drop duplicate rows
    for i in range(len(ps)):
        if i == 0:
            ind = np.argmax(ps[i][:, 0])
            ps[i] = np.flip(np.roll(ps[i], -ind - 1, axis=0), axis=0)
        else:
            ps[i] = np.flip(np.roll(ps[i], 1, axis=0), axis=0)
        _, idx = np.unique(ps[i], axis=0, return_index=True)
        ps[i] = ps[i][np.sort(idx)]
        if closed:
            ps[i] = np.concatenate([ps[i], [ps[i][0]]])
    return np.array(ps)


class Collision(Exception):

    def __init__(self, object1, object2):
        self.object1 = object1
        self.object2 = object2


def rearrange_contour(ps0):
    ps_plus = [p for p in ps0 if p[1] >= 0]
    ps_plus.sort(key=lambda x: x[0], reverse=True)
    ps_minus = [p for p in ps0 if p[1] < 0]
    ps_minus.sort(key=lambda x: x[0], reverse=False)
    return ps_plus + ps_minus


def comp_bearing(xs, ys, ors, loc=(0.0, 0.0), in_deg=True):
    """
    Compute the bearing (azimuth) of a set of oriented 2D point-vectors relative to a location point.

    Parameters:
    ----------
    xs : array-like
        x-coordinates of the points.
    ys : array-like
        y-coordinates of the points.
    ors : float or array-like
        The orientations (in degrees) of the point-vectors.
    loc : tuple, optional
        The reference location's coordinates as a (x, y) tuple. Default is (0.0, 0.0).
    in_deg : bool, optional
        If True, returns bearings in degrees (default). If False, returns bearings in radians.

    Returns:
    -------
    array-like
        An array of bearing angles in degrees or radians, depending on the 'in_deg' parameter.
        Positive angles indicate clockwise rotation from the positive x-axis.

    Examples:
    --------
    xs = [1.0, 2.0, 3.0]
    ys = [1.0, 2.0, 0.0]
    ors = 90.0
    comp_bearing(xs, ys, ors)

    array([-135., -135.,  -90.])
    """

    x0, y0 = loc
    dxs = x0 - np.array(xs)
    dys = y0 - np.array(ys)
    rads = np.arctan2(dys, dxs)
    drads = (ors - np.rad2deg(rads)) % 360
    drads[drads > 180] -= 360
    return drads if in_deg else np.deg2rad(drads)


def compute_dispersal_solo(xy, min_valid_proportion=0.2, max_start_proportion=0.1, min_end_proportion=0.9):
    """
    Compute dispersal values for a given trajectory.

    This function calculates dispersal values based on a trajectory represented as a 2D array or DataFrame.
    It checks for the validity of the input trajectory and returns dispersal values accordingly.

    Parameters:
    ----------
    xy : array-like or DataFrame
        The trajectory data, where each row represents a point in 2D space.
    min_valid_proportion : float, optional
        The minimum proportion of valid data points required in the trajectory.
        Defaults to 0.2, meaning at least 20% of non-missing data points are required.
    max_start_proportion : float, optional
        The maximum proportion of missing data allowed before the first valid point.
        Defaults to 0.1, meaning up to 10% of missing data is allowed at the start.
    min_end_proportion : float, optional
        The minimum proportion of data allowed before the last valid point.
        Defaults to 0.9, meaning up to 10% of missing data is allowed at the end.

    Returns:
    -------
    array-like
        An array of dispersal values or NaNs based on the input trajectory's validity.

    Notes:
    ------
    - The input trajectory should be a 2D array or a DataFrame with columns representing x and y coordinates.
    - The function checks for the proportion of valid data points and the presence of missing data at the start and end.
    - If the trajectory is valid, dispersal values are computed using a custom function (eudi5x).


    """

    if isinstance(xy, pd.DataFrame):
        xy = xy.values
    N = xy.shape[0]
    idx = np.where(~np.isnan(xy))[0]
    if idx.shape[0] < N * min_valid_proportion or idx[0] > N * max_start_proportion or idx[-1] < N * min_end_proportion:
        return np.zeros(N) * np.nan
    else:
        return eudi5x(xy, xy[idx[0]])


# def get_timeseries_slice(df, dt=0.1, time_range=None):
#     if time_range is None :
#         return df
#     else :
#         t0,t1=time_range
#         s0 = int(t0 / dt)
#         s1 = int(t1 / dt)
#         df_slice = df.loc[(slice(s0, s1), slice(None)), :]
#         return df_slice



def compute_dispersal_multi(xy0, t0, t1, dt, **kwargs):
    """
    Compute dispersal values for multiple agents over a time range.

    Parameters:
    ----------
    xy0 : pd.DataFrame
        A DataFrame containing agent positions and timestamps.
    t0 : float
        The start time for dispersal computation in sec.
    t1 : float
        The end time for dispersal computation in sec.
    dt : float
        Timestep of the timeseries.
    **kwargs : keyword arguments
        Additional arguments to pass to compute_dispersal_solo.

    Returns:
    -------
    np.ndarray
        An array of dispersal values for all agents at each time step.
    int
        The number of time steps.

    Example:
    --------
    xy0 = pd.DataFrame({'AgentID': [1, 1, 2, 2],
                       'Step': [0, 1, 0, 1],
                       'x': [0.0, 1.0, 2.0, 3.0],
                       'y': [0.0, 1.0, 2.0, 3.0]})

    AA, Nt = compute_dispersal_multi(xy0, t0=0, t1=1, dt=1)

    # AA will contain dispersal values, and Nt will be the number of time steps.
    """

    # xy=get_timeseries_slice(xy0, dt=dt, time_range=(t0,t1))

    s0 = int(t0 / dt)
    s1 = int(t1 / dt)
    xy = xy0.loc[(slice(s0, s1), slice(None)), ['x', 'y']]

    AA = apply_per_level(xy, compute_dispersal_solo, **kwargs)
    Nt = AA.shape[0]
    N = xy0.index.unique('AgentID').size
    Nticks = xy0.index.unique('Step').size

    AA0 = np.zeros([Nticks, N]) * np.nan
    AA0[s0:s0 + Nt, :] = AA

    return AA0.flatten(), Nt


def compute_component_velocity(xy, angles, dt, return_dst=False):
    """
    Compute the component velocity along a given orientation angle.

    This function calculates the component velocity of a set of 2D points (xy) along
    the specified orientation angles. It can optionally return the displacement along
    the orientation vector as well.

    Parameters:
    ----------
    xy : ndarray
        An array of shape (n, 2) representing the x and y coordinates of the points.
    angles : ndarray
        An array of shape (n,) containing the orientation angles in radians.
    dt : float
        The time interval for velocity calculation.
    return_dst : bool, optional
        If True, the function returns both velocities and displacements.
        If False (default), it returns only velocities.

    Returns:
    -------
    ndarray
        An array of component velocities calculated along the specified angles.

    ndarray (optional)
        An array of displacements along the specified orientation angles.
        Returned only if `return_dst` is True.

    """
    dx = np.diff(xy[:, 0], prepend=np.nan)
    dy = np.diff(xy[:, 1], prepend=np.nan)
    d_temp = np.sqrt(dx ** 2 + dy ** 2)

    # This is the angle of the displacement vector relative to x-axis
    rads = np.arctan2(dy, dx)
    # And this is the angle of the displacement vector relative to the front-segment orientation vector
    angles2ref = rads - angles
    angles2ref %= 2 * np.pi
    d = d_temp * np.cos(angles2ref)
    v = d / dt
    if return_dst:
        return v, d
    else:
        return v


def compute_velocity_threshold(v, Nbins=500, max_v=None, kernel_width=0.02):
    """
    Compute a velocity threshold using a density-based approach.

    Parameters:
    ----------
    v : array-like
        The input velocity data.
    Nbins : int, optional
        Number of bins for the velocity histogram. Default is 500.
    max_v : float or None, optional
        Maximum velocity value. If None, it is computed from the data. Default is None.
    kernel_width : float, optional
        Width of the Gaussian kernel for density estimation. Default is 0.02.

    Returns:
    -------
    float
        The computed velocity threshold.

    Notes:
    -----
    This function calculates a velocity threshold by estimating the density of the velocity data.
    It uses a histogram with `Nbins` bins, applies a Gaussian kernel of width `kernel_width`,
    and identifies the minimum between local maxima and minima in the density curve.

    """

    import matplotlib.pyplot as plt
    if max_v is None:
        max_v = np.nanmax(v)
    bins = np.linspace(0, max_v, Nbins)

    hist, bin_edges = np.histogram(v, bins=bins, density=True)
    vals = bin_edges[0:-1] + 0.5 * np.diff(bin_edges)
    hist += 1 / len(v)
    hist /= np.sum(hist)
    plt.figure()
    plt.semilogy(vals, hist)
    ker = sp.signal.gaussian(len(vals), kernel_width * Nbins / max_v)
    ker /= np.sum(ker)

    density = np.exp(np.convolve(np.log(hist), ker, 'same'))
    plt.semilogy(vals, density)

    mi, ma = sp.signal.argrelextrema(density, np.less)[0], sp.signal.argrelextrema(density, np.greater)[0]
    try:
        minimum = vals[mi][0]
    except:
        minimum = np.nan
    return minimum


def get_display_dims():
    import pygame
    pygame.init()
    W, H = pygame.display.Info().current_w, pygame.display.Info().current_h
    return int(W * 2 / 3 / 16) * 16, int(H * 2 / 3 / 16) * 16


def get_window_dims(arena_dims):
    X, Y = np.array(arena_dims)
    W0, H0 = get_display_dims()
    R0, R = W0 / H0, X / Y
    if R0 < R:

        return W0, int(W0 / R / 16) * 16
    else:
        return int(H0 * R / 16) * 16, H0


def get_arena_bounds(arena_dims, s=1):
    X, Y = np.array(arena_dims) * s
    return np.array([-X / 2, X / 2, -Y / 2, Y / 2])


def screen2space_pos(pos, screen_dims, space_dims):
    X, Y = space_dims
    X0, Y0 = screen_dims
    p = (2 * pos[0] / X0 - 1), -(2 * pos[1] / Y0 - 1)
    pp = p[0] * X / 2, p[1] * Y / 2
    return pp


def space2screen_pos(pos, screen_dims, space_dims):
    X, Y = space_dims
    X0, Y0 = screen_dims

    p = pos[0] * 2 / X, pos[1] * 2 / Y
    pp = ((p[0] + 1) * X0 / 2, (-p[1] + 1) * Y0)
    return pp


def circle_to_polygon(N, r):
    one_segment = np.pi * 2 / N

    points = [
        (np.sin(one_segment * i) * r,
         np.cos(one_segment * i) * r)
        for i in range(N)]

    return points


def boolean_indexing(v, fillval=np.nan):
    lens = np.array([len(item) for item in v])
    mask = lens[:, None] > np.arange(lens.max())
    out = np.full(mask.shape, fillval)
    out[mask] = np.concatenate(v)
    return out


def concat_datasets(ddic, key='end', unit='sec'):
    dfs = []
    for l, d in ddic.items():
        if key == 'end':
            try:
                df = d.endpoint_data
            except:
                df = d.read('end')
        elif key == 'step':
            try:
                df = d.step_data
            except:
                df = d.read('step')
        else:
            raise
        df['DatasetID'] = l
        df['GroupID'] = d.group_id
        dfs.append(df)
    df0 = pd.concat(dfs)
    if key == 'step':
        df0.reset_index(level='Step', drop=False, inplace=True)
        dts = np.unique([d.config.dt for l, d in ddic.items()])
        if len(dts) == 1:
            dt = dts[0]
            dic = {'sec': 1, 'min': 60, 'hour': 60 * 60, 'day': 24 * 60 * 60}
            df0['Step'] *= dt / dic[unit]
    return df0


def moving_average(a, n=3):
    return np.convolve(a, np.ones((n,)) / n, mode='same')


def body_contour(points=[(0.9, 0.1), (0.05, 0.1)], start=(1, 0), stop=(0, 0)):
    xy = np.zeros([len(points) * 2 + 2, 2]) * np.nan
    xy[0, :] = start
    xy[len(points) + 1, :] = stop
    for i in range(len(points)):
        x, y = points[i]
        xy[1 + i, :] = x, y
        xy[-1 - i, :] = x, -y
    return xy


def apply_per_level(s, func, level='AgentID', **kwargs):
    """
    Apply a function to each subdataframe of a MultiIndex DataFrame after grouping by a specified level.

    Parameters:
    ----------
    s : pandas.DataFrame
        A MultiIndex DataFrame with levels ['Step', 'AgentID'].
    func : function
        The function to apply to each subdataframe.
    level : str, optional
        The level by which to group the DataFrame. Default is 'AgentID'.
    **kwargs : dict
        Additional keyword arguments to pass to the 'func' function.

    Returns:
    -------
    numpy.ndarray
        An array of dimensions [N_ticks, N_ids], where N_ticks is the number of unique 'Step' values,
        and N_ids is the number of unique 'AgentID' values.

    Notes:
    -----
    This function groups the DataFrame 's' by the specified 'level', applies 'func' to each subdataframe, and
    returns the results as a numpy array.
    """

    def init_A(Ndims):
        ids = s.index.unique('AgentID').values
        Nids = len(ids)
        N = s.index.unique('Step').size
        if Ndims == 1:
            return np.zeros([N, Nids]) * np.nan
        elif Ndims == 2:
            return np.zeros([N, Nids, Ai.shape[1]]) * np.nan
        else:
            raise ValueError('Not implemented')

    A = None

    for i, (v, ss) in enumerate(s.groupby(level=level)):

        ss = ss.droplevel(level)
        Ai = func(ss, **kwargs)
        if A is None:
            A = init_A(len(Ai.shape))
        if level == 'AgentID':
            A[:, i] = Ai
        elif level == 'Step':
            A[i, :] = Ai
    return A


def unwrap_deg(a):
    if isinstance(a, pd.Series):
        a = a.values
    b = np.copy(a)
    b[~np.isnan(b)] = np.unwrap(b[~np.isnan(b)] * np.pi / 180) * 180 / np.pi
    return b


def unwrap_rad(a):
    if isinstance(a, pd.Series):
        a = a.values
    b = np.copy(a)
    b[~np.isnan(b)] = np.unwrap(b[~np.isnan(b)])
    return b


def rate(a, dt):
    if isinstance(a, pd.Series):
        a = a.values
    v = np.diff(a) / dt
    return np.insert(v, 0, np.nan)


def eudist(xy):
    if isinstance(xy, pd.DataFrame):
        xy = xy.values
    A = np.sqrt(np.nansum(np.diff(xy, axis=0) ** 2, axis=1))
    A = np.insert(A, 0, 0)
    return A


def eudi5x(a, b):
    """
    Calculate the Euclidean distance between points in arrays 'a' and 'b'.

    Parameters:
    ----------
    a : numpy.ndarray
        An array containing the coordinates of the first set of points.
    b : numpy.ndarray
        An array containing the coordinates of the second set of points.

    Returns:
    -------
    numpy.ndarray
        An array of Euclidean distances between each pair of points from 'a' and 'b'.
    """
    return np.sqrt(np.sum((a - np.array(b)) ** 2, axis=1))


def eudiNxN(a, b):
    b = np.array(b)
    return np.sqrt(np.sum(np.array([a - b[i] for i in range(b.shape[0])]) ** 2, axis=2))


def compute_dst(s, point=''):
    s[nam.dst(point)] = apply_per_level(s[nam.xy(point)], eudist).flatten()


def comp_extrema(a, order=3, threshold=None, return_2D=True):
    """
    Compute local extrema in a one-dimensional array or time series.

    Parameters:
    ----------
    a : pd.Series
        The input time series data as a pandas Series.
    order : int, optional
        The order of the extrema detection. Default is 3.
    threshold : tuple, optional
        A tuple (min_threshold, max_threshold) to filter extrema based on values.
        Default is None, which means no thresholding is applied.
    return_2D : bool, optional
        If True, returns a 2D array with flags for minima and maxima.
        If False, returns a 1D array with -1 for minima, 1 for maxima, and NaN for non-extrema.
        Default is True.

    Returns:
    -------
    np.ndarray
        An array with extrema flags based on the specified criteria.

    Notes:
    ------
    - This function uses `scipy.signal.argrelextrema` for extrema detection.
    """
    A = a.values
    N = A.shape[0]
    i_min = sp.signal.argrelextrema(A, np.less_equal, order=order)[0]
    i_max = sp.signal.argrelextrema(A, np.greater_equal, order=order)[0]

    # i_min_dif = np.diff(i_min, append=order)
    # i_max_dif = np.diff(i_max, append=order)
    # i_min = i_min[i_min_dif >= order]
    # i_max = i_max[i_max_dif >= order]

    if threshold is not None:
        t0 = a.index.min()
        thr_min, thr_max = threshold
        i_min = i_min[a.loc[i_min + t0] < thr_min]
        i_max = i_max[a.loc[i_max + t0] > thr_max]

    if return_2D:
        aa = np.zeros([N, 2]) * np.nan
        aa[i_min, 0] = True
        aa[i_max, 1] = True
    else:
        aa = np.zeros(N) * np.nan
        aa[i_min] = -1
        aa[i_max] = 1
    return aa
