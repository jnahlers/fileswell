import numpy as np
import scipy.ndimage as ndi
from scipy.interpolate import UnivariateSpline
from uncertainties import ufloat, unumpy as unp
from skfda.preprocessing.registration import LeastSquaresShiftRegistration
from skfda import FDataGrid
import shapely.geometry
import skimage.filters as skfilt
import skimage.measure as skmeas
import skimage.feature as skfeat
import skimage.morphology as skmorph

from fileswell._roi import ROI


def order_line_points(x, y):
    """Order the coordinates of a 2D line skeleton.

    This function takes a set of coordinates that represent a line skeleton and
    orders them in such a way that they form a continuous path. The function
    assumes that the line skeleton does not branch.

    Implements https://stackoverflow.com/a/37744549.

    Parameters
    ----------
    x : array
        The x-coordinates of the line skeleton.
    y : array
        The y-coordinates of the line skeleton.

    Returns
    -------
    tuple
        A tuple containing the ordered x and y coordinates of the line skeleton.
    """
    points = np.c_[x, y]
    from sklearn.neighbors import NearestNeighbors

    clf = NearestNeighbors(n_neighbors=2).fit(points)
    G = clf.kneighbors_graph()

    import networkx as nx

    T = nx.from_scipy_sparse_array(G)

    paths = [list(nx.dfs_preorder_nodes(G=T, source=i)) for i in range(len(points))]

    mindist = np.inf
    minidx = 0

    for i in range(len(points)):
        p = paths[i]  # order of nodes
        ordered = points[p]  # ordered nodes
        # find cost of that order by the sum of euclidean distances between points (i) and (i+1)
        cost = (((ordered[:-1] - ordered[1:]) ** 2).sum(1)).sum()
        if cost < mindist:
            mindist = cost
            minidx = i

    opt_order = paths[minidx]

    return points[opt_order][:, 0], points[opt_order][:, 1]



def extract_line_profile(im, edgewidth=5, linelength=10, linewidth=3, roi=None, ax=None):
    """Extract a line profile along an edge in an image.

    This function extracts an averaged line profile along an edge in an image.
    It was first designed to extract line profiles from edges in CT slices of
    fairly homogenous materials. If there is a lot of noise, or the materials are
    very inhomogenous, it may not work so well.

    The function works by segmenting the images into two regions, one with a high
    intensity and one with a low intensity. It then finds the edge between these
    two regions using a Canny edge detector. At each point on the edge it takes
    a line profile perpendicular to the edge, and then averages these line profiles.

    Therefore, you need to supply a region-of-interest (ROI) that contains the edge
    you want to extract the line profile from, and some of the two regions surrounding
    that edge, but no other features.

    If the image already contains nothing more than the edge and the two regions, you
    don't need to supply a ROI.

    See also: https://stackoverflow.com/a/52020098
              https://stackoverflow.com/q/37742358

    Parameters
    ----------
    im : ndarray
        The image to extract the line profile from.
    edgewidth : int, optional (default=5)
        The estimated width of the edge, in pixels.
        For example, in a propagation-based phase-contrast CT slice, this would be the
        width of the phase fringes. Does not need to be exact.
    linelength : int, optional (default=10)
        The length of the line profile either side of the edge, in pixels.
    linewidth : int, optional (default=3)
        The width of each line profile, in pixels.
    roi : ROI, optional (default=None)
        The region-of-interest containing the edge and the two regions surrounding it.
        If None, the ROI is the whole image.
    ax : array of matplotlib axes, optional (default=None)
        Needs to be a list or array of two matplotlib axes. The first axis will be used
        to show an image of the section of the image containing the edge and the line,
        with a sample of line profiles drawn on top. The second axis will be used to
        plot the line profiles. If None, no plots will be made.

    Returns
    -------
    dict
        A dictionary containing the following:
        - line_profile : array
            The line profile along the edge, as a numpy array of uncertainties.
        - intensity_high : ufloat
            The mean intensity (and std) in the high intensity region.
        - intensity_low : ufloat
            The mean intensity (and std) in the low intensity region.
    """

    # If no region-of-interest is provided, the ROI is the whole image
    if roi is None:
        Ny, Nx = im.shape
        roi = ROI(np.array([[0, 0], [0, Nx], [Ny, Nx], [Ny, 0]]))

    # Crop the image to the region-of-interest
    im = im[roi.bounding_box_slice]

    # Run a median filter
    im_median = ndi.median_filter(im, size=2*edgewidth)

    # Get the mask in the coordinates of the image cropped to the ROI
    mask = roi.local_mask

    # Get the max and min values within the median filtered image
    max = np.max(im_median[mask])
    min = np.min(im_median[mask])

    # Normalise the image to 0-255 and convert to 8-bit
    im_norm = (im_median - min) / (max - min) * 255
    im_norm = im_norm.astype(np.uint8)

    # Make the masked image
    im_masked = np.ma.masked_where(~mask, im_norm)

    # Threshold using Otsu's method
    thresh_value = skfilt.threshold_otsu(im_masked)
    im_thresh = im_masked > thresh_value

    # Remove small holes that may have been caused by noise
    im_thresh = skmorph.remove_small_holes(im_thresh)
    im_thresh = skmorph.remove_small_objects(im_thresh)

    # Ensure that the thresholded image consists of two path-connected regions
    im_label = np.ma.masked_where(~mask, skmeas.label(im_thresh))
    unique = np.unique(im_label)
    # np.unique will include all unique elements. However, annoyingly, it will also
    # include one element 'representing' the masked elements (if used on with a masked
    # array). Worse, this element is not included if the mask covers the entire image.
    # So we need to check if the mask covered the whole image, in which case the
    # array.mask is simply False (instead of a boolean array).
    if isinstance(unique.mask, np.bool_):
        max_length = 2
    else:
        max_length = 3
    if len(unique) > max_length:
        raise ValueError(
            "Thresholded image does not consist of two path-connected regions")

    # Apply a Canny edge detector to find the edge between the two regions
    edges = skfeat.canny(im_thresh, sigma=edgewidth)

    # Remove any edges that do not lie on a slightly shrunk version of the mask,
    # to ensure we don't include boundaries of the mask.
    shrunk_mask = ndi.binary_erosion(mask, iterations=1)
    edges[~shrunk_mask] = 0

    # Calculate the mean intensities in each of the two regions.
    # As the intensities close to the edge are affected by the edge, we exclude these
    # by eroding the thresholded image and taking the mean intensity in the eroded region.
    mask_high = im_thresh.astype(bool) & mask
    mask_high = ndi.binary_erosion(
        np.pad(mask_high, 4 * edgewidth, constant_values=True),
        iterations=2 * edgewidth
    )
    mask_high = mask_high[
                4 * edgewidth: -4 * edgewidth, 4 * edgewidth: -4 * edgewidth
                ]
    intensity_high = ufloat(np.mean(im[mask_high]), np.std(im[mask_high]))
    mask_low = np.logical_not(ndi.binary_dilation(im_thresh, iterations=2 * edgewidth)) & mask
    intensity_low = ufloat(np.mean(im[mask_low]), np.std(im[mask_low]))

    # In order to take profiles across the edge, we need to find out what the normal
    # vector is at each point on the edge. To do this we will fit a spline to the edge
    # and then take the derivative of the spline at each point.

    # We start by getting the points on the edge. This gives us a line skeleton.
    y, x = edges.nonzero()

    # nonzero() returns the points in returned in row-major, C-style order.
    # We need to sort the points by the distance along the edge in order to fit a
    # spline. This is non-trivial, as the edge may swerve and swirl and double back '
    # on itself.
    # See https://stackoverflow.com/q/37742358 for a discussion of how to do this.
    x, y = order_line_points(x, y)

    # Cumulative distance along the edge
    dist = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))

    # Remove the last coord
    x = x[:-1]
    y = y[:-1]

    # Build a list of spline functions, one for each coordinate
    splines = [UnivariateSpline(dist, coord, k=3, s=None) for coord in [x, y]]

    # Find the normal vector at each point on the edge
    dx = splines[0].derivative()(dist)
    dy = splines[1].derivative()(dist)
    grad_x = -dy
    grad_y = dx

    # Normalize the gradients
    grad_norm = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_x = -grad_x / grad_norm
    grad_y = -grad_y / grad_norm

    # For each edge point, draw a line of length line_length in the direction of the
    # normal vector. Because we want the line to go from low to high, we will reverse
    # the direction here, going against the gradient.
    x1 = x - linelength * grad_x
    y1 = y - linelength * grad_y
    x2 = x + linelength * grad_x
    y2 = y + linelength * grad_y

    # scikit-image's profile_line function calculates the length of the line as
    # int(np.ceil(np.hypot(d_row, d_col) + 1)). Because of numerical errors, the part
    # that the ceil function is applied to may be just below or just above the actual
    # correct length. For example, if profile_width is 10, then the length should be
    # 2 * 10 + 1 = 21 (as the last point is included). However, the length may be
    # calculated as 20.99999999999999 or 21.000000000000007. This will cause the line
    # profile to have a different length for each line, which will hard to align and
    # also have a different spacing dx between the points in the line profile. To
    # avoid this, we will add a small offset to the end points of the lines, slightly
    # reducing the overall length of the line profile.
    x2 = x2 - 1E-5 * grad_x
    y2 = y2 - 1E-5 * grad_y

    # We need to reject any lines that are not fully contained within the
    # region-of-interest.
    # Need to change mask coordinates from y, x to x, y for shapely
    mask_coordinates = np.roll(roi.local_coords, 1, axis=1)
    mask_poly = shapely.geometry.Polygon(mask_coordinates)
    lines = [shapely.geometry.LineString([(x1[i], y1[i]), (x2[i], y2[i])]) for i in range(len(x))]
    lines = [line for line in lines if mask_poly.contains(line)]

    # Find the line profile along each line
    line_profiles = []
    for line in lines:
        try:
            src = (line.coords[0][1], line.coords[0][0])
            dst = (line.coords[1][1], line.coords[1][0])
            line_profile = skmeas.profile_line(
                im, src, dst, order=3, mode="constant", cval=0,
                linewidth=linewidth,
            )
        except ValueError:
            # If the line profile goes outside the image (it shouldn't) ignore it
            continue

        # Ensure the length is close to 2 * profile_width + 1
        if not np.isclose(line.length, 2 * linelength):
            raise ValueError("Line profile has incorrect length")

        # Ensure the distances between the points in the line profile are close to 1
        dx = line.length / (len(line_profile) - 1)
        if not np.isclose(dx, 1):
            raise ValueError("Line profile has incorrect spacing")

        # The line profile may have nan values at the ends if we went outside the image
        # or masked region. In this case, ignore the line profile.
        if np.isnan(line_profile).any():
            continue

        line_profiles.append(line_profile)

    # Make the line profiles into a FDataGrid
    fd = FDataGrid(data_matrix=line_profiles)

    # Align the line profiles using least squares shift registration
    shift_registration = LeastSquaresShiftRegistration()
    fd_registered = shift_registration.fit_transform(fd)
    line_profiles_aligned = np.squeeze(fd_registered.data_matrix)

    # Average the aligned line profiles, ignoring nan values
    line_profile_avg = np.nanmean(line_profiles_aligned, axis=0)
    line_profile_std = np.nanstd(line_profiles_aligned, axis=0)
    line_profile = unp.uarray(line_profile_avg, line_profile_std)

    # Plot the aligned line profiles
    if ax is not None:
        for i in range(len(line_profiles_aligned)):
            ax[0].plot(line_profiles_aligned[i], alpha=0.1, color="gray")
        # Plot the mean and std line profile
        ax[0].plot(line_profile_avg, "r-")
        y_err_low = line_profile_avg - line_profile_std
        y_err_high = line_profile_avg + line_profile_std
        ax[0].fill_between(np.arange(len(line_profile_avg)), y_err_low, y_err_high,
                           color="r", alpha=0.3)
        # Plot horizontal lines at the two intensity levels
        ax[0].axhline(intensity_high.nominal_value, color="b", linestyle="-", alpha=0.5)
        ax[0].axhline(intensity_low.nominal_value, color="b", linestyle="-", alpha=0.5)
        # Set the x limits
        ax[0].set_xlim(0, len(line_profile_avg)-1)

    # Draw an image of the line profiles if ax is provided
    if ax is not None:
        ax[1].imshow(im, cmap="gray")
        # Draw the phase edge
        ax[1].plot(x, y, "b-", markersize=2, alpha=0.5)
        # Draw the line profiles
        for line in lines[0::10]:
            ax[1].plot(*line.xy, "r-")
        # Draw the mask regions
        ax[1].imshow(np.ma.masked_where(mask_high == 0, mask_high), cmap="cool",
                     alpha=0.1)
        ax[1].imshow(np.ma.masked_where(mask_low == 0, mask_low), cmap="cool",
                     alpha=0.1)

    results = {
        "line_profile": line_profile,
        "intensity_high": intensity_high,
        "intensity_low": intensity_low
    }

    return results

