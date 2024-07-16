from pathlib import Path
from zipfile import ZipFile
import tempfile

import numpy as np
import scipy.ndimage as ndi
from uncertainties import ufloat, unumpy as unp
from skfda.preprocessing.registration import LeastSquaresShiftRegistration
from skfda import FDataGrid
import shapely.geometry
import skimage.filters as skfilt
import skimage.measure as skmeas
import skimage.feature as skfeat
import skimage.draw as skdraw
from roifile import ImagejRoi


class ROI(object):
    def __init__(self, coords):
        self.global_coords = coords

        self.top = round(np.min(coords[:, 0]))
        self.bottom = round(np.max(coords[:, 0]))
        self.left = round(np.min(coords[:, 1]))
        self.right = round(np.max(coords[:, 1]))

        self.local_coords = coords - np.array([self.top, self.left])

    @property
    def bounding_box_slice(self):
        return np.s_[self.top:self.bottom, self.left:self.right]

    @property
    def local_shape(self):
        return (self.bottom - self.top, self.right - self.left)

    @property
    def local_mask(self):
        return skdraw.polygon2mask(self.local_shape, self.local_coords)

    def __repr__(self):
        return f"ROI(top={self.top}, bottom={self.bottom}, left={self.left}, right={self.right})"

    def __str__(self):
        return self.__repr__()

    @classmethod
    def from_imagej_roi(cls, imagej_roi: ImagejRoi):
        coords = np.roll(imagej_roi.coordinates(), 1, axis=1)
        return cls(coords)


def load_imagej_rois(path: Path):
    rois = []
    if path.name.endswith('.zip'):
        with ZipFile(path, "r") as zf:
            with tempfile.TemporaryDirectory() as tempdirname:
                tempdir = Path(tempdirname)
                zf.extractall(tempdir)
                for file in tempdir.iterdir():
                    if file.name.endswith('.roi'):
                        rois.append(ROI.from_imagej_roi(ImagejRoi.fromfile(file)))
    elif path.name.endswith('.roi'):
        rois.append(ROI.from_imagej_roi(ImagejRoi.fromfile(path)))

    return rois


def extract_line_profile(im, edgewidth=5, linelength=10, linewidth=3, roi=None, ax=None):

    # If no region-of-interest is provided, the ROI is the whole image
    if roi is None:
        Ny, Nx = im.shape
        roi = ROI(np.array([[0, 0], [0, Nx], [Ny, Nx], [Ny, 0]]))

    # Crop the image to the region-of-interest
    im = im[roi.bounding_box_slice]

    # Run a median filter
    im_median = ndi.median_filter(im, size=10)

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

    # Find the gradients in x and y direction of the filtered image
    grad_x = skfilt.scharr_v(im_median)
    grad_y = skfilt.scharr_h(im_median)

    # The gradients can be a bit noisy, but we assume the edge is not bending quickly,
    # and so we smooth the gradients with a filter.
    grad_x = ndi.uniform_filter(grad_x, size=edgewidth)
    grad_y = ndi.uniform_filter(grad_y, size=edgewidth)

    # Find all points on the phase edge
    y, x = np.where(edges)

    # Find the gradient at each point on the phase edge
    grad_x = grad_x[y, x]
    grad_y = grad_y[y, x]

    # Normalize the gradients
    grad_norm = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_x = grad_x / grad_norm
    grad_y = grad_y / grad_norm

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

    # Plot the aligned line profiles
    if ax is not None:
        for i in range(len(line_profiles_aligned)):
            ax[0].plot(line_profiles_aligned[i], alpha=0.1)

    # Average the aligned line profiles, ignoring nan values
    line_profile_avg = np.nanmean(line_profiles_aligned, axis=0)
    line_profile_std = np.nanstd(line_profiles_aligned, axis=0)
    line_profile = unp.uarray(line_profile_avg, line_profile_std)

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

    return line_profile, intensity_high, intensity_low

