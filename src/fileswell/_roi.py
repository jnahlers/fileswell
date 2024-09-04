import os

from pathlib import Path
from zipfile import ZipFile
import tempfile

import numpy as np
import skimage.draw as skdraw
from roifile import ImagejRoi
from natsort import natsorted


class ROI(object):
    """A region-of-interest (ROI) defined by a set of coordinates."""
    def __init__(self, coords):
        """Constructs a ROI

        Parameters
        ----------
        coords : ndarray
            An Nx2 array of coordinates defining the ROI, with the first column
            being the y-coordinates and the second column being the x-coordinates.

        Returns
        -------
        ROI
            A ROI object.
        """
        self.global_coords = coords

        self.top = round(np.min(coords[:, 0]))
        self.bottom = round(np.max(coords[:, 0]))
        self.left = round(np.min(coords[:, 1]))
        self.right = round(np.max(coords[:, 1]))

        self.local_coords = coords - np.array([self.top, self.left])

        self.imagejroi = None

    @property
    def bounding_box_slice(self):
        return np.s_[self.top:self.bottom, self.left:self.right]

    @property
    def local_shape(self):
        return self.bottom - self.top, self.right - self.left

    @property
    def local_mask(self):
        return skdraw.polygon2mask(self.local_shape, self.local_coords)

    def __repr__(self):
        return f"ROI(top={self.top}, bottom={self.bottom}, left={self.left}, right={self.right})"

    def __str__(self):
        return self.__repr__()

    @classmethod
    def from_imagej_roi(cls, imagej_roi: ImagejRoi):
        """Create a ROI from an ImagejRoi object."""
        coords = np.roll(imagej_roi.coordinates(), 1, axis=1)
        roi = cls(coords)
        roi.imagejroi = imagej_roi
        return roi


def load_imagej_rois(path: str | os.PathLike) -> list[ROI]:
    """Load ImageJ ROIs from a .roi or .zip (containing .roi files) file.

    Parameters
    ----------
    path : Path
        The path to the .roi or .zip file.

    Returns
    -------
    list of ROI
        A list of ROI objects.
    """
    path = Path(path)
    rois = []
    if path.name.endswith('.zip'):
        with ZipFile(path, "r") as zf:
            with tempfile.TemporaryDirectory() as tempdirname:
                tempdir = Path(tempdirname)
                zf.extractall(tempdir)
                for file in natsorted(tempdir.iterdir()):
                    if file.name.endswith('.roi'):
                        rois.append(ROI.from_imagej_roi(ImagejRoi.fromfile(file)))
    elif path.name.endswith('.roi'):
        rois.append(ROI.from_imagej_roi(ImagejRoi.fromfile(path)))

    return rois