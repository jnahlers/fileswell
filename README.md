<h1 align="center">
  <a href="https://github.com/jnahlers/fileswell">
    <img src="https://github.com/jnahlers/fileswell/blob/main/resources/icon_256x256.png" alt="Fileswell"/>
  </a>
  <br/>
  Fileswell [<em>faɪlz·wel</em>]
</h1>

<p align="center"><strong>Fileswell is a tool for extracting line profiles. It is therefore a profiler. A pro filer. 
</strong></p>


## Installation

The package is not yet available on PyPI. You can install it from the source code by 
running the following command in the root directory of the repository:

```bash
pip install .
```

## Usage

The package provides two functions: `load_imagej_rois` and `extract_line_profile`. 
`load_imagej_rois` can be used to load regions of interest (ROIs) that were selected 
and saved using the ROI Manager in ImageJ/Fiji, either as a `.roi` file or as a `.zip` 
file containing multiple `.roi` files.

```python
from pathlib import Path

from fileswell import load_imagej_rois, extract_line_profile

rois = load_imagej_rois(Path("path/to/rois.zip"))  # or .roi file

# Select the roi you want to extract the line profile from
roi = rois[0]
```

If you don't have regions of interest (ROIs) saved in the ImageJ format, you can also 
manually create a ROI object from a set of coordinates:

```python
import numpy as np

from fileswell import ROI

# Coordinates in the form [[y1, x1], [y2, x2], ...]
coordinates = np.array([[0, 0], [0, 10], [10, 10], [10, 0]])

roi = ROI(coordinates)
```

Using the image and the selected ROI, you can extract the line profile using the 
`extract_line_profile` function:


```python
results = extract_line_profile(im, roi=roi)
``` 

The `results` are a dictionary containing the following keys:
- `line_profile_mean`: The mean line profile along the edge.
- `line_profile_std`: The standard deviation of the line profile along the edge.
- `intensity_high_mean`: The mean intensity in the high intensity region.
- `intensity_high_std`: The standard deviation of the intensity in the high intensity region.
- `intensity_low_mean`: The mean intensity in the low intensity region.
- `intensity_low_std`: The standard deviation of the intensity in the low intensity region.

## API

#### extract\_line\_profile

```
Extract a line profile along an edge in an image.

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
smoothing : int, optional (default=2)
    Before thresholding, the image is smoothed using a median filter with size
    smoothing * edgewidth. The default of 2 should be sufficient for most images.
thresh_method : str, optional (default='otsu')
    The method used to binary threshold the image. Supported methods are:
    - 'otsu': Otsu's method (default)
    - 'triangle': Triangle method
    - 'minimum': Minimum method
    - 'isodata': ISODATA method
    - 'li': Li’s iterative Minimum Cross Entropy method
ax : array of matplotlib axes, optional (default=None)
    Needs to be a list or array of two matplotlib axes. The first axis will be used
    to show an image of the section of the image containing the edge and the line,
    with a sample of line profiles drawn on top. The second axis will be used to
    plot the line profiles. If None, no plots will be made.
return_profiles : bool, optional (default=False)
    Return the line profiles within the results.
return_shifts : bool, optional (default=False)
    Return the profile shifts within the results.

Returns
-------
dict
    A dictionary containing the following:
    - line_profile_mean: The mean line profile, averaged over all line profiles.
    - line_profile_std: The standard deviation of the line profile, averaged over
        all line profiles.
    - intensity_high_mean: The mean intensity of the high intensity region.
    - intensity_high_std: The standard deviation of the intensity of the high
        intensity region.
    - intensity_low_mean: The mean intensity of the low intensity region.
    - intensity_low_std: The standard deviation of the intensity of the low
        intensity region.
    - line_profiles: The individual line profiles, if return_profiles is True.
    - shifts: The shifts of the line profiles, if return_shifts is True.
```

#### load\_imagej\_rois

```
Load ImageJ ROIs from a .roi or .zip (containing .roi files) file.

Parameters
----------
path : Path
    The path to the .roi or .zip file.

Returns
-------
list of ROI
    A list of ROI objects
```

#### ROI

```
A region-of-interest (ROI) defined by a set of coordinates.

Constructor parameters
----------
coords : ndarray
    An Nx2 array of coordinates defining the ROI, with the first column
    being the y-coordinates and the second column being the x-coordinates.

Returns
-------
ROI
    A ROI object.
```
