import time
import numpy as np
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure
from skimage.measure import label


def calculate_runtime(t):
    t2 = time.gmtime(time.time() - t)
    return f'{t2.tm_hour:02.0f}:{t2.tm_min:02.0f}:{t2.tm_sec:02.0f}'


def bbox2_3D(img):
    x = np.any(img, axis=(1, 2))
    y = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    xmin, xmax = np.where(x)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return xmin, xmax, ymin, ymax, zmin, zmax+1


def bbox2_2D(img):
    x = np.any(img, axis=1)
    y = np.any(img, axis=0)

    xmin, xmax = np.where(x)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]

    return xmin, xmax, ymin, ymax


def assd_and_hd(result, reference, voxelspacing=None, connectivity: int = 1, crop_to_relevant_scope: bool = True):
    """
    The concept is taken from medpy.metric.assd and medpy.metric.hd

    Average symmetric surface distance and Hausdorff Distance.

    Computes the average symmetric surface distance (ASD) and the (symmetric) Hausdorff Distance (HD) between the binary objects in
    two images.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        The decision on the connectivity is important, as it can influence the results
        strongly. If in doubt, leave it as it is.
    crop_to_relevant_scope : bool
        If set to True (by default) the two cases holding the objects will be cropped to the relevant region
        to save running time.

    Returns
    -------
    (assd, hd) : Tuple(float, float)
        The average symmetric surface distance and The symmetric Hausdorff Distance between the object(s) in ``result`` and the
        object(s) in ``reference``. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    Notes
    -----
    These are real metrics. The binary images can therefore be supplied in any order.
    """

    if crop_to_relevant_scope:
        result, reference = crop_to_relevant_joint_bbox(result, reference)

    sds1 = _surface_distances(result, reference, voxelspacing, connectivity)
    sds2 = _surface_distances(reference, result, voxelspacing, connectivity)

    assd_res = np.mean((sds1.mean(), sds2.mean()))
    hd_res = max(sds1.max(), sds2.max())

    return assd_res, hd_res



def crop_to_relevant_joint_bbox(result, reference):
    relevant_joint_case = np.logical_or(result, reference)

    if result.ndim == 3:
        xmin, xmax, ymin, ymax, zmin, zmax = bbox2_3D(relevant_joint_case)
        xmax += 1
        ymax += 1
        zmax += 1
        slc = np.s_[xmin: xmax, ymin: ymax, zmin: zmax]
    else:
        xmin, xmax, ymin, ymax = bbox2_2D(relevant_joint_case)
        xmax += 1
        ymax += 1
        slc = np.s_[xmin: xmax, ymin: ymax]
    result = result[slc]
    reference = reference[slc]

    return result, reference

def _surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    This function is copied from medpy.metric version 0.3.0

    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # test for emptiness
    if 0 == np.count_nonzero(result):
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference):
        raise RuntimeError('The second supplied array does not contain any binary object.')

        # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]

    return sds


def get_largest_connected_component(segmentation):
    labels = label(segmentation, connectivity=1)
    assert (labels.max() != 0)  # assume at least 1 CC
    largest = (labels == (np.argmax(np.bincount(labels.flat)[1:]) + 1)).astype(segmentation.dtype)
    return largest
