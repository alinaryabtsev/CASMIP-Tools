import nibabel as nib
import numpy as np
import os
from glob import glob
from skimage import morphology
from skimage import measure
from skimage.morphology import cube
from tqdm import tqdm


def postprocessing_segmentation(file):
    nifti = nib.load(file)
    nifti = nib.as_closest_canonical(nifti)
    seg = nifti.get_fdata()
    after_erosion = morphology.binary_erosion(seg, cube(3))
    labels = measure.label(after_erosion)
    largest = (labels == np.argmax(np.bincount(labels.flat, weights=labels.flat))).astype(np.uint)
    largest = morphology.binary_dilation(largest, cube(3)).astype(np.uint)
    largest = morphology.remove_small_holes(largest).astype(np.uint)
    obj_to_save = nib.Nifti1Image(largest, nifti.affine)
    path_to_save = os.path.join(os.path.dirname(file), f"{os.path.basename(file).rsplit('.', 2)[0]}_postprocessed.nii.gz")
    nib.save(obj_to_save, path_to_save)


def postprocessing_directory(dir_path):
    files = glob(os.path.join(dir_path, "*.nii.gz"))
    for file in tqdm(files):
        postprocessing_segmentation(file)
    