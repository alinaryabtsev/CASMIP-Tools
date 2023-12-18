import nibabel as nib
from skimage.measure import label, regionprops
import numpy as np


class SegmentationStats:
    def __init__(self, mask_filename):
        self.mask_filename = mask_filename
        self.mask_nifti = nib.load(self.mask_filename)
        self.mask = self.mask_nifti.get_fdata()
        self.connected_components_num, self.mask_labels = self.get_num_of_connected_components(get_labels=True)

    def get_mask_shape(self):
        return self.mask.shape

    def get_num_of_connected_components(self, get_labels=False):
        labeled_image, count = label(self.mask, return_num=True)
        return (count, labeled_image) if get_labels else count

    def get_num_of_connected_components_by_area(self, min_size=0, max_size=np.inf, get_labels=False):
        """
        Returns an amount or a list of connected components by its area and an amount.
        :param min_size: the minimum size of a connected component in voxels
        :param max_size: the maximum size of a connected component in voxels
        :param get_labels: to return a list of connected components or not
        :return: number of connected components and a list of connected components (optional) according to area
        """
        regions = regionprops(self.mask_labels)
        object_areas = [obj["area"] for obj in regions]
        object_areas_in_size = [area for area in object_areas if min_size <= area <= max_size]
        if get_labels:
            return len(object_areas_in_size), object_areas_in_size
        return len(object_areas_in_size)

    def get_num_of_connected_components_by_size(self, min_size=0, max_size=np.inf, get_labels=False):
        """
        Returns a size or a list of connected components by size and a size.
        :param min_size: the minimum size of a connected component in mm
        :param max_size: the maximum size of a connected component in mm
        :param get_labels: to return a list of connected components or not
        :return: number of connected components and a list of connected components (optional) according to size
        """
        regions = regionprops(self.mask_labels)
        voxel_volume = np.prod(self.mask_nifti.header.get_zooms())
        object_volumes = [obj.image.sum() * voxel_volume for obj in regions]
        object_areas_in_size = [obj for vol, obj in zip(object_volumes, self.mask_labels.reshape(-1, 512, 512))
                                if min_size <= SegmentationStats.approximate_diameter(vol) <= max_size]
        if get_labels:
            return len(object_areas_in_size), object_areas_in_size
        return len(object_areas_in_size)

    def get_histogram_of_connected_components(self, min_size=0, max_size=np.inf):
        object_areas_in_size = self.get_num_of_connected_components_by_size(min_size=min_size, max_size=max_size, get_labels=True)
        return np.histogram(object_areas_in_size, bins=100)

    @staticmethod
    def approximate_diameter(tumor_volume):
        """
        approximate the diameter of a tumor from its volume
        """
        r = ((3 * tumor_volume) / (4 * np.pi)) ** (1 / 3)
        diameter = 2 * r
        return diameter