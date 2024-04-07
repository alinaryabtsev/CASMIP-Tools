from copy import deepcopy
from typing import List

import pandas as pd
import numpy as np
from scipy import ndimage
from skimage import measure
import nibabel as nib
import operator
import os
from functools import partial
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects

from utils import assd_and_hd
from multiprocessing import Pool
import tqdm

LIVER = "liver"
TUMORS = "tumors"
LIVER_MEASUREMENTS = ['Num_of_Connected_Components', 'Dice', 'ASSD (mm)', 'Hausdorff (mm)']
TUMOR_MEASUREMENTS = ['Num of lesion',
                      'Dice',
                      'Dice with FN',
                      'Mean ASSD (mm)',
                      'Mean Hausdorff (mm)',
                      'Max Hausdorff (mm)',
                      'Segmentation TP (cc)',
                      'Segmentation FP (cc)',
                      'Segmentation FN (cc)',
                      'Total tumor volume GT (cc)',
                      'Total tumor volume Predictions (cc)',
                      'Delta between total tumor volumes (cc)',
                      'Delta between total tumor volumes (%)',
                      'Delta between total tumor volumes (TP only) (cc)',
                      'ABS Delta between total tumor volumes (TP only) (cc)',
                      'Delta between total tumor volumes (TP only) (%)',
                      'ABS Delta between total tumor volumes (TP only) (%)',
                      'Tumor Burden GT (%)',
                      'Tumor Burden Pred (%)',
                      'Tumor Burden Delta (%)',
                      'Detection TP (per lesion)',
                      'Detection FP (per lesion)',
                      'Detection FN (per lesion)',
                      'Precision',
                      'Recall',
                      'F1 Score']
TUMORS_SIZES = (0, 5, 10)
CATEGORIES_TO_CALCULATE = [f">{s}" for s in TUMORS_SIZES]
BINARY_FILLER_MATRIX = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).reshape([3, 3, 1])  # matrix for semantic operations


def calculate_measures(measures_type, predictions_list, gt_list, threshold_range, excel_path, roi_list=None,
                       tumor_sizes=TUMORS_SIZES):
    """
    calculate measures (detection and segmentation) for a list of predictions and ground truth, creates an excel file
    that summarizes all the information
    :param measures_type: Are we measuring liver (or any big object) or tumors
    :param predictions_list: the list with predictions file paths
    :param gt_list: the list with ground truth file paths
    :param threshold_range: the range of thresholds to calculate measures for (if there is only one threshold, the range is (1,2))
    :param excel_path: the path to save the excel file
    :param roi_list: a list with roi file paths
    :param tumor_sizes: Texonomy of tumor sizes
    """
    cm = CalculateMeasures(measures_type, tumor_sizes)

    working_dir = os.path.dirname(excel_path)

    # if there are multiple thresholds, create a directory according to the name of the excel (excel path) and put
    # there all threshold analysis files
    if len(threshold_range) > 1:
        working_dir = os.path.join(working_dir, os.path.splitext(os.path.basename(excel_path))[0])

    os.makedirs(working_dir, exist_ok=True)
    for th in threshold_range:
        if len(threshold_range) > 1:
            excel_path = os.path.join(working_dir, f"Threshold_{th}.xlsx")
        excel_writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')

        print(f"Calculating measures for threshold {th}")

        with Pool(os.cpu_count() - 2) as p:
            if roi_list:
                zipped_predictions = zip(predictions_list, gt_list, roi_list)
            else:
                print("No ROI list provided")
                zipped_predictions = zip(predictions_list, gt_list)
            results = p.starmap(partial(cm.calculate_stats, th=th),
                                tqdm.tqdm(zipped_predictions, total=len(predictions_list)))

        results_df = dict([(cat, pd.DataFrame()) for cat in CATEGORIES_TO_CALCULATE])

        for res in results:
            for k, cat in enumerate(CATEGORIES_TO_CALCULATE):
                results_df[cat] = pd.concat([results_df[cat], pd.DataFrame(res[k], index=[0])], ignore_index=True)

        for cat in CATEGORIES_TO_CALCULATE:
            cm.write_to_excel(excel_writer, cat, results_df[cat])
        print(f"Writing measures for threshold {th} finished")

        excel_writer.save()


def calculate_measures_dataframe(measures_type, predictions_list, gt_list, threshold_range, roi_list=None,
                                 pix_dim=False, tumor_sizes=TUMORS_SIZES):
    cm = CalculateMeasures(measures_type, tumor_sizes)
    results_th = {t: [] for t in threshold_range}

    for th in threshold_range:
        print(f"Calculating measures for threshold {th}")

        with Pool(os.cpu_count() - 2) as p:
            if roi_list:
                zipped_predictions = zip(predictions_list, gt_list, roi_list)
            else:
                print("No ROI list provided")
                zipped_predictions = zip(predictions_list, gt_list)

            if pix_dim:
                print("Setting pixel dimensions to (1,1,1)")
                results = p.starmap(partial(cm.calculate_stats, th=th, pix_dims=[1, 1, 1]),
                                    tqdm.tqdm(zipped_predictions, total=len(gt_list)))
            else:
                results = p.starmap(partial(cm.calculate_stats, th=th),
                                    tqdm.tqdm(zipped_predictions, total=len(gt_list)))

        results_diameter_0, results_diameter_5, results_diameter_10 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        print("All stats are calculated, rearranging data")
        for res in results:
            results_diameter_0 = results_diameter_0.append(res[0], ignore_index=True)
            results_diameter_5 = results_diameter_5.append(res[1], ignore_index=True)
            results_diameter_10 = results_diameter_10.append(res[2], ignore_index=True)

        results_dfs = [pd.DataFrame(results_diameter_0).agg(['mean', 'std', 'min', 'max', 'sum']),
                       pd.DataFrame(results_diameter_5).agg(['mean', 'std', 'min', 'max', 'sum']),
                       pd.DataFrame(results_diameter_10).agg(['mean', 'std', 'min', 'max', 'sum'])]

        results_th[th] = results_dfs
        print(f"Calculating measures for threshold {th} finished")
    return results_th


class CalculateMeasures:
    def __init__(self, measures_type, tumor_sizes=None):
        self.measures_type = measures_type  # liver or tumors
        if self.measures_type == TUMORS:
            self.tumor_sizes = tumor_sizes

    def calculate_stats(self, prediction, gt, roi=None, th=1, calculate_ASSD=True, calculate_HD=True, pix_dims=None):
        if type(gt) == np.ndarray:
            file_name = "scan"
        else:
            file_name = os.path.dirname(gt) + os.path.basename(gt)
        if self.measures_type == LIVER:
            stats = SegmentationStatsLiver(file_name, gt, prediction, gt, th)
            return stats.calculate_stats_by_diameter(diameter=0)
        else:  # TUMOR
            if roi:
                stats = SegmentationStatsTumors(roi, gt, prediction, file_name, th, pix_dims)
            else:
                stats = SegmentationStatsTumors(None, gt, prediction, file_name, th, pix_dims)
                print(f"calculating stats {prediction}")
            return (stats.calculate_statistics_by_diameter(self.tumor_sizes[0],
                                                           calculate_ASSD=calculate_ASSD,
                                                           calculate_HD=calculate_HD),
                    stats.calculate_statistics_by_diameter(self.tumor_sizes[1],
                                                           calculate_ASSD=calculate_ASSD,
                                                           calculate_HD=calculate_HD),
                    stats.calculate_statistics_by_diameter(self.tumor_sizes[2],
                                                           calculate_ASSD=calculate_ASSD,
                                                           calculate_HD=calculate_HD)
                    )

    def write_to_excel(self, excel_writer, sheet_name, results):
        if self.measures_type == LIVER:
            columns_order = LIVER_MEASUREMENTS
        else:
            columns_order = TUMOR_MEASUREMENTS

        results = results.set_index('Filename')
        results = results.append(results.agg(['mean', 'std', 'min', 'max', 'sum']), ignore_index=False)

        # format excel workbook
        workbook = excel_writer.book
        cell_format = workbook.add_format({'num_format': '#,##0.00'})
        cell_format.set_font_size(16)
        results.to_excel(excel_writer, sheet_name=sheet_name, columns=columns_order, startrow=1, startcol=1,
                         header=False, index=False)

        header_format, last_format, max_format, min_format = self._format_header(workbook)

        worksheet = excel_writer.sheets[sheet_name]
        worksheet.freeze_panes(1, 1)
        worksheet.conditional_format('$B$2:$E$' + str(len(results.axes[0]) - 4),
                                     {'type': 'formula',
                                      'criteria': '=B2=B$' + str(len(results.axes[0])),
                                      'format': max_format})
        worksheet.conditional_format('$B$2:$E$' + str(len(results.axes[0]) - 4),
                                     {'type': 'formula',
                                      'criteria': '=B2=B$' + str(len(results.axes[0]) - 1),
                                      'format': min_format})

        for i in range(len(results.axes[0]) - 4, len(results.axes[0]) + 1):
            worksheet.set_row(i, None, last_format)
        for col_num, value in enumerate(columns_order):
            worksheet.write(0, col_num + 1, value, header_format)
        for row_num, value in enumerate(results.axes[0].astype(str)):
            worksheet.write(row_num + 1, 0, value, header_format)

        # Fix first column
        column_len = results.axes[0].astype(str).str.len().max() + results.axes[0].astype(str).str.len().max() * 0.5
        worksheet.set_column(0, 0, column_len, cell_format)

        # Fix all  the rest of the columns
        for i, col in enumerate(columns_order):
            # find length of column i
            column_len = results[col].astype(str).str.len().max()
            # Setting the length if the column header is larger than the max column value length
            column_len = max(column_len, len(col))
            column_len += column_len * 0.5
            # set the column length
            worksheet.set_column(i + 1, i + 1, column_len, cell_format)

    @staticmethod
    def _format_header(workbook):
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'font_size': 16,
            'valign': 'top',
            'border': 1})
        max_format = workbook.add_format({
            'font_size': 16,
            'bg_color': '#E6FFCC'})
        min_format = workbook.add_format({
            'font_size': 16,
            'bg_color': '#FFB3B3'})
        last_format = workbook.add_format({
            'font_size': 16,
            'bg_color': '#C0C0C0',
            'border': 1,
            'num_format': '#,##0.00'})
        return header_format, last_format, max_format, min_format


class Utils:

    @staticmethod
    def get_largest_connected_component(segmentation, connectivity=None):
        """
        get the largest connected component from a segmentation mask
        """
        labels = measure.label(segmentation, connectivity=connectivity)
        assert (labels.max() != 0), "No connected component was found"  # assume at least 1 CC
        largest_cc = (labels == np.argmax(np.bincount(labels.flat)[1:]) + 1).astype(segmentation.dtype)
        return largest_cc

    @staticmethod
    def get_connected_components(map):
        """
        Remove Small connected components from a binary mask
        :param map: the binary mask
        :return: the binary mask with small connected components removed and the number of connected components
        """
        label_img = measure.label(map)
        cc_num = label_img.max()
        cc_areas = ndimage.sum(map, label_img, range(cc_num + 1))
        area_mask = (cc_areas <= 10)
        label_img[area_mask[label_img]] = 0
        return_value = measure.label(label_img)
        return return_value, return_value.max()

    @staticmethod
    def dice(gt_seg, prediction_seg):
        """
        compute dice coefficient
        :param gt_seg: ground truth segmentation
        :param prediction_seg: predicted segmentation
        :return: dice coefficient between gt and predictions
        """
        seg1 = np.asarray(gt_seg).astype(np.bool)
        seg2 = np.asarray(prediction_seg).astype(np.bool)

        # Compute Dice coefficient
        intersection = np.logical_and(seg1, seg2)
        if seg1.sum() + seg2.sum() == 0:
            return 1
        return float(format(2. * intersection.sum() / (seg1.sum() + seg2.sum()), '.3f'))

    @staticmethod
    def approximate_diameter(tumor_volume):
        """
        approximate the diameter of a tumor from its volume
        """
        r = ((3 * tumor_volume) / (4 * np.pi)) ** (1 / 3)
        diameter = 2 * r
        return diameter

    @staticmethod
    def mask_by_diameter(gt_shape, voxel_volume, labeled_unique, diameter, operator):
        """
        classifies predicted shapes into diameters
        :param gt_shape: ground truth shape
        :param voxel_volume: voxel volume from nifti's header
        :param labeled_unique: the predicted shape
        :param diameter: diameter to classify
        :param operator: operator
        :return: a list of labels, list of masks and their indices according to diameter
        """
        tumors_with_diameter_list = []
        debug = []
        tumors_with_diameter_mask = np.zeros(gt_shape)
        for i in range(1, labeled_unique[1] + 1):
            current_1_tumor = (labeled_unique[0] == i)
            num_of_voxels = current_1_tumor.sum()
            tumor_volume = num_of_voxels * voxel_volume
            approx_diameter = Utils.approximate_diameter(tumor_volume)
            if operator(approx_diameter, diameter):
                tumors_with_diameter_list.append(i)
                debug.append(num_of_voxels)
                tumors_with_diameter_mask[current_1_tumor] = 1
        tumors_with_diameter_labeled = measure.label(tumors_with_diameter_mask)
        tumors_with_diameter_labeled = tuple((tumors_with_diameter_labeled, tumors_with_diameter_labeled.max()))
        return tumors_with_diameter_labeled, tumors_with_diameter_mask, tumors_with_diameter_list

    @staticmethod
    def segmentation_statistics(tumors_with_diameter_gt, predict_lesions_touches, debug=False, nifti_affine=None):
        seg_tp = (tumors_with_diameter_gt * predict_lesions_touches)
        seg_fp = (predict_lesions_touches - (tumors_with_diameter_gt * predict_lesions_touches))
        seg_fn = (tumors_with_diameter_gt - (tumors_with_diameter_gt * predict_lesions_touches))
        if debug:
            unique_gt = nib.Nifti1Image(seg_fp, nifti_affine.affine)
            nib.save(unique_gt, 'FP.nii.gz')
            unique_gt = nib.Nifti1Image(seg_fn, nifti_affine.affine)
            nib.save(unique_gt, 'FN.nii.gz')

        return seg_tp, seg_fp, seg_fn

    @staticmethod
    def segmentation_statistics_tumors(tumors_with_diameter_gt_matrix, predict_lesions_with_diameter_TP,
                                       tumors_with_diameter_predictions_matrix, gt_lesions_with_diameter_TP,
                                       debug=False, nifti_affine=None):
        seg_TP = (tumors_with_diameter_gt_matrix * predict_lesions_with_diameter_TP)
        seg_FP = tumors_with_diameter_predictions_matrix * (1 - gt_lesions_with_diameter_TP)
        seg_FN = tumors_with_diameter_gt_matrix * (1 - predict_lesions_with_diameter_TP)
        if debug:
            unique_gt = nib.Nifti1Image(seg_FP, nifti_affine)
            nib.save(unique_gt, 'FP.nii.gz')
            unique_gt = nib.Nifti1Image(seg_FN, nifti_affine)
            nib.save(unique_gt, 'FN.nii.gz')

        return seg_TP, seg_FP, seg_FN

    @staticmethod
    def detection_statistics(predict_lesions_touches, tumors_with_diameter_gt_unique,
                             tumors_with_diameter_predictions_matrix_unique):
        detection_tp = len(list(np.unique((predict_lesions_touches * tumors_with_diameter_gt_unique[0])))) - 1
        detection_fp = int(tumors_with_diameter_predictions_matrix_unique[1] - (len(list(np.unique(
            (predict_lesions_touches * tumors_with_diameter_predictions_matrix_unique[0])))) - 1))
        detection_fn = int(tumors_with_diameter_gt_unique[1] - detection_tp)

        try:
            precision = detection_tp / (detection_tp + detection_fp)
        except ZeroDivisionError:
            precision = 1

        try:
            recall = detection_tp / (detection_fn + detection_tp)
        except ZeroDivisionError:
            recall = 1

        return detection_tp, detection_fp, detection_fn, precision, recall

    @staticmethod
    def detection_statistics_tumors(unique_gt_matrix, predict_lesions_with_diameter_TP, tumors_with_diameter_gt_unique,
                                    tumors_with_diameter_predictions_matrix_unique):
        all_gt_tumors = (unique_gt_matrix >= 1)
        detection_TP = len(list(np.unique((tumors_with_diameter_gt_unique[0] * predict_lesions_with_diameter_TP)))) - 1

        detection_FP = int(tumors_with_diameter_predictions_matrix_unique[1] - (len(list(np.unique(
            (all_gt_tumors * tumors_with_diameter_predictions_matrix_unique[0])))) - 1))
        detection_FN = int(tumors_with_diameter_gt_unique[1] - detection_TP)

        try:
            precision = detection_TP / (detection_TP + detection_FP)
        except ZeroDivisionError:
            precision = 1
        try:
            recall = detection_TP / (detection_FN + detection_TP)
        except ZeroDivisionError:
            recall = 1

        return detection_TP, detection_FP, detection_FN, precision, recall


class SegmentationStatsLiver:
    def __init__(self, filename, roi, prediction, gt, threshold, liver_roi=None):
        self.file_name = filename
        self.prediction = prediction
        self.gt_nifti = nib.load(gt)
        self.th = threshold
        self.liver_roi = liver_roi

        # Getting voxel_volume
        self.pix_dims = self.gt_nifti.header.get_zooms()
        self.voxel_volume = self.pix_dims[0] * self.pix_dims[1] * self.pix_dims[2]

        roi_nifti = nib.load(roi)
        roi_temp = roi_nifti.get_fdata()
        self.roi = Utils.get_largest_connected_component(roi_temp)

        self.gt = Utils.get_largest_connected_component(self.gt_nifti.get_fdata())
        pred_nifti = nib.load(prediction)
        self.prediction = pred_nifti.get_fdata()
        self.prediction[self.prediction < self.th] = 0
        self.prediction[self.prediction >= self.th] = 1
        self.prediction = Utils.get_largest_connected_component(self.prediction)
        self.prediction = binary_fill_holes(self.prediction, BINARY_FILLER_MATRIX.astype(self.prediction.dtype))

        # unique lesions for gt and predictions
        self.unique_gt = Utils.get_connected_components(self.gt)
        self.unique_predictions = Utils.get_connected_components(self.prediction)

    def calculate_stats_by_diameter(self, diameter, op=operator.gt, seg_stats_debug=False):
        """
        calculate stats by diameter
        :param diameter: the diameter to calculate stats for
        :param op: the operator to use for the diameter comparison
        :param seg_stats_debug: if true, debug files will be saved
        """
        predict_liver_touches = np.zeros(self.gt.shape)

        # calculate diameter for each connected component in GT
        liver_with_diameter_gt_unique, liver_with_diameter_gt, liver_with_diameter_labels = \
            Utils.mask_by_diameter(self.gt.shape, self.voxel_volume, self.unique_gt, diameter, op)

        # calculate diameter for each connected component in Predictions
        predicted_liver_with_diameter_unique, predicted_liver_with_diameter_matrix, predicted_liver_with_diameter_labels = \
            Utils.mask_by_diameter(self.gt.shape, self.voxel_volume, self.unique_predictions, diameter, op)

        # calculating ASSDs ans Hausdorff metrics
        ASSDs = []
        HDs = []
        for i in liver_with_diameter_labels:
            current_shape = (self.unique_gt[0] == i)
            unique_predictions = list(np.unique((current_shape * predicted_liver_with_diameter_unique[0])))
            unique_predictions.pop(0)
            for j in unique_predictions:
                predict_liver_touches[predicted_liver_with_diameter_unique[0] == j] = 1
                assd_res, hd_res = assd_and_hd(current_shape, predicted_liver_with_diameter_unique[0] == j,
                                               voxelspacing=self.pix_dims, connectivity=2, crop_to_relevant_scope=True)
                ASSDs += [assd_res]
                HDs += [hd_res]

        mean_ASSDs = float(format(np.mean(ASSDs), '.3f')) if ASSDs else np.nan
        mean_HDs = float(format(np.mean(HDs), '.3f')) if HDs else np.nan
        max_HDs = float(format(np.max(HDs), '.3f')) if HDs else np.nan

        # Segmentation statistics
        seg_tp, seg_fp, seg_fn = \
            Utils.segmentation_statistics(liver_with_diameter_gt, predict_liver_touches, debug=seg_stats_debug,
                                          nifti_affine=self.gt_nifti.affine)

        total_liver_gt = float(format((liver_with_diameter_gt > 0).sum() * self.voxel_volume * 0.001, '.3f'))
        total_liver_pred = float(
            format((predicted_liver_with_diameter_matrix > 0).sum() * self.voxel_volume * 0.001, '.3f'))
        liver_cc = self.roi.sum() * self.voxel_volume * 0.001

        if total_liver_gt == 0:
            delta_percentage = 0
        else:
            delta_percentage = ((total_liver_gt - total_liver_pred) / (total_liver_gt + total_liver_pred)) * 100

        return {'Filename': self.file_name,
                'Num_of_lesion': len(liver_with_diameter_labels),
                'Dice': Utils.dice(liver_with_diameter_gt, predict_liver_touches),
                'Segmentation TP (cc)': float(format(seg_tp.sum() * self.voxel_volume * 0.001, '.3f')),
                'Segmentation FP (cc)': float(format(seg_fp.sum() * self.voxel_volume * 0.001, '.3f')),
                'Segmentation FN (cc)': float(format(seg_fn.sum() * self.voxel_volume * 0.001, '.3f')),
                'Total tumor volume GT (cc)': total_liver_gt,
                'Total tumor volume Predictions (cc)': total_liver_pred,
                'Delta between total Liver volumes (cc)': total_liver_gt - total_liver_pred,
                'Delta between total Liver volumes (%)': delta_percentage,
                'Liver Burden GT (%)': float(format(total_liver_gt / liver_cc, '.3f')) * 100,
                'Liver Burden Pred (%)': float(format(total_liver_pred / liver_cc, '.3f')) * 100,
                'Liver Burden Delta (%)': float(format((total_liver_gt - total_liver_pred) / liver_cc, '.3f')) * 100,
                'Mean ASSD (mm)': mean_ASSDs,
                'Mean Hausdorff (mm)': mean_HDs,
                'Max Hausdorff (mm)': max_HDs}


class SegmentationStatsTumors:
    def __init__(self, roi, gt, predictions, file_name, th, pix_dims=None):
        self.file_name = file_name
        if not type(gt) == np.ndarray:
            self.gt_nifti = nib.load(gt)
            self.gt = nib.as_closest_canonical(self.gt_nifti).get_fdata().astype("float64")
        else:
            self.gt = gt
        if not type(predictions) == np.ndarray:
            pred_nifti = nib.load(predictions)
            self.predictions = nib.as_closest_canonical(pred_nifti).get_fdata().astype("float64")
        else:
            self.predictions = predictions
        if roi:
            roi_nifti = nib.load(roi)
            self.roi = roi_nifti.get_fdata().astype("float64")
        else:
            self.roi = np.ones_like(self.gt)
        self.predictions *= self.roi

        # Getting voxel_volume
        if pix_dims:
            self.pix_dims = pix_dims
            self.voxel_volume = self.pix_dims[0] * self.pix_dims[1] * self.pix_dims[2]
        else:
            self.pix_dims = self.gt_nifti.header.get_zooms()
            self.voxel_volume = self.pix_dims[0] * self.pix_dims[1] * self.pix_dims[2]

        # preprocessing on GT and ROI
        self.gt = binary_fill_holes(self.gt, BINARY_FILLER_MATRIX.astype(self.gt.dtype))
        self.gt = remove_small_objects(self.gt.astype(np.bool), min_size=20).astype(self.gt.dtype)
        try:
            self.roi = Utils.get_largest_connected_component(self.roi, connectivity=1)
        except AssertionError:
            print(f"Segmentation {self.file_name} has no liver mask")
            raise
        self.roi = binary_fill_holes(self.roi, BINARY_FILLER_MATRIX.astype(self.roi.dtype))

        self.predictions[self.predictions < th] = 0
        self.predictions[self.predictions >= th] = 1

        # preprocessing on the prediction
        self.predictions = binary_fill_holes(self.predictions, BINARY_FILLER_MATRIX.astype(np.bool))
        self.predictions = remove_small_objects(self.predictions, min_size=20).astype(self.predictions.dtype)

        # unique lesions for gt and predictions
        self.unique_gt = Utils.get_connected_components(self.gt)
        self.unique_predictions = Utils.get_connected_components(self.predictions)

    def calculate_statistics_by_diameter(self, diameter, oper=operator.gt, calculate_ASSD=False,
                                         calculate_HD=False):
        predict_lesions_with_diameter_TP = np.zeros(self.gt.shape)
        gt_lesions_with_diameter_TP = np.zeros(self.gt.shape)
        predict_lesions_TP = np.zeros(self.gt.shape)

        # calculate diameter for each lesion in GT
        tumors_with_diameter_gt_unique, tumors_with_diameter_gt_matrix, tumors_with_diameter_gt = \
            Utils.mask_by_diameter(self.gt.shape, self.voxel_volume, self.unique_gt, diameter, oper)

        # calculate diameter for each lesion in Predictions
        tumors_with_diameter_predictions_matrix_unique, tumors_with_diameter_predictions_matrix, tumors_with_diameter_predictions = \
            Utils.mask_by_diameter(self.gt.shape, self.voxel_volume, self.unique_predictions, diameter, oper)

        # Find predicted tumor that touches 1 tumor of the prediction and calculating ASSDs ans Hausdorff metrics
        ASSDs: List[float] = []
        HDs: List[float] = []
        for i in tumors_with_diameter_gt:
            current_1_tumor_gt = (self.unique_gt[0] == i)
            unique_predictions_touch_current_gt_tumor = np.unique((current_1_tumor_gt * self.unique_predictions[0]))
            unique_predictions_touch_current_gt_tumor = list(
                unique_predictions_touch_current_gt_tumor[unique_predictions_touch_current_gt_tumor != 0])
            if len(unique_predictions_touch_current_gt_tumor) > 0:
                gt_lesions_with_diameter_TP[current_1_tumor_gt] = 1
            for j in unique_predictions_touch_current_gt_tumor:
                current_1_tumor_pred = (self.unique_predictions[0] == j)
                if calculate_HD and calculate_ASSD:
                    _assd, _hd = assd_and_hd(current_1_tumor_gt, current_1_tumor_pred,
                                             voxelspacing=self.pix_dims, connectivity=2)
                    ASSDs += [_assd]
                    HDs += [_hd]
                predict_lesions_TP[current_1_tumor_pred] = 1

        for i in tumors_with_diameter_predictions:
            current_1_tumor_pred = (self.unique_predictions[0] == i)
            unique_gt_touch_current_pred_tumor = np.unique((current_1_tumor_pred * self.unique_gt[0]))
            unique_gt_touch_current_pred_tumor = list(
                unique_gt_touch_current_pred_tumor[unique_gt_touch_current_pred_tumor != 0])
            if len(unique_gt_touch_current_pred_tumor) > 0:
                predict_lesions_with_diameter_TP[current_1_tumor_pred] = 1

        mean_ASSDs = float(format(np.mean(ASSDs), '.3f')) if ASSDs else np.nan
        mean_HDs = float(format(np.mean(HDs), '.3f')) if HDs else np.nan
        max_HDs = float(format(np.max(HDs), '.3f')) if HDs else np.nan

        # Segmentation statistics
        seg_TP, seg_FP, seg_FN = \
            Utils.segmentation_statistics_tumors(tumors_with_diameter_gt_matrix, self.predictions,
                                                 tumors_with_diameter_predictions_matrix, self.gt, debug=False)

        Total_tumor_GT = float(format((tumors_with_diameter_gt_matrix > 0).sum() * self.voxel_volume * 0.001, '.3f'))
        Total_tumor_pred = float(
            format((tumors_with_diameter_predictions_matrix > 0).sum() * self.voxel_volume * 0.001, '.3f'))
        Total_tumor_GT_without_FN = float(
            format((gt_lesions_with_diameter_TP > 0).sum() * self.voxel_volume * 0.001, '.3f'))
        Total_tumor_pred_without_FP = float(
            format((predict_lesions_with_diameter_TP > 0).sum() * self.voxel_volume * 0.001, '.3f'))
        Liver_cc = self.roi.sum() * self.voxel_volume * 0.001

        if (Total_tumor_GT + Total_tumor_pred) == 0:
            delta_percentage = 0
        else:
            delta_percentage = ((Total_tumor_GT - Total_tumor_pred) / (Total_tumor_GT + Total_tumor_pred)) * 100

        if (Total_tumor_GT_without_FN + Total_tumor_pred_without_FP) == 0:
            delta_percentage_TP_only = 0
        else:
            delta_percentage_TP_only = ((Total_tumor_GT_without_FN - Total_tumor_pred_without_FP) / (
                    Total_tumor_GT_without_FN + Total_tumor_pred_without_FP)) * 100

        # Detection statistics
        detection_TP, detection_FP, detection_FN, precision, recall = \
            Utils.detection_statistics_tumors(self.unique_gt[0], self.predictions, tumors_with_diameter_gt_unique,
                                              tumors_with_diameter_predictions_matrix_unique)
        try:
            f1_score = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            f1_score = 0

        return {'Filename': self.file_name,
                'Num of lesion': len(tumors_with_diameter_gt),
                'Dice': Utils.dice(gt_lesions_with_diameter_TP, predict_lesions_TP),
                # 'Dice with FP and FN': self.dice(tumors_with_diameter_gt, tumors_with_diameter_predictions_matrix),
                'Dice with FN': Utils.dice(tumors_with_diameter_gt_matrix, predict_lesions_TP),
                'Segmentation TP (cc)': float(format(seg_TP.sum() * self.voxel_volume * 0.001, '.3f')),
                'Segmentation FP (cc)': float(format(seg_FP.sum() * self.voxel_volume * 0.001, '.3f')),
                'Segmentation FN (cc)': float(format(seg_FN.sum() * self.voxel_volume * 0.001, '.3f')),
                'Total tumor volume GT (cc)': Total_tumor_GT,
                'Total tumor volume Predictions (cc)': Total_tumor_pred,
                'Delta between total tumor volumes (cc)': Total_tumor_GT - Total_tumor_pred,
                'Delta between total tumor volumes (%)': delta_percentage,
                'Delta between total tumor volumes (TP only) (cc)': Total_tumor_GT_without_FN - Total_tumor_pred_without_FP,
                'ABS Delta between total tumor volumes (TP only) (cc)': abs(
                    Total_tumor_GT_without_FN - Total_tumor_pred_without_FP),
                'Delta between total tumor volumes (TP only) (%)': delta_percentage_TP_only,
                'ABS Delta between total tumor volumes (TP only) (%)': abs(delta_percentage_TP_only),
                'Tumor Burden GT (%)': float(format(Total_tumor_GT / Liver_cc, '.3f')) * 100,
                'Tumor Burden Pred (%)': float(format(Total_tumor_pred / Liver_cc, '.3f')) * 100,
                'Tumor Burden Delta (%)': float(format((Total_tumor_GT - Total_tumor_pred) / Liver_cc, '.3f')) * 100,
                'Detection TP (per lesion)': detection_TP,
                'Detection FP (per lesion)': detection_FP,
                'Detection FN (per lesion)': detection_FN,
                'Precision': float(format(precision, '.3f')),
                'Recall': float(format(recall, '.3f')),
                'F1 Score': float(format(f1_score, '.3f')),
                'Mean ASSD (mm)': mean_ASSDs,
                'Mean Hausdorff (mm)': mean_HDs,
                'Max Hausdorff (mm)': max_HDs}
