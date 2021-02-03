import numpy as np
from copy import deepcopy
from skimage.transform import resize
from scipy.ndimage import binary_fill_holes
from skimage.measure import regionprops
from src.Utils.configuration_parser import *
import subprocess
from nibabel.processing import resample_to_output
from src.Utils.io import load_nifti_volume, convert_and_export_to_nifti


def crop_CT(filepath, volume, lungs_mask_filename, new_spacing, storage_prefix):
    # @TODO. Should name the lungs mask file with a specific name, otherwise will be multiple instances when ensemble...
    if lungs_mask_filename is None or not os.path.exists(lungs_mask_filename):
        script_path = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-2]) + '/main.py'
        #output_prefix = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-2]) + '/tmp'
        subprocess.call(['python3', '{script}'.format(script=script_path),
                         '-i{input}'.format(input=filepath),
                         '-o{output}'.format(output=storage_prefix),
                         '-m{model}'.format(model='CT_Lungs'),
                         '-g{gpu}'.format(gpu=os.environ["CUDA_VISIBLE_DEVICES"])])
        lungs_mask_filename = storage_prefix + '-pred_Lungs.nii.gz'
    else:
        ext_split = lungs_mask_filename.split('.')
        extension = '.'.join(ext_split[1:])

        if extension != 'nii' or extension != 'nii.gz':
            lungs_mask_filename = convert_and_export_to_nifti(input_filepath=lungs_mask_filename)

    lungs_mask_ni = load_nifti_volume(lungs_mask_filename)
    resampled_volume = resample_to_output(lungs_mask_ni, new_spacing, order=0)
    lungs_mask = resampled_volume.get_data().astype('float32')
    lungs_mask[lungs_mask < 0.5] = 0
    lungs_mask[lungs_mask >= 0.5] = 1
    lungs_mask = lungs_mask.astype('uint8')


    lung_region = regionprops(lungs_mask)
    min_row, min_col, min_depth, max_row, max_col, max_depth = lung_region[0].bbox
    print('cropping params', min_row, min_col, min_depth, max_row, max_col, max_depth)

    cropped_volume = volume[min_row:max_row, min_col:max_col, min_depth:max_depth]
    bbox = [min_row, min_col, min_depth, max_row, max_col, max_depth]

    return cropped_volume, bbox


def resize_volume(volume, new_slice_size, slicing_plane, order=1):
    new_volume = None
    if len(new_slice_size) == 2:
        if slicing_plane == 'axial':
            new_val = int(volume.shape[2] * (new_slice_size[1] / volume.shape[1]))
            new_volume = resize(volume, (new_slice_size[0], new_slice_size[1], new_val), order=order)
        elif slicing_plane == 'sagittal':
            new_val = new_slice_size[0]
            new_volume = resize(volume, (new_val, new_slice_size[0], new_slice_size[1]), order=order)
        elif slicing_plane == 'coronal':
            new_val = new_slice_size[0]
            new_volume = resize(volume, (new_slice_size[0], new_val, new_slice_size[1]), order=order)
    elif len(new_slice_size) == 3:
        new_volume = resize(volume, new_slice_size, order=order)
    return new_volume


def intensity_normalization_CT(volume, parameters):
    result = deepcopy(volume).astype('float32')

    result[volume < parameters.intensity_clipping_values[0]] = parameters.intensity_clipping_values[0]
    result[volume > parameters.intensity_clipping_values[1]] = parameters.intensity_clipping_values[1]

    if parameters.normalization_method == 'zeromean':
        mean_val = np.mean(result)
        var_val = np.std(result)
        tmp = (result - mean_val) / var_val
        result = tmp
    else:
        min_val = np.min(result)
        max_val = np.max(result)
        if (max_val - min_val) != 0:
            tmp = (result - min_val) / (max_val - min_val)
            result = tmp

    return result


def intensity_normalization(volume, parameters):
    return intensity_normalization_CT(volume, parameters)


def padding_for_inference(data, slab_size, slicing_plane):
    new_data = data
    if slicing_plane == 'axial':
        missing_dimension = (slab_size - (data.shape[2] % slab_size)) % slab_size
        if missing_dimension != 0:
            new_data = np.pad(data, ((0, 0), (0, 0), (0, missing_dimension), (0, 0)), mode='edge')
    elif slicing_plane == 'sagittal':
        missing_dimension = (slab_size - (data.shape[0] % slab_size)) % slab_size
        if missing_dimension != 0:
            new_data = np.pad(data, ((0, missing_dimension), (0, 0), (0, 0), (0, 0)), mode='edge')
    elif slicing_plane == 'coronal':
        missing_dimension = (slab_size - (data.shape[1] % slab_size)) % slab_size
        if missing_dimension != 0:
            new_data = np.pad(data, ((0, 0), (0, missing_dimension), (0, 0), (0, 0)), mode='edge')

    return new_data, missing_dimension


def padding_for_inference_both_ends(data, slab_size, slicing_plane):
    new_data = data
    padding_val = int(slab_size / 2)
    if slicing_plane == 'axial':
        new_data = np.pad(data, ((0, 0), (0, 0), (padding_val, padding_val), (0, 0)), mode='edge')
    elif slicing_plane == 'sagittal':
        new_data = np.pad(data, ((padding_val, padding_val), (0, 0), (0, 0), (0, 0)), mode='edge')
    elif slicing_plane == 'coronal':
        new_data = np.pad(data, ((0, 0), (padding_val, padding_val), (0, 0), (0, 0)), mode='edge')

    return new_data
