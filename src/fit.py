import os
import sys
import time
import numpy as np
from copy import deepcopy
from src.Utils.configuration_parser import *
from src.PreProcessing.pre_processing import run_pre_processing, run_pre_processing_guided
from src.Inference.predictions import run_predictions
from src.Inference.predictions_reconstruction import reconstruct_post_predictions, perform_ensemble
from src.Utils.io import dump_predictions

MODELS_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../', 'resources/models')
print(MODELS_PATH)
sys.path.insert(1, MODELS_PATH)


def fit(input_filename, output_path, selected_model, lungs_mask_filename, anatomical_priors_filename,
        user_runtime=None):
    """

    """
    print("Starting inference for file: {}, with model: {}.\n".format(input_filename, selected_model))
    overall_start = time.time()
    pre_processing_parameters = PreProcessingParser(model_name=selected_model)
    if user_runtime is None:
        user_runtime = RuntimeConfigParser()
    valid_extensions = ['.h5', '.hd5', '.hdf5', '.hdf', '.ckpt']
    model_path = ''
    for e, ext in enumerate(valid_extensions):
        model_path = os.path.join(MODELS_PATH, selected_model, 'model' + ext)
        if os.path.exists(model_path):
            break

    if not os.path.exists(model_path):
        raise ValueError('Could not find any model matching the requested type \'{}\'.'.format(selected_model))

    if 'APG' in selected_model:
        nib_volume, resampled_volume, data, crop_bbox = run_pre_processing_guided(filename=input_filename,
                                                                                  pre_processing_parameters=pre_processing_parameters,
                                                                                  lungs_mask_filename=lungs_mask_filename,
                                                                                  storage_prefix=output_path,
                                                                                  anatomical_priors_filename=anatomical_priors_filename)
    else:
        nib_volume, resampled_volume, data, crop_bbox = run_pre_processing(filename=input_filename,
                                                                           pre_processing_parameters=pre_processing_parameters,
                                                                           lungs_mask_filename=lungs_mask_filename,
                                                                           storage_prefix=output_path)
        data = np.expand_dims(data, axis=-1)
    start = time.time()
    predictions = run_predictions(data=data, model_path=model_path, training_parameters=pre_processing_parameters,
                                  runtime_parameters=user_runtime)
    print('Model loading + inference time: {} seconds.'.format(time.time() - start))

    final_predictions = reconstruct_post_predictions(predictions=predictions, parameters=user_runtime,
                                                     crop_bbox=crop_bbox, nib_volume=nib_volume,
                                                     resampled_volume=resampled_volume)

    dump_predictions(predictions=final_predictions, training_parameters=pre_processing_parameters,
                     runtime_parameters=user_runtime, nib_volume=nib_volume,
                     storage_prefix=output_path)
    print('Total processing time: {:.2f} seconds.\n'.format(time.time() - overall_start))


def fit_ensemble(input_filename, output_path, model_list, lungs_mask_filename, anatomical_priors_filename):
    overall_start = time.time()
    user_runtime = RuntimeConfigParser()

    for model in model_list:
        ensemble_runtime = RuntimeConfigParser()
        ensemble_runtime.set_default_runtime()
        outpath = output_path + model
        fit(input_filename, outpath, model, lungs_mask_filename, anatomical_priors_filename=anatomical_priors_filename,
            user_runtime=user_runtime)

    perform_ensemble(input_filename, output_path, model_list, user_runtime)
    print('Total ensemble processing time: {:.2f} seconds.\n'.format(time.time() - overall_start))
