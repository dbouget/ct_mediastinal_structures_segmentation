import configparser
import os
import sys


class PreProcessingParser:
    def __init__(self, model_name):
        self.preprocessing_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../',
                                                   'resources/models', model_name, 'pre_processing.ini')
        if not os.path.exists(self.preprocessing_filename):
            raise ValueError('Missing configuration file with pre-processing parameters: {}'.
                             format(self.preprocessing_filename))

        self.pre_processing_config = configparser.ConfigParser()
        self.pre_processing_config.read(self.preprocessing_filename)
        self.__parse_content()

    def __parse_content(self):
        self.__parse_pre_processing_content()
        self.__parse_training_content()

    def __parse_training_content(self):
        self.training_nb_classes = None
        self.training_class_names = None
        self.training_slab_size = None
        self.training_optimal_thresholds = None
        self.training_deep_supervision = False

        if self.pre_processing_config.has_option('Training', 'nb_classes'):
            self.training_nb_classes = int(self.pre_processing_config['Training']['nb_classes'].split('#')[0])

        if self.pre_processing_config.has_option('Training', 'classes'):
            if self.pre_processing_config['Training']['classes'].split('#')[0].strip() != '':
                self.training_class_names = [x.strip() for x in self.pre_processing_config['Training']['classes'].split('#')[0].split(',')]

        if self.pre_processing_config.has_option('Training', 'slab_size'):
            self.training_slab_size = int(self.pre_processing_config['Training']['slab_size'].split('#')[0])

        if self.pre_processing_config.has_option('Training', 'optimal_thresholds'):
            if self.pre_processing_config['Training']['optimal_thresholds'].split('#')[0].strip() != '':
                self.training_optimal_thresholds = [float(x.strip()) for x in self.pre_processing_config['Training']['optimal_thresholds'].split('#')[0].split(',')]

        if self.pre_processing_config.has_option('Training', 'deep_supervision'):
            if self.pre_processing_config['Training']['deep_supervision'].split('#')[0].strip() != '':
                self.training_deep_supervision = True if self.pre_processing_config['Training']['deep_supervision'].split('#')[0].strip().lower() == 'true' else False

    def __parse_pre_processing_content(self):
        self.preprocessing_library = 'nibabel'
        self.output_spacing = None
        self.crop_background = False
        self.intensity_clipping_values = None
        self.intensity_clipping_range = [0.0, 100.0]
        self.intensity_target_range = [0.0, 1.0]
        self.new_axial_size = None
        self.slicing_plane = 'axial'
        self.normalization_method = None

        if self.pre_processing_config.has_option('PreProcessing', 'library'):
            if self.pre_processing_config['PreProcessing']['library'].split('#')[0].strip() == 'dipy':
                self.preprocessing_library = 'dipy'

        if self.pre_processing_config.has_option('PreProcessing', 'output_spacing'):
            if self.pre_processing_config['PreProcessing']['output_spacing'].split('#')[0].strip() != '':
                self.output_spacing = [float(x) for x in self.pre_processing_config['PreProcessing']['output_spacing'].split('#')[0].split(',')]

        if self.pre_processing_config.has_option('PreProcessing', 'intensity_clipping_values'):
            if self.pre_processing_config['PreProcessing']['intensity_clipping_values'].split('#')[0].strip() != '':
                self.intensity_clipping_values = [float(x) for x in self.pre_processing_config['PreProcessing']['intensity_clipping_values'].split('#')[0].split(',')]

        if self.pre_processing_config.has_option('PreProcessing', 'intensity_clipping_range'):
            if self.pre_processing_config['PreProcessing']['intensity_clipping_range'].split('#')[0].strip() != '':
                self.intensity_clipping_range = [float(x) for x in self.pre_processing_config['PreProcessing']['intensity_clipping_range'].split('#')[0].split(',')]

        if self.pre_processing_config.has_option('PreProcessing', 'intensity_final_range'):
            if self.pre_processing_config['PreProcessing']['intensity_final_range'].split('#')[0].strip() != '':
                self.intensity_target_range = [float(x) for x in self.pre_processing_config['PreProcessing']['intensity_final_range'].split('#')[0].split(',')]

        if self.pre_processing_config.has_option('PreProcessing', 'background_cropping'):
            if self.pre_processing_config['PreProcessing']['background_cropping'].split('#')[0].strip() != '':
                self.crop_background = True if self.pre_processing_config['PreProcessing']['background_cropping'].split('#')[0].lower()\
                                           == 'true' else False

        if self.pre_processing_config.has_option('PreProcessing', 'new_axial_size'):
            if self.pre_processing_config['PreProcessing']['new_axial_size'].split('#')[0].strip() != '':
                self.new_axial_size = [int(x) for x in self.pre_processing_config['PreProcessing']['new_axial_size'].split('#')[0].split(',')]

        if self.pre_processing_config.has_option('PreProcessing', 'slicing_plane'):
            if self.pre_processing_config['PreProcessing']['slicing_plane'].split('#')[0].strip() != '':
                self.slicing_plane = self.pre_processing_config['PreProcessing']['slicing_plane'].split('#')[0]

        if self.pre_processing_config.has_option('PreProcessing', 'normalization_method'):
            self.normalization_method = self.pre_processing_config['PreProcessing']['normalization_method'].split('#')[0]


class RuntimeConfigParser:
    def __init__(self):
        self.runtime_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../', 'resources/data',
                                             'runtime_config.ini')
        if not os.path.exists(self.runtime_filename):
            raise ValueError('Missing configuration file with runtime parameters: {}'.
                             format(self.runtime_filename))

        self.runtime_config = configparser.ConfigParser()
        self.runtime_config.read(self.runtime_filename)
        self.__parse_content()

    def __parse_content(self):
        self.predictions_non_overlapping = True
        self.predictions_reconstruction_method = None
        self.predictions_reconstruction_order = None
        self.predictions_probability_thresholds = [0.5]

        if self.runtime_config.has_option('Predictions', 'non_overlapping'):
            self.predictions_non_overlapping = True if self.runtime_config['Predictions']['non_overlapping'].split('#')[0].lower().strip()\
                                                   == 'true' else False

        if self.runtime_config.has_option('Predictions', 'reconstruction_method'):
            self.predictions_reconstruction_method = self.runtime_config['Predictions']['reconstruction_method'].split('#')[0].strip()

        if self.runtime_config.has_option('Predictions', 'reconstruction_order'):
            self.predictions_reconstruction_order = self.runtime_config['Predictions']['reconstruction_order'].split('#')[0].strip()

        if self.runtime_config.has_option('Predictions', 'probability_threshold'):
            if self.runtime_config['Predictions']['probability_threshold'].split('#')[0].strip() != '':
                self.predictions_probability_thresholds = [float(x.strip()) for x in self.runtime_config['Predictions']['probability_threshold'].split('#')[0].split(',')]

    def set_default_runtime(self):
        self.predictions_non_overlapping = True
        self.predictions_reconstruction_method = 'probabilities'
        self.predictions_reconstruction_order = 'resample_first'
        self.predictions_probability_thresholds = [0.5]
