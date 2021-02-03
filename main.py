import getopt
import os
import sys
from src.fit import fit, fit_ensemble


def main(argv):
    input_filename = ''
    output_prefix = ''
    model_list = ''
    lungs_mask_filename = None
    apg_mask_filename = None
    gpu_id = '-1'
    try:
        opts, args = getopt.getopt(argv, "hi:o:m:l:a:g:", ["Input=", "Output=", "Model=", "Lungs=", "APG=", "GPU="])
    except getopt.GetoptError:
        print('usage: main.py --Input <CT volume path> --Output <Results output path> --Model <Inference model name>'
              ' --Lungs <Lung mask filepath> --APG <Anatomical mask filename> --GPU <GPU id>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('main.py --Input <CT volume path> --Output <Results output path> --Model <Inference model name>'
                  ' --Lungs <Lung mask filepath> --APG <Anatomical mask filename> --GPU <GPU id>')
            sys.exit()
        elif opt in ("-i", "--Input"):
            input_filename = arg
        elif opt in ("-o", "--Output"):
            output_prefix = arg
        elif opt in ("-m", "--Model"):
            model_list = arg
        elif opt in ("-l", "--Lungs"):
            lungs_mask_filename = arg
        elif opt in ("-a", "--APG"):
            apg_mask_filename = arg
        elif opt in ("-g", "--GPU"):
            if arg.isnumeric():
                gpu_id = arg
    if input_filename == '':
        print('usage: main.py --Input <CT volume path> --Output <Results output path> --Model <Inference model name>'
              ' --Lungs <Lung mask filepath> --APG <Anatomical mask filename> --GPU <GPU id>')
        sys.exit()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    if os.path.exists(input_filename):
        real_path = os.path.realpath(os.path.dirname(input_filename))
        input_filename = os.path.join(real_path, os.path.basename(input_filename))
    else:
        print('Input filename does not exist on disk, with argument: {}'.format(input_filename))
        sys.exit(2)

    if os.path.exists(os.path.dirname(output_prefix)):
        real_path = os.path.realpath(os.path.dirname(output_prefix))
        output_prefix = os.path.join(real_path, os.path.basename(output_prefix))
    else:
        print('Directory name for the output prefix does not exist on disk, with argument: {}'.format(input_filename))
        sys.exit(2)

    model_list = model_list.split(',')
    if len(model_list) == 1:
        fit(input_filename=input_filename, output_path=output_prefix, selected_model=model_list[0],
            lungs_mask_filename=lungs_mask_filename, anatomical_priors_filename=apg_mask_filename)
    else:
        fit_ensemble(input_filename=input_filename, output_path=output_prefix, model_list=model_list,
                     lungs_mask_filename=lungs_mask_filename, anatomical_priors_filename=apg_mask_filename)


if __name__ == "__main__":
    main(sys.argv[1:])

