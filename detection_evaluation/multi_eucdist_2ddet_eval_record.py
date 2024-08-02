import os
import argparse
import yaml
import numpy as np
import math
import datetime

from detection_evaluation.eucdist_2ddet_evaluation import eucdist_evaluate_2ddet_data

def pretty_str_evaluation_results(results):
    result_str = "\nFinal Evaluation Results:\n"
    for name in results.dtype.names:
        values = results[name]
        result_str += f"{name.capitalize()}: {['{:.2f}'.format(val) for val in values]}\n"
    return result_str

def print_log(message, log_file=None):
    print(message)
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(message + '\n')

def multi_eval_record(config_data):

    results_recording_enabled = config_data['results_recording']['enabled']
    results_recording_output_dir = config_data['results_recording']['output_dir']

    logging_file = None

    if results_recording_enabled:
        if not os.path.exists(results_recording_output_dir):
            os.makedirs(results_recording_output_dir)
        elif not os.path.isdir(results_recording_output_dir):
            raise NotADirectoryError(f"Results recording output directory {results_recording_output_dir} is not a directory")
        else:
            raise FileExistsError(f"Results recording output directory {results_recording_output_dir} already exists")
        
        logging_file = os.path.join(results_recording_output_dir, 'log.txt')
        config_copy_file = os.path.join(results_recording_output_dir, 'config_copy.yaml')
        metadata_file = os.path.join(results_recording_output_dir, 'metadata.yaml')

        with open(config_copy_file, 'w') as file:
            yaml.dump(config_data, file)

        with open(metadata_file, 'w') as file:
            metadata = {
                'date': datetime.datetime.now().strftime("%Y-%m-%d"),
                'time': datetime.datetime.now().strftime("%H:%M:%S")
            }
            yaml.dump(metadata, file)
        
        print_log(f"Results recording enabled. Results will be saved to: {results_recording_output_dir}", logging_file)
    else:
        print_log("Results recording disabled", logging_file)

    evaluation_data = config_data['evaluation_data']

    geometry_mode = config_data['geometry_mode']
    
    euclidean_distance_thresholds = config_data['euclidean_distance_thresholds']
    camera_position = config_data['camera_position']
    camera_rotation = config_data['camera_rotation']

    evaluation_data_name_ids = [data['name_id'] for data in evaluation_data]

    print_log(f"\nEvaluating data: {evaluation_data_name_ids}", logging_file)

    sims_results = []

    for data in evaluation_data:
        print_log(f"\nEvaluating data: {data['name_id']}", logging_file)
        data['euclidean_distance_thresholds'] = euclidean_distance_thresholds
        data['camera_position'] = camera_position
        data['camera_rotation'] = camera_rotation
        data['geometry_mode'] = geometry_mode
        
        results = eucdist_evaluate_2ddet_data(data, verbose=False)
        results_str = pretty_str_evaluation_results(results)
        print_log(results_str, logging_file)

        sims_results.append(results)

    # calculate average results for all data

    precision_values = [[] for _ in euclidean_distance_thresholds]
    recall_values = [[] for _ in euclidean_distance_thresholds]
    translation_error_values = [[] for _ in euclidean_distance_thresholds]
    scale_error_values = [[] for _ in euclidean_distance_thresholds]

    for results in sims_results:
        precision = results['precision']
        for i, value in enumerate(precision):
            precision_values[i].append(value)
        recall = results['recall']
        for i, value in enumerate(recall):
            recall_values[i].append(value)
        translation_error = results['translation_error']
        for i, value in enumerate(translation_error):
            if not math.isnan(value):
                translation_error_values[i].append(value)
        scale_error = results['scale_error']
        for i, value in enumerate(scale_error):
            if not math.isnan(value):
                scale_error_values[i].append(value)

    average_precision = np.array([np.mean(values) if len(values) > 0 else 0.0 for values in precision_values])
    average_recall = np.array([np.mean(values) if len(values) > 0 else 0.0 for values in recall_values])
    average_translation_error = np.array([np.mean(values) if len(values) > 0 else 0.0 for values in translation_error_values])
    average_scale_error = np.array([np.mean(values) if len(values) > 0 else 0.0 for values in scale_error_values])

    evaluation_results_dtype = [
        ('thresholds', 'f4', len(euclidean_distance_thresholds)),
        ('precision', 'f4', len(euclidean_distance_thresholds)),
        ('recall', 'f4', len(euclidean_distance_thresholds)),
        ('translation_error', 'f4', len(euclidean_distance_thresholds)),
        ('scale_error', 'f4', len(euclidean_distance_thresholds))
    ]

    final_evaluation_results = np.array((
        euclidean_distance_thresholds,
        average_precision,
        average_recall,
        average_translation_error,
        average_scale_error
    ), dtype=evaluation_results_dtype)

    print_log(f"\n Mean Average Results:", logging_file)
    print_log(pretty_str_evaluation_results(final_evaluation_results), logging_file)


def main():
    parser = argparse.ArgumentParser(description='Benchmark detection against synthetic data')
    parser.add_argument('config_file', type=str, help='Path to the config file')
    args = parser.parse_args()
    
    # config file in yaml format
    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f"Config file {args.config_file} not found")
    
    with open(args.config_file, 'r') as file:
        config_data = yaml.safe_load(file)

    multi_eval_record(config_data)

if __name__ == "__main__":
    main()