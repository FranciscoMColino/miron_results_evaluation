import os
import argparse
import yaml
import numpy as np
import pandas as pd
import datetime

from detection_evaluation.geom_3ddet_evaluation import geom_evaluate_3ddet_data

FIXED_PRETTY_PRINT_LENGTH = 20

def pretty_str_evaluation_results(results, fixed_length=FIXED_PRETTY_PRINT_LENGTH):
    result_str = "\nFinal Evaluation Results:\n"
    for name in results.dtype.names:
        values = results[name]
        name_str = f"{name.capitalize()}:".ljust(fixed_length)
        result_str += f"{name_str} {['{:.2f}'.format(val) for val in values]}\n"
    return result_str

def print_log(message, log_file=None):
    print(message)
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(message + '\n')

def calculate_ap(recall, precision):
    recall = np.array(recall)
    precision = np.array(precision)
    
    # Append sentinel values at the end
    recall = np.concatenate(([0], recall, [1]))
    precision = np.concatenate(([0], precision, [0]))
    
    # Ensure precision is non-increasing
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    
    # Calculate AP
    ap = 0.0
    for i in range(1, len(recall)):
        ap += (recall[i] - recall[i - 1]) * precision[i]
    
    return ap

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
        # csv file for each of the metrics
        precision_file = os.path.join(results_recording_output_dir, 'precision.csv')
        recall_file = os.path.join(results_recording_output_dir, 'recall.csv')
        iou_file = os.path.join(results_recording_output_dir, 'iou.csv')

        index_row = np.concatenate((['name_id'], config_data['iou_thresholds']))

        with open(precision_file, 'w') as file:
            file.write('sep=,\n')
        with open(recall_file, 'w') as file:
            file.write('sep=,\n')
        with open(iou_file, 'w') as file:
            file.write('sep=,\n')

        precision_df = pd.DataFrame(columns=index_row)
        recall_df = pd.DataFrame(columns=index_row)
        iou_df = pd.DataFrame(columns=index_row)

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

    if geometry_mode != '3d':
        raise ValueError("Geometry mode must be 3d")
    
    iou_thresholds = config_data['iou_thresholds']
    iou_thresholds = np.sort(np.array(iou_thresholds))
    camera_position = config_data['camera_position']
    camera_rotation = config_data['camera_rotation']

    evaluation_data_name_ids = [data['name_id'] for data in evaluation_data]

    print_log(f"\nEvaluating data: {evaluation_data_name_ids}", logging_file)

    sims_results = []
    sims_ap_values = [[] for _ in iou_thresholds]
 

    for data in evaluation_data:
        print_log(f"\nEvaluating data: {data['name_id']}", logging_file)
        data['iou_thresholds'] = iou_thresholds
        data['camera_position'] = camera_position
        data['camera_rotation'] = camera_rotation
        
        results = geom_evaluate_3ddet_data(data, verbose=False)
        results_str = pretty_str_evaluation_results(results)
        print_log(results_str, logging_file)

        precision = results['precision']
        recall = results['recall']

        for i in range(0, len(iou_thresholds)):
            c_recall = recall[0:i+1]
            c_precision = precision[0:i+1]
            ap = calculate_ap(c_recall, c_precision)
            sims_ap_values[i].append(ap)

        #{['{:.2f}'.format(ap[-1]) for ap in sims_ap_values]}
        name_str = "Average Precision:".ljust(FIXED_PRETTY_PRINT_LENGTH)
        print_log(f"{name_str} {['{:.2f}'.format(ap[-1]) for ap in sims_ap_values]}", logging_file)
        
        sims_results.append(results)

        if results_recording_enabled:
            precision_values = np.concatenate(([data['name_id']], results['precision']))
            recall_values = np.concatenate(([data['name_id']], results['recall']))
            iou_values = np.concatenate(([data['name_id']], results['iou']))

            precision_df = pd.concat([precision_df, pd.DataFrame([precision_values], columns=index_row)], ignore_index=True)
            recall_df = pd.concat([recall_df, pd.DataFrame([recall_values], columns=index_row)], ignore_index=True)
            iou_df = pd.concat([iou_df, pd.DataFrame([iou_values], columns=index_row)], ignore_index=True)

    # calculate average results for all data

    precision_values = [[] for _ in iou_thresholds]
    recall_values = [[] for _ in iou_thresholds]
    iou_values = [[] for _ in iou_thresholds]

    for results in sims_results:
        precision = results['precision']
        for i, value in enumerate(precision):
            precision_values[i].append(value)
        recall = results['recall']
        for i, value in enumerate(recall):
            recall_values[i].append(value)
        iou = results['iou']
        for i, value in enumerate(iou):
            if value > 0:
                iou_values[i].append(value)

    average_precision = np.array([np.mean(values) for values in precision_values])
    average_recall = np.array([np.mean(values) for values in recall_values])
    average_iou = np.array([np.mean(values) for values in iou_values])
    mean_ap = np.array([np.mean(values) if len(values) > 0 else 0.0 for values in sims_ap_values])

    if results_recording_enabled:
        precision_df = pd.concat([precision_df, pd.DataFrame([['mean'] + list(average_precision)], columns=index_row)], ignore_index=True)
        recall_df = pd.concat([recall_df, pd.DataFrame([['mean'] + list(average_recall)], columns=index_row)], ignore_index=True)
        iou_df = pd.concat([iou_df, pd.DataFrame([['mean'] + list(average_iou)], columns=index_row)], ignore_index=True)

        precision_df.to_csv(precision_file, index=False, mode='a')
        recall_df.to_csv(recall_file, index=False, mode='a')
        iou_df.to_csv(iou_file, index=False, mode='a')

    evaluation_results_dtype = [
        ('thresholds' , 'f4', len(iou_thresholds)),
        ('precision', 'f4', len(iou_thresholds)),
        ('recall', 'f4', len(iou_thresholds)),
        ('iou', 'f4', len(iou_thresholds)),
    ]

    final_evaluation_results = np.array((
        iou_thresholds,
        average_precision,
        average_recall,
        average_iou
    ), dtype=evaluation_results_dtype)

    print_log(f"\n Mean Average Results:", logging_file)
    print_log(pretty_str_evaluation_results(final_evaluation_results), logging_file)

    name_str = "Average Precision:".ljust(FIXED_PRETTY_PRINT_LENGTH)
    print_log(f"{name_str} {['{:.2f}'.format(ap) for ap in mean_ap]}", logging_file)



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