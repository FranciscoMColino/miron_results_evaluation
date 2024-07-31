import os
import argparse
import yaml
import numpy as np

from bbox_evaluation.geom_3ddet_evaluation import geom_evaluate_3ddet_data

def pretty_print_evaluation_results(results):
    print("\nFinal Evaluation Results:")
    for name in results.dtype.names:
        values = results[name]
        print(f"{name.capitalize()}: {['{:.2f}'.format(val) for val in values]}")

def multi_eval_record(config_data, verbose=False):

    evaluation_data = config_data['evaluation_data']

    geometry_mode = config_data['geometry_mode']

    if geometry_mode != '3d':
        raise ValueError("Geometry mode must be 3d")
    
    iou_thresholds = config_data['iou_thresholds']
    camera_position = config_data['camera_position']
    camera_rotation = config_data['camera_rotation']

    print(f"Lenght of evaluation data: {len(evaluation_data)}")

    sims_results = []

    for data in evaluation_data:
        print(f"\nEvaluating data: {data['name_id']}")
        data['iou_thresholds'] = iou_thresholds
        data['camera_position'] = camera_position
        data['camera_rotation'] = camera_rotation
        
        results = geom_evaluate_3ddet_data(data, verbose=False)
        pretty_print_evaluation_results(results)

        sims_results.append(results)

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
            iou_values[i].append(value)

    average_precision = np.array([np.mean(values) for values in precision_values])
    average_recall = np.array([np.mean(values) for values in recall_values])
    average_iou = np.array([np.mean(values) for values in iou_values])

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

    print(f"\n Mean Average Results:")
    pretty_print_evaluation_results(final_evaluation_results)


def main():
    parser = argparse.ArgumentParser(description='Benchmark detection against synthetic data')
    parser.add_argument('config_file', type=str, help='Path to the config file')
    args = parser.parse_args()
    
    # config file in yaml format
    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f"Config file {args.config_file} not found")
    
    with open(args.config_file, 'r') as file:
        config_data = yaml.safe_load(file)

    multi_eval_record(config_data, verbose=True)

if __name__ == "__main__":
    main()