import os
import argparse
import yaml

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

    print(f"Lenght of evaluation data: {len(evaluation_data)}")

    for data in evaluation_data:
        print(f"Data: {data}")

    pass

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