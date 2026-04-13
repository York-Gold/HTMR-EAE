import argparse

from runner.baseline_generation import BaselineGeneration
from config import get_baseline_generation_config, get_baseline_runner_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_type', type=int, choices=[0,1,2], help='The type of the dataset, 0: ACE-E, 1: ACE-Plus, 2: ERE')
    parser.add_argument('--result_path', type=str, required=True, help='The path to save the results')
    parser.add_argument('--log_path', type=str, required=True, default="generation.log", help='The path to save the logs')

    args = parser.parse_args()

    run_config = get_baseline_runner_config(args)
    gen_config = get_baseline_generation_config(args)
    generator = BaselineGeneration(run_config, gen_config)
    generator.generate()