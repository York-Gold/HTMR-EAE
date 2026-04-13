from .dataset2rl import convert_dataset_to_rl
from .alpaca2rl import convert_alpaca_to_rl, json_to_parquet
import argparse

"""
This module provides utilities to convert datasets and Alpaca formatted data into RL format.
If you specify a Parquet file path, the converted data will also be saved in Parquet format.

To convert an Alpaca formatted file to RL format, run:
```sh
python -m utils.data_format alpaca2rl \
    --input_file ... \
    --output_file ... \
    --parquet_file ...  # Optional
```

To convert ACE-E, ACE-Plus, or ERE dev datasets to RL format, run:
```sh
python -m utils.data_format dataset2rl \
    --dataset_type ... \
    --test_data_path ... \
    --result_path ... \
    --parquet_file ...  # Optional
```
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Convert dataset/Aplaca to RL format")

    subparsers = parser.add_subparsers(dest="opration", required=True, help="Operation to perform")

    # Subparser for Alpaca to RL dataset
    alpaca_parser = subparsers.add_parser("alpaca2rl", help="Convert Alpaca format to RL format")
    alpaca_parser.add_argument('--input_file', type=str, required=True, help='Path to the input Alpaca format JSON file')
    alpaca_parser.add_argument('--output_file', type=str, required=True, help='Path to save the converted RL format JSON file') 
    alpaca_parser.add_argument('--parquet_file', type=str, help='Path to save the Parquet file (optional)')

    # Subparser for Dataset to RL dataset
    dataset_parser = subparsers.add_parser("dataset2rl", help="Convert dataset to RL format")
    dataset_parser.add_argument('--dataset_type', type=int, choices=[0,1,2], help='The type of the dataset, 0: ACE-E, 1: ACE-Plus, 2: ERE')
    dataset_parser.add_argument('--result_path', type=str, help='The path to save the converted RL format dataset')
    dataset_parser.add_argument('--test_data_path', type=str, help='The path to the dev file')
    dataset_parser.add_argument('--parquet_file', type=str, help='Path to save the Parquet file (optional)')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    if args.opration == "alpaca2rl":
        rl_data = convert_alpaca_to_rl(args)
    elif args.opration == "dataset2rl":
        rl_data = convert_dataset_to_rl(args)
    
    if args.parquet_file:
        json_to_parquet(rl_data, args.parquet_file)
        print(f"Saved Parquet file to {args.parquet_file}")