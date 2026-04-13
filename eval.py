import logging
import argparse

from evaluate import PerformanceEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate model performance")
    parser.add_argument("--result_path", type=str, required=True, help="Path to the result JSON file")
    args = parser.parse_args()

    evaluator = PerformanceEvaluator(args.result_path)
    evaluator.evaluate()