import os.path
import time

from torchvnnlib import TorchVNNLIB


def convert_category(converter: TorchVNNLIB, benchmark_dir: str, target_dir: str):
    instances_csv = os.path.join(benchmark_dir, "instances.csv")
    if not os.path.exists(instances_csv):
        raise FileNotFoundError(f"Instances CSV file not found: {instances_csv}. ")

    # Read the instances from the CSV file
    # The second column contains the vnnlib file paths
    vnnlib_files = []
    with open(instances_csv, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) > 1:
                vnnlib_files.append(parts[1].strip())

    if not vnnlib_files:
        raise ValueError(f"No vnnlib files found in {instances_csv}.")

    print(f"Converting {len(vnnlib_files)} instances in category '{benchmark_dir}'...")
    for vnnlib_file in vnnlib_files:
        vnnlib_file_path = os.path.join(benchmark_dir, vnnlib_file)
        if not os.path.exists(vnnlib_file_path):
            raise FileNotFoundError(f"VNNLIB file not found: {vnnlib_file_path}. ")
        converter.convert(vnnlib_file_path, target_dir)


if __name__ == "__main__":

    dir_path = "../../../../PythonProjects/vnncomp2025_benchmarks/benchmarks"
    categories = [
        "test",
        # "acasxu_2023",
        # "cersyve",
        # "cgan_2023",
        # "cifar100_2024",
        # "collins_rul_cnn_2022",
        # "cora_2024",
        # "dist_shift_2023",
        # "malbeware",
        # "metaroom_2023",
        # "nn4sys",
        # "safenlp_2024",
        # "sat_relu",
        # "soundnessbench",
        # "tinyimagenet_2024",
        # "tllverifybench_2023",
        # "vit_2023",
        # "yolo_2023",
        # "cctsdb_yolo_2023",
        # "collins_aerospace_benchmark",
        # "lsnc_relu",
        # "ml4acopf_2024",
        # "relusplitter",
        # "traffic_signs_recognition_2023",
        # "vggnet16_2023",
    ]
    target_dir_path = ".tmp/vnncomp2025_benchmarks/benchmarks"

    converter = TorchVNNLIB(verbose=False)

    dir_path = os.path.normpath(dir_path)
    for category in categories:
        time_start = time.perf_counter()

        benchmark_dir = os.path.join(dir_path, category)
        target_benchmark_dir = os.path.join(target_dir_path, category)
        convert_category(converter, benchmark_dir, target_benchmark_dir)

        time_end = time.perf_counter()
        elapsed_time = time_end - time_start
        print(f"Category '{category}' converted in {elapsed_time:.2f} seconds.")
