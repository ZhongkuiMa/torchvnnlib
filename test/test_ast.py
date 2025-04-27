from torchvnnlib.torchvnnlib.ast import *


def parse_vnnlib_file(path: str):
    with open(path) as f:
        lines = f.readlines()
    lines = pre_process_vnnlib(lines)
    tokens_list = tokenize(lines)
    expr = parse(tokens_list)
    expr = optimize(expr)


if __name__ == "__main__":
    file_path = "../data/example.vnnlib"
    # file_path = (
    #     "../../../vnncomp2024_benchmarks/benchmarks/nn4sys_2023/vnnlib/"
    #     "cardinality_0_1_2048.vnnlib"
    # )
    # file_path = (
    #     "../../../vnncomp2024_benchmarks/benchmarks/acasxu_2023/vnnlib/prop_1.vnnlib"
    # )
    # file_path = (
    #     "../../../vnncomp2024_benchmarks/benchmarks/acasxu_2023/vnnlib/prop_5.vnnlib"
    # )
    # file_path = (
    #     "../../../vnncomp2024_benchmarks/benchmarks/linearizenn/vnnlib/"
    #     "prop_10_10_0_io.vnnlib"
    # )
    asts = parse_vnnlib_file(file_path)
