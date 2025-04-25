from torchvnnlib import TorchVNNLIB

if __name__ == "__main__":

    converter = TorchVNNLIB(verbose=True)
    file_path = (
        "../../vnncomp2024_benchmarks/benchmarks/cctsdb_yolo_2023/vnnlib/"
        "spec_onnx_patch-1_idx_00087_1.vnnlib"
    )
    converter.convert(file_path)
