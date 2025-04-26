from torchvnnlib import TorchVNNLIB

if __name__ == "__main__":

    converter = TorchVNNLIB(verbose=True)
    file_path = (
        "../../vnncomp2024_benchmarks/benchmarks/cgan_2023/vnnlib/"
        "cGAN_imgSz32_nCh_1_prop_0_input_eps_0.010_output_eps_0.015.vnnlib"
    )
    converter.convert(file_path)
