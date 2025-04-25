from torchvnnlib import TorchVNNLIB

if __name__ == "__main__":

    converter = TorchVNNLIB(verbose=True)
    file_path = (
        "../../vnncomp2024_benchmarks/benchmarks/cgan_2023/vnnlib/"
        "cGAN_imgSz32_nCh_3_small_transformer_prop_0_input_eps_0.005_output_eps_0.010.vnnlib"
    )
    converter.convert(file_path)
