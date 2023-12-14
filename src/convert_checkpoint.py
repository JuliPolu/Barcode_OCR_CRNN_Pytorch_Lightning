import torch
from lightning_module import OCRModule
import argparse 
# import onnx


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--model_path', type=str, default='./models/ts_script_model/final_ocr.pt', help='Path to the *pt file')
    parser.add_argument('--device', type=str, default='cpu', help='specify device')
    return parser.parse_args()
    

if __name__ == '__main__':

    args = arg_parse()
    checkpoint_name = args.checkpoint  # Set checkpoint_name from command-line argument
    core_model = OCRModule.load_from_checkpoint(checkpoint_name, map_location=torch.device(args.device))._model
    core_model.eval()  

    dummy_input = torch.randn(1, 3, 224, 224)
    traced_scripted_model = torch.jit.script(core_model, dummy_input)
    torch.jit.save(traced_scripted_model, args.model_path)