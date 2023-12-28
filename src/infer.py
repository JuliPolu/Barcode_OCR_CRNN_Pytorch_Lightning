import torch
import argparse 
import cv2
from transforms import get_transforms
from predict_utils import matrix_to_string
from src.config import Config


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="config file")
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image file')
    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parse()
    config = Config.from_yaml(args.config_file)
    
    image_path = args.image_path  # Set image_path from command-line argument
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transforms = get_transforms(
        width=config.data_config.width,
        height=config.data_config.height,
        text_size=config.data_config.text_size,
        vocab=config.data_config.vocab,
        postprocessing=True,
        augmentations=False
    )

    model_path = config.weights_path
    model = torch.jit.load(model_path, map_location=config.accelerator)

    transformed_image = transforms(image=img, text='')['image']
    predict = model(transformed_image[None].to(config.accelerator)).cpu().detach()
    string_pred, _ = matrix_to_string(predict, config.data_config.vocab)

    pr_text = string_pred[0]
    pr_dict = {'value': pr_text}

    print(pr_dict)
