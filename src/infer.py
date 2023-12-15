import torch
import argparse 
import cv2
from transforms import get_transforms
from predict_utils import matrix_to_string


DEVICE = 'cpu'
VOCAB = '0123456789'
WIDTH = 416
HEIGHT = 96
TEXT_SIZE = 13
VOCAB = '0123456789'
MODEL_PATH = './models/ts_script_model/final_ocr.pt'
IMAGE_PATH = "./data/images/000a8eff-08fb-4907-8b34-7a13ca7e37ea--ru.8e3b8a9a-9090-46ba-9c6c-36f5214c606d.jpg"


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=MODEL_PATH, help='Path to the checkpoint file')
    parser.add_argument('--image_path', type=str, default=IMAGE_PATH, help='Path to the image file')
    return parser.parse_args()

if __name__ == '__main__':

    args = arg_parse()
    
    image_path = args.image_path  # Set image_path from command-line argument
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transforms = get_transforms(width=WIDTH, height=HEIGHT, text_size=TEXT_SIZE, vocab=VOCAB, postprocessing=True, augmentations=False)

    model_path = args.model_path
    model = torch.jit.load(model_path, map_location=DEVICE)

    transformed_image = transforms(image=img, text='')['image']
    predict = model(transformed_image[None].to(DEVICE)).cpu().detach()
    string_pred, _ = matrix_to_string(predict, VOCAB)

    pr_text = string_pred[0]
    pr_dict = {'value': pr_text}
    
    print(pr_dict)
