import csv
import glob
import shutil
from funcitons import *
from model.utils import *
from PIL import Image, ImageDraw, ImageFont
import json
import os.path as osp
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet50, resnet101


class args_eval():
    def __init__(self):
        self.data_location = "data/images"
        self.nms_thresh = 0.3
        self.prob_thresh = 0.03
        self.checkpoint = "weights/checkpoint_50.pth"
        self.results_dir = "data/output_faces"
        self.template_file = "data/templates.json"


def main():
    args = args_eval()

    ## getting templates
    templates = json.load(open(args.template_file))
    json.dump(templates, open(args.template_file, "w"))
    templates = np.round_(np.array(templates), decimals=8)
    num_templates = templates.shape[0]

    ## getting transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_transforms = transforms.Compose([transforms.ToTensor(), normalize])

    ## get model
    model = get_model(args.checkpoint, num_templates=num_templates)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    rf = {
        'size': [859, 859],
        'stride': [8, 8],
        'offset': [-1, -1]
    }

    # get all images to be processed
    images = list(glob.glob(args.data_location + '/*.jpg'))

    for img in images:

        # process image
        img_tensor = transforms.functional.to_tensor(Image.open(img).convert('RGB'))
        dets = get_detections(model, img_tensor, templates, rf, val_transforms,
                              prob_thresh=args.prob_thresh, nms_thresh=args.nms_thresh, device=device)
        savescores(dets, "results/img02/scores.csv")
        cut_bboxes(dets, args.results_dir + "unqiue", Image.open(img))

        # cut image into processed
        shutil.move(img, args.data_location + "/processed")


if __name__ == "__main__":
    main()
