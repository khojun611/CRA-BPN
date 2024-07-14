import os
import time
import cv2
import pickle
import natsort
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from dataset import TotalText, Ctw1500Text, Icdar15Text, Mlt2017Text, TD500Text
from network.textnet import TextNet
from util.augmentation import BaseTransform
from cfglib.config import config as cfg, update_config, print_config
from cfglib.option import BaseOptions
from util.misc import to_device, mkdirs, rescale_result
from util.eval import deal_eval_total_text, deal_eval_ctw1500, deal_eval_icdar15, \
    deal_eval_TD500, data_transfer_ICDAR, data_transfer_TD500, data_transfer_MLT2017
from using_craft import get14p
import argparse
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

def osmkdir(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

def write_to_file(contours, file_path):
    """
    :param contours: [[x1, y1], [x2, y2]... [xn, yn]]
    :param file_path: target file path
    """
    # according to total-text evaluation method, output file should be formatted to: y0,x0, ..... yn,xn
    with open(file_path, 'w') as f:
        for cont in contours:
            cont = np.stack([cont[:, 0], cont[:, 1]], 1)
            cont = cont.flatten().astype(str).tolist()
            cont = ','.join(cont)
            f.write(cont + '\n')

def get_craft_results(craft_results, i):
    print("i shape",i)
    return craft_results[i]

def inference(model, image_path, output_dir, craft_results, idx):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    transformer = BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
    img_transformed, _ = transformer(img, None)
    img_transformed = torch.from_numpy(img_transformed).permute(2, 0, 1).unsqueeze(0).float()

    input_dict = dict()
    input_dict['img'] = to_device(img_transformed)
    input_dict["craft"] = get_craft_results(craft_results, idx)
    input_dict["meta_HW"] = np.array([h, w])
    print("H",h)
    print("W",w)


    # get detection result
    start = time.time()
    torch.cuda.synchronize()
    output_dict = model(input_dict)
    end = time.time()
    fps = 1.0 / (end - start)
    
    print(f'Inference time: {end - start:.4f}s, FPS: {fps:.2f}')

    contours = output_dict["py_preds"][-1].int().cpu().numpy()
    img_show, contours = rescale_result(img, contours, h, w)

    # write to file
    fname = os.path.basename(image_path).replace('jpg', 'txt')
    write_to_file(contours, os.path.join(output_dir, fname))

def main(image_dir, output_dir):
    osmkdir(output_dir)

    # Model
    model = TextNet(is_training=False, backbone="resnet50")
    model_path = "./model/TextBPN_resnet50_70.pth"

    model.load_model(model_path)
    model = model.to(cfg.device)  # copy to cuda
    model.eval()
    if cfg.cuda:
        cudnn.benchmark = True

    print('Start testing TextBPN.')
    get14p("TextBPN/ctw_test_coord")  # 여기서 변환

    with open("inference.pkl", "rb") as f:
        craft_results = pickle.load(f)
    
    image_files = [os.path.join(image_dir, f) for f in natsort.natsorted(os.listdir(image_dir)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for idx, image_path in enumerate(image_files):
        inference(model, image_path, output_dir, craft_results, idx)

if __name__ == "__main__":
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    print_config(cfg)

    osmkdir(args.output_dir)

    main(args.image_dir, args.output_dir)
