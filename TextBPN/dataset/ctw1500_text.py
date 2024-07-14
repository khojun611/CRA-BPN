#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = '古溪'
import os
import numpy as np
from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset, TextInstance, TextInstance_craft
from util.io import read_lines
import cv2
import sys
np.random.seed(2020)
# from using_craft_coords import get14p

class Ctw1500Text(TextDataset):

    def __init__(self, data_root, is_training=True, transform=None, ignore_list=None):
        super().__init__(transform, is_training)
        self.data_root = data_root
        self.is_training = is_training

        self.image_root = os.path.join(data_root, 'train' if is_training else 'test', "text_image")
        self.annotation_root = os.path.join(data_root, 'train' if is_training else 'test', "text_label_circum")
        self.craft_root = os.path.join(data_root, 'train' if is_training else 'test', "craft_coords")
        self.image_list = os.listdir(self.image_root)
        self.annotation_list = ['{}'.format(img_name.replace('.jpg', '')) for img_name in self.image_list]
        self.craft_list = ['res_{}.txt'.format(img_name.replace('.jpg', '')) for img_name in self.image_list]
    @staticmethod
    def linear_eq(x1,y1,x2,y2):
        m = (y2-y1)/(x2-x1)
        n = y1 - (m*x1)
        return m,n
    @staticmethod
    def horizantal_point(x1,y1,x2,y2,poly_list):
        if x1 != x2:
            m,n = Ctw1500Text.linear_eq(x1,y1,x2,y2)
            pt_x1 = (int(np.random.uniform(x1,x2)))
            pt_y1 = (m*pt_x1) + n
            pt_x2 = (int(np.random.uniform(pt_x1,x2)))
            pt_y2 = (m*pt_x2) + n 
            pt_x3 = (int(np.random.uniform(pt_x2,x2)))
            pt_y3 = (m*pt_x3) + n 
        else:
            pt_x1 = x1
            pt_y1 = int(np.random.uniform(y1,y2))
            pt_x2 = x1
            pt_y2 = int(np.random.uniform(pt_y1,y2))
            pt_x3 = x1
            pt_y3 = int(np.random.uniform(pt_y2,y2))
        
        
        return poly_list.extend([x1,y1]),poly_list.extend([pt_x1,pt_y1]),poly_list.extend([pt_x2,pt_y2]),poly_list.extend([pt_x3,pt_y3]),poly_list.extend([x2,y2])
    

    @staticmethod
    def vetex_point(x1,y1,x2,y2,poly_list):
        if x1 != x2:
            m,n = Ctw1500Text.linear_eq(x1,y1,x2,y2)
            pt_x1 = (int(np.random.uniform(x1,x2)))
            pt_y1 = (m*pt_x1) + n
            pt_x2 = (int(np.random.uniform(pt_x1,x2)))
            pt_y2 = (m*pt_x2) + n
        else:
            pt_x1 = x1
            pt_y1 = int(np.random.uniform(y1,y2))
            pt_x2 = x1
            pt_y2 = int(np.random.uniform(pt_y1,y2))
        
        
        return poly_list.extend([pt_x1,pt_y1]),poly_list.extend([pt_x2,pt_y2])
    @staticmethod
    def get14p(txt_list):
        poly2 = []
        
            # array1 = np.zeros((14,2))

        
        tl_x = txt_list[0] # top_left
        tl_y = txt_list[1]
        tr_x = txt_list[2] # top_right
        tr_y = txt_list[3]
        br_x = txt_list[4] # bottom right
        br_y = txt_list[5]
        bl_x = txt_list[6] # bottom left
        bl_y = txt_list[7]
        origin = [tl_x,tl_y]

                
        Ctw1500Text.horizantal_point(tl_x,tl_y,tr_x,tr_y,poly2) # top 5 point
        Ctw1500Text.vetex_point(tr_x,tr_y,br_x,br_y,poly2) # right 2 point
        Ctw1500Text.horizantal_point(br_x,br_y,bl_x,bl_y,poly2) # bottom 5 point
        Ctw1500Text.vetex_point(bl_x,bl_y,tl_x,tl_y,poly2)
        
        # print("poly2",poly2)

                
        # poly4 = np.array(poly3).tolist()
        # poly4 = sorted(poly4,key=clockwiseangle_and_distance)
        #poly3[:5] = sorted[poly3[:5],axis=0]
        # print("poly2",poly2)
        return poly2
                
                
            
        
        # return poly2.append((np.array(txt_list[j]).reshape(14,2)).tolist())        
    
    @staticmethod
    def divide_line(x1, y1, x2, y2, divisions):
       
        points = []
        for i in range(1, divisions):
            fraction = i / divisions
            pt_x = x1 + (x2 - x1) * fraction
            pt_y = y1 + (y2 - y1) * fraction
            points.append([pt_x, pt_y])
        return points
    
    @staticmethod
    def expand_polygon(polygon, divisions=3):
        
        if len(polygon) != 8:
            raise ValueError("Polygon must have 4 points (8 coordinates).")

        expanded_polygon = []
        for i in range(0, 8, 2):
            x1, y1 = polygon[i], polygon[i + 1]
            x2, y2 = polygon[(i + 2)], polygon[(i + 3)]
            expanded_polygon.append(x1, y1)
            expanded_polygon.extend(Ctw1500Text.divide_line(x1, y1, x2, y2, divisions))

        return expanded_polygon
    
    
    
    
    @staticmethod
    def using_craft(craft_path):
        
        """
        .mat file parser
        :param gt_path: (str), mat file path
        :return: (list), TextInstance
        """
        lines = read_lines(craft_path)
        craft_coords = []
        for line in lines:
            # line = strs.remove_all(line.strip('\ufeff'), '\xef\xbb\xbf')
            gt = list(map(int, line.split(',')))
            # print(gt)
            # print("craft",gt)
            """
            if len(gt) != 28:
                
                expand_poly = Ctw1500Text.get14p(gt)
                # print("expand_poly",expand_poly)
                pts = np.stack([expand_poly[0::2], expand_poly[1::2]]).T.astype(np.int32)
                
            else:
                pts = np.stack([gt[0::2], gt[1::2]]).T.astype(np.int32)
            """
        # print("craft_pts",pts.shape)
            pts = np.stack([gt[0::2], gt[1::2]]).T.astype(np.int32)
            # print(pts)
            # print("craft_pts",pts)
            craft_coords.append(TextInstance_craft(pts, 'c', "**"))
            # print("craft_coords",craft_coords.points.shape)
            
            # print("craft pts",pts)
        return craft_coords
    
    
    @staticmethod
    def parse_carve_txt(gt_path):
        """
        .mat file parser
        :param gt_path: (str), mat file path
        :return: (list), TextInstance
        """
        lines = read_lines(gt_path + ".txt")
        polygons = []
        for line in lines:
            # line = strs.remove_all(line.strip('\ufeff'), '\xef\xbb\xbf')
            gt = list(map(int, line.split(',')))
            
            # print("gt",gt)
            pts = np.stack([gt[4::2], gt[5::2]]).T.astype(np.int32)
            # print("gt pts",pts)
            pts[:, 0] = pts[:, 0] + gt[0]
            pts[:, 1] = pts[:, 1] + gt[1]
            
            
            polygons.append(TextInstance(pts, 'c', "**"))
        # print("polygons",polygons)
        # print("poygons",polygons)
        return polygons
    
    def pad_shorter_list(list1, list2, pad_value=0):
        # 두 리스트의 길이 차이 계산
        length_diff = abs(len(list1) - len(list2))

        # 두 리스트 중 더 짧은 리스트를 찾아 패딩
        if len(list1) < len(list2):
            list1.extend([pad_value] * length_diff)
        elif len(list2) < len(list1):
            list2.extend([pad_value] * length_diff)

        return list1, list2

    def __getitem__(self, item):

        image_id = self.image_list[item]
        # print("image_id",image_id)
        image_path = os.path.join(self.image_root, image_id)
        print("image_path", image_path)
        # Read image data
        image = pil_load_img(image_path)
        print("img1 shape", image.shape)
        try:
            h, w, c = image.shape
            assert(c == 3)
        except:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.array(image)
        # print("item",item)
        # Read annotation
        annotation_id = self.annotation_list[item]
        # print("annotation_id",annotation_id)
        craft_id = self.craft_list[item]
        # print("craft_id",craft_id)
        annotation_path = os.path.join(self.annotation_root, annotation_id)
        craft_path = os.path.join(self.craft_root, craft_id)
        polygons = self.parse_carve_txt(annotation_path)
        craft_coords = self.using_craft(craft_path)
        print("ctw image shape", image.shape)
        # print(image_id)
        """
        if image_id == "0658.jpg":
            
            print("image_id",image_id)
            print("craft_coord",craft_coords)
            print("annotation_poly",polygons)
        """
        # print("len polygons",len(polygons))
        # print(polygons)
        # print("len craft_coords",len(craft_coords))
        # print(craft_coords)

        if self.is_training:
            return self.get_training_data(image, polygons,craft_coords, image_id=image_id, image_path=image_path)
        else:
            return self.get_test_data(image, polygons, image_id=image_id, image_path=image_path)

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    from util.augmentation import Augmentation, BaseTransform
    from util.misc import regularize_sin_cos
    from nmslib import lanms
    from util.pbox import bbox_transfor_inv, minConnectPath
    from util import canvas as cav
    import time

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = BaseTransform(
        size=640, mean=means, std=stds
    )
    
    trainset = Ctw1500Text(
        data_root='../data/ctw1500',
        is_training=True,
        transform=transform
    )
    

    # img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta = trainset[944]
    for idx in range(0, len(trainset)):
        t0 = time.time()
        img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, gt_roi = trainset[idx]
        img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, gt_roi \
            = map(lambda x: x.cpu().numpy(), (img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, gt_roi))

        img = img.transpose(1, 2, 0)
        img = ((img * stds + means) * 255).astype(np.uint8)
        print(idx, img.shape)
        top_map = radius_map[:, :, 0]
        bot_map = radius_map[:, :, 1]

        print(radius_map.shape)

        sin_map, cos_map = regularize_sin_cos(sin_map, cos_map)
        ret, labels = cv2.connectedComponents(tcl_mask[:, :, 0].astype(np.uint8), connectivity=8)
        cv2.imshow("labels0", cav.heatmap(np.array(labels * 255 / np.max(labels), dtype=np.uint8)))
        print(np.sum(tcl_mask[:, :, 1]))

        t0 = time.time()
        for bbox_idx in range(1, ret):
            bbox_mask = labels == bbox_idx
            text_map = tcl_mask[:, :, 0] * bbox_mask

            boxes = bbox_transfor_inv(radius_map, sin_map, cos_map, text_map, wclip=(2, 8))
            # nms
            boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), 0.25)
            boxes = boxes[:, :8].reshape((-1, 4, 2)).astype(np.int32)
            if boxes.shape[0] > 1:
                center = np.mean(boxes, axis=1).astype(np.int32).tolist()
                paths, routes_path = minConnectPath(center)
                boxes = boxes[routes_path]
                top = np.mean(boxes[:, 0:2, :], axis=1).astype(np.int32).tolist()
                bot = np.mean(boxes[:, 2:4, :], axis=1).astype(np.int32).tolist()

                boundary_point = top + bot[::-1]
                # for index in routes:

                for ip, pp in enumerate(top):
                    if ip == 0:
                        color = (0, 255, 255)
                    elif ip == len(top) - 1:
                        color = (255, 255, 0)
                    else:
                        color = (0, 0, 255)
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 2, color, -1)
                for ip, pp in enumerate(bot):
                    if ip == 0:
                        color = (0, 255, 255)
                    elif ip == len(top) - 1:
                        color = (255, 255, 0)
                    else:
                        color = (0, 255, 0)
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 2, color, -1)
                cv2.drawContours(img, [np.array(boundary_point)], -1, (0, 255, 255), 1)
        # print("nms time: {}".format(time.time() - t0))
        # # cv2.imshow("", img)
        # # cv2.waitKey(0)

        # print(meta["image_id"])
        cv2.imshow('imgs', img)
        cv2.imshow("", cav.heatmap(np.array(labels * 255 / np.max(labels), dtype=np.uint8)))
        cv2.imshow("tr_mask", cav.heatmap(np.array(tr_mask * 255 / np.max(tr_mask), dtype=np.uint8)))
        cv2.imshow("tcl_mask",
                   cav.heatmap(np.array(tcl_mask[:, :, 1] * 255 / np.max(tcl_mask[:, :, 1]), dtype=np.uint8)))
        # cv2.imshow("top_map", cav.heatmap(np.array(top_map * 255 / np.max(top_map), dtype=np.uint8)))
        # cv2.imshow("bot_map", cav.heatmap(np.array(bot_map * 255 / np.max(bot_map), dtype=np.uint8)))
        cv2.waitKey(0)
