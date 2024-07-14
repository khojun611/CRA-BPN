import numpy as np
import math
import cv2
import copy
import numpy.random as random
import torch

from shapely.geometry import Polygon



class SquarePadding(object):

    def __call__(self, image, polygons=None):

        H, W, _ = image.shape

        if H == W:
            return image, polygons

        padding_size = max(H, W)
        (h_index, w_index) = (np.random.randint(0, H*7//8),np.random.randint(0, W*7//8))
        img_cut = image[h_index:(h_index+H//9),w_index:(w_index+W//9)]
        expand_image = cv2.resize(img_cut,(padding_size, padding_size))
        #expand_image = np.zeros((padding_size, padding_size, 3), dtype=image.dtype)
        #expand_image=img_cut[:,:,:]
        if H > W:
            y0, x0 = 0, (H - W) // 2
        else:
            y0, x0 = (W - H) // 2, 0
        if polygons is not None:
            for polygon in polygons:
                polygon.points += np.array([x0, y0])
        expand_image[y0:y0+H, x0:x0+W] = image
        image = expand_image

        return image, polygons

def crop_first(image, polygons, scale =10):
    polygons_new = copy.deepcopy(polygons)
    h, w, _ = image.shape
    pad_h = h // scale
    pad_w = w // scale
    h_array = np.zeros((h + pad_h * 2), dtype=np.int32)
    w_array = np.zeros((w + pad_w * 2), dtype=np.int32)

    text_polys = []
    pos_polys = []
    for polygon in polygons_new:
        rect = cv2.minAreaRect(polygon.points.astype(np.int32))
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        text_polys.append([box[0], box[1], box[2], box[3]])
        if polygon.label != -1:
            pos_polys.append([box[0], box[1], box[2], box[3]])

    polys = np.array(text_polys, dtype=np.int32)
    for poly in polys:
        poly = np.round(poly, decimals=0).astype(np.int32)  # 四舍五入
        minx = np.min(poly[:, 0])
        maxx = np.max(poly[:, 0])
        w_array[minx + pad_w:maxx + pad_w] = 1
        miny = np.min(poly[:, 1])
        maxy = np.max(poly[:, 1])
        h_array[miny + pad_h:maxy + pad_h] = 1
    # ensure the cropped area not across a text 保证截取区域不会横穿文字
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]
    pp_polys = np.array(pos_polys, dtype=np.int32)

    return h_axis, w_axis, pp_polys


class RandomResizeScale(object):
    def __init__(self, size=512, ratio=(3./4, 5./2)):
        self.size = size
        self.ratio = ratio

    def __call__(self, image, polygons=None):

        aspect_ratio = np.random.uniform(self.ratio[0], self.ratio[1])
        h, w, _ = image.shape
        scales = self.size*1.0/max(h, w)
        aspect_ratio = scales * aspect_ratio
        aspect_ratio = int(w * aspect_ratio)*1.0/w
        image = cv2.resize(image, (int(w * aspect_ratio), int(h*aspect_ratio)))
        scales = np.array([aspect_ratio, aspect_ratio])
        if polygons is not None:
            for polygon in polygons:
                polygon = polygon * scales
                
        return image, polygons

class RandomCropFlip(object):

    def __init__(self, min_crop_side_ratio=0.2):
        self.scale=10
        self.ratio =0.5
        self.epsilon =1e-2
        self.min_crop_side_ratio = min_crop_side_ratio

    def __call__(self, image, polygons=None):

        if polygons is None:
            return image, polygons

        if np.random.random() <= self.ratio:
            return image, polygons

        
        h_axis, w_axis, pp_polys = crop_first(image, polygons, scale =self.scale)
        if len(h_axis) == 0 or len(w_axis) == 0:
            return image, polygons

        # TODO try crop
        attempt = 0
        h, w, _ = image.shape
        area = h * w
        pad_h = h // self.scale
        pad_w = w // self.scale
        while attempt < 10:
            attempt += 1
            polygons_new = []
            xx = np.random.choice(w_axis, size=2)
            xmin = np.min(xx) - pad_w
            xmax = np.max(xx) - pad_w
            xmin = np.clip(xmin, 0, w - 1)
            xmax = np.clip(xmax, 0, w - 1)
            yy = np.random.choice(h_axis, size=2)
            ymin = np.min(yy) - pad_h
            ymax = np.max(yy) - pad_h
            ymin = np.clip(ymin, 0, h - 1)
            ymax = np.clip(ymax, 0, h - 1)
            if (xmax - xmin) * (ymax - ymin) < area *  self.min_crop_side_ratio:
                # area too small
                continue

            pts = np.stack([[xmin, xmax, xmax, xmin], [ymin, ymin, ymax, ymax]]).T.astype(np.int32)
            pp = Polygon(pts).buffer(0)
            Fail_flag = False
            for polygon in polygons:
                ppi = Polygon(polygon.points).buffer(0)
                ppiou = float(ppi.intersection(pp).area)
                if np.abs(ppiou - float(ppi.area)) >self.epsilon  and np.abs(ppiou)> self.epsilon:
                    Fail_flag = True
                    break
                elif np.abs(ppiou - float(ppi.area)) <self.epsilon:
                    polygons_new.append(polygon)
                else:
                    pass

            if Fail_flag:
                continue
            else:
                break

        if len(polygons_new) == 0:
            cropped = image[ymin:ymax, xmin:xmax, :]
            select_type = random.randint(3)
            if select_type == 0:
                img = np.ascontiguousarray(cropped[:, ::-1])
            elif select_type == 1:
                img = np.ascontiguousarray(cropped[::-1, :])
            else:
                img = np.ascontiguousarray(cropped[::-1, ::-1])
            image[ymin:ymax, xmin:xmax, :] = img
            return image, polygons

        else:

            cropped = image[ymin:ymax, xmin:xmax, :]
            height, width, _ = cropped.shape
            select_type = random.randint(3)
            if select_type == 0:
                img = np.ascontiguousarray(cropped[:, ::-1])
                for polygon in polygons_new:
                    polygon.points[:, 0] = width - polygon.points[:, 0] + 2 * xmin
            elif select_type == 1:
                img = np.ascontiguousarray(cropped[::-1, :])
                for polygon in polygons_new:
                    polygon.points[:, 1] = height - polygon.points[:, 1] + 2 * ymin
            else:
                img = np.ascontiguousarray(cropped[::-1, ::-1])
                for polygon in polygons_new:
                    polygon.points[:, 0] = width - polygon.points[:, 0] + 2 * xmin
                    polygon.points[:, 1] = height - polygon.points[:, 1] + 2 * ymin
            image[ymin:ymax, xmin:xmax, :] = img

        return image, polygons
    
class RandomResizedCrop(object):
    def __init__(self, min_crop_side_ratio = 0.2):
        self.scale = 10
        self.epsilon = 1e-2
        self.min_crop_side_ratio = min_crop_side_ratio

    def __call__(self, image, polygons):

        if polygons is None:
            return image, polygons

        # 计算 有效的Crop区域, 方便选取有效的种子点
        h_axis, w_axis, pp_polys = crop_first(image, polygons, scale =self.scale)
        if len(h_axis) == 0 or len(w_axis) == 0:
            return image, polygons

        # TODO try crop
        attempt = 0
        h, w, _ = image.shape
        area = h * w
        pad_h = h // self.scale
        pad_w = w // self.scale
        while attempt < 10:
            attempt += 1
            xx = np.random.choice(w_axis, size=2)
            xmin = np.min(xx) - pad_w
            xmax = np.max(xx) - pad_w
            xmin = np.clip(xmin, 0, w - 1)
            xmax = np.clip(xmax, 0, w - 1)
            yy = np.random.choice(h_axis, size=2)
            ymin = np.min(yy) - pad_h
            ymax = np.max(yy) - pad_h
            ymin = np.clip(ymin, 0, h - 1)
            ymax = np.clip(ymax, 0, h - 1)
            if (xmax - xmin)*(ymax - ymin) <area*self.min_crop_side_ratio:
                # area too small
                continue
            if pp_polys.shape[0] != 0:
                poly_axis_in_area = (pp_polys[:, :, 0] >= xmin) & (pp_polys[:, :, 0] <= xmax) \
                                    & (pp_polys[:, :, 1] >= ymin) & (pp_polys[:, :, 1] <= ymax)
                selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
            else:
                selected_polys = []

            if len(selected_polys) == 0:
                continue
            else:
                pts = np.stack([[xmin, xmax, xmax, xmin], [ymin, ymin, ymax, ymax]]).T.astype(np.int32)
                pp = Polygon(pts).buffer(0)
                polygons_new = []
                Fail_flag = False
                for polygon in copy.deepcopy(polygons):
                    ppi = Polygon(polygon.points).buffer(0)
                    ppiou = float(ppi.intersection(pp).area)
                    if np.abs(ppiou - float(ppi.area)) > self.epsilon and np.abs(ppiou) > self.epsilon:
                        Fail_flag = True
                        break
                    elif np.abs(ppiou - float(ppi.area)) < self.epsilon:
                        # polygon.points -= np.array([xmin, ymin])
                        polygons_new.append(polygon)

                if Fail_flag:
                    continue
                else:
                    cropped = image[ymin:ymax + 1, xmin:xmax + 1, :]
                    for polygon in polygons_new:
                        polygon.points -= np.array([xmin, ymin])

                    return cropped, polygons_new

        return image, polygons
    
class RotatePadding(object):
    def __init__(self, up=60,colors=True):
        self.up = up
        self.colors = colors
        self.ratio = 0.5

    @staticmethod
    def rotate(center, pt, theta, movSize=[0, 0], scale=1):  # 二维图形学的旋转
        (xr, yr) = center
        yr = -yr
        x, y = pt[:, 0], pt[:, 1]
        y = -y

        theta = theta / 180 * math.pi
        cos = math.cos(theta)
        sin = math.sin(theta)

        x = (x - xr) * scale
        y = (y - yr) * scale

        _x = xr + x * cos - y * sin + movSize[0]
        _y = -(yr + x * sin + y * cos) + movSize[1]

        return _x, _y

    @staticmethod
    def shift(size, degree):
        angle = degree * math.pi / 180.0
        width = size[0]
        height = size[1]

        alpha = math.cos(angle)
        beta = math.sin(angle)
        new_width = int(width * math.fabs(alpha) + height * math.fabs(beta))
        new_height = int(width * math.fabs(beta) + height * math.fabs(alpha))

        size = [new_width, new_height]
        return size

    def __call__(self, image, polygons=None, scale=1.0):
        if np.random.random() <= self.ratio:
            return image, polygons
        angle = np.random.normal(loc=0.0, scale=0.5) * self.up  # angle 按照高斯分布
        rows, cols = image.shape[0:2]
        center = (cols / 2.0, rows / 2.0)
        newSize = self.shift([cols * scale, rows * scale], angle)
        movSize = [int((newSize[0] - cols) / 2), int((newSize[1] - rows) / 2)]

        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += int((newSize[0] - cols) / 2)
        M[1, 2] += int((newSize[1] - rows) / 2)

        if self.colors:
            H, W, _ = image.shape
            mask = np.zeros_like(image)
            (h_index, w_index) = (np.random.randint(0, H * 7 // 8), np.random.randint(0, W * 7 // 8))
            img_cut = image[h_index:(h_index + H // 9), w_index:(w_index + W // 9)]
            img_cut = cv2.resize(img_cut, (newSize[0], newSize[1]))
            mask = cv2.warpAffine(mask, M, (newSize[0], newSize[1]), borderValue=[1, 1, 1])
            image = cv2.warpAffine(image, M, (newSize[0], newSize[1]), borderValue=[0,0,0])
            image=image+img_cut*mask
        else:
            color = [0, 0, 0]
            image = cv2.warpAffine(image, M, (newSize[0], newSize[1]), borderValue=color)

        if polygons is not None:
            for polygon in polygons:
                x, y = self.rotate(center, polygon.points, angle,movSize,scale)
                pts = np.vstack([x, y]).T
                polygon.points = pts
        return image, polygons
    
    
class ResizeLimitSquare(object):
    def __init__(self, size=512, ratio=0.6):
        self.size = size
        self.ratio = ratio
        self.SP = SquarePadding()

    def __call__(self, image, polygons=None):
        if np.random.random() <= self.ratio:
            image, polygons = self.SP(image, polygons)
        h, w, _ = image.shape
        image = cv2.resize(image, (self.size,self.size))
        scales = np.array([self.size*1.0/ w, self.size*1.0 / h])

        if polygons is not None:
            for polygon in polygons:
                polygon.points = polygon.points * scales

        return image, polygons
class RandomMirror(object):
    # 镜像
    def __init__(self):
        pass

    def __call__(self, image, polygons=None):
        if polygons is None:
            return image, polygons
        if np.random.randint(2):
            image = np.ascontiguousarray(image[:, ::-1])
            _, width, _ = image.shape
            for polygon in polygons:
                polygon.points[:, 0] = width - polygon.points[:, 0]
        return image, polygons
    
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image, polygons=None):
        image = image.astype(np.float32)
        image /= 255.0
        image -= self.mean
        image /= self.std
        return image, polygons