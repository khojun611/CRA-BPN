import copy
import cv2
import torch
import numpy as np
from PIL import Image
from scipy import ndimage as ndimg
from cfglib.config import config as cfg
from util.misc import find_bottom, find_long_edges, split_edge_seqence, \
    vector_sin, get_sample_point
np.random.seed(2020)

def pil_load_img(path):
    image = Image.open(path)
    image = np.array(image)
    return image


class TextInstance_craft(object):
    def __init__(self, points, orient, text):
        self.orient = orient
        self.text = text
        self.bottoms = None
        self.e1 = None
        self.e2 = None
        if self.text != "#":
            self.label = 1
        else:
            self.label = -1

        # remove_points = []
        
        self.points = np.array(points)

    def find_bottom_and_sideline(self):
        self.bottoms = find_bottom(self.points)  # find two bottoms of this Text
        self.e1, self.e2 = find_long_edges(self.points, self.bottoms)  # find two long edge sequence

    def get_sample_point(self, size=None):
        mask = np.zeros(size, np.uint8)
        cv2.fillPoly(mask, [self.points.astype(np.int32)], color=(1,))
        control_points = get_sample_point(mask, cfg.num_points, cfg.approx_factor)

        return control_points

    def get_control_points(self, size=None):
        n_disk = cfg.num_control_points // 2 - 1
        sideline1 = split_edge_seqence(self.points, self.e1, n_disk)
        sideline2 = split_edge_seqence(self.points, self.e2, n_disk)[::-1]
        if sideline1[0][0] > sideline1[-1][0]:
            sideline1 = sideline1[::-1]
            sideline2 = sideline2[::-1]
        p1 = np.mean(sideline1, axis=0)
        p2 = np.mean(sideline2, axis=0)
        vpp = vector_sin(p1 - p2)
        if vpp >= 0:
            top = sideline2
            bot = sideline1
        else:
            top = sideline1
            bot = sideline2

        control_points = np.concatenate([np.array(top), np.array(bot[::-1])], axis=0).astype(np.float64)

        return control_points

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, item):
        return getattr(self, item)



class TextInstance(object):
    def __init__(self, points, orient, text):
        self.orient = orient
        self.text = text
        self.bottoms = None
        self.e1 = None
        self.e2 = None
        if self.text != "#":
            self.label = 1
        else:
            self.label = -1

        remove_points = []
        if len(points) > 4:
            # remove point if area is almost unchanged after removing it
            ori_area = cv2.contourArea(points)
            for p in range(len(points)):
                # attempt to remove p
                index = list(range(len(points)))
                index.remove(p)
                area = cv2.contourArea(points[index])
                if np.abs(ori_area - area)/ori_area < 0.0017 and len(points) - len(remove_points) > 4:
                    remove_points.append(p)
            
            self.points = np.array([point for i, point in enumerate(points) if i not in remove_points])
            # print("if > 4",self.points.shape)
        else:
            self.points = np.array(points)
            # print("if < 4",self.points.shape)
            
            

    def find_bottom_and_sideline(self):
        self.bottoms = find_bottom(self.points)  # find two bottoms of this Text
        self.e1, self.e2 = find_long_edges(self.points, self.bottoms)  # find two long edge sequence

    def get_sample_point(self, size=None):
        mask = np.zeros(size, np.uint8)
        cv2.fillPoly(mask, [self.points.astype(np.int32)], color=(1,))
        control_points = get_sample_point(mask, cfg.num_points, cfg.approx_factor)

        return control_points

    def get_control_points(self, size=None):
        n_disk = cfg.num_control_points // 2 - 1
        sideline1 = split_edge_seqence(self.points, self.e1, n_disk)
        sideline2 = split_edge_seqence(self.points, self.e2, n_disk)[::-1]
        if sideline1[0][0] > sideline1[-1][0]:
            sideline1 = sideline1[::-1]
            sideline2 = sideline2[::-1]
        p1 = np.mean(sideline1, axis=0)
        p2 = np.mean(sideline2, axis=0)
        vpp = vector_sin(p1 - p2)
        if vpp >= 0:
            top = sideline2
            bot = sideline1
        else:
            top = sideline1
            bot = sideline2

        control_points = np.concatenate([np.array(top), np.array(bot[::-1])], axis=0).astype(np.float64)

        return control_points

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, item):
        return getattr(self, item)


class TextDataset(object):

    def __init__(self, transform, is_training=False):
        super().__init__()
        self.transform = transform
        self.is_training = is_training
        self.min_text_size = 4
        self.jitter = 0.8
        self.th_b = 0.4
        self.random_states = {}

    @staticmethod
    def sigmoid_alpha(x, k):
        betak = (1 + np.exp(-k)) / (1 - np.exp(-k))
        dm = max(np.max(x), 0.0001)
        res = (2 / (1 + np.exp(-x * k / dm)) - 1) * betak
        return np.maximum(0, res)

    @staticmethod
    def fill_polygon(mask, pts, value):
        """
        fill polygon in the mask with value
        :param mask: input mask
        :param pts: polygon to draw
        :param value: fill value
        """
        # cv2.drawContours(mask, [polygon.astype(np.int32)], -1, value, -1)
        cv2.fillPoly(mask, [pts.astype(np.int32)], color=(value,))
        # rr, cc = drawpoly(polygon[:, 1], polygon[:, 0], shape=(mask.shape[0],mask.shape[1]))
        # mask[rr, cc] = value

    @staticmethod
    def generate_proposal_point(text_mask, num_points, approx_factor, jitter=0.0, distance=10.0):
        # get  proposal point in contours
        h, w = text_mask.shape[0:2]
        contours, _ = cv2.findContours(text_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        epsilon = approx_factor * cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], epsilon, True).reshape((-1, 2))
        pts_num = approx.shape[0]
        e_index = [(i, (i + 1) % pts_num) for i in range(pts_num)]
        ctrl_points = split_edge_seqence(approx, e_index, num_points)
        ctrl_points = np.array(ctrl_points[:num_points, :]).astype(np.int32)

        if jitter > 0:
            x_offset = (np.random.rand(ctrl_points.shape[0]) - 0.5) * distance*jitter
            y_offset = (np.random.rand(ctrl_points.shape[0]) - 0.5) * distance*jitter
            ctrl_points[:, 0] += x_offset.astype(np.int32)
            ctrl_points[:, 1] += y_offset.astype(np.int32)
        ctrl_points[:, 0] = np.clip(ctrl_points[:, 0], 1, w - 2)
        ctrl_points[:, 1] = np.clip(ctrl_points[:, 1], 1, h - 2)
        return ctrl_points

    @staticmethod
    def compute_direction_field(inst_mask, h, w):
        _, labels = cv2.distanceTransformWithLabels(inst_mask, cv2.DIST_L2,
                                                    cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL)
        # # compute the direction field
        index = np.copy(labels)
        index[inst_mask > 0] = 0
        place = np.argwhere(index > 0)
        nearCord = place[labels - 1, :]
        y = nearCord[:, :, 0]
        x = nearCord[:, :, 1]
        nearPixel = np.zeros((2, h, w))
        nearPixel[0, :, :] = y
        nearPixel[1, :, :] = x
        grid = np.indices(inst_mask.shape)
        grid = grid.astype(float)
        diff = nearPixel - grid

        return diff

    def make_text_region(self, img, polygons, craft_coords, image_id):
        h, w = img.shape[0], img.shape[1]
        mask_zeros = np.zeros(img.shape[:2], np.uint8)

        train_mask = np.ones((h, w), np.uint8)
        tr_mask = np.zeros((h, w), np.uint8)
        weight_matrix = np.zeros((h, w), dtype=np.float64)
        direction_field = np.zeros((2, h, w), dtype=np.float64)
        distance_field = np.zeros((h, w), np.float64)

        gt_points = np.zeros((cfg.max_annotation, cfg.num_points, 2), dtype=np.float64)
        proposal_points = np.zeros((cfg.max_annotation, cfg.num_points, 2), dtype=np.float64)
        proposal_crafts = np.zeros((cfg.max_annotation, cfg.num_points, 2), dtype=np.float64)
        ignore_tags = np.zeros((cfg.max_annotation,), dtype=np.int64)

        
        if polygons is None:
            return train_mask, tr_mask, \
                   distance_field, direction_field, \
                   weight_matrix, gt_points, proposal_points, ignore_tags
        # print("len polygons",len(polygons))
        # print(polygons)
        # print("len craft_coords",len(craft_coords))
        # print(craft_coords)
        # proposal_crafts = []
        
        for idx, craft_coord in enumerate(craft_coords):
            
            craft_coord.points[:,0] = np.clip(craft_coord.points[:,0], 1, w - 2)
            craft_coord.points[:,1] = np.clip(craft_coord.points[:,1], 1, h - 2)
            
            # print("craft_coords shape",craft_coord.points.shape)
            # proposal_crafts = proposal_crafts.extend[craft_coord.points]
            proposal_crafts[idx, :, :] = craft_coord.get_sample_point(size=(h,w))
            
            ignore_tags[idx] = 1
        
        
        for idx, polygon in enumerate(polygons):
            if idx >= cfg.max_annotation:
                break
        
            # print("polygon",polygon.points)
            polygon.points[:, 0] = np.clip(polygon.points[:, 0], 1, w - 2)
            polygon.points[:, 1] = np.clip(polygon.points[:, 1], 1, h - 2)
            gt_points[idx, :, :] = polygon.get_sample_point(size=(h, w))
            # print("polygon",polygon.points)
            # print("len polygon",len(polygon))
            # print("len polygon",len(polygon.points))
            # print("len craft",len(craft_coords[idx].points))
            # print("gt_points shape",gt_points.shape)
            # print("craft_coords",craft_coords[idx].points)
            """
            craft_coords[idx].points[:,0] = np.clip(craft_coords[idx].points[:,0], 1, w - 2)
            craft_coords[idx].points[:,1] = np.clip(craft_coords[idx].points[:,0], 1, h - 2)
            proposal_crafts[idx, :, :] = craft_coords[idx].get_sample_point(size=(h,w))
            """
            cv2.fillPoly(tr_mask, [polygon.points.astype(np.int64)], color=(idx + 1,))

            inst_mask = mask_zeros.copy()
            cv2.fillPoly(inst_mask, [polygon.points.astype(np.int32)], color=(1,))
            dmp = ndimg.distance_transform_edt(inst_mask)  # distance transform
            """
            if polygon.text == '#' or np.max(dmp) < self.min_text_size:
                cv2.fillPoly(train_mask, [polygon.points.astype(np.int32)], color=(0,))
                ignore_tags[idx] = -1
            else:
                ignore_tags[idx] = 1
            
            proposal_points[idx, :, :] = \
                self.generate_proposal_point(dmp / (np.max(dmp)+1e-9) >= self.th_b, cfg.num_points,
                                             cfg.approx_factor, jitter=self.jitter, distance=self.th_b * np.max(dmp))
            """
            
            distance_field[:, :] = np.maximum(distance_field[:, :], dmp / (np.max(dmp)+1e-9))

            weight_matrix[inst_mask > 0] = 1. / np.sqrt(inst_mask.sum())
            # weight_matrix[inst_mask > 0] = 1. / inst_mask.sum()
            diff = self.compute_direction_field(inst_mask, h, w)
            direction_field[:, inst_mask > 0] = diff[:, inst_mask > 0]

        # ### background ######
        weight_matrix[tr_mask == 0] = 1. / np.sqrt(np.sum(tr_mask == 0))
        # weight_matrix[tr_mask == 0] = 1. / np.sum(tr_mask == 0)
        # diff = self.compute_direction_field((tr_mask == 0).astype(np.uint8), h, w)
        # direction_field[:, tr_mask == 0] = diff[:, tr_mask == 0]

        train_mask = np.clip(train_mask, 0, 1)
        # print("proposal points shape",proposal_points.shape)
        return train_mask, tr_mask, \
               distance_field, direction_field, \
               weight_matrix, gt_points, proposal_points, ignore_tags, proposal_crafts

    def get_training_data(self, image, polygons,craft_coords, image_id=None, image_path=None):
        
        H, W, _ = image.shape
        if self.transform:
            # original_image = image.copy()
            image, polygons, craft_points = self.transform(image, polygons, craft_coords)
            # image,polygons = self.transform(original_image,polygons)
            # print("22222222222222")
            
            # _, craft_points = self.transform(original_image, craft_coords)
            
            #    print("polygons",polygons[0])
                # print("polygons[0]",len(polygons))
            '''
            if image_id is not None and image_id not in self.random_states:
                np.random.seed(2020)
                self.random_states[image_id] = np.random.get_state()
            
            np.random.set_state(self.random_states[image_id])
            image, polygons = self.transform(original_image, polygons)
            
            np.random.set_state(self.random_states[image_id])
            image2, craft_points = self.transform(original_image, craft_coords)
            '''
            
            """
            np.random.seed(2020)
            random_state = np.random.get_state()
            if image_id == "0650.jpg":
                print("rand1",np.random.randint(1,100))
            image, polygons = self.transform(original_image, polygons)
            
            
            np.random.set_state(random_state)
            if image_id == "0650.jpg":
                print('rand2',np.random.randint(1,100))
            image2, craft_points = self.transform(original_image, craft_coords)
            """
            
            
            
            # print("next")
            """
            np.random.seed(2020)
            _, craft_points = self.transform(image, craft_coords)
            """
            """
            if image_id == "0001.jpg":
                # print("gt",polygons[0])
                # print("gt2",polygons[1])
                # print("craft_points",craft_points[0])
                # print("craft_points2",craft_points[1])
                # print("image",image)
                
                cv2.imwrite('a1.jpg', image*255)
                print("save done")
                cv2.imwrite("a2.jpg", image2*255)
                print("save2 done")
            """
            # print("next2")
        train_mask, tr_mask, \
        distance_field, direction_field, \
        weight_matrix, gt_points, proposal_points, ignore_tags,craft = self.make_text_region(image, polygons, craft_points,image_id)

        # # to pytorch channel sequence
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()

        train_mask = torch.from_numpy(train_mask).bool()
        tr_mask = torch.from_numpy(tr_mask).int()
        weight_matrix = torch.from_numpy(weight_matrix).float()
        direction_field = torch.from_numpy(direction_field).float()
        distance_field = torch.from_numpy(distance_field).float()
        gt_points = torch.from_numpy(gt_points).float()
        proposal_points = torch.from_numpy(proposal_points).float()
        # print("proposal points shape",proposal_points.shape)
        ignore_tags = torch.from_numpy(ignore_tags).int()
        image_id = image_id

        return image, train_mask, tr_mask, distance_field, \
               direction_field, weight_matrix, gt_points, proposal_points, ignore_tags, image_id, H, W, craft

    def get_test_data(self, image, polygons, image_id=None, image_path=None):
        H, W, _ = image.shape
        print("H",H)
        print("W",W)
        if self.transform:
            image, polygons = self.transform(image, polygons)
            # print("image shape",image.shape)

        points = np.zeros((cfg.max_annotation, cfg.max_points, 2))
        length = np.zeros(cfg.max_annotation, dtype=int)
        label_tag = np.zeros(cfg.max_annotation, dtype=int)
        if polygons is not None:
            for i, polygon in enumerate(polygons):
                pts = polygon.points
                points[i, :pts.shape[0]] = polygon.points
                length[i] = pts.shape[0]
                if polygon.text != '#':
                    label_tag[i] = 1
                else:
                    label_tag[i] = -1

        meta = {
            'image_id': image_id,
            'image_path': image_path,
            'annotation': points,
            'n_annotation': length,
            'label_tag': label_tag,
            'Height': H,
            'Width': W
        }

        # to pytorch channel sequence
        image = image.transpose(2, 0, 1)

        return image, meta

    def __len__(self):
        raise NotImplementedError()
