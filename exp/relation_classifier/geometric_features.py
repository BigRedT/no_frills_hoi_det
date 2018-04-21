import copy
import numpy as np

import utils.bbox_utils as bbox_utils


class GeometricFeatures():
    def __init__(self):
        pass

    def compute_bbox_center(self,bbox):
        center = np.array([
            0.5*(bbox[0] + bbox[2]),
            0.5*(bbox[1] + bbox[3])])
        return center

    def normalize_center(self,c,im_wh):
        return c / im_wh

    def compute_bbox_wh(self,bbox):
        wh = np.array([
            0.5*(bbox[2]-bbox[0]),
            0.5*(bbox[3]-bbox[1])])
        return wh

    def compute_offset(self,c1,c2,wh1,normalize):
        offset = c2 - c1
        if normalize:
            offset = offset / wh1
        return offset

    def compute_aspect_ratio(self,wh,take_log):
        aspect_ratio = wh[0] / (wh[1] + 1e-6)
        if take_log:
            aspect_ratio = np.log2(aspect_ratio+1e-6)
        return aspect_ratio

    def compute_bbox_size_ratio(self,wh1,wh2,take_log):
        return np.log2((wh2[0]*wh2[1])/(wh1[0]*wh1[1]))
            
    def compute_bbox_area(self,wh,im_wh,normalize):
        bbox_area = wh[0]*wh[1]
        if normalize:
            norm_factor = im_wh[0]*im_wh[1]
        else:
            norm_factor = 1
        bbox_area = bbox_area / norm_factor
        return bbox_area

    def compute_features(self,bbox1,bbox2,img_size):
        imh, imw = [float(v) for v in img_size[:2]]
        im_wh = np.array([imw,imh],dtype=np.float32)
        # bbox1 = copy.deepcopy(bbox1).astype(np.float32)
        # bbox2 = copy.deepcopy(bbox2).astype(np.float32)
        c1 = self.compute_bbox_center(bbox1)
        c2 = self.compute_bbox_center(bbox2)
        c1_normalized = self.normalize_center(c1,im_wh)
        c2_normalized = self.normalize_center(c2,im_wh)
        wh1 = self.compute_bbox_wh(bbox1)
        wh2 = self.compute_bbox_wh(bbox2)
        aspect_ratio1 = self.compute_aspect_ratio(wh1,take_log=True)
        aspect_ratio2 = self.compute_aspect_ratio(wh2,take_log=True)
        area1 = self.compute_bbox_area(wh1,im_wh,normalize=True)
        area2 = self.compute_bbox_area(wh2,im_wh,normalize=True)
        offset = self.compute_offset(c1,c2,wh1,normalize=True)
        bbox_size_ratio = self.compute_bbox_size_ratio(wh1,wh2,take_log=True)
        iou = bbox_utils.compute_iou(bbox1,bbox2)
        geometric_feat = np.concatenate((
            offset,
            [bbox_size_ratio],
            [iou],
            [aspect_ratio1,aspect_ratio2],
            [area1,area2],
            c1_normalized,
            c2_normalized,
        ))
        return geometric_feat


class GeometricFeaturesBatch():
    def __init__(self):
        pass

    def compute_bbox_center(self,bbox):
        num_boxes = bbox.shape[0]
        center = np.zeros([num_boxes,2])
        center[:,0] = 0.5*(bbox[:,0] + bbox[:,2])
        center[:,1] = 0.5*(bbox[:,1] + bbox[:,3])
        return center

    def normalize_center(self,c,im_wh):
        return c / im_wh

    def compute_bbox_wh(self,bbox):
        num_boxes = bbox.shape[0]
        wh = np.zeros([num_boxes,2])
        wh[:,0] = 0.5*(bbox[:,2]-bbox[:,0])
        wh[:,1] = 0.5*(bbox[:,3]-bbox[:,1])
        return wh

    def compute_offset(self,c1,c2,wh1,normalize):
        offset = c2 - c1
        if normalize:
            offset = offset / wh1
        return offset

    def compute_aspect_ratio(self,wh,take_log):
        aspect_ratio = wh[:,0] / (wh[:,1] + 1e-6)
        if take_log:
            aspect_ratio = np.log2(aspect_ratio+1e-6)
        return aspect_ratio

    def compute_bbox_size_ratio(self,wh1,wh2,take_log):
        return np.log2((wh2[:,0]*wh2[:,1])/(wh1[:,0]*wh1[:,1]))
            
    def compute_bbox_area(self,wh,im_wh,normalize):
        bbox_area = wh[:,0]*wh[:,1]
        if normalize:
            norm_factor = im_wh[:,0]*im_wh[:,1]
        else:
            norm_factor = 1
        bbox_area = bbox_area / norm_factor
        return bbox_area

    def compute_features(self,bbox1,bbox2,im_wh):
        # bbox1 = copy.deepcopy(bbox1).astype(np.float32)
        # bbox2 = copy.deepcopy(bbox2).astype(np.float32)
        c1 = self.compute_bbox_center(bbox1)
        c2 = self.compute_bbox_center(bbox2)
        c1_normalized = self.normalize_center(c1,im_wh)
        c2_normalized = self.normalize_center(c2,im_wh)
        wh1 = self.compute_bbox_wh(bbox1)
        wh2 = self.compute_bbox_wh(bbox2)
        aspect_ratio1 = self.compute_aspect_ratio(wh1,take_log=True)
        aspect_ratio2 = self.compute_aspect_ratio(wh2,take_log=True)
        area1 = self.compute_bbox_area(wh1,im_wh,normalize=True)
        area2 = self.compute_bbox_area(wh2,im_wh,normalize=True)
        offset = self.compute_offset(c1,c2,wh1,normalize=True)
        bbox_size_ratio = self.compute_bbox_size_ratio(wh1,wh2,take_log=True)
        iou = bbox_utils.compute_iou_batch(bbox1,bbox2)
        geometric_feat = np.concatenate((
            offset,
            bbox_size_ratio[:,np.newaxis],
            iou[:,np.newaxis],
            aspect_ratio1[:,np.newaxis],
            aspect_ratio2[:,np.newaxis],
            area1[:,np.newaxis],
            area2[:,np.newaxis],
            c1_normalized,
            c2_normalized,
        ),1)
        return geometric_feat