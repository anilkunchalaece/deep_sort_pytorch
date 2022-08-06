import numpy as np
import torch

from .deep.feature_extractor import Extractor, FastReIDExtractor, FSINetFeatureExtractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker
import json
import torchvision

__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self, model_path, model_config=None, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap

        if model_config is None:
            self.extractor = Extractor(model_path, use_cuda=use_cuda)
        else :
            self.extractor = FSINetFeatureExtractor(model_config,model_path, use_cuda=use_cuda)
        # else :
        #     self.extractor = FastReIDExtractor(model_config, model_path, use_cuda=use_cuda)

        max_cosine_distance = max_dist
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, bbox_xywh, confidences, ori_img, attrFile=None):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_xywh, ori_img, attrFile)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i,conf in enumerate(confidences) if conf>self.min_confidence]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs,axis=0)
        return outputs


    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2.
        bbox_tlwh[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2.
        return bbox_tlwh


    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w),self.width-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),self.height-1)
        return x1,y1,x2,y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1,y1,x2,y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2-x1)
        h = int(y2-y1)
        return t,l,w,h

    def resizeBbox(self,bboxes):
        #TODO -> convert these into configurable
        out_img_width = 432 # x_scaled
        out_img_height = 240 # y_scaled
        in_img_width = 1920 # x_org
        in_img_height = 1080 # y_org
        
        scaled_bboxes_yolo = []
        scaled_bboxes_pascal = []

        x_scale = out_img_width/in_img_width
        y_scale = out_img_height/in_img_height
        
        for bbox in bboxes :
            # print(F"original {bbox}")
            bbox[0] = bbox[0]*x_scale
            bbox[1] = bbox[1]*y_scale 
            bbox[2] = bbox[2]*x_scale
            bbox[3] = bbox[3]*y_scale

            # converting the Pascal BBOX to YoLo BBOX
            # Ref - https://github.com/mkocabas/multi-person-tracker/blob/2803ac529dc77328f0f1ff6cd9d36041e57e7288/multi_person_tracker/mpt.py#L133
            # Since the multiperson tracker used in the VIBE and PARE depends on the YoLo bboxes
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            c_x, c_y = bbox[0] + w / 2, bbox[1] + h / 2
            w = h = np.where(w / h > 1, w, h)
            
            scaled_bboxes_yolo.append([c_x, c_y, w, h])
            scaled_bboxes_pascal.append(bbox)
            # print(F"scaled {bbox}")


        # return {
        #     "scaled_yolo" : np.array(scaled_bboxes_yolo),
        #     "scaled_pascal" : np.array(scaled_bboxes_pascal)
        # }
        return scaled_bboxes_pascal
    
    def _get_features(self, bbox_xywh, ori_img,attrFile=None):
        im_crops = []
        crop_attr = []

        if attrFile != None : 
            with open(attrFile) as fd :
                bboxes_attr_org = json.load(fd)["metadata"]
                bboxes_attr = torch.tensor(self.resizeBbox([v['bbox'] for v in bboxes_attr_org]),dtype=torch.float32)
                # print(bboxes_attr)

        for box in bbox_xywh:
            x1,y1,x2,y2 = self._xywh_to_xyxy(box)
            # print(x1,y1,x2,y2)
            im = ori_img[y1:y2,x1:x2]
            im_crops.append(im)
            if attrFile != None :
                # print(torchvision.ops.box_iou(torch.tensor([[x1,y1,x2,y2]],dtype=torch.float32),bboxes_attr))
                attr_idx = torch.argmax(torchvision.ops.box_iou(torch.tensor([[x1,y1,x2,y2]],dtype=torch.float32),bboxes_attr))
                crop_attr.append(bboxes_attr_org[attr_idx]["attr"])
        if im_crops:
            if attrFile != None :
                features = self.extractor(im_crops,crop_attr)
            else :
                features = self.extractor(im_crops)
            # print(features.shape)
        else:
            features = np.array([])
        return features


