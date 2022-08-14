import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging

from .FSINet import FusedSimilarityNet
from .model import Net
# from fastreid.config import get_cfg
# from fastreid.engine import DefaultTrainer
# from fastreid.utils.checkpoint import Checkpointer

class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['net_dict']
        self.net.load_state_dict(state_dict)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        


    def _preprocess(self, im_crops,crop_attr=None):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch


    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()

class FastReIDExtractor(object):
    def __init__(self, model_config, model_path, use_cuda=True):
        cfg = get_cfg()
        cfg.merge_from_file(model_config)
        cfg.MODEL.BACKBONE.PRETRAIN = False
        self.net = DefaultTrainer.build_model(cfg)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"

        Checkpointer(self.net).load(model_path)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.net.eval()
        height, width = cfg.INPUT.SIZE_TEST
        self.size = (width, height)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    
    def _preprocess(self, im_crops):
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch


    def __call__(self, im_crops,crop_attr=None):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()

class FSINetFeatureExtractor(object):
    def __init__(self,model_config, model_path, use_cuda=True):
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu" 
        self.net = FusedSimilarityNet(model_config)
        print(F"loading weights from {model_path}")
        try :
            state_dict , _ = torch.load(model_path)
        except ValueError :
            state_dict = torch.load(model_path)
            
        self.net.load_state_dict(state_dict)
        self.net = self.net.to(self.device)
        self.net.eval()

        self.size = (64,128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.defineAttr()

    def defineAttr(self):
        self.GENDER_ATTR = ['Female', 'Male']
        self.AGE_ATTR = ['AgeLess16','Age17-30','Age31-45','Age46-60','AgeBiger60','NA']
        self.BODY_SHAPE_ATTR = ['BodyFatter','BodyFat','BodyNormal','BodyThin','BodyThiner',"NA"]
        self.ATTACHMENT_ATTR = ['attachment-Backpack','attachment-ShoulderBag','attachment-HandBag','attachment-WaistBag',
                                'attachment-Box','attachment-PlasticBag','attachment-PaperBag','attachment-HandTrunk',
                                'attachment-Baby','attachment-Other', 'NA']
        self.UPPER_BODY_ATTR = ['ub-Shirt','ub-Sweater','ub-Vest','ub-TShirt','ub-Cotton','ub-Jacket','ub-SuitUp',
                                'ub-Tight','ub-ShortSleeve','ub-Others','ub-ColorBlack','ub-ColorWhite','ub-ColorGray',
                                'up-ColorRed','ub-ColorGreen','ub-ColorBlue','ub-ColorSilver','ub-ColorYellow',
                                'ub-ColorBrown','ub-ColorPurple','ub-ColorPink','ub-ColorOrange','ub-ColorMixture','ub-ColorOther',
                                'NA']
        self.LOWER_BODY_ATTR = ['lb-LongTrousers','lb-Shorts','lb-Skirt','lb-ShortSkirt','lb-LongSkirt','lb-Dress','lb-Jeans',
                                'lb-TightTrousers','lb-ColorBlack','lb-ColorWhite','lb-ColorGray','lb-ColorRed','lb-ColorGreen',
                                'lb-ColorBlue','lb-ColorSilver','lb-ColorYellow','lb-ColorBrown','lb-ColorPurple',
                                'lb-ColorPink','lb-ColorOrange','lb-ColorMixture','lb-ColorOther', 'NA'
                                ]        

    def _preprocess(self, im_crops,crop_attr):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        attr_batch = torch.tensor([self.attributeNamesToIndex(attr) for attr in crop_attr])
        # _data = []
        # for idx,im in enumerate(im_crops):
        #     _data.append({
        #         "imageVal" : self.norm(_resize(im, self.size)).unsqueeze(0),
        #         "attrIdxs" : torch.tensor(self.attributeNamesToIndex(crop_attr[idx]))
        #     })
        
        im_batch = {
            "imageVal" : im_batch,
            "attrIdxs" : attr_batch
        } #torch.cat(_data,dim=0).float()

        return im_batch

    def __call__(self, im_crops,crop_attr):
        im_batch = self._preprocess(im_crops,crop_attr)
        with torch.no_grad():
            # im_batch = im_batch.to(self.device)
            im_batch["imageVal"] = im_batch["imageVal"].to(self.device)
            im_batch["attrIdxs"] = im_batch["attrIdxs"].to(self.device)#[ x.to(self.device) for x in im_batch["attrIdxs"]]

            features = self.net(im_batch)
            
        return features.cpu()#.numpy()

    
    def attributeNamesToIndex(self,attrNames) :
        gender_idx = self.GENDER_ATTR.index(attrNames[0])
        age_idx = self.AGE_ATTR.index(attrNames[1])
        bodyShape_idx = self.BODY_SHAPE_ATTR.index(attrNames[2])
        attachment_idx = self.ATTACHMENT_ATTR.index(attrNames[5])
        upperBody_idx = self.UPPER_BODY_ATTR.index(attrNames[3])
        lowerBody_idx = self.LOWER_BODY_ATTR.index(attrNames[4])
        
        return [gender_idx,age_idx,bodyShape_idx,attachment_idx,upperBody_idx,lowerBody_idx]


if __name__ == '__main__':
    img = cv2.imread("demo/1.jpg")[:,:,(2,1,0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    # print(feature.shape)

