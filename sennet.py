# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import List, Union

import numpy as np
from mmengine.fileio import get_local_path
from torch.utils.data import Dataset

from mmdet.registry import DATASETS
from .api_wrappers import COCO
from .base_det_dataset import BaseDetDataset
from torch.utils.data import Dataset
import torch
import numpy as np
import albumentations as A
import torchvision.transforms as T
import cv2
import os
import yaml
from typing import Literal, Any, Union
from tqdm import tqdm
import json
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from mmengine.registry import TRANSFORMS

class Compose:
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict, callable], optional): Sequence of transform
            object or config dict to be composed.
    """

    def __init__(self, transforms: Optional[Sequence[Union[dict, Callable]]]):
        self.transforms: List[Callable] = []

        if transforms is None:
            transforms = []

        for transform in transforms:
            # `Compose` can be built with config dict with type and
            # corresponding arguments.
            if isinstance(transform, dict):
                transform = TRANSFORMS.build(transform)
                if not callable(transform):
                    raise TypeError(f'transform should be a callable object, '
                                    f'but got {type(transform)}')
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(
                    f'transform must be a callable object or dict, '
                    f'but got {type(transform)}')

    def __call__(self, data: dict) -> Optional[dict]:
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """
        for t in self.transforms:
            # print(data['img_path'])
            # print("t mask",len(data['masks']),len(data["bboxes"]),data['img_path'])
            data = t(data)
            # The transform will return None when it failed to load images or
            # cannot find suitable augmentation parameters to augment the data.
            # Here we simply return None if the transform returns None and the
            # dataset will handle it by randomly selecting another data sample.
            if data is None:
                return None
        return data

    def __repr__(self):
        """Print ``self.transforms`` in sequence.

        Returns:
            str: Formatted string.
        """
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string

@DATASETS.register_module()
class SennetDataset(Dataset):

    # METAINFO = {'classes':('blood_vessel'),'palette':[(220,20,68)]}
    CLASSES = ('blood_vessel',)

    def __init__(self,
                 pipeline,   
                 stage: Literal["train", "val"],
                 config_path: str,
                 transforms: Union[A.Compose, T.Compose] = None,
                 *args, **kwargs
                 ):

        print("initing hubmap dataset")
        self.config = self.load_config(config_path)
        self.images_dirpath = None
        self.labels_dirpath = None
        self.__define_paths(stage)
        self.data_list: List[dict] = []
        
        self.names = self.config["names"]
        self.num_classes = len(self.names) 
        self.samples = os.listdir(self.labels_dirpath)[:3]
        print(self.samples[:3])
        self.flag = np.array([0 if i % 2 == 0 else 1 for i in range(len(self.samples))])
        self.transforms = None
        self.pipeline = Compose(pipeline)
        if transforms:
            self.bbox_params = {
                "format":"pascal_voc",
                "min_area": 0,
                "min_visibility": 0,
                "label_fields": ["category_id"]
            }
            self.transforms = A.Compose(transforms, bbox_params=self.bbox_params)
        self.init()
    
    @property
    def metainfo(self) -> dict:
        """Get meta information of dataset.

        Returns:
            dict: meta information collected from ``BaseDataset.METAINFO``,
            annotation file and metainfo argument during instantiation.
        """
        return dict({'classes':('blood_vessel'),'palette':[(220,20,68)]})

    def init(self):
        print("get item")
        for idx in tqdm(range(len(self.samples))):
            filename = self.samples[idx].split(".")[0]
            paths = self._get_paths(filename)
            # image = cv2.imread(paths["image"], cv2.COLOR_BGR2RGB)  
            # print("image shape", image.shape)
            target = self._get_target(paths["label"])
            # print("get target ", target)
            target["image_id"] = torch.tensor([idx])
            # if self.transforms:
            #     image, target = self.transform(image, target)
            #print(image.shape,image.dtype, target)
            
            image_info = dict()
            image_info['filename'] = paths['image']
            # print(paths['image'])
            # ann_info = dict()
            # ann_info['bboxes'] = np.array(target['bboxes']).astype('float32')
            # ann_info['masks'] = np.array(target['masks'])#[[x.tolist()] for x in target['masks']]
            # ann_info['labels'] = target['labels']
            # results = dict(img_info=image_info, ann_info=ann_info)
            results= {}
            results['img_path'] = paths["image"]
            results['img_id'] = idx
            results['height'] = 1303
            results['width'] = 912
            instances = []
            # print(paths["label"], len(target['bboxes']))
            for i in range(len(target["bboxes"])):
                instance = {}
                instance['bbox'] = list(target['bboxes'][i])
                instance['mask'] = [list(target['masks'][i])]
                instance['bbox_label'] = target['labels'][i]
                instance['ignore_flag'] = 0
                instances.append(instance)
            
            results['instances'] = instances
            self.data_list.append(results)

    @staticmethod
    def load_config(path: str) -> dict:
        with open(path, mode="r") as f:
            data = yaml.load(stream=f, Loader=yaml.SafeLoader)
        return data

    def __len__(self) -> int:
        return len(self.samples)

    def get_data_info(self, idx:int):
        data_info = self.data_list[idx]
        data_info['sample_idx'] = idx
        # print("get_data_info",data_info)
        return data_info

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        results = self.data_list[idx]
        # print("getitem",results['instances'])
        # print("result mask",results['masks'][0].shape[0],"pat",paths['image'])
        #print("dddddddtypes", ann_info['bboxes'].dtype, target['masks'][0].dtype)
        #print("rrrrrrrrrrrresults", results)
        # print(results)
        return self.pipeline(results)
    
    def transform(self, image: np.ndarray, target: dict) -> tuple[torch.Tensor, dict]:
        transformed = self.transforms(
            image=image, masks=target["masks"],
            bboxes=target["bboxes"], 
            category_id=target["labels"]
        )
    
        image = transformed["image"]
        target["masks"] = torch.as_tensor(
            np.array(list(map(np.array, transformed["masks"])), dtype=np.uint8)
        ) 
        
        target["labels"] = torch.tensor(transformed["category_id"])
        target["bboxes"] = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
        target["area"] = self.__get_area(target["bboxes"])
        return image, target
        
    def __define_paths(self, stage: Literal["train", "val"]) -> None:
        data_dirpath = self.config[stage]
        self.images_dirpath = os.path.join(data_dirpath, "images")
        self.labels_dirpath = os.path.join(data_dirpath, "labels")
        
    def _get_paths(self, filename: str) -> dict:
        image_path = os.path.join(self.images_dirpath, f"{filename}.tif")
        label_path = os.path.join(self.labels_dirpath, f"{filename}.txt")
        return {
            "image": image_path,
            "label": label_path
        }
    
    @staticmethod
    def _get_target_sample() -> dict:
        return {
            "bboxes": [],
            "masks": [],
            "area": [],
            "labels": [],
            "iscrowd": None,
            "image_id": None,
            "img_prefix": None
        }

    def _get_target(self, annotations_path: str) -> dict:
        target = self._get_target_sample()
        #print(annotations_path)
        with open(annotations_path, "r") as file:
            for line in file:
                label = int(line[0])
                #if label == 2:
                #    continue
                coordinates = np.array(list(map(int, line[1:].split())))
                # print("coordinats", coordinates)
                #print(coordinates.shape)
                mask = self.__get_mask(label, coordinates.reshape(1, -1, 2))
                #print("mask", mask)
                # print(mask.shape)

                box = self.__get_box(mask)
                target["masks"].append(coordinates)
                target["bboxes"].append(box)
                target["labels"].append(label)
        target["labels"] = np.array(target["labels"], dtype=np.int64)
        num_objs = len(target["labels"])
        target["iscrowd"] = torch.zeros((num_objs,), dtype=torch.int64)
        return target 

    @staticmethod
    def __get_mask(label: int, coordinates: np.ndarray) -> np.ndarray:
        mask = np.zeros((1303, 912), dtype=np.uint8)
        return cv2.fillPoly(
            mask, pts=coordinates,
            color=(255, 255, 255)
        )
    
    @staticmethod
    def __get_box(mask: np.ndarray) -> list[np.ndarray, ...]:
        pos = np.nonzero(mask)
        #print("pos", pos)
        if pos[0].shape[0] == 0:
            xmin = 0
            xmax = 0
            ymin = 0
            ymax = 0
            #print("no bbbbbbbbbbbbbbbbbbbbbbbox")
            return [0, 0, 1, 1]
        #print("has bbbbbbbbbbbbbbbbbbbbbbbox")
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        # print(xmin,xmax,ymin,ymax)
        return [xmin, ymin, xmax, ymax]
    
    @staticmethod
    def __get_area(boxes: list[list, ...]) -> torch.Tensor:
         return (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    
    
    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        
        np.set_printoptions(threshold=np.inf)
        #counts = results[0][1][0][0]['counts']
        #if isinstance(counts, bytes):
        #    print("result is bytes")
        #    print(counts.decode())
        # data_name = "hubmap"
        # f = open(f"E:/mmdetection-master/mmdetection-master/mmdet/datasets/data/eval_{data_name}.txt", "a")
        # f.write(str(results))
        # f.write("\n")
        # f.close()
        return dict()
