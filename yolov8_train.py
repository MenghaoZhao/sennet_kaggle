from ultralytics import YOLO
import os 

# yaml_text = """
# # YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# # Hyperparameters for high-augmentation COCO training from scratch
# # python train.py --batch 32 --cfg yolov5m6.yaml --weights '' --data coco.yaml --img 1280 --epochs 300
# # See tutorials for hyperparameter evolution https://github.com/ultralytics/yolov5#tutorials

# lr0: 0.0005  # initial learning rate (SGD=1E-2, Adam=1E-3)
# lrf: 0.1  # final OneCycleLR learning rate (lr0 * lrf)
# momentum: 0.937  # SGD momentum/Adam beta1
# weight_decay: 0.0005  # optimizer weight decay 5e-4
# warmup_epochs: 3.0  # warmup epochs (fractions ok)
# warmup_momentum: 0.8  # warmup initial momentum
# warmup_bias_lr: 0.1  # warmup initial bias lr
# box: 0.05  # box loss gain
# cls: 0.3  # cls loss gain
# cls_pw: 1.0  # cls BCELoss positive_weight
# obj: 0.7  # obj loss gain (scale with pixels)
# obj_pw: 1.0  # obj BCELoss positive_weight
# iou_t: 0.20  # IoU training threshold
# anchor_t: 4.0  # anchor-multiple threshold
# # anchors: 3  # anchors per output layer (0 to ignore)
# fl_gamma: 0.0  # focal loss gamma (efficientDet default gamma=1.5)
# hsv_h: 0.015  # image HSV-Hue augmentation (fraction)
# hsv_s: 0.7  # image HSV-Saturation augmentation (fraction)
# hsv_v: 0.4  # image HSV-Value augmentation (fraction)
# degrees: 0.0  # image rotation (+/- deg)
# translate: 0.1  # image translation (+/- fraction)
# scale: 0.9  # image scale (+/- gain)
# shear: 0.0  # image shear (+/- deg)
# perspective: 0.0  # image perspective (+/- fraction), range 0-0.001
# flipud: 0.0  # image flip up-down (probability)
# fliplr: 0.5  # image flip left-right (probability)
# mosaic: 1.0  # image mosaic (probability)
# mixup: 0  # image mixup (probability)
# copy_paste: 0  # segment copy-paste (probability)
# """
# with open('./yolov8_hyp.yaml', 'w') as text_file:
#     text_file.write(yaml_text)



if __name__ == '__main__':
    
    os.chdir("G:/mmdetection-master/mmdetection-master/mmdet/datasets/data/sennet/")
    os.listdir()

    model = YOLO("yolov8m-seg.pt")
    results = model.train(
            batch=4,
            data="yolo-coco.yaml",
            epochs=24,
            imgsz=[1303,912],
        )

