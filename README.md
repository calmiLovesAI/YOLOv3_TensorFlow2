# YOLOv3_TensorFlow2
A tensorflow2 implementation of YOLO_V3.

## Requirements:
+ Python == 3.7
+ TensorFlow == 2.0.0
+ numpy == 1.17.0

## Usage
### Train on PASCAL VOC 2012
1. Download the [PASCAL VOC 2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/).
2. Unzip the file and place it in the 'dataset' folder, make sure the directory is like this : 
```
|——dataset
    |——VOCdevkit
        |——VOC2012
            |——Annotations
            |——ImageSets
            |——JPEGImages
            |——SegmentationClass
            |——SegmentationObject
```
3. Change the parameters in **configuration.py** according to the specific situation. Specially, you can set *"load_weights_before_training"* to **True** if you would like to restore training from saved weights.
4. Run **train_from_scratch.py** to start training.

### Test
1. Change *"test_picture_dir"* in **configuration.py**.
2. Run **test_on_single_image.py** to test single picture.


## Reference
1. YOLO_v3 paper: https://pjreddie.com/media/files/papers/YOLOv3.pdf or https://arxiv.org/abs/1804.02767
2. Keras implementation of YOLOV3: https://github.com/qqwweee/keras-yolo3
3. https://www.cnblogs.com/wangxinzhe/p/10592184.html
4. https://www.cnblogs.com/wangxinzhe/p/10648465.html
5. 李金洪. 深度学习之TensorFlow工程化项目实战[M]. 北京: 电子工业出版社, 2019: 343-375
6. https://zhuanlan.zhihu.com/p/49556105
7. https://blog.csdn.net/leviopku/article/details/82660381
8. https://blog.csdn.net/qq_37541097/article/details/81214953
9. https://blog.csdn.net/Gentleman_Qin/article/details/84349144
10. https://blog.csdn.net/qq_34199326/article/details/84109828
11. https://blog.csdn.net/weixin_38145317/article/details/95349201