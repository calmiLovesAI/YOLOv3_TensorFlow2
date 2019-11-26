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

### Train on custom dataset
1. Turn your custom dataset's labels into this form: 
```
xxx.jpg 100 200 300 400 1 300 600 500 800 2
```
The first position is the image name, and the next 5 elements are [xmin, ymin, xmax, ymax, class_id]. If there are multiple boxes, continue to add elements later. <br>**Considering that the image will be resized before it is entered into the network, the values of xmin, ymin, xmax, and ymax will also change accordingly.**<br>
The original picture and resized picture:<br>
Create a new file *data.txt* in the data_process directory and write the label of each picture into it,each line is a label for an image.

2. Change the parameters *CATEGORY_NUM*, *use_dataset*, *custom_dataset_dir*, *custom_dataset_classes* in **configuration.py**.
3. Run **train_from_scratch.py** to start training.

### Test
1. Change *"test_picture_dir"* in **configuration.py**.
2. Run **test_on_single_image.py** to test single picture.


## References
1. YOLO_v3 paper: https://pjreddie.com/media/files/papers/YOLOv3.pdf or https://arxiv.org/abs/1804.02767
2. Keras implementation of YOLOV3: https://github.com/qqwweee/keras-yolo3
3. [blog 1](https://www.cnblogs.com/wangxinzhe/p/10592184.html), [blog 2](https://www.cnblogs.com/wangxinzhe/p/10648465.html), [blog 3](https://blog.csdn.net/leviopku/article/details/82660381), [blog 4](https://blog.csdn.net/qq_37541097/article/details/81214953), [blog 5](https://blog.csdn.net/Gentleman_Qin/article/details/84349144), [blog 6](https://blog.csdn.net/qq_34199326/article/details/84109828), [blog 7](https://blog.csdn.net/weixin_38145317/article/details/95349201)
5. 李金洪. 深度学习之TensorFlow工程化项目实战[M]. 北京: 电子工业出版社, 2019: 343-375
6. https://zhuanlan.zhihu.com/p/49556105