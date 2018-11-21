# YOLOv3-Object-Detection-with-OpenCV

This project implements an image and video object detection classifier using pretrained yolov3 models. 
The yolov3 models are taken from the offical yolov3 paper which was released in 2018. The yolov3 implementation is from [darknet](https://github.com/pjreddie/darknet). Also, this project implements an option to perform classification real-time using the webcam.

## How to use?

1) Clone the repository

```
git clone https://github.com/iArunava/YOLOv3-Object-Detection-with-OpenCV.git
```

2) Move to the directory
```
cd YOLOv3-Object-Detection-with-OpenCV
```

3) To infer on an image that is stored on your local machine
```
python3 yolov3.py --image-path='/path/to/image/'
```
4) To infer on a video that is stored on your local machine
```
python3 yolov3.py --video-path='/path/to/video/'
```
5) To infer real-time on webcam
```
python3 yolov3.py
```

Note: This works considering you have the `weights` and `config` files at the yolov3-coco directory.
<br/>
If the files are located somewhere else then mention the path while calling the `yolov3.py`. For more details
```
yolov3.py --help
```

## Inference on images

## Inference on Video

## Inference in Real-time

## References

1) [PyImageSearch YOLOv3 Object Detection with OpenCV Blog](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/)

## License

The code in this project is distributed under the MIT License.
