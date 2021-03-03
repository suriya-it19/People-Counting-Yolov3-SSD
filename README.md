# People-Counting-Yolov3-SSD

# YoloV3
  Run using the command
  python yolov3_copy.py --input videos/example.mp4 --weights weights/yolov3-608.weights --config weights/yolov3-608.cfg --size 608
  Note: Download the weights file, 
  YoloV3-608 -> https://pjreddie.com/media/files/yolov3.weights 
  YoloV3-416 -> https://pjreddie.com/media/files/yolov3.weights
  and store the weights inside the weights folder. Create a folder name videos and store any example video for demo
 
# SSD
  Run using the command
  python SSD.py --input videos/example.mp4 --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel
  Note: Download the weights file here and store the weights inside the weights folder
