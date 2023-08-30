import cv2
import numpy
import time
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Loads the config
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 22  # Sets number of classes
cfg.MODEL.WEIGHTS = r"C:\Users\katta\mscproject\VOCdevkit\VOC2012\output\model_final.pth"  # Paths to your checkpoint file
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Sets the testing threshold 
cfg.MODEL.DEVICE = "cuda"  # Use GPU for inferences

# Creates predictor
predictor = DefaultPredictor(cfg)

# Gets metadata for visualizer
metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

# Starts webcam
cap = cv2.VideoCapture(0)

# Variables for FPS calculations
frame_counter = 0
start_time = time.time()
font = cv2.FONT_HERSHEY_SIMPLEX
fps = 0  # Initial FPS values

while True:
    # Reads frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Makes prediction
    outputs = predictor(frame)

    # FPS calculations
    frame_counter += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:  # Update FPS value every second
        fps = frame_counter / elapsed_time
        frame_counter = 0
        start_time = time.time()

    # Displays FPS on the original frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), font, 1, (0, 255, 0), 2)

    # Visualizes result on a copy of the frame
    v = Visualizer(frame[:, :, ::-1].copy(), metadata=metadata, scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result = v.get_image()[:, :, ::-1]

    # Shows result
    cv2.imshow("object detection", result)

    if cv2.waitKey(1) == 27:
        break  # stops if 'ESC' is pressed

cap.release()
cv2.destroyAllWindows()