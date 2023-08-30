import torch
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2 import model_zoo

#  path to the Pascal VOC 2012 dataset
VOC2012_PATH = r"C:\Users\katta\mscproject\VOCdevkit\VOC2012"

# Convert Pascal VOC 2012 annotations to COCO format
# ...

def train_faster_rcnn(cfg):
    # Creates a trainer with the given config
    trainer = DefaultTrainer(cfg)

    # Starts the training
    try:
        trainer.resume_or_load(resume=False)
        trainer.train()
    except Exception as e:
        # Print any exception that occurred during training
        print("Exception during training:")
        print(e)
        raise

def main():
    # Load the Faster R-CNN configuration file
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"))

    # Update the configuration with Pascal VOC 2012 dataset settings
    cfg.DATASETS.TRAIN = ("pascal_voc_2012_train",)
    cfg.DATASETS.TEST = ("pascal_voc_2012_val",)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.INPUT.MIN_SIZE_TRAIN = (800,)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000  # Saves a checkpoint every 1000 iterations
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 22  # Pascal VOC 2012 has 20 classes + 1 background class
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml")
    cfg.MODEL.DEVICE = "cpu"
    
    # Register the Pascal VOC 2012 dataset in Detectron2
    register_coco_instances("pascal_voc_2012_train", {}, f"{VOC2012_PATH}/pascal_voc_2012_train.json", VOC2012_PATH)
    
    # Train the Faster R-CNN model
    train_faster_rcnn(cfg)

if __name__ == "__main__":
    main()
