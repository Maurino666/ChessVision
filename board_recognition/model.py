import torchvision
from torchvision.models.detection import keypointrcnn_resnet50_fpn

def get_keypoint_model(num_keypoints=4):
    model = keypointrcnn_resnet50_fpn(weights="COCO_V1")
    in_channels = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
    model.roi_heads.keypoint_predictor = \
        torchvision.models.detection.keypoint_rcnn.KeypointRCNNPredictor(
            in_channels=in_channels,
            num_keypoints=num_keypoints
        )
    return model
