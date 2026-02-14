import math
import cv2
from sam2.build_sam import build_sam2_video_predictor
import torch
from frame_predictor import FramePredictor
from frame_tools import VideoStreamReader
from utils import match_masks, get_segmented_ratio
from visualize import visualize_masks
global stuff_list
import copy
from class_tools import stuff_list

# import faulthandler
# faulthandler.enable()

SAM2_FILE_DICT = {
    "TINY": {
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "checkpoint": "pretrained_models/SAM2/sam2.1_hiera_tiny.pt"
    },
    "SMALL": {
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "checkpoint": "pretrained_models/SAM2/sam2.1_hiera_small.pt"
    },
    "BASE": {
        "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "checkpoint": "pretrained_models/SAM2/sam2.1_hiera_base_plus.pt"
    },
    "LARGE": {
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "checkpoint": "pretrained_models/SAM2/sam2.1_hiera_large.pt"
    }
}


class SAM2Predictor:
    def __init__(self, sam2_type: str = "LARGE", device: str = None,
                 matched_iou_threshold: float = 0.5,
                 use_llm: bool = False, refine_sam2: bool = True,
                 covered_iou_tolerance: float = 0.001,
                 refine_per_second: int = 5,
                 other_config=None):
        if other_config is None:
            other_config = {"FramePredictor": {},
                            "CropFormer": {},
                            "OneFormer": {}
                            }
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.sam2_video_predictor = build_sam2_video_predictor(SAM2_FILE_DICT[sam2_type]["config"],
                                                               SAM2_FILE_DICT[sam2_type]["checkpoint"],
                                                               device=device)
        self.video_reader = None
        self.frame_predictor = FramePredictor(device=device, **other_config["FramePredictor"],
                                              other_config=other_config)
        self.inference_state = None
        self.video_segments = {}
        self.object_count = 0
        self.use_llm = use_llm
        self.matched_iou_threshold = matched_iou_threshold
        self.refine_sam2 = refine_sam2
        self.refine_per_second = refine_per_second
        self.refine_frequency = None
        self.video_path = None
        self.covered_iou_tolerance = covered_iou_tolerance
        self.semantic_labels = []

    def init_state(self, video_path: str = None):
        if video_path is not None:
            self.video_path = video_path
            self.video_reader = VideoStreamReader(video_path)
            self.refine_frequency = math.ceil(self.video_reader.fps / self.refine_per_second)
            self.inference_state = self.sam2_video_predictor.init_state(video_path, async_loading_frames=True)
            self.semantic_labels = []
            self.object_count = 0
            self.video_segments = {}
        self.sam2_video_predictor.reset_state(self.inference_state)

    def predict_frame(self, global_frame_idx: int = 0):
        frame = self.video_reader.read_frame(global_frame_idx)
        model_masks, model_labels = self.frame_predictor.predict(frame, use_llm=self.use_llm)
        segmented_ratio = get_segmented_ratio(model_masks)
        if global_frame_idx != 0:
            sam2_masks = self.video_segments[global_frame_idx]["masks"]
            sam2_obj_ids = self.video_segments[global_frame_idx]["object_ids"]
            sam2_masks = sam2_masks.to(model_masks.device)
            (matched_new_masks_indices, matched_sam2_masks_indices, unmatched_new_masks_indices) = (
                match_masks(model_masks, sam2_masks, self.matched_iou_threshold))
        else:
            matched_new_masks_indices = None
            matched_sam2_masks_indices = None
            unmatched_new_masks_indices = range(len(model_masks))
        if matched_new_masks_indices is not None:
            for new_mask_index, sam2_mask_index in zip(matched_new_masks_indices, matched_sam2_masks_indices):
                obj_id = sam2_obj_ids[sam2_mask_index]
                self.semantic_labels[obj_id] = model_labels[new_mask_index]
                self.sam2_video_predictor.add_new_mask(
                    inference_state=self.inference_state,
                    frame_idx=global_frame_idx,
                    obj_id=obj_id,
                    mask=model_masks[new_mask_index]
                )
        for unsegmented_index in unmatched_new_masks_indices:
            model_mask = model_masks[unsegmented_index]
            label = model_labels[unsegmented_index]
            if label in self.semantic_labels and label in stuff_list:
                stuff_id = self.semantic_labels.index(label)
                if stuff_id in sam2_obj_ids:
                    stuff_id_index = sam2_obj_ids.index(stuff_id)
                    model_mask = sam2_masks[stuff_id_index] | model_mask
                self.sam2_video_predictor.add_new_mask(
                    inference_state=self.inference_state,
                    frame_idx=global_frame_idx,
                    obj_id=stuff_id,
                    mask=model_mask
                )
            else:
                self.sam2_video_predictor.add_new_mask(
                    inference_state=self.inference_state,
                    frame_idx=global_frame_idx,
                    obj_id=self.object_count,
                    mask=model_mask
                )
                self.semantic_labels.append(label)
                self.object_count += 1
        return segmented_ratio

    def predict(self, video_path: str):
        self.init_state(video_path)
        first_segmented_ratio = self.predict_frame()
        segmented_ratio_threshold = first_segmented_ratio - (self.covered_iou_tolerance * self.refine_frequency)
        finished = False
        global_frame_idx = 0
        last_frame_idx = 0
        updated = False
        while not finished:
            finished = True
            for sam2_frame_idx, sam2_obj_ids, sam2_masks_logits in self.sam2_video_predictor.propagate_in_video(
                    self.inference_state,
                    start_frame_idx=global_frame_idx):
                sam2_masks = sam2_masks_logits.squeeze(1) > 0.0
                segmented_ratio = get_segmented_ratio(sam2_masks)
                while_refine = (self.refine_sam2 and updated and
                                global_frame_idx % self.refine_frequency == 0 and
                                segmented_ratio < segmented_ratio_threshold)
                if while_refine:
                    global_frame_idx = last_frame_idx
                    torch.cuda.empty_cache()
                    self.init_state()
                    self.predict_frame(global_frame_idx)
                    finished = False
                    updated = False
                    break
                if segmented_ratio < segmented_ratio_threshold and last_frame_idx != sam2_frame_idx:
                    last_frame_idx = sam2_frame_idx
                    updated = True
                save_masks_detach = sam2_masks.detach().cpu()
                self.video_segments[global_frame_idx] = {"masks": save_masks_detach,
                                                         "object_ids": copy.deepcopy(sam2_obj_ids)}
                global_frame_idx += 1
                del sam2_masks
                torch.cuda.empty_cache()
        self.video_reader.close()

    def visualize_result(self, save_path: str):
        save_frames = []
        frame_idx_list = sorted(list(self.video_segments.keys()))
        for frame_idx in frame_idx_list:
            colored_image = visualize_masks(self.video_segments[frame_idx]["masks"],
                                            mask_id_list=self.video_segments[frame_idx]["object_ids"])
            save_frames.append(colored_image)
        height, width, _ = save_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(save_path, fourcc, self.video_reader.fps, (width, height))
        for frame in save_frames:
            video.write(frame)
        video.release()


if __name__ == "__main__":
    video_path = "test_video.mp4"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam2_predictor = SAM2Predictor(device=device, use_llm=True, sam2_type="LARGE", refine_sam2=True)
    sam2_predictor.predict(video_path)
    sam2_predictor.visualize_result("test_video_seg.mp4")
