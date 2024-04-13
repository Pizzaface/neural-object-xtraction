import cv2
import numpy as np
import torch
import yaml
import torch.nn.functional as F
from tracker.model.network import XMem
from tracker.inference.inference_core import InferenceCore
from tracker.util.mask_mapper import MaskMapper
from torchvision import transforms
from tracker.util.range_transform import im_normalization

from tools.painter import mask_painter


class BaseTracker:
    def __init__(self, xmem_checkpoint, device) -> None:
        """
        device: model device
        xmem_checkpoint: checkpoint of XMem model
        """
        # load configurations
        with open('tracker/config/config.yaml', 'r') as stream:
            config = yaml.safe_load(stream)

        self.size = config['size']

        # initialise XMem
        network = XMem(config, xmem_checkpoint).to(device).eval()
        # initialise IncerenceCore
        self.tracker = InferenceCore(network, config)
        # data transformation
        self.im_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                im_normalization,
            ]
        )
        self.device = device

        # changable properties
        self.mapper = MaskMapper()
        self.initialised = False

        # # SAM-based refinement
        # self.sam_model = sam_model
        # self.resizer = Resize([256, 256])

    @torch.no_grad()
    def resize_mask(self, mask):
        # mask transform is applied AFTER mapper, so we need to post-process it in eval.py
        h, w = mask.shape[-2:]
        min_hw = min(h, w)
        return F.interpolate(
            mask,
            (int(h / min_hw * self.size), int(w / min_hw * self.size)),
            mode='nearest',
        )

    @torch.no_grad()
    def track(
        self, frame, first_frame_annotation=None, brightness_threshold=110
    ):
        mask, labels = self._prepare_mask_and_labels(first_frame_annotation)
        frame_tensor = self._prepare_frame_tensor(frame)
        out_mask = self._get_output_mask(frame_tensor, mask, labels)
        painted_image = self._paint_image(
            frame, out_mask, brightness_threshold
        )
        return out_mask, out_mask, painted_image

    def _prepare_mask_and_labels(self, first_frame_annotation):
        if first_frame_annotation is not None:
            mask, labels = self.mapper.convert_mask(first_frame_annotation)
            mask = torch.Tensor(mask).to(self.device)
            self.tracker.set_all_labels(list(self.mapper.remappings.values()))
        else:
            mask = labels = None
        return mask, labels

    def _prepare_frame_tensor(self, frame):
        return self.im_transform(frame).to(self.device)

    def _get_output_mask(self, frame_tensor, mask, labels):
        probs, _ = self.tracker.step(frame_tensor, mask, labels)
        out_mask = torch.argmax(probs, dim=0)
        return (out_mask.detach().cpu().numpy()).astype(np.uint8)

    def _paint_image(self, frame, out_mask, brightness_threshold):
        painted_image = frame.copy()
        objs = np.unique(out_mask)[1:]  # exclude background

        # run all objects concurrently

        for obj in objs:
            painted_image = self._apply_mask_to_image(
                out_mask, obj, painted_image, brightness_threshold
            )
        return painted_image

    def _apply_mask_to_image(
        self, out_mask, obj, painted_image, brightness_threshold
    ):
        mask = ((1 - out_mask) == obj).astype('uint8')
        masked_portion = mask_painter(
            painted_image, mask, mask_alpha=1, mask_color=0
        )
        x, y, w, h = cv2.boundingRect(mask)
        bright_pixels = cv2.threshold(
            painted_image[y : y + h, x : x + w],
            brightness_threshold,
            255,
            cv2.THRESH_BINARY_INV,
        )[1]
        painted_image[np.where(bright_pixels != 0)] = masked_portion[
            np.where(bright_pixels != 0)
        ]
        painted_image[np.where(bright_pixels == 0)] = 0.0
        return painted_image

    @torch.no_grad()
    def clear_memory(self):
        self.tracker.clear_memory()
        self.mapper.clear_labels()
        torch.cuda.empty_cache()
