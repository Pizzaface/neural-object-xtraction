import cv2
from tqdm import tqdm

from tools.interact_tools import SamController
from tracker.base_tracker import BaseTracker
import numpy as np
import argparse


class TrackingAnything:
    def __init__(
        self, sam_checkpoint, xmem_checkpoint, e2fgvi_checkpoint, args
    ):
        self.args = args
        self.sam_checkpoint = sam_checkpoint
        self.xmem_checkpoint = xmem_checkpoint
        self.e2fgvi_checkpoint = e2fgvi_checkpoint
        self.samcontroller = SamController(
            self.sam_checkpoint, args.sam_model_type, args.device
        )
        self.xmem = BaseTracker(self.xmem_checkpoint, device=args.device)

    def first_frame_click(
        self,
        image: np.ndarray,
        points: np.ndarray,
        labels: np.ndarray,
        multimask=True,
    ):
        mask, logit, painted_image = self.samcontroller.first_frame_click(
            image, points, labels, multimask
        )
        return mask, logit, painted_image

    def generator(
        self, images: list, template_mask: np.ndarray, brightness_threshold=110
    ):

        masks = []
        logits = []
        painted_images = []
        for i in tqdm(range(len(images)), desc='Tracking image'):
            image_copy = images[i].copy()
            if i == 0:
                mask, logit, painted_image = self.xmem.track(
                    image_copy, template_mask, brightness_threshold
                )
                masks.append(mask)
                logits.append(logit)
                painted_images.append(painted_image)

            else:
                mask, logit, painted_image = self.xmem.track(image_copy, brightness_threshold=brightness_threshold)

                mask = 1 - mask
                mask = mask.astype(np.uint8)

                masks.append(mask)
                logits.append(logit)

                # remove the mask from the image
                painted_image = image_copy
                painted_image[mask == 1] = 0

                painted_image = cv2.cvtColor(painted_image, cv2.COLOR_BGR2GRAY)
                # apply the mask to the image

                # threshold the image
                bright_pixels = cv2.threshold(
                    painted_image, brightness_threshold, 255, cv2.THRESH_BINARY_INV
                )[1]

                painted_image[np.where(bright_pixels == 0)] = 255. # 1. is white
                painted_image[np.where(bright_pixels != 0)] = 0.

                # Make it black and white onlys
                painted_image = cv2.cvtColor(painted_image, cv2.COLOR_GRAY2BGR)

                painted_images.append(painted_image)


        return masks, logits, painted_images


def parse_augment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--sam_model_type', type=str, default='vit_h')
    parser.add_argument(
        '--port',
        type=int,
        default=12212,
        help='only useful when running gradio applications',
    )
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--mask_save', default=False)
    args = parser.parse_args()

    if args.debug:
        print(args)
    return args
