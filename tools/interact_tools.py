from PIL import Image
import numpy as np

from .base_segmenter import BaseSegmenter
from .painter import mask_painter, point_painter


mask_color = 3
mask_alpha = 0.7
contour_color = 1
contour_width = 5
point_color_ne = 8
point_color_ps = 50
point_alpha = 0.9
point_radius = 15
contour_color = 2
contour_width = 5


class SamController:
    def __init__(self, SAM_checkpoint, model_type, device):
        self.sam_controler = BaseSegmenter(SAM_checkpoint, model_type, device)

    def first_frame_click(
        self,
        image: np.ndarray,
        points: np.ndarray,
        labels: np.ndarray,
        multimask=True,
        mask_color=3,
    ):
        """
        When you click the first frame, you will get the mask, logit and painted image.
        Args:
            image: np.ndarray
            points: np.ndarray
            labels: np.ndarray
            multimask: bool
            mask_color: int

        Returns: mask, logit, painted_image
        """
        neg_flag = labels[-1]
        if neg_flag == 1:
            prompts = {
                'point_coords': points,
                'point_labels': labels,
            }
            masks, scores, logits = self.sam_controler.predict(
                prompts, 'point', multimask
            )
            mask, logit = (
                masks[np.argmax(scores)],
                logits[np.argmax(scores), :, :],
            )
            prompts = {
                'point_coords': points,
                'point_labels': labels,
                'mask_input': logit[None, :, :],
            }
            masks, scores, logits = self.sam_controler.predict(
                prompts, 'both', multimask
            )
            mask, logit = (
                masks[np.argmax(scores)],
                logits[np.argmax(scores), :, :],
            )
        else:
            prompts = {
                'point_coords': points,
                'point_labels': labels,
            }
            masks, scores, logits = self.sam_controler.predict(
                prompts, 'point', multimask
            )
            mask, logit = (
                masks[np.argmax(scores)],
                logits[np.argmax(scores), :, :],
            )

        assert len(points) == len(labels)

        painted_image = mask_painter(
            image.copy(),
            mask.astype('uint8'),
            mask_color,
            mask_alpha
        )
        painted_image = point_painter(
            painted_image,
            np.squeeze(points[np.argwhere(labels > 0)], axis=1),
            point_color_ne,
            point_alpha,
            point_radius,
            contour_color,
            contour_width,
        )
        painted_image = point_painter(
            painted_image,
            np.squeeze(points[np.argwhere(labels < 1)], axis=1),
            point_color_ps,
            point_alpha,
            point_radius,
            contour_color,
            contour_width,
        )
        painted_image = Image.fromarray(painted_image)

        return mask, logit, painted_image
