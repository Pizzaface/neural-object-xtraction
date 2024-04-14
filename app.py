import gradio as gr
import gdown
import cv2
import numpy as np
import os
from track_anything import TrackingAnything
from track_anything import parse_augment
import requests
import json
import torchvision
import torch
from tools.painter import mask_painter
import psutil
import time

try:
    from mmcv.cnn import ConvModule
except ImportError:
    os.system('mim install mmcv')


def download_checkpoint(url, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print('download checkpoints ......')
        response = requests.get(url, stream=True)
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print('download successfully!')

    return filepath


def download_checkpoint_from_google_drive(file_id, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = str(os.path.join(folder, filename))

    if not os.path.exists(filepath):
        print(
            'Downloading checkpoints from Google Drive... tips: If you cannot see the progress bar, please try to download it manually \
              and put it in the checkpoints directory. E2FGVI-HQ-CVPR22.pth: https://github.com/MCG-NKU/E2FGVI(E2FGVI-HQ model)'
        )
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, filepath, quiet=False)
        print('Downloaded successfully!')

    return filepath


# convert points input to prompt state
def get_prompt(click_state, click_input):
    inputs = json.loads(click_input)
    points = click_state[0]
    labels = click_state[1]
    for input in inputs:
        points.append(input[:2])
        labels.append(input[2])
    click_state[0] = points
    click_state[1] = labels
    prompt = {
        'prompt_type': ['click'],
        'input_point': click_state[0],
        'input_label': click_state[1],
        'multimask_output': 'True',
    }
    return prompt


# extract frames from upload video
def get_frames_from_video(video_input, video_state):
    video_path = video_input
    frames = []
    user_name = time.time()
    operation_log = [
        ('', ''),
        (
            'Extracted frames from video. Please select a frame for tracking, then click the image to add a point to the tracking mask. Press "Add mask" when you are satisfied with the mask.',
            'Normal',
        ),
    ]
    fps = video_state.get('fps', 30)
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                current_memory_usage = psutil.virtual_memory().percent
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if current_memory_usage > 90:
                    operation_log = [
                        (
                            'Memory usage is too high (>90%). Stop the video extraction. Please reduce the video resolution or frame rate.',
                            'Error',
                        )
                    ]
                    print(
                        'Memory usage is too high (>90%). Please reduce the video resolution or frame rate.'
                    )
                    break
            else:
                break
    except (OSError, TypeError, ValueError, KeyError, SyntaxError) as e:
        print('read_frame_source:{} error. {}\n'.format(video_path, str(e)))

    if len(frames) == 0:
        operation_log = [
            ('', ''),
            ('Error! Please upload a video first.', 'Error'),
        ]
        return (
            {},
            '',
            None,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True, value=operation_log),
        )

    image_size = (frames[0].shape[0], frames[0].shape[1])
    # initialize video_state
    video_state = {
        'user_name': user_name,
        'video_name': os.path.split(video_path)[-1],
        'origin_images': frames,
        'painted_images': frames.copy(),
        'masks': [np.zeros((frames[0].shape[0], frames[0].shape[1]), np.uint8)]
        * len(frames),
        'logits': [None] * len(frames),
        'select_frame_number': 0,
        'fps': fps,
    }
    video_info = (
        'Video Name: {}, FPS: {}, Total Frames: {}, Image Size:{}'.format(
            video_state['video_name'],
            video_state['fps'],
            len(frames),
            image_size,
        )
    )
    model.samcontroller.sam_controler.reset_image()
    model.samcontroller.sam_controler.set_image(
        video_state['origin_images'][0]
    )
    return (
        video_state,
        video_info,
        video_state['origin_images'][0],
        gr.update(visible=True, maximum=len(frames), value=1),
        gr.update(visible=True, maximum=len(frames), value=len(frames)),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True, value=operation_log),
    )


# get the select frame from gradio slider
def select_template(
    image_selection_slider, video_state, interactive_state, mask_dropdown
):
    # images = video_state[1]
    image_selection_slider -= 1
    video_state['select_frame_number'] = image_selection_slider

    # once select a new template frame, set the image in sam

    model.samcontroller.sam_controler.reset_image()
    model.samcontroller.sam_controler.set_image(
        video_state['origin_images'][image_selection_slider]
    )

    # update the masks when select a new template frame
    # if video_state["masks"][image_selection_slider] is not None:
    # video_state["painted_images"][image_selection_slider] = mask_painter(video_state["origin_images"][image_selection_slider], video_state["masks"][image_selection_slider])
    if mask_dropdown:
        print('ok')
    operation_log = [
        ('', ''),
        (
            'Select frame {}. Try click image and add mask for tracking.'.format(
                image_selection_slider
            ),
            'Normal',
        ),
    ]

    return (
        video_state['painted_images'][image_selection_slider],
        video_state,
        interactive_state,
        operation_log,
    )


# set the tracking end frame
def get_end_number(track_pause_number_slider, video_state, interactive_state):
    interactive_state['track_end_number'] = track_pause_number_slider
    operation_log = [
        ('', ''),
        (
            'Set the tracking finish at frame {}'.format(
                track_pause_number_slider
            ),
            'Normal',
        ),
    ]

    return (
        video_state['painted_images'][track_pause_number_slider],
        interactive_state,
        operation_log,
    )


def get_resize_ratio(resize_ratio_slider, interactive_state):
    interactive_state['resize_ratio'] = resize_ratio_slider

    return interactive_state


def get_brightness_threshold(brightness_threshold_slider, interactive_state):
    interactive_state['brightness_threshold'] = brightness_threshold_slider

    return interactive_state


# use sam to get the mask
def sam_refine(
    video_state,
    point_prompt,
    click_state,
    interactive_state,
    evt: gr.SelectData,
):
    """
    Use the SAM model to refine the mask based on the user's input.
    Args:
        video_state: The current state of the video.
        point_prompt: The type of point to add (positive or negative).
        click_state: The current state of the click points.
        interactive_state: The current state of the interactive elements.
        evt: The event data from the click event.

    Returns: The painted image, updated video state, updated interactive state, and operation log.

    """
    if point_prompt == 'Positive':
        coordinate = '[[{},{},1]]'.format(evt.index[0], evt.index[1])
        interactive_state['positive_click_times'] += 1
    else:
        coordinate = '[[{},{},0]]'.format(evt.index[0], evt.index[1])
        interactive_state['negative_click_times'] += 1

    # prompt for sam model
    model.samcontroller.sam_controler.reset_image()
    first_frame = video_state['origin_images'][
        video_state['select_frame_number']
    ].copy()
    model.samcontroller.sam_controler.set_image(first_frame)
    prompt = get_prompt(click_state=click_state, click_input=coordinate)

    mask, logit, painted_image = model.first_frame_click(
        image=first_frame,
        points=np.array(prompt['input_point']),
        labels=np.array(prompt['input_label']),
        multimask=prompt['multimask_output'],
    )
    video_state['masks'][video_state['select_frame_number']] = mask
    video_state['logits'][video_state['select_frame_number']] = logit
    video_state['painted_images'][
        video_state['select_frame_number']
    ] = painted_image

    operation_log = [
        ('', ''),
        (
            'Click the image to add points for tracking. Press "Add mask" when you are satisfied with the mask.',
            'Normal',
        ),
        (
            'Click "Clear clicks" to clear the points history and reset the points.',
            'Normal',
        ),
        (
            'Click "Track Video" to track the video with the selected mask(s).',
            'Normal',
        )
    ]
    return painted_image, video_state, interactive_state, operation_log


def add_multi_mask(video_state, interactive_state, mask_dropdown):
    try:
        mask = video_state['masks'][video_state['select_frame_number']].copy()
        interactive_state['multi_mask']['masks'].append(mask)
        interactive_state['multi_mask']['mask_names'].append(
            'mask_{:03d}'.format(len(interactive_state['multi_mask']['masks']))
        )
        mask_dropdown.append(
            'mask_{:03d}'.format(len(interactive_state['multi_mask']['masks']))
        )
        select_frame, run_status = show_mask(
            video_state, interactive_state, mask_dropdown
        )

        operation_log = [
            ('', ''),
            (
                'Added a mask, use the mask select for target tracking.',
                'Normal',
            ),
        ]
    except:
        operation_log = [
            ('Please click the left image to generate mask.', 'Error'),
            ('', ''),
        ]
        return (
            video_state,
            interactive_state,
            mask_dropdown,
            [[], []],
            operation_log,
        )

    return (
        interactive_state,
        gr.update(
            choices=interactive_state['multi_mask']['mask_names'],
            value=mask_dropdown,
        ),
        select_frame,
        [[], []],
        operation_log,
    )


def clear_click(video_state, click_state):
    click_state = [[], []]
    if not video_state['origin_images']:
        operation_log = [('', ''), ('Please upload a video first.', 'Error')]
        return video_state, click_state, operation_log

    template_frame = video_state['origin_images'][
        video_state['select_frame_number']
    ].copy()
    operation_log = [
        ('', ''),
        ('Clear points history and refresh the image.', 'Normal'),
    ]
    return template_frame, click_state, operation_log


def remove_multi_mask(interactive_state, mask_dropdown):
    if len(mask_dropdown) == 0:
        operation_log = [
            ('', ''),
            ('No mask to remove. Please add a mask first.', 'Error'),
        ]
        return interactive_state, mask_dropdown, operation_log

    interactive_state['multi_mask']['mask_names'] = []
    interactive_state['multi_mask']['masks'] = []

    operation_log = [
        ('', ''),
        ('Remove all mask, please add new masks', 'Normal'),
    ]
    return interactive_state, gr.update(choices=[], value=[]), operation_log


def show_mask(video_state, interactive_state, mask_dropdown):
    mask_dropdown.sort()
    if not video_state['origin_images']:
        operation_log = [('', ''), ('Please upload a video first.', 'Error')]
        return video_state, operation_log

    select_frame = video_state['origin_images'][
        video_state['select_frame_number']
    ].copy()
    for i in range(len(mask_dropdown)):
        mask_number = int(mask_dropdown[i].split('_')[1]) - 1
        mask = interactive_state['multi_mask']['masks'][mask_number]
        select_frame = mask_painter(
            select_frame, mask.astype('uint8'), mask_color=mask_number + 2
        )

    operation_log = [
        ('', ''),
        (
            'Select {} for tracking'.format(mask_dropdown),
            'Normal',
        ),
    ]
    return select_frame, operation_log


def vos_tracking_video(video_state, interactive_state, mask_dropdown):
    operation_log = [
        ('', ''),
        (
            'Track the selected masks, and then you can select the masks.',
            'Normal',
        ),
    ]
    model.xmem.clear_memory()
    if interactive_state['track_end_number']:
        following_frames = video_state['origin_images'][
            video_state['select_frame_number'] : interactive_state[
                'track_end_number'
            ]
        ].copy()
    else:
        following_frames = video_state['origin_images'][
            video_state['select_frame_number'] :
        ].copy()

    if interactive_state['multi_mask']['masks']:
        if len(mask_dropdown) == 0:
            mask_dropdown = ['mask_001']
        mask_dropdown.sort()
        template_mask = interactive_state['multi_mask']['masks'][
            int(mask_dropdown[0].split('_')[1]) - 1
        ] * (int(mask_dropdown[0].split('_')[1]))
        for i in range(1, len(mask_dropdown)):
            mask_number = int(mask_dropdown[i].split('_')[1]) - 1
            template_mask = np.clip(
                template_mask
                + interactive_state['multi_mask']['masks'][mask_number]
                * (mask_number + 1),
                0,
                mask_number + 1,
            )
        video_state['masks'][
            video_state['select_frame_number']
        ] = template_mask
    else:
        template_mask = video_state['masks'][
            video_state['select_frame_number']
        ]
    fps = video_state['fps']

    # operation error
    if len(np.unique(template_mask)) == 1:
        template_mask[0][0] = 1
        operation_log = [
            (
                'Error! Please add at least one mask to track by clicking the left image.',
                'Error',
            ),
            ('', ''),
        ]
        # return video_output, video_state, interactive_state, operation_error

    masks, logits, painted_images = model.generator(
        images=following_frames,
        template_mask=template_mask,
        brightness_threshold=interactive_state['brightness_threshold'],
    )

    # clear GPU memory
    model.xmem.clear_memory()

    video_state['masks'][video_state['select_frame_number'] :] = masks
    video_state['logits'][video_state['select_frame_number'] :] = logits
    video_state['painted_images'] = painted_images

    video_output = generate_video_from_frames(
        video_state['painted_images'],
        output_path='./result/track/{}'.format(video_state['video_name']),
        fps=fps,
    )  # import video_input to name the output video
    interactive_state['inference_times'] += 1

    print(
        'For generating this tracking result, inference times: {}, click times: {}, positive: {}, negative: {}'.format(
            interactive_state['inference_times'],
            interactive_state['positive_click_times']
            + interactive_state['negative_click_times'],
            interactive_state['positive_click_times'],
            interactive_state['negative_click_times'],
        )
    )

    if interactive_state['mask_save']:
        if not os.path.exists(
            './result/mask/{}'.format(video_state['video_name'].split('.')[0])
        ):
            os.makedirs(
                './result/mask/{}'.format(
                    video_state['video_name'].split('.')[0]
                )
            )
        i = 0
        print('save mask')
        for mask in video_state['masks']:
            np.save(
                os.path.join(
                    './result/mask/{}'.format(
                        video_state['video_name'].split('.')[0]
                    ),
                    '{:05d}.npy'.format(i),
                ),
                mask,
            )
            i += 1

    return video_output, video_state, interactive_state, operation_log


# generate video after vos inference
def generate_video_from_frames(frames, output_path, fps=30):
    """
    Generates a video from a list of frames.

    Args:
        frames (list of numpy arrays): The frames to include in the video.
        output_path (str): The path to save the generated video.
        fps (int, optional): The frame rate of the output video. Defaults to 30.
    """
    frames = torch.from_numpy(np.asarray(frames))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    torchvision.io.write_video(
        output_path, frames, fps=fps, video_codec='libx264'
    )
    return output_path


if __name__ == '__main__':
    # args, defined in track_anything.py
    args = parse_augment()

    # check and download checkpoints if needed
    SAM_checkpoint_dict = {
        'vit_h': 'sam_vit_h_4b8939.pth',
        'vit_l': 'sam_vit_l_0b3195.pth',
        'vit_b': 'sam_vit_b_01ec64.pth',
    }
    SAM_checkpoint_url_dict = {
        'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
        'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
        'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
    }
    sam_checkpoint = SAM_checkpoint_dict[args.sam_model_type]
    sam_checkpoint_url = SAM_checkpoint_url_dict[args.sam_model_type]
    xmem_checkpoint = 'XMem-s012.pth'
    xmem_checkpoint_url = 'https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth'
    e2fgvi_checkpoint = 'E2FGVI-HQ-CVPR22.pth'
    e2fgvi_checkpoint_id = '10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3'

    folder = './checkpoints'
    SAM_checkpoint = download_checkpoint(
        sam_checkpoint_url, folder, sam_checkpoint
    )
    xmem_checkpoint = download_checkpoint(
        xmem_checkpoint_url, folder, xmem_checkpoint
    )
    e2fgvi_checkpoint = download_checkpoint_from_google_drive(
        e2fgvi_checkpoint_id, folder, e2fgvi_checkpoint
    )
    args.port = 12212
    args.device = 'cuda:0'
    # args.mask_save = True

    # initialize sam, xmem, e2fgvi models
    model = TrackingAnything(
        SAM_checkpoint, xmem_checkpoint, e2fgvi_checkpoint, args
    )

    title = """<p><h1 align="center">NOX-ify</h1></p>
        """
    description = """<p>
    An altered version of the Gradio demo for <a href='https://github.com/gaomingqi/Track-Anything'>Track Anything</a> that provides a matte mask for a tracked segment.
    </p>"""

    with gr.Blocks() as iface:
        """
        state for
        """
        click_state = gr.State([[], []])
        interactive_state = gr.State(
            {
                'inference_times': 0,
                'negative_click_times': 0,
                'positive_click_times': 0,
                'mask_save': args.mask_save,
                'multi_mask': {'mask_names': [], 'masks': []},
                'track_end_number': None,
                'resize_ratio': 1,
                'brightness_threshold': 70,
            }
        )

        video_state = gr.State(
            {
                'user_name': '',
                'video_name': '',
                'origin_images': None,
                'painted_images': None,
                'masks': None,
                'logits': None,
                'select_frame_number': 0,
                'fps': 30,
            }
        )
        gr.Markdown(title)
        gr.Markdown(description)

        with gr.Row():
            with gr.Column():
                video_input = gr.Video(
                    width=500,
                    height=300,
                    label='Upload video',
                )

            with gr.Column():
                extract_frames_button = gr.Button(
                    value='Process Video',
                    interactive=True,
                    variant='primary',
                )

                video_info = gr.Textbox(label='Video Info')
                resize_ratio_slider = gr.Slider(
                    minimum=0.02,
                    maximum=1,
                    step=0.02,
                    value=1,
                    label='Resize ratio',
                    visible=True,
                )


        with gr.Row():
            run_status = gr.HighlightedText(
                value=[
                ],
                visible=False,
                label='Operation Log',
            )

        with gr.Row():
            # put the template frame under the radio button
            with gr.Column():
                mask_dropdown = gr.Dropdown(
                    multiselect=True,
                    value=[],
                    label='Mask selection',
                    info='.',
                    visible=False,
                )

                brightness_threshold_slider = gr.Slider(
                    minimum=1,
                    maximum=255,
                    step=1,
                    value=70,
                    label='Brightness Threshold',
                    visible=False,
                )
                image_selection_slider = gr.Slider(
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=1,
                    label='Track start frame',
                    visible=False,
                )
                track_pause_number_slider = gr.Slider(
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=1,
                    label='Track end frame',
                    visible=False,
                )

                with gr.Row():
                    point_prompt = gr.Radio(
                        choices=['Positive', 'Negative'],
                        value='Positive',
                        label='Point prompt',
                        interactive=True,
                        visible=False,
                    )

                    with gr.Column():

                        clear_button_click = gr.Button(
                            value='Clear clicks',
                            interactive=True,
                            visible=False,
                        )
                        with gr.Row():
                            remove_mask_button = gr.Button(
                                value='Remove mask',
                                interactive=True,
                                visible=False,
                            )
                            add_mask_button = gr.Button(
                                value='Add mask',
                                interactive=True,
                                visible=False,
                            )

            with gr.Column():
                template_frame = gr.Image(
                    type='pil',
                    interactive=True,
                    label="Preview (click to add points)",
                    elem_id='template_frame',
                    visible=False,
                )
                tracking_video_predict_button = gr.Button(
                    value='Track Video', visible=False
                )

        with gr.Row():
            video_output = gr.Video(visible=False, label='Tracking result')

        # first step: get the video information
        extract_frames_button.click(
            fn=get_frames_from_video,
            inputs=[video_input, video_state],
            outputs=[
                video_state,
                video_info,
                template_frame,
                image_selection_slider,
                track_pause_number_slider,
                brightness_threshold_slider,
                point_prompt,
                clear_button_click,
                add_mask_button,
                template_frame,
                tracking_video_predict_button,
                video_output,
                mask_dropdown,
                remove_mask_button,
                run_status,
            ],
        )

        # second step: select images from slider
        image_selection_slider.release(
            fn=select_template,
            inputs=[image_selection_slider, video_state, interactive_state],
            outputs=[
                template_frame,
                video_state,
                interactive_state,
                run_status,
            ],
            api_name='select_image',
        )
        track_pause_number_slider.release(
            fn=get_end_number,
            inputs=[track_pause_number_slider, video_state, interactive_state],
            outputs=[template_frame, interactive_state, run_status],
            api_name='end_image',
        )
        resize_ratio_slider.release(
            fn=get_resize_ratio,
            inputs=[resize_ratio_slider, interactive_state],
            outputs=[interactive_state],
            api_name='resize_ratio',
        )

        brightness_threshold_slider.release(
            fn=get_brightness_threshold,
            inputs=[brightness_threshold_slider, interactive_state],
            outputs=[interactive_state],
            api_name='brightness_threshold',
        )

        # click select image to get mask using sam
        template_frame.select(
            fn=sam_refine,
            inputs=[video_state, point_prompt, click_state, interactive_state],
            outputs=[
                template_frame,
                video_state,
                interactive_state,
                run_status,
            ],
        )

        # add different mask
        add_mask_button.click(
            fn=add_multi_mask,
            inputs=[video_state, interactive_state, mask_dropdown],
            outputs=[
                interactive_state,
                mask_dropdown,
                template_frame,
                click_state,
                run_status,
            ],
        )

        remove_mask_button.click(
            fn=remove_multi_mask,
            inputs=[interactive_state, mask_dropdown],
            outputs=[interactive_state, mask_dropdown, run_status],
        )

        # tracking video from select image and mask
        tracking_video_predict_button.click(
            fn=vos_tracking_video,
            inputs=[video_state, interactive_state, mask_dropdown],
            outputs=[video_output, video_state, interactive_state, run_status],
        )

        # click to get mask
        mask_dropdown.change(
            fn=show_mask,
            inputs=[video_state, interactive_state, mask_dropdown],
            outputs=[template_frame, run_status],
        )

        # clear input
        video_input.clear(
            lambda: (
                {
                    'user_name': '',
                    'video_name': '',
                    'origin_images': None,
                    'painted_images': None,
                    'masks': None,
                    'logits': None,
                    'select_frame_number': 0,
                    'fps': 60,
                    'brightness_threshold': 70,
                },
                {
                    'inference_times': 0,
                    'negative_click_times': 0,
                    'positive_click_times': 0,
                    'mask_save': args.mask_save,
                    'multi_mask': {'mask_names': [], 'masks': []},
                    'track_end_number': 0,
                    'resize_ratio': 1,
                },
                [[], []],
                None,
                None,
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False, value=[]),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            ),
            [],
            [
                video_state,
                interactive_state,
                click_state,
                video_output,
                template_frame,
                tracking_video_predict_button,
                image_selection_slider,
                track_pause_number_slider,
                brightness_threshold_slider,
                point_prompt,
                clear_button_click,
                add_mask_button,
                template_frame,
                tracking_video_predict_button,
                video_output,
                mask_dropdown,
                remove_mask_button,
                run_status,
            ],
            queue=False,
            show_progress='full',
        )

        # points clear
        clear_button_click.click(
            fn=clear_click,
            inputs=[
                video_state,
                click_state,
            ],
            outputs=[template_frame, click_state, run_status],
        )

    iface.queue(max_size=2)
    iface.launch(debug=True, server_port=args.port, prevent_thread_lock=True)
