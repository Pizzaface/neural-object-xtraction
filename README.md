<h1 align="center">
Neural Object eXtractor (NOX)
</h2>
<p align="center">
Uses Track Anything to generate a Matte Mask for a video file.

The code is based on the original code from the [Track Anything](https://github.com/gaomingqi/Track-Anything) with some changes to focus on the Mask generation rather than inpainting.
</p>

<details>
<summary>

## Requirements
</summary>

The following items are required to run the code (download and set them up if you haven't already)
1. Python 3.11
    - [Windows](https://www.python.org/downloads/release/python-3119/)
    - [Mac](https://www.python.org/downloads/macos/)

2. Once installed, you'll need to install [Poetry](https://python-poetry.org/docs/)
    - Run the following command in your terminal:
    ```bash
   pip install poetry
    ```
</details>

<details>
<summary>

## Installation
</summary>

#### This part requires you to be able to utilize your terminal:

- On Windows, press `Win + R`, type `powershell` and press `Enter`
- On Mac, press `Cmd + Space`, type `terminal` and press `Enter`
- On Linux, press `Ctrl + Alt + T`

To navigate to the folder where you want to download the code, use the `cd` command. For example, to navigate to the Desktop, you would type:
```bash
cd ~/<path>/<to>/<folder>
```

i.e.
```bash
cd ~/Downloads/NOX-ify
```

Once you're in the code directory, you need to install the required packages. We'll use Poetry to do this. Run the following command:
```bash
poetry install
```

This will create a new virtual environment and install the required packages. Once this is done, you can run the code using the following command:
```bash
poetry run python app.py
```

or to run using your CPU (not recommended):
```bash
poetry run python app.py --device "cpu"
```
Using the CPU will be **significantly** slower than using a GPU, but it's an option if you don't have a GPU available.)

<hr>

**Note:** The first time you run the code, it will take a while to start as it downloads the required models. Subsequent runs will be faster.

Once the application starts, you can access it by going to `http://localhost:12212` in your browser.

</details>

<details>
<summary>

## Usage
</summary>

The application is pretty straightforward to use.

Here's the steps to reproduce the Matte Mask:

1. Once you open the UI, upload your video file by clicking on the `Choose File` button.
2. You'll be presented with a preview of the video. You can resize the video if it's super large, or you don't have a lot of VRAM.
3. Then, click "Get Video Info" to get the video information.
4. Once the video's frames are chopped - the UI will show you the selection screen.
5. You'll select the "Brightness Threshold" to properly mask things like Teeth or Glasses
6. You'll want to pan through the frames, and find a frame where the object you want to mask is visible.
   - Use the Slider to pan through the frames, or select a frame number on the right.   
7. Begin Selecting your masks by clicking on regions in the image
   - When you want a mask to include a selected region select **"Positive"**
   - When you want to exclude a region - select **"Negative"**
   
8. Once you're done, click "Add Mask" to add the mask to the list.
9. Do this until you have all the masks you need.
10. Then, click "Tracking" to start the tracking process.
11. Once the tracking is done, the video mask will be generated and show below the editing UI (you can download it by clicking the "Download" button in the top right corner).
    - If it doesn't look right, you can always go back and edit the **Brightness Threshold**, which will help "clean" the mask to be tighter. The higher the threshold, the brighter the object needs to be to be included in the masked region.
    - It's a bit of a trial and error process, but it's pretty quick once you get the hang of it.

</details>

<hr>

## **Tips:**
   - I recommend selecting "Negative" regions first, then "Positive" regions.
   - If you make a mistake, you can always click "Clear Clicks" to start over.
   - Shorter videos are easier to work with, so if you can, try to keep the video length to a minimum.
   - This process takes a while:
     - The more masks you have, the longer it will take to track.
     - The more frames you have, the longer it will take to track.
     - The higher the resolution, the longer it will take to track.
     - The longer the video, the longer it will take to track.
   - You'll want a decent GPU to run this, especially if you have a lot of frames or a high-resolution video.
   - If you're running into issues, try reducing the resolution of the video, or the number of frames.


