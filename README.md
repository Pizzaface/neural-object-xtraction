## Neural Object eXtractor (NOX)

This will help you set up the user interface to create Matte Masks from tracked footage.

The code has been changed a lot to remove the inpainting features which I find unnecessary.

The code is based on the original code from the [Track Anything](https://github.com/gaomingqi/Track-Anything) with some changes to focus on the Mask generation rather than inpainting.

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

This will start the application and you can access it by going to `http://localhost:12212` in your browser.

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
6. When you want a mask to include a selected region - use "Positive" and when you want to exclude a region - use "Negative"
   - I recommend selecting "Negative" regions first, then "Positive" regions.
   - If you make a mistake, you can always click "Clear Clicks" to start over.
7. Once you're done, click "Add Mask" to add the mask to the list.
8. Do this until you have all the masks you need.
9. Then, click "Tracking" to start the tracking process.
10. Once the tracking is done, the video mask will be generated and show below the editing UI
   - If it doesn't look right, you can always go back and edit the brightness, which will help "clean" the mask to be tighter.
   - It's a bit of a trial and error process, but it's pretty quick once you get the hang of it.

</details>