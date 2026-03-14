import spaces  # must be first!
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
GRADIO_TMP = REPO_ROOT / ".gradio_cache"
GRADIO_TMP.mkdir(parents=True, exist_ok=True)

os.environ["GRADIO_TEMP_DIR"] = str(GRADIO_TMP)
print(f"Gradio temp/cache dir: {GRADIO_TMP}")

import torch
from argparse import Namespace
from test_single_video import *

import gradio as gr

device = "cuda" if torch.cuda.is_available() else "cpu"
yaml_args = OmegaConf.load(f"{REPO_ROOT}/../ckpt/model_config.yaml")
pipeline = None


@spaces.GPU
def fn(input_video):
    global pipeline, yaml_args, device
    if pipeline is None:
        pipeline = load_model(f"{REPO_ROOT}/../ckpt", yaml_args)
    
    input_video_basename = os.path.basename(input_video)
    input_tensor, orig_size, origin_fps = load_video_data(Namespace(
        input_video=input_video,
        height=480,
        width=640,
    ))
    depth = predict_depth(pipeline, input_tensor, orig_size, Namespace(
        window_size=81,
        overlap=21
    ))
    output_video = save_results(depth, origin_fps, Namespace(
        input_video=input_video,
        output_dir=GRADIO_TMP,
        grayscale=False
    ))

    return output_video


if __name__ == "__main__":
    inputs = [
        gr.Video(label="Input Video", autoplay=True),
    ]
    outputs = [
        gr.Video(label="Output Video", autoplay=True),
    ]

    demo = gr.Interface(
        fn=fn,
        title="DVD: Deterministic Video Depth Estimation with Generative Priors",
        description="""
            <strong>Please consider starring <span style="color: orange">&#9733;</span> our <a href="https://github.com/EnVision-Research/DVD" target="_blank" rel="noopener noreferrer">GitHub Repo</a> if you find this demo useful!</strong>
        """,
        inputs=inputs,
        outputs=outputs,
        examples=[
            [f"{REPO_ROOT}/../demo/drone.mp4"],
            [f"{REPO_ROOT}/../demo/robot_navi.mp4"]
        ]
    )

    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name="0.0.0.0",
        server_port=1324,
    )
