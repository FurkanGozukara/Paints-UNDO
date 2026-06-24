# https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags


import os
import csv
import numpy as np
import onnxruntime as ort

from PIL import Image
from onnxruntime import InferenceSession
from torch.hub import download_url_to_file


global_model = None
global_csv = None


def create_session_options():
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = min(4, os.cpu_count() or 1)
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    sess_options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return sess_options


def preload_onnxruntime_cuda_libraries():
    if not hasattr(ort, "preload_dlls"):
        return
    try:
        ort.preload_dlls(directory="")
    except Exception as e:
        print(f"WD14 Tagger CUDA library preload skipped: {e}")


def create_inference_session(model_onnx_filename):
    sess_options = create_session_options()
    available_providers = ort.get_available_providers()

    if "CUDAExecutionProvider" in available_providers:
        preload_onnxruntime_cuda_libraries()
        try:
            model = InferenceSession(
                model_onnx_filename,
                sess_options=sess_options,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            active_providers = model.get_providers()
            if "CUDAExecutionProvider" in active_providers:
                print("WD14 Tagger using GPU (CUDA)")
            else:
                print(f"WD14 Tagger using CPU (CUDA provider failed, active providers: {active_providers})")
            return model
        except Exception as e:
            print(f"WD14 Tagger CUDA session failed, falling back to CPU: {e}")

    model = InferenceSession(
        model_onnx_filename,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    print("WD14 Tagger using CPU (CUDA not available)")
    return model


def download_model(url, local_path):
    if os.path.exists(local_path):
        return local_path

    temp_path = local_path + '.tmp'
    download_url_to_file(url=url, dst=temp_path)
    os.rename(temp_path, local_path)
    return local_path


def default_interrogator(image, threshold=0.35, character_threshold=0.85, exclude_tags=""):
    global global_model, global_csv

    model_name = "wd-v1-4-moat-tagger-v2"

    model_onnx_filename = download_model(
        url=f'https://huggingface.co/lllyasviel/misc/resolve/main/{model_name}.onnx',
        local_path=f'./{model_name}.onnx',
    )

    model_csv_filename = download_model(
        url=f'https://huggingface.co/lllyasviel/misc/resolve/main/{model_name}.csv',
        local_path=f'./{model_name}.csv',
    )

    if global_model is not None:
        model = global_model
    else:
        model = create_inference_session(model_onnx_filename)
        global_model = model

    input = model.get_inputs()[0]
    height = input.shape[1]

    if isinstance(image, str):
        image = Image.open(image)  # RGB
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    else:
        image = image

    ratio = float(height) / max(image.size)
    new_size = tuple([int(x*ratio) for x in image.size])
    image = image.resize(new_size, Image.LANCZOS)
    square = Image.new("RGB", (height, height), (255, 255, 255))
    square.paste(image, ((height-new_size[0])//2, (height-new_size[1])//2))

    image = np.array(square).astype(np.float32)
    image = image[:, :, ::-1]  # RGB -> BGR
    image = np.expand_dims(image, 0)

    if global_csv is not None:
        csv_lines = global_csv
    else:
        csv_lines = []
        with open(model_csv_filename) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                csv_lines.append(row)
        global_csv = csv_lines

    tags = []
    general_index = None
    character_index = None
    for line_num, row in enumerate(csv_lines):
        if general_index is None and row[2] == "0":
            general_index = line_num
        elif character_index is None and row[2] == "4":
            character_index = line_num
        tags.append(row[1])

    label_name = model.get_outputs()[0].name
    probs = model.run([label_name], {input.name: image})[0]

    result = list(zip(tags, probs[0]))

    general = [item for item in result[general_index:character_index] if item[1] > threshold]
    character = [item for item in result[character_index:] if item[1] > character_threshold]

    all = character + general
    remove = [s.strip() for s in exclude_tags.lower().split(",")]
    all = [tag for tag in all if tag[0] not in remove]

    res = ", ".join((item[0].replace("(", "\\(").replace(")", "\\)") for item in all)).replace('_', ' ')
    return res
