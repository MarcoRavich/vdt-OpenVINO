import argparse
import cv2
import numpy as np
from openvino.runtime import Core, PartialShape

DEVICE_PRIORITY = "AUTO:GPU,CPU"

def load_model(model_path, static_shape, device=DEVICE_PRIORITY):
    core = Core()
    model = core.read_model(model_path)
    input_layer = model.inputs[0]

    # If static_shape is provided (tuple), set it
    if static_shape is not None:
        # static_shape: (N, C, H, W)
        model.reshape({input_layer.any_name: PartialShape(static_shape)})

    compiled_model = core.compile_model(model, device)
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    return compiled_model, input_layer, output_layer

def preprocess_frame(frame, input_shape):
    _, c, h, w = input_shape
    frame_resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC)
    frame_transposed = frame_resized.transpose(2, 0, 1)
    frame_norm = frame_transposed.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(frame_norm, axis=0)
    return input_tensor, frame_resized

def postprocess_output(output, orig_shape):
    output = np.squeeze(output)
    if output.ndim == 3:
        output = output.transpose(1, 2, 0)
    output = np.clip(output, 0, 1)
    output_uint8 = (output * 255).round().astype(np.uint8)
    if output_uint8.shape[:2] != orig_shape[:2]:
        output_uint8 = cv2.resize(output_uint8, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_CUBIC)
    return output_uint8

def run_video_inference(model_path, input_video, output_video, device=DEVICE_PRIORITY, force_hw=None):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {frame_count} frames...")

    # Read first frame to get shape (or use --force-hw)
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Cannot read first frame from video.")

    # Model expects [N, C, H, W] (usually 1,3,H,W), use video frame size or --force-hw
    if force_hw:
        h, w = force_hw
    else:
        h, w = frame.shape[0], frame.shape[1]

    static_shape = (1, 3, h, w)
    compiled_model, input_layer, output_layer = load_model(model_path, static_shape, device)
    input_shape = (1, 3, h, w)

    # Process first frame
    input_tensor, _ = preprocess_frame(frame, input_shape)
    result = compiled_model({input_layer.any_name: input_tensor})
    output = result[output_layer.any_name]
    out_frame = postprocess_output(output, frame.shape)
    out.write(out_frame)

    frame_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        input_tensor, _ = preprocess_frame(frame, input_shape)
        result = compiled_model({input_layer.any_name: input_tensor})
        output = result[output_layer.any_name]
        out_frame = postprocess_output(output, frame.shape)
        out.write(out_frame)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx}/{frame_count} frames")

    cap.release()
    out.release()
    print(f"Inference finished. Output saved as {output_video}")

def main():
    parser = argparse.ArgumentParser(description="OpenVINO 2025.1 video inference for VHS enhancement models with AUTO device selection and dynamic input support.")
    parser.add_argument("model_path", help="Path to the ONNX or OpenVINO IR model file")
    parser.add_argument("input_video", help="Path to the input video file")
    parser.add_argument("output_video", help="Path to save the enhanced output video")
    parser.add_argument("--device", default=DEVICE_PRIORITY, help="Device for OpenVINO inference (default: AUTO:GPU,CPU)")
    parser.add_argument("--force-hw", nargs=2, type=int, metavar=('H', 'W'),
                        help="Force input height and width (overrides video frame size, e.g. --force-hw 576 720 for PAL)")
    args = parser.parse_args()
    run_video_inference(args.model_path, args.input_video, args.output_video, device=args.device, force_hw=args.force_hw)

if __name__ == "__main__":
    main()
