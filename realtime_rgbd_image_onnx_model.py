import numpy as np
import cv2
import onnxruntime
import pyrealsense2 as rs

# интринсики для d405
fx = 640.873108  # Фокусное расстояние по x
fy = 640.873108  # Фокусное расстояние по y
cx = 641.508728  # Оптический центр x
cy = 356.237122  # Оптический центр y
baseline = 0.018

Q = np.float32(
    [[1, 0, 0, -cx],
     [0, 1, 0, -cy],
     [0, 0, 0, fx],
     [0, 0, 1 / baseline, 0]]
)

# Инициализация пайплайна RealSense
pipeline = rs.pipeline()
config = rs.config()

# Получение IR изображений (левого и правого)
config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)

# Запуск пайплайна
pipeline.start(config)


def inference(left, right, model, no_flow_model):
    # Get onnx model layer names (see convert_to_onnx.py for what these are)
    input1_name = model.get_inputs()[0].name
    input2_name = model.get_inputs()[1].name
    input3_name = model.get_inputs()[2].name
    output_name = model.get_outputs()[0].name

    # Decimate the image to half the original size for flow estimation network
    imgL_dw2 = cv2.resize(
        left, (left.shape[1] // 2, left.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
    imgR_dw2 = cv2.resize(
        right, (right.shape[1] // 2, right.shape[0] // 2), interpolation=cv2.INTER_LINEAR)

    # Reshape inputs to match what is expected
    imgL = left.transpose(2, 0, 1)
    imgR = right.transpose(2, 0, 1)
    imgL = np.ascontiguousarray(imgL[None, :, :, :]).astype("float32")
    imgR = np.ascontiguousarray(imgR[None, :, :, :]).astype("float32")

    imgL_dw2 = imgL_dw2.transpose(2, 0, 1)
    imgR_dw2 = imgR_dw2.transpose(2, 0, 1)
    imgL_dw2 = np.ascontiguousarray(imgL_dw2[None, :, :, :]).astype("float32")
    imgR_dw2 = np.ascontiguousarray(imgR_dw2[None, :, :, :]).astype("float32")

    print("Model Forwarding...")
    # First pass it just to get the flow
    pred_flow_dw2 = no_flow_model.run(
        [output_name], {input1_name: imgL_dw2, input2_name: imgR_dw2})[0]
    # Second pass gets us the disparity
    pred_disp = model.run([output_name], {
                          input1_name: imgL, input2_name: imgR, input3_name: pred_flow_dw2})[0]
    
    pred_disp = np.squeeze(pred_disp[:, 0, :, :])

    return pred_disp


while True:
    # Ожидание одного кадра
    frames = pipeline.wait_for_frames()

    # Получение IR кадров
    left_infrared = frames.get_infrared_frame(1)
    right_infrared = frames.get_infrared_frame(2)

    # Преобразование кадров в массивы NumPy
    imgL_raw = np.asanyarray(left_infrared.get_data())
    imgR_raw = np.asanyarray(right_infrared.get_data())
    imgL = np.stack((imgL_raw,)*3, axis=-1)
    imgR = np.stack((imgR_raw,)*3, axis=-1)

    no_flow_model_path = "models/crestereo_without_flow.onnx"
    model_path = "models/crestereo.onnx"

    model = onnxruntime.InferenceSession(model_path)
    no_flow_model = onnxruntime.InferenceSession(no_flow_model_path)

    pred = inference(imgL, imgR, model, no_flow_model)

    depth_view = pred
    depth_view = depth_view.astype("uint8")
    cv2.namedWindow("rgbd image", cv2.WINDOW_NORMAL)
    cv2.imshow("rgbd image", depth_view)
    if cv2.waitKey(1) == ord('q'):
        break

# Остановка пайплайна
pipeline.stop()