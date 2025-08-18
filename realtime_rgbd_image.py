import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from nets import Model

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

device = "cuda"
model_path = "models/crestereo_eth3d.pth"
model = Model(max_disp=256, mixed_precision=False, test_mode=True)
model.load_state_dict(torch.load(model_path), strict=True)
model.to(device)
model.eval()

# Инициализация пайплайна RealSense
pipeline = rs.pipeline()
config = rs.config()

# Получение IR изображений (левого и правого)
config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)

# Запуск пайплайна
pipeline.start(config)


def inference(left, right, model, n_iter=20):
    print("Model Forwarding...")
    imgL = left.transpose(2, 0, 1)
    imgR = right.transpose(2, 0, 1)
    imgL = np.ascontiguousarray(imgL[None, :, :, :])
    imgR = np.ascontiguousarray(imgR[None, :, :, :])

    imgL = torch.tensor(imgL.astype("float32")).to(device)
    imgR = torch.tensor(imgR.astype("float32")).to(device)

    imgL_dw2 = F.interpolate(
        imgL,
        size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
        mode="bilinear",
        align_corners=True,
    )
    imgR_dw2 = F.interpolate(
        imgR,
        size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
        mode="bilinear",
        align_corners=True,
    )
    with torch.inference_mode():
        pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=n_iter, flow_init=None)

        pred_flow = model(imgL, imgR, iters=n_iter, flow_init=pred_flow_dw2)
    pred_disp = torch.squeeze(pred_flow[:, 0, :, :]).cpu().detach().numpy()

    return pred_disp


def disparity_to_pointcloud(disparity, Q):
    disparity = np.float32(disparity)

    # Преобразование в 3D точки
    points_3d = cv2.reprojectImageTo3D(disparity, Q)

    return points_3d


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

    # Получение облака точек
    pred = inference(imgL, imgR, model, n_iter=20)
    # point_cloud = disparity_to_pointcloud(pred, Q)

    depth_view = pred
    depth_view = depth_view.astype("uint8")
    cv2.namedWindow("rgbd image", cv2.WINDOW_NORMAL)
    cv2.imshow("rgbd image", depth_view)
    if cv2.waitKey(1) == ord('q'):
        break

# Остановка пайплайна
pipeline.stop()
