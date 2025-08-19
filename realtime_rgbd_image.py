import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from nets import Model

# intrinsics for d405
fx = 640.873108
fy = 640.873108
cx = 641.508728
cy = 356.237122
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

# Initializing the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Receiving IR images (left and right)
config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)

# Start pipeline
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

    # convert disparity to 3D point cloud
    points_3d = cv2.reprojectImageTo3D(disparity, Q)

    return points_3d


while True:
    # Waiting for one frame
    frames = pipeline.wait_for_frames()

    # Receiving IR frames
    left_infrared = frames.get_infrared_frame(1)
    right_infrared = frames.get_infrared_frame(2)

    # Converting frames to NumPy arrays
    imgL_raw = np.asanyarray(left_infrared.get_data())
    imgR_raw = np.asanyarray(right_infrared.get_data())
    imgL = np.stack((imgL_raw,)*3, axis=-1)
    imgR = np.stack((imgR_raw,)*3, axis=-1)

    # Getting a point cloud
    pred = inference(imgL, imgR, model, n_iter=4)
    # point_cloud = disparity_to_pointcloud(pred, Q)

    depth_view = pred
    depth_view = depth_view.astype("uint8")
    cv2.namedWindow("rgbd image", cv2.WINDOW_NORMAL)
    cv2.imshow("rgbd image", depth_view)
    if cv2.waitKey(1) == ord('q'):
        break

# Stop pipeline
pipeline.stop()
