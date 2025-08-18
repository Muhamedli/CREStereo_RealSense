import torch
import torch.nn.functional as F
import numpy as np
import cv2
import open3d as o3d
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

    # Маска для валидных точек (исключаем точки с бесконечной/некорректной глубиной)
    mask = (disparity > disparity.min()) & (disparity != 0)

    # Фильтрация точек
    points_3d = points_3d[mask]

    return points_3d


def create_grid(plane, size, step, color):
    lines = []
    # Генерация линий XY
    if plane == "xy":
        for i in np.arange(-size, size + step, step):
            lines.append([[i, -size, 0], [i, size, 0]])  # Линии параллельные Y
            lines.append([[-size, i, 0], [size, i, 0]])  # Линии параллельные X
    # Генерация линий YZ
    elif plane == "yz":
        for i in np.arange(-size, size + step, step):
            lines.append([[0, i, -size], [0, i, size]])  # Линии параллельные Z
            lines.append([[0, -size, i], [0, size, i]])  # Линии параллельные Y
    # Генерация линий XZ
    elif plane == "xz":
        for i in np.arange(-size, size + step, step):
            lines.append([[i, 0, -size], [i, 0, size]])  # Линии параллельные Z
            lines.append([[-size, 0, i], [size, 0, i]])  # Линии параллельные X

    # Создаем LineSet
    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(np.array(lines).reshape(-1, 3))
    grid.lines = o3d.utility.Vector2iVector(
        np.arange(len(lines) * 2).reshape(-1, 2))
    grid.colors = o3d.utility.Vector3dVector(
        [color for _ in range(len(lines))])

    return grid


if __name__ == "__main__":

    left_img = cv2.imread("d405/img/left_Infrared.png")
    right_img = cv2.imread("d405/img/right_Infrared.png")
    in_h, in_w = left_img.shape[:2]

    # Resize image in case the GPU memory overflows
    eval_h, eval_w = (in_h, in_w)
    assert eval_h % 8 == 0, "input height should be divisible by 8"
    assert eval_w % 8 == 0, "input width should be divisible by 8"

    imgL = cv2.resize(left_img, (eval_w, eval_h),
                      interpolation=cv2.INTER_LINEAR)
    imgR = cv2.resize(right_img, (eval_w, eval_h),
                      interpolation=cv2.INTER_LINEAR)

    pred = inference(imgL, imgR, model, n_iter=20)

    depth_view = pred
    depth_view = depth_view.astype("uint8")
    cv2.namedWindow("rgbd image", cv2.WINDOW_NORMAL)
    cv2.imshow("rgbd image", depth_view)
    cv2.waitKey(0)

    point_cloud = disparity_to_pointcloud(pred, Q)

    # Добавляем оси
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0]
    )

    # Создаем сетку
    grid_xy = create_grid(plane="xy", size=0.5, step=0.05, color=[0, 0, 0])
    grid_yz = create_grid(plane="yz", size=0.5, step=0.05, color=[0, 0, 0])
    grid_xz = create_grid(plane="xz", size=0.5, step=0.05, color=[0, 0, 0])

    # Создаем объект PointCloud из Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    # Разрежение облака точек
    pcd = pcd.voxel_down_sample(voxel_size=0.002)

    # Визуализация облака точек
    o3d.visualization.draw_geometries(
        [pcd, coord_frame, grid_xy, grid_yz, grid_xz],
        window_name="point cloud",
        width=1280,
        height=720)
