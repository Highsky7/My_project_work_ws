#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import argparse
import time
import cv2
import torch
import numpy as np
import cv2.ximgproc as ximgproc
import torch.backends.cudnn as cudnn
import threading
import math
from collections import deque

from pathlib import Path as PathlibPath
from nav_msgs.msg import Path as RosPath
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float64

# (중요) utils.py 내 함수들을 그대로 사용한다고 가정
from utils.utils import (
    time_synchronized,
    select_device,
    increment_path,
    AverageMeter,
    LoadCamera,
    LoadImages,
    lane_line_mask,
    letterbox
)

##########################################
# 1. argparse
##########################################
def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='/home/highsky/yolopv2.pt',
                        help='YOLOPv2 모델(.pt) 파일 경로 (차선 인식 전용)')
    parser.add_argument('--source', type=str,
                        default='/home/highsky/Videos/Webcam/bev영상1.mp4',
                        help='비디오/이미지 경로 또는 웹캠 소스(예: 2=/dev/video2)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='추론 시 입력 사이즈 (pixels)')
    parser.add_argument('--device', default='0',
                        help='cuda device (예: 0,1) 또는 cpu')
    parser.add_argument('--lane-thres', type=float, default=0.8,
                        help='차선 세그멘테이션 스코어 임계값 (0~1)')
    parser.add_argument('--nosave', action='store_false',
                        help='결과 영상을 저장하지 않음')
    parser.add_argument('--project', default='/home/highsky/My_project_work_ws/runs/detect',
                        help='결과 저장 기본 디렉터리')
    parser.add_argument('--name', default='exp',
                        help='프로젝트/이름 => runs/detect/exp')
    parser.add_argument('--exist-ok', action='store_false',
                        help='이미 디렉터리가 존재해도 덮어쓸지 여부')
    parser.add_argument('--frame-skip', type=int, default=0,
                        help='N 프레임을 건너뛸 때 사용 (현재 미사용)')
    return parser

##########################################
# 2. MiDaS Depth Estimator
##########################################
class DepthEstimator:
    def __init__(self, model_type="MiDaS_small"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rospy.loginfo(f"[INFO] Loading MiDaS model ({model_type})...")
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        self.midas.to(self.device)
        self.midas.eval()
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        if "DPT" in model_type:
            self.transform = self.midas_transforms.dpt_transform
        else:
            self.transform = self.midas_transforms.small_transform
        rospy.loginfo("[INFO] MiDaS model loaded successfully.")

    def run(self, cv2_image):
        img_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)
        with torch.no_grad():
            prediction = self.midas(input_batch)
            depth_map = torch.squeeze(prediction).cpu().numpy()
        min_val, max_val = depth_map.min(), depth_map.max()
        denom = (max_val - min_val) if (max_val > min_val) else 1e-8
        depth_map = (depth_map - min_val) / denom
        return depth_map

##########################################
# 3. IPM (인버스 퍼스펙티브 맵) 보조 함수들
##########################################
def get_birds_eye_view(frame, src_points, dst_size=(400, 600)):
    w, h = dst_size
    dst_points = np.array([
        [0,   h-1],
        [w-1, h-1],
        [w-1, 0],
        [0,   0]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_points.astype(np.float32), dst_points)
    Minv = cv2.getPerspectiveTransform(dst_points, src_points.astype(np.float32))
    birds_eye = cv2.warpPerspective(frame, M, (w, h), flags=cv2.INTER_LINEAR)
    return birds_eye, M, Minv

def lane_spline_fitting_in_birdeye(lane_mask_bev):
    indices = np.argwhere(lane_mask_bev > 127)
    if len(indices) < 10:
        return []
    indices = indices[indices[:, 0].argsort()]
    step = 5
    height = lane_mask_bev.shape[0]
    y_vals, x_vals = [], []
    for y in range(0, height, step):
        y_bin = indices[(indices[:, 0] >= y) & (indices[:, 0] < y + step)]
        if len(y_bin) > 0:
            mean_x = np.mean(y_bin[:, 1])
            y_center = y + (step / 2)
            y_vals.append(y_center)
            x_vals.append(mean_x)
    if len(y_vals) < 3:
        return []
    y_vals = np.array(y_vals, dtype=np.float32)
    x_vals = np.array(x_vals, dtype=np.float32)
    poly_coefs = np.polyfit(y_vals, x_vals, deg=2)
    y_new = np.linspace(y_vals[0], y_vals[-1], num=50)
    x_new = poly_coefs[0]*y_new**2 + poly_coefs[1]*y_new + poly_coefs[2]
    curve_xy = list(zip(x_new, y_new))
    return curve_xy

def project_lane_back(curve_xy, Minv):
    if not curve_xy:
        return []
    ones = np.ones((len(curve_xy), 1), dtype=np.float32)
    pts_bev = np.hstack([np.array(curve_xy, dtype=np.float32), ones])
    pts_bev = pts_bev.reshape(-1, 3).T
    pts_src = np.dot(Minv, pts_bev)
    pts_src[0, :] /= (pts_src[2, :] + 1e-8)
    pts_src[1, :] /= (pts_src[2, :] + 1e-8)
    pts_src = pts_src[:2, :].T
    return [(float(p[0]), float(p[1])) for p in pts_src]

##########################################
# 4. 경로 시각화 보조
##########################################
def project_path_to_image(disp_bgr, path_pts, fx, fy, cx, cy):
    h, w = disp_bgr.shape[:2]
    color = (0, 0, 255)
    thickness = 2
    prev_px, prev_py = None, None
    for (x_3d, z_3d) in path_pts:
        if z_3d <= 0.01:
            continue
        px = (x_3d * fx / z_3d) + cx
        py = (0.0 * fy / z_3d) + cy
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(disp_bgr, (int(px), int(py)), 4, color, -1)
            if prev_px is not None and prev_py is not None:
                cv2.line(disp_bgr, (int(prev_px), int(prev_py)), (int(px), int(py)), color, thickness)
            prev_px, prev_py = px, py
        else:
            prev_px, prev_py = None, None

##########################################
# 5. ROS Marker 퍼블리시
##########################################
def publish_vehicle_marker(pub_vehicle_marker):
    marker = Marker()
    marker.header.stamp = rospy.Time.now()
    marker.header.frame_id = "map"
    marker.ns = "vehicle"
    marker.id = 1
    marker.type = Marker.ARROW
    marker.action = Marker.ADD
    marker.scale.x = 0.5
    marker.scale.y = 0.2
    marker.scale.z = 0.2
    marker.color.r = 0.0
    marker.color.g = 0.0
    marker.color.b = 1.0
    marker.color.a = 1.0
    marker.points = [Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0)]
    pub_vehicle_marker.publish(marker)

def publish_lanes_3d(points_3d, pub_lane_marker):
    marker = Marker()
    marker.header.stamp = rospy.Time.now()
    marker.header.frame_id = "map"
    marker.ns = "lane"
    marker.id = 0
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.05
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    for (x_3d, y_3d, z_3d) in points_3d:
        pt = Point()
        pt.x = z_3d   # map 기준: x=전방
        pt.y = x_3d   # map 기준: y=좌우
        pt.z = y_3d
        marker.points.append(pt)
    pub_lane_marker.publish(marker)

##########################################
# 6. Pure Pursuit 기반 조향각 계산
##########################################
def compute_steering_angle_pure_pursuit(path_pts, lookahead_distance=5.0, wheelbase=2.5):
    """
    path_pts: [(x, z), ...] (차량 좌표계, x: 좌우, z: 전방)
    """
    for pt in path_pts:
        x, z = pt
        dist = math.sqrt(x*x + z*z)
        if dist >= lookahead_distance and z > 0.001:
            alpha = math.atan2(x, z)
            delta = math.atan2(2 * wheelbase * math.sin(alpha), dist)
            return math.degrees(delta)
    return 0.0

##########################################
# 7. 경로 smoothing
##########################################
def smooth_path(path_buffer):
    all_x, all_z = [], []
    for path in path_buffer:
        for (x, y, z) in path:
            all_x.append(x)
            all_z.append(z)
    if len(all_x) < 3:
        return []
    poly_coefs = np.polyfit(all_z, all_x, deg=2)
    z_new = np.linspace(min(all_z), max(all_z), num=50)
    x_new = poly_coefs[0]*z_new**2 + poly_coefs[1]*z_new + poly_coefs[2]
    smoothed_path = [(x, 0.0, z) for x, z in zip(x_new, z_new)]
    return smoothed_path

##########################################
# 8. 차선 기반 3D 경로 생성 (IPM 기반)
##########################################
def build_robust_path_ipm(thin_mask, depth_map, depth_scale,
                          fx, fy, cx, cy,
                          trap_points):
    h, w = thin_mask.shape[:2]
    mask_3ch = cv2.cvtColor(thin_mask, cv2.COLOR_GRAY2BGR)
    bev_mask, M_bev, Minv_bev = get_birds_eye_view(
        mask_3ch,
        src_points=trap_points.astype(np.float32),
        dst_size=(400, 600)
    )
    bev_gray = cv2.cvtColor(bev_mask, cv2.COLOR_BGR2GRAY)
    curve_xy = lane_spline_fitting_in_birdeye(bev_gray)
    curve_uv = project_lane_back(curve_xy, Minv_bev)
    path_pts_3d = []
    for (u, v) in curve_uv:
        if not (0 <= int(u) < w and 0 <= int(v) < h):
            continue
        d_val = depth_map[int(v), int(u)]
        z_val = depth_scale / (d_val + 1e-3)
        x_3d = (u - cx) * z_val / fx
        path_pts_3d.append((x_3d, 0.0, z_val))
    path_pts_3d.sort(key=lambda p: p[2])
    return path_pts_3d

##########################################
# 9. [변경] MiDaS 기반 꼬깔(traffic cone) 검출 함수
##########################################
def parse_cone_detections_depth(depth_map, im0s, depth_scale, fx, fy, cx, cy,
                                min_area=50, max_area=5000, depth_thresh=0.2, color_ratio_thresh=0.5):
    """
    depth_map: 0~1로 정규화된 깊이 맵
    im0s: 원본 이미지 (BGR)
    depth_scale: 사용자가 지정한 깊이 스케일
    fx, fy, cx, cy: 카메라 내재 파라미터
    min_area, max_area: 컨투어 영역 최소/최대 면적
    depth_thresh: 가까운 객체로 간주할 depth 임계값 (예: 0.2 이하)
    color_ratio_thresh: 후보 영역 내 오렌지 색(traffic cone의 대표 색상)의 비율 임계값 (0~1)
    """
    # 1. 깊이 맵을 8비트로 변환 후, 낮은 depth(즉, 가까운 영역) 추출
    depth_8u = np.uint8(depth_map * 255)
    ret, thresh = cv2.threshold(depth_8u, int(depth_thresh * 255), 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 2. 원본 이미지에서 오렌지 색 영역 검출 (HSV 기준)
    hsv = cv2.cvtColor(im0s, cv2.COLOR_BGR2HSV)
    # OpenCV의 Hue 범위: 0~179, 오렌지는 대략 5~25, Saturation 및 Value는 충분히 높은 값으로
    mask_color = cv2.inRange(hsv, (5, 100, 100), (25, 255, 255))
    
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        if aspect_ratio < 0.5 or aspect_ratio > 1.5:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # 후보 영역 내 오렌지 색 비율 계산
        roi_mask = mask_color[y:y+h, x:x+w]
        if roi_mask.size == 0:
            continue
        orange_ratio = np.count_nonzero(roi_mask) / roi_mask.size
        if orange_ratio < color_ratio_thresh:
            continue
        d_val = depth_map[cY, cX]
        z_val = depth_scale / (d_val + 1e-3)
        x_3d = (cX - cx) * z_val / fx
        y_3d = 0.0
        candidates.append((x_3d, y_3d, z_val))
    return candidates

##########################################
# 10. 웹캠 영상 녹화 보조
##########################################
def make_webcam_video(record_frames, save_dir: PathlibPath, stem_name: str, real_duration: float):
    if len(record_frames) == 0:
        rospy.loginfo("[INFO] 저장할 웹캠 프레임이 없습니다.")
        return
    num_frames = len(record_frames)
    if real_duration <= 0:
        real_duration = 1e-6
    real_fps = num_frames / real_duration
    rospy.loginfo("[INFO] 웹캠 녹화: 총 %d 프레임, %.2f초 => FPS 약 %.2f", num_frames, real_duration, real_fps)
    save_path = str(save_dir / f"{stem_name}_webcam.mp4")
    h, w = record_frames[0].shape[:2]
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), real_fps, (w, h))
    for f in record_frames:
        out.write(f)
    out.release()
    rospy.loginfo("[INFO] 웹캠 결과 영상 저장 완료: %s", save_path)

##########################################
# 11. 메인 detect_and_publish
##########################################
def detect_and_publish(opt, pub_mask, pub_path,
                       pub_lane_marker, pub_vehicle_marker, pub_steer):
    cv2.setUseOptimized(True)
    cv2.setNumThreads(0)
    cudnn.benchmark = True
    bridge = CvBridge()

    source = opt.source
    weights = opt.weights
    lane_thr = opt.lane_thres
    imgsz = opt.img_size
    save_img = not opt.nosave

    save_dir = PathlibPath(increment_path(PathlibPath(opt.project) / opt.name, exist_ok=opt.exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)
    inf_time = AverageMeter()
    vid_path, vid_writer = None, None

    rospy.loginfo("[INFO] Loading YOLOPv2 (차선 인식 전용) 모델: %s", weights)
    device = select_device(opt.device)
    model = torch.jit.load(weights, map_location=device)
    half = (device.type != 'cpu')
    model = model.half() if half else model.float()
    model.eval()

    depth_est = DepthEstimator(model_type="MiDaS_small")

    if source.isdigit():
        rospy.loginfo(f"[INFO] 웹캠(장치 ID={source}) 입력...")
        dataset = LoadCamera(source, img_size=imgsz, stride=32)
    else:
        rospy.loginfo(f"[INFO] 파일 입력: {source}")
        dataset = LoadImages(source, img_size=imgsz, stride=32)

    record_frames = []
    start_time = None

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    fx = 663.13788013 * 2
    fy = 663.85577581 * 2
    cx = 327.61915049 * 2
    cy = 249.78533605 * 2
    depth_scale = 0.5724

    trap_points = np.array([
        [0,   720],
        [980, 720],
        [980, 0],
        [0,   0]
    ], dtype=np.int32)

    path_buffer = deque(maxlen=10)

    if dataset.mode == 'stream':
        frame_queue = deque(maxlen=5)
        def frame_producer():
            for item in dataset:
                frame_queue.append(item)
            frame_queue.append(None)
        prod_thread = threading.Thread(target=frame_producer)
        prod_thread.daemon = True
        prod_thread.start()
        rospy.loginfo("[DEBUG] 웹캠 비동기 프레임 생산 시작")

        while not rospy.is_shutdown():
            if len(frame_queue) == 0:
                continue
            frame_data = frame_queue.popleft()
            if frame_data is None:
                rospy.loginfo("[INFO] 프레임 종료 신호 수신")
                break

            path_item, img, im0s, vid_cap = frame_data
            if start_time is None:
                start_time = time.time()

            enhanced_im0s = cv2.cvtColor(im0s, cv2.COLOR_BGR2RGB)
            enhanced_im0s = cv2.cvtColor(enhanced_im0s, cv2.COLOR_RGB2BGR)
            img, ratio, pad = letterbox(enhanced_im0s, (imgsz, imgsz), stride=32)
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img_t = torch.from_numpy(img).to(device)
            img_t = img_t.half() if half else img_t.float()
            img_t /= 255.0
            if img_t.ndimension() == 3:
                img_t = img_t.unsqueeze(0)

            t1 = time_synchronized()
            with torch.no_grad():
                # YOLOPv2는 차선 인식에만 사용됨 (출력: [det_seg, seg, ll])
                _, seg, ll = model(img_t)
            t2 = time_synchronized()
            inf_time.update(t2 - t1, img_t.size(0))

            ll_seg = lane_line_mask(ll, threshold=lane_thr)
            binary_mask = (ll_seg * 255).astype(np.uint8)
            thin_mask = ximgproc.thinning(binary_mask, thinningType=ximgproc.THINNING_ZHANGSUEN)

            roi_mask = np.zeros_like(thin_mask, dtype=np.uint8)
            cv2.fillPoly(roi_mask, [trap_points], 255)
            thin_mask = cv2.bitwise_and(thin_mask, roi_mask)

            depth_map = depth_est.run(im0s)
            if depth_map.shape[:2] != (im0s.shape[0], im0s.shape[1]):
                depth_map = cv2.resize(depth_map, (im0s.shape[1], im0s.shape[0]),
                                       interpolation=cv2.INTER_LINEAR)

            lane_pixel_count = cv2.countNonZero(thin_mask)
            lane_exist_threshold = 300
            use_lane = (lane_pixel_count > lane_exist_threshold)

            if use_lane:
                path_3d = build_robust_path_ipm(thin_mask, depth_map, depth_scale,
                                                fx, fy, cx, cy, trap_points)
            else:
                # YOLO 기반 꼬깔 검출 함수는 제거하고, 오직 MiDaS 기반 검출 사용
                cone_candidates = parse_cone_detections_depth(depth_map, im0s, depth_scale, fx, fy, cx, cy,
                                                              min_area=50, max_area=5000, depth_thresh=0.2, color_ratio_thresh=0.5)
                if cone_candidates:
                    cone_candidates.sort(key=lambda p: p[2])
                    nearest_cone = cone_candidates[0]
                    num_pts = 50
                    path_3d = []
                    for i in range(num_pts):
                        t = float(i) / (num_pts - 1)
                        path_3d.append(( (1-t)*0.0 + t*nearest_cone[0],
                                         (1-t)*0.0 + t*nearest_cone[1],
                                         (1-t)*0.0 + t*nearest_cone[2] ))
                else:
                    path_3d = []

            if path_3d:
                path_buffer.append(path_3d)
                smoothed_path_3d = smooth_path(path_buffer)
            else:
                smoothed_path_3d = []

            path_for_angle = [(pt[0], pt[2]) for pt in smoothed_path_3d]
            steer_angle = compute_steering_angle_pure_pursuit(path_for_angle,
                                                              lookahead_distance=5.0,
                                                              wheelbase=2.5)
            angle_msg = Float64()
            angle_msg.data = steer_angle
            pub_steer.publish(angle_msg)

            path_msg = RosPath()
            path_msg.header.stamp = rospy.Time.now()
            path_msg.header.frame_id = "map"
            for (x_3d, y_3d, z_3d) in smoothed_path_3d:
                pose = PoseStamped()
                pose.header = path_msg.header
                pose.pose.position.x = float(z_3d)
                pose.pose.position.y = float(x_3d)
                pose.pose.position.z = float(y_3d)
                pose.pose.orientation.w = 1.0
                path_msg.poses.append(pose)
            pub_path.publish(path_msg)

            try:
                ros_mask = bridge.cv2_to_imgmsg(thin_mask, encoding="mono8")
                pub_mask.publish(ros_mask)
            except CvBridgeError as e:
                rospy.logerr("CvBridge Error: %s", str(e))

            publish_vehicle_marker(pub_vehicle_marker)
            publish_lanes_3d(smoothed_path_3d, pub_lane_marker)

            disp_depth = (depth_map * 255).astype(np.uint8)
            disp_color = cv2.applyColorMap(disp_depth, cv2.COLORMAP_JET)
            lane_indices = np.where(thin_mask == 255)
            disp_color[lane_indices] = (255, 255, 255)
            project_path_to_image(disp_color, path_for_angle, fx, fy, cx, cy)
            cv2.polylines(disp_color, [trap_points], isClosed=True,
                          color=(0,255,255), thickness=2)
            cv2.putText(disp_color, f"SteerAngle(deg)={steer_angle:.3f}", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            cv2.imshow("Depth + Lane(or Cone) + Path", disp_color)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                rospy.loginfo("[INFO] 'q' 키 입력 → 종료")
                break

            record_frames.append(disp_color.copy())

        if save_img and len(record_frames) > 0:
            end_time = time.time()
            real_duration = end_time - start_time if start_time else 0
            stem_name = "webcam0"
            make_webcam_video(record_frames, save_dir, stem_name, real_duration)
            rospy.loginfo("[INFO] 웹캠 처리 평균 시간: %.4fs/frame", inf_time.avg)

    else:
        start_time = time.time()
        record_frames = []
        delay = 30
        if dataset.mode == 'video' and dataset.cap is not None:
            fps = dataset.cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                delay = int(1000/fps)
        for frame_data in dataset:
            path_item, img, im0s, vid_cap = frame_data
            enhanced_im0s = cv2.cvtColor(im0s, cv2.COLOR_BGR2RGB)
            enhanced_im0s = cv2.cvtColor(enhanced_im0s, cv2.COLOR_RGB2BGR)
            img, ratio, pad = letterbox(enhanced_im0s, (imgsz, imgsz), stride=32)
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img_t = torch.from_numpy(img).to(device)
            img_t = img_t.half() if half else img_t.float()
            img_t /= 255.0
            if img_t.ndimension() == 3:
                img_t = img_t.unsqueeze(0)

            t1 = time_synchronized()
            with torch.no_grad():
                _, seg, ll = model(img_t)
            t2 = time_synchronized()
            inf_time.update(t2 - t1, img_t.size(0))

            binary_mask = (lane_line_mask(ll, threshold=lane_thr) * 255).astype(np.uint8)
            thin_mask = ximgproc.thinning(binary_mask, thinningType=ximgproc.THINNING_ZHANGSUEN)
            roi_mask = np.zeros_like(thin_mask, dtype=np.uint8)
            cv2.fillPoly(roi_mask, [trap_points], 255)
            thin_mask = cv2.bitwise_and(thin_mask, roi_mask)

            depth_map = depth_est.run(im0s)
            if depth_map.shape[:2] != (im0s.shape[0], im0s.shape[1]):
                depth_map = cv2.resize(depth_map, (im0s.shape[1], im0s.shape[0]),
                                       interpolation=cv2.INTER_LINEAR)

            lane_pixel_count = cv2.countNonZero(thin_mask)
            lane_exist_threshold = 300
            use_lane = (lane_pixel_count > lane_exist_threshold)

            if use_lane:
                path_3d = build_robust_path_ipm(thin_mask, depth_map, depth_scale,
                                                fx, fy, cx, cy, trap_points)
            else:
                cone_candidates = parse_cone_detections_depth(depth_map, im0s, depth_scale, fx, fy, cx, cy,
                                                              min_area=50, max_area=5000, depth_thresh=0.2, color_ratio_thresh=0.5)
                if cone_candidates:
                    cone_candidates.sort(key=lambda p: p[2])
                    nearest_cone = cone_candidates[0]
                    num_pts = 50
                    path_3d = []
                    for i in range(num_pts):
                        t = float(i) / (num_pts - 1)
                        path_3d.append(( (1-t)*0.0 + t*nearest_cone[0],
                                         (1-t)*0.0 + t*nearest_cone[1],
                                         (1-t)*0.0 + t*nearest_cone[2] ))
                else:
                    path_3d = []

            if path_3d:
                path_buffer.append(path_3d)
                smoothed_path_3d = smooth_path(path_buffer)
            else:
                smoothed_path_3d = []

            path_for_angle = [(pt[0], pt[2]) for pt in smoothed_path_3d]
            steer_angle = compute_steering_angle_pure_pursuit(path_for_angle,
                                                              lookahead_distance=5.0,
                                                              wheelbase=2.5)
            angle_msg = Float64()
            angle_msg.data = steer_angle
            pub_steer.publish(angle_msg)

            path_msg = RosPath()
            path_msg.header.stamp = rospy.Time.now()
            path_msg.header.frame_id = "map"
            for (x_3d, y_3d, z_3d) in smoothed_path_3d:
                pose = PoseStamped()
                pose.header = path_msg.header
                pose.pose.position.x = float(z_3d)
                pose.pose.position.y = float(x_3d)
                pose.pose.position.z = float(y_3d)
                pose.pose.orientation.w = 1.0
                path_msg.poses.append(pose)
            pub_path.publish(path_msg)

            try:
                ros_mask = bridge.cv2_to_imgmsg(thin_mask, encoding="mono8")
                pub_mask.publish(ros_mask)
            except CvBridgeError as e:
                rospy.logerr("CvBridge Error: %s", str(e))

            publish_vehicle_marker(pub_vehicle_marker)
            publish_lanes_3d(smoothed_path_3d, pub_lane_marker)

            disp_depth = (depth_map * 255).astype(np.uint8)
            disp_color = cv2.applyColorMap(disp_depth, cv2.COLORMAP_JET)
            lane_indices = np.where(thin_mask == 255)
            disp_color[lane_indices] = (255, 255, 255)
            project_path_to_image(disp_color, path_for_angle, fx, fy, cx, cy)
            cv2.polylines(disp_color, [trap_points], isClosed=True,
                          color=(0,255,255), thickness=2)
            cv2.putText(disp_color, f"SteerAngle(deg)={steer_angle:.3f}", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            cv2.imshow("Depth + Lane(or Cone) + Path", disp_color)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                rospy.loginfo("[INFO] 'q' 키 입력 → 종료")
                break

            if dataset.mode == 'image':
                save_path = str(save_dir / PathlibPath(path_item).name)
                sp = PathlibPath(save_path)
                if sp.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                    sp = sp.with_suffix('.jpg')
                cv2.imwrite(str(sp), disp_color)
                rospy.loginfo("[INFO] 이미지 저장: %s", sp)
            elif dataset.mode == 'video':
                save_path = str(save_dir / (PathlibPath(path_item).stem + '_output.mp4'))
                if vid_path != save_path:
                    vid_path = save_path
                    if vid_writer is not None:
                        vid_writer.release()
                    fps = vid_cap.get(cv2.CAP_PROP_FPS) or 30
                    wv, hv = disp_color.shape[1], disp_color.shape[0]
                    rospy.loginfo("[INFO] 비디오 저장 시작: %s (FPS=%s, size=(%d,%d))", 
                                  vid_path, fps, wv, hv)
                    vid_writer = cv2.VideoWriter(
                        vid_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps, (wv, hv)
                    )
                vid_writer.write(disp_color)
            else:
                record_frames.append(disp_color.copy())

        if vid_writer is not None:
            vid_writer.release()
            rospy.loginfo("[INFO] 비디오 저장 완료: %s", vid_path)
        rospy.loginfo("[INFO] 동기 처리 추론 평균 시간: %.4fs/frame", inf_time.avg)
        end_time = time.time()

    cv2.destroyAllWindows()
    rospy.loginfo("[INFO] 추론 완료.")

##########################################
# 12. ros_main()
##########################################
def ros_main():
    rospy.init_node('yolopv2_cone_lane_node', anonymous=True)
    parser = make_parser()
    opt, _ = parser.parse_known_args()

    # ROS 퍼블리셔
    pub_mask = rospy.Publisher('yolopv2/lane_mask', Image, queue_size=1)
    pub_path = rospy.Publisher('yolopv2/driving_path', RosPath, queue_size=1)
    pub_lane_marker = rospy.Publisher('yolopv2/lane_marker', Marker, queue_size=1)
    pub_vehicle_marker = rospy.Publisher('yolopv2/vehicle_marker', Marker, queue_size=1)
    pub_steer = rospy.Publisher('steering_angle', Float64, queue_size=1)

    detect_and_publish(opt, pub_mask, pub_path,
                       pub_lane_marker, pub_vehicle_marker, pub_steer)

    rospy.loginfo("[INFO] Pseudo-Lidar Lane+Cone node finished. spin() for keepalive.")
    rospy.spin()

if __name__ == '__main__':
    try:
        with torch.no_grad():
            ros_main()
    except rospy.ROSInterruptException:
        pass
