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

from math import atan2, degrees
from pathlib import Path
import threading
import queue

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# ========== (프로젝트 내 다른 유틸) ==========
from utils.utils import (
    time_synchronized,
    select_device,
    increment_path,
    lane_line_mask,  # YOLOPv2 세그멘테이션 결과 기반 이진화
    AverageMeter,
    LoadCamera,
    LoadImages,
    letterbox,
    apply_clahe  # CLAHE 함수
)

# ===================================================
# Kalman Filter 클래스 (단순 예시)
# ---------------------------------------------------
class LaneKalmanFilter:
    """
    2차 다항식 계수 [a, b, c]를 추적하기 위한 간단한 Kalman Filter 예시.
    - state: [a, b, c]
    - measurement: [a_meas, b_meas, c_meas]
    """
    def __init__(self):
        # 상태차원=3, 관측차원=3
        self.dim_x = 3
        self.dim_z = 3

        # 초기 상태 벡터
        self.x = np.zeros((self.dim_x, 1), dtype=np.float32)

        # 상태 천이 행렬(F): 기본적으로 “정적” 모델(상태가 급변하지 않는다고 가정)
        self.F = np.eye(self.dim_x, dtype=np.float32)

        # 측정 행렬(H): 관측값이 곧바로 [a, b, c]라 가정
        self.H = np.eye(self.dim_z, self.dim_x, dtype=np.float32)

        # 상태 오차 공분산(P) 초기값
        self.P = np.eye(self.dim_x, dtype=np.float32) * 10.0

        # 프로세스 노이즈 공분산(Q)
        self.Q = np.eye(self.dim_x, dtype=np.float32) * 0.1

        # 측정 노이즈 공분산(R)
        self.R = np.eye(self.dim_z, dtype=np.float32) * 5.0

        # 초기화 여부
        self.initialized = False

    def reset(self):
        self.x[:] = 0
        self.P = np.eye(self.dim_x, dtype=np.float32) * 10.0
        self.initialized = False

    def predict(self):
        # x = F x
        self.x = np.dot(self.F, self.x)
        # P = F P F^T + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        """
        z: 관측 (shape=(3,) 또는 (3,1))
        """
        if z is None:
            return

        # z.shape = (3,) 을 (3,1) 형태로 맞춤
        z = np.array(z, dtype=np.float32).reshape(self.dim_z, 1)

        if not self.initialized:
            # 초기 관측으로 상태를 바로 설정
            self.x = z.copy()
            self.initialized = True
            return

        # y = z - Hx
        y = z - np.dot(self.H, self.x)
        # S = H P H^T + R
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        # K = P H^T S^-1
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        # x = x + K y
        self.x = self.x + np.dot(K, y)
        # P = (I - K H) P
        I = np.eye(self.dim_x, dtype=np.float32)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

# ===================================================
# 폴리피팅 & 곡률 계산 유틸
# ---------------------------------------------------
def polyfit_lane(points_y, points_x, order=2):
    """
    points_y: 세로 방향(전방), points_x: 가로 방향(차폭)
    2차 다항식 x = a*(y^2) + b*y + c 형태로 피팅 (OpenCV 좌표계상 y가 아래로 증가한다고 가정)
    """
    if len(points_y) < 5:
        return None
    fit_coeff = np.polyfit(points_y, points_x, order)
    return fit_coeff  # [a, b, c] (2차)

def eval_poly(coeff, y_vals):
    """ coeff = [a, b, c], y_vals는 배열 → x = a*y^2 + b*y + c """
    if coeff is None:
        return None
    x_vals = np.polyval(coeff, y_vals)
    return x_vals

def calc_curvature(coeff, y_eval):
    """
    단순 픽셀 단위 곡률 계산 예시
    - 2차 다항식 x = a*y^2 + b*y + c
    - 곡률 반경 R = [ (1 + (2ay + b)^2)^(3/2 ) ] / |2a|
    - y_eval: 곡률을 측정하고자 하는 y 지점
    """
    if coeff is None:
        return None
    a, b, c = coeff
    # (2a*y + b)
    first_deriv = 2*a*y_eval + b
    # (2a)
    second_deriv = 2*a
    denom = abs(second_deriv) * ((1 + first_deriv**2)**1.5)
    if denom < 1e-6:
        return None
    R = 1.0 / denom
    return R

def extract_lane_points(bev_mask):
    """
    전체 차선 마스크에서 픽셀 좌표 (y, x)를 추출
    (시연용으로 단순히 모든 픽셀을 통째로 피팅; 실제로는 좌/우 차선 분리 가능)
    """
    ys, xs = np.where(bev_mask > 0)
    return ys, xs

def overlay_polyline(image, coeff, color=(0, 0, 255), step=5):
    """
    2차 다항식 coeff에 맞춰 y좌표를 순회하며 x좌표를 구해
    image 위에 폴리라인을 그려주는 시각화 예시
    """
    if coeff is None:
        return image
    h, w = image.shape[:2]
    draw_points = []
    for y in range(0, h, step):
        x = np.polyval(coeff, y)
        if 0 <= x < w:
            draw_points.append((int(x), int(y)))
    if len(draw_points) > 1:
        for i in range(len(draw_points) - 1):
            cv2.line(image, draw_points[i], draw_points[i+1], color, 2)
    return image

# ===================================================
# 모폴로지 및 연결요소 기반 필터
# ---------------------------------------------------
def morph_open(binary_mask, ksize=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

def morph_close(binary_mask, ksize=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

def remove_small_components(binary_mask, min_size=100):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == i] = 255
    return cleaned

def keep_top2_components(binary_mask, min_area=50):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 2:
        return binary_mask
    comps = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
    comps.sort(key=lambda x: x[1], reverse=True)
    keep_indices = [i for i, area in comps[:2] if area >= min_area]
    cleaned = np.zeros_like(binary_mask)
    for idx in keep_indices:
        cleaned[labels == idx] = 255
    return cleaned

def line_fit_filter(binary_mask, max_line_fit_error=2.0, min_angle_deg=70.0, max_angle_deg=110.0):
    h, w = binary_mask.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    out_mask = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        comp_mask = (labels == i).astype(np.uint8)
        if stats[i, cv2.CC_STAT_AREA] < 5:
            continue
        ys, xs = np.where(comp_mask > 0)
        pts = np.column_stack((xs, ys)).astype(np.float32)
        if len(pts) < 2:
            continue
        line_param = cv2.fitLine(pts, distType=cv2.DIST_L2, param=0, reps=0.01, aeps=0.01)
        vx, vy, x0, y0 = line_param.flatten()
        angle_deg = abs(degrees(atan2(vy, vx)))
        if angle_deg > 180:
            angle_deg -= 180
        if not (min_angle_deg <= angle_deg <= max_angle_deg):
            continue
        norm_len = (vx**2 + vy**2)**0.5
        if norm_len < 1e-12:
            continue
        vx_n, vy_n = vx/norm_len, vy/norm_len
        dist_sum = sum(abs((xx - x0)*(-vy_n) + (yy - y0)*vx_n) for xx, yy in pts)
        if (dist_sum / len(pts)) <= max_line_fit_error:
            out_mask[labels == i] = 255
    return out_mask

def advanced_filter_pipeline(binary_mask):
    step1 = morph_open(binary_mask, ksize=3)
    step2 = morph_close(step1, ksize=5)
    step3 = remove_small_components(step2, min_size=100)
    step4 = keep_top2_components(step3, min_area=150)
    step5 = line_fit_filter(step4, max_line_fit_error=5.0, min_angle_deg=20.0, max_angle_deg=160.0)
    return step5

def final_filter(bev_mask):
    f1 = morph_open(bev_mask, ksize=3)
    f2 = morph_close(f1, ksize=8)
    f3 = remove_small_components(f2, min_size=300)
    f4 = keep_top2_components(f3, min_area=300)
    f5 = line_fit_filter(f4, max_line_fit_error=5, min_angle_deg=15.0, max_angle_deg=165.0)
    return f5

# ===================================================
# BEV 변환 함수
# ---------------------------------------------------
def do_bev_transform(image, bev_param_file):
    params = np.load(bev_param_file)
    src_points = params['src_points']
    dst_points = params['dst_points']
    warp_w = int(params['warp_w'])
    warp_h = int(params['warp_h'])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    bev = cv2.warpPerspective(image, M, (warp_w, warp_h), flags=cv2.INTER_LINEAR)
    return bev

# ===================================================
# argparse 설정
# ---------------------------------------------------
def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='/home/highsky/yolopv2.pt',
                        help='model.pt 경로')
    parser.add_argument('--source', type=str,
                        default='/home/highsky/Videos/Webcam/bev영상1.mp4', # '2'
                        help='source: 0(webcam) 또는 영상/이미지 파일 경로')
    parser.add_argument('--img-size', type=int, default=640,
                        help='YOLO 추론 해상도')
    parser.add_argument('--device', default='0',
                        help='cuda device: 0 또는 cpu')
    parser.add_argument('--lane-thres', type=float, default=0.5,
                        help='차선 세그 임계값 (0.0~1.0)')
    parser.add_argument('--nosave', action='store_false',
                        help='저장하지 않으려면 사용')
    parser.add_argument('--project', default='runs/detect',
                        help='결과 저장 폴더')
    parser.add_argument('--name', default='exp',
                        help='결과 저장 폴더 이름')
    parser.add_argument('--exist-ok', action='store_false',
                        help='기존 폴더 사용 허용')
    parser.add_argument('--frame-skip', type=int, default=0,
                        help='프레임 건너뛰기 (0이면 건너뛰지 않음)')
    parser.add_argument('--param-file', type=str,
                        default='/home/highsky/My_project_work_ws/bev_params.npz',
                        help='BEV 파라미터 (src_points, dst_points, warp_w, warp_h)')
    return parser

# ===================================================
# 결과 영상을 저장하는 함수
# ---------------------------------------------------
def make_video(record_frames, save_dir: Path, stem_name: str, real_duration: float):
    if len(record_frames) == 0:
        rospy.loginfo("[INFO] 저장할 프레임이 없습니다.")
        return
    num_frames = len(record_frames)
    if real_duration <= 0:
        real_duration = 1e-6
    real_fps = num_frames / real_duration
    rospy.loginfo("[INFO] 녹화: 총 %d 프레임, %.2f초 => FPS 약 %.2f", num_frames, real_duration, real_fps)
    save_path = str(save_dir / f"{stem_name}_output.mp4")
    h, w = record_frames[0].shape[:2]
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), real_fps, (w, h))
    if not out.isOpened():
        rospy.logerr(f"[ERROR] 비디오 라이터 열기 실패: {save_path}")
        return
    for f in record_frames:
        out.write(f)
    out.release()
    rospy.loginfo("[INFO] 결과 영상 저장 완료: %s", save_path)

# ===================================================
# 메인 처리 함수: detect_and_publish
# ---------------------------------------------------
def detect_and_publish(opt, pub_mask):
    cv2.setUseOptimized(True)
    cv2.setNumThreads(0)
    cudnn.benchmark = True

    bridge = CvBridge()
    source, weights = opt.source, opt.weights
    imgsz = opt.img_size
    lane_threshold = opt.lane_thres
    save_img = not opt.nosave
    bev_param_file = opt.param_file

    # 결과 저장 폴더 생성
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)

    inf_time = AverageMeter()
    vid_path = None
    vid_writer = None
    current_save_size = None

    # 모델 로드 및 GPU 최적화
    stride = 32
    model = torch.jit.load(weights)
    device = select_device(opt.device)
    half = (device.type != 'cpu')
    model = model.to(device)
    if half:
        model.half()
    model.eval()

    # Kalman Filter (2차 계수 추적) 인스턴스
    kf = LaneKalmanFilter()

    # 입력 소스 결정 (웹캠 vs. 파일)
    if source.isdigit():
        rospy.loginfo("[INFO] 웹캠(장치=%s) 열기", source)
        dataset = LoadCamera(source, img_size=imgsz, stride=stride)
    else:
        rospy.loginfo("[INFO] 파일(영상/이미지): %s", source)
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # ──────────────────────────
    # 함수: 매 프레임 처리 로직
    # ──────────────────────────
    def process_frame(im0s):
        """
        im0s: BGR 원본 이미지
        1) YOLOPv2 추론
        2) lane_line_mask → thinning/필터
        3) BEV 변환 후 final_filter
        4) 폴리피팅 + 곡률 계산 + Kalman Filter 업데이트
        5) 최종 시각화/퍼블리시
        """
        # 전처리: CLAHE + letterbox
        enhanced_im0s = apply_clahe(im0s)
        net_input_img, ratio, pad = letterbox(enhanced_im0s, (imgsz, imgsz), stride=stride)
        net_input_img = net_input_img[:, :, ::-1].transpose(2, 0, 1)
        net_input_img = np.ascontiguousarray(net_input_img)

        # 모델 추론 준비
        img_t = torch.from_numpy(net_input_img).to(device)
        img_t = img_t.half() if half else img_t.float()
        img_t /= 255.0
        if img_t.ndimension() == 3:
            img_t = img_t.unsqueeze(0)

        # 추론
        t1 = time_synchronized()
        with torch.no_grad():
            [_, _], seg, ll = model(img_t)
        t2 = time_synchronized()
        inf_time.update(t2 - t1, img_t.size(0))

        # 차선 세그멘테이션 → 이진화
        binary_mask = lane_line_mask(ll, threshold=lane_threshold, method='otsu')
        # thinning (2D 라인화)
        thin_mask = ximgproc.thinning(binary_mask, thinningType=ximgproc.THINNING_ZHANGSUEN)
        if thin_mask is None or thin_mask.size == 0:
            rospy.logwarn("[WARNING] Thinning 결과 비어 있음 → binary_mask 사용")
            thin_mask = binary_mask

        # BEV 변환
        bev_mask = do_bev_transform(thin_mask, bev_param_file)

        # 추가 필터링
        bevfilter_mask = final_filter(bev_mask)
        final_mask = ximgproc.thinning(bevfilter_mask, thinningType=ximgproc.THINNING_ZHANGSUEN)
        if final_mask is None or final_mask.size == 0:
            rospy.logwarn("[WARNING] Thinning 결과 비어 있음 → bevfilter_mask 사용")
            final_mask = bevfilter_mask

        # 차선 픽셀 추출 → 폴리피팅
        ys, xs = extract_lane_points(final_mask)
        coeff = polyfit_lane(ys, xs, order=2)  # [a, b, c]

        # Kalman Filter로 계수 추적
        kf.predict()
        if coeff is not None:
            kf.update(coeff)  # 관측값은 [a, b, c]
        smoothed_coeff = kf.x.flatten()  # KF 후 상태 추정 [a, b, c]

        # 곡률 계산 (y_eval은 이미지 하단부 근처 등)
        y_eval = final_mask.shape[0] - 1  # 맨 아래 행
        curvature = calc_curvature(smoothed_coeff, y_eval)

        # 시각화를 위해 BEV 컬러로 변환
        bev_im_color = cv2.cvtColor(bev_mask, cv2.COLOR_GRAY2BGR)
        # 폴리라인 오버레이 (KF 추정 계수 사용)
        bev_im_color = overlay_polyline(bev_im_color, smoothed_coeff, color=(0, 0, 255), step=5)

        # 곡률 텍스트 표시
        if curvature is not None:
            cv2.putText(bev_im_color, f"Curvature ~ {curvature:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        bev_im = do_bev_transform(im0s, bev_param_file)

        # ROS 퍼블리시
        try:
            ros_mask = bridge.cv2_to_imgmsg(final_mask, encoding="mono8")
            pub_mask.publish(ros_mask)
        except CvBridgeError as e:
            rospy.logerr("[ERROR] CvBridge 변환 실패: %s", str(e))

        # 결과 디스플레이
        cv2.imshow("BEV Image", bev_im)
        cv2.imshow("BEV Mask + Polyfit", bev_im_color)
        cv2.imshow("Thin Mask", final_mask)

        return final_mask, bev_im_color

    # ──────────────────────────
    # 웹캠 vs. 동영상/이미지 처리 분기
    # ──────────────────────────
    record_frames = []
    start_time = None

    if dataset.mode == 'stream':
        # 웹캠(스트림) 비동기 처리
        frame_skip = opt.frame_skip
        frame_counter = 0
        frame_queue = queue.Queue(maxsize=5)

        def frame_producer():
            for item in dataset:
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                frame_queue.put(item)
            frame_queue.put(None)

        producer_thread = threading.Thread(target=frame_producer)
        producer_thread.daemon = True
        producer_thread.start()

        rospy.loginfo("[DEBUG] 웹캠 비동기 프레임 생산 시작")
        t0 = time.time()

        while not rospy.is_shutdown():
            try:
                frame_data = frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if frame_data is None:
                rospy.loginfo("[INFO] 프레임 종료 신호 수신")
                break
            path_item, net_input_img, im0s, vid_cap = frame_data

            if frame_skip > 0:
                if frame_counter % (frame_skip + 1) != 0:
                    frame_counter += 1
                    continue
                frame_counter += 1

            if start_time is None:
                start_time = time.time()

            final_mask, bev_vis = process_frame(im0s)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                rospy.loginfo("[INFO] q 입력 → 종료")
                break

            # 기록용
            out_bgr = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
            record_frames.append(out_bgr)

        # 웹캠 종료 후
        if save_img and len(record_frames) > 0:
            end_time = time.time()
            real_duration = end_time - start_time if start_time else 0
            make_video(record_frames, save_dir, 'webcam0', real_duration)
        rospy.loginfo("[INFO] 웹캠 추론 평균 시간: %.4fs/frame, 전체: %.3fs", inf_time.avg, time.time()-t0)

    else:
        # 동기 처리 (영상/이미지)
        rospy.loginfo("[DEBUG] 저장된 영상/이미지 동기 처리 시작")
        delay = 30
        if dataset.mode == 'video' and dataset.cap is not None:
            fps = dataset.cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                delay = int(1000 / fps)
        start_time = time.time()

        for frame_data in dataset:
            path_item, net_input_img, im0s, vid_cap = frame_data
            final_mask, bev_vis = process_frame(im0s)

            if cv2.waitKey(delay) & 0xFF == ord('q'):
                rospy.loginfo("[INFO] q 입력 → 종료")
                break

            # 영상 파일의 경우 바로 저장하기 위해
            thin_bgr = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
            if dataset.mode == 'image':
                save_path = str(save_dir / Path(path_item).name)
                sp = Path(save_path)
                if sp.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                    sp = sp.with_suffix('.jpg')
                cv2.imwrite(str(sp), thin_bgr)
                rospy.loginfo("[INFO] 이미지 저장: %s", sp)
            elif dataset.mode == 'video':
                if vid_path != (str(save_dir / (Path(path_item).stem + '_output.mp4'))):
                    vid_path = str(save_dir / (Path(path_item).stem + '_output.mp4'))
                    if vid_writer is not None:
                        vid_writer.release()
                    if vid_cap:
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        if fps <= 0:
                            fps = 30
                    else:
                        fps = 30
                    wv, hv = thin_bgr.shape[1], thin_bgr.shape[0]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    vid_writer = cv2.VideoWriter(vid_path, fourcc, fps, (wv, hv))
                    if not vid_writer.isOpened():
                        rospy.logerr("[ERROR] 비디오 라이터 열기 실패: %s", vid_path)
                        vid_writer = None
                    current_save_size = (wv, hv)
                    rospy.loginfo("[INFO] 영상 저장 시작: %s (fps=%d, size=(%d,%d))", vid_path, fps, wv, hv)
                if vid_writer is not None:
                    if (thin_bgr.shape[1], thin_bgr.shape[0]) != current_save_size:
                        rospy.logwarn("[WARNING] 프레임 크기 불일치 → 리사이즈")
                        try:
                            thin_bgr = cv2.resize(thin_bgr, current_save_size, interpolation=cv2.INTER_LINEAR)
                        except cv2.error as e:
                            rospy.logerr("[ERROR] 리사이즈 실패: %s", str(e))
                            continue
                    vid_writer.write(thin_bgr)

            record_frames.append(thin_bgr.copy())

        if vid_writer is not None:
            vid_writer.release()
            rospy.loginfo("[INFO] 영상 저장 완료: %s", vid_path)
        end_time = time.time()
        real_duration = end_time - start_time if start_time else 0
        if dataset.mode == 'video' and save_img and len(record_frames) > 0:
            make_video(record_frames, save_dir, 'video_debug', real_duration)
        rospy.loginfo("[INFO] 동기 처리 추론 평균 시간: %.4fs/frame, 전체: %.3fs", inf_time.avg, time.time()-start_time)

    cv2.destroyAllWindows()
    rospy.loginfo("[INFO] 추론 완료.")

# ===================================================
# 메인 ros_main
# ---------------------------------------------------
def ros_main():
    rospy.init_node('bev_lane_thinning_node', anonymous=True)
    parser = make_parser()
    opt, _ = parser.parse_known_args()
    pub_mask = rospy.Publisher('camera_lane_segmentation/lane_mask', Image, queue_size=1)
    detect_and_publish(opt, pub_mask)
    rospy.loginfo("[INFO] bev_lane_thinning_node 종료, spin() 호출")
    rospy.spin()

# ===================================================
# 메인 실행
# ---------------------------------------------------
if __name__=='__main__':
    try:
        with torch.no_grad():
            ros_main()
    except rospy.ROSInterruptException:
        pass
