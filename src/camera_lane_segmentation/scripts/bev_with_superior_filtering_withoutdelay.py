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

# utils.py
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

"""
[최종 개선판]
- 웹캠 스트림인 경우: 멀티스레딩(비동기 프레임 읽기)를 사용하여 최신 프레임만 처리.
- 저장된 영상(또는 이미지)인 경우: 동기식 for 루프를 사용하여 한 프레임씩 처리하면서 디버깅 및 결과 저장.
- 두 경우 모두 동일한 알고리즘(차선 세그멘테이션 → 이진화 → thinning → BEV 변환 → 후처리)을 적용하며,
  디스플레이와 결과 저장(영상 또는 이미지 파일)이 가능하도록 분기하였습니다.
"""

# ---------------------------------------------------------------------------
# 모폴로지 및 연결요소 기반 필터 함수들
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 필터 파이프라인 및 후처리 함수들
# ----------------------------------------------------------------------------
def advanced_filter_pipeline(binary_mask):
    step1 = morph_open(binary_mask, ksize=3)
    step2 = morph_close(step1, ksize=5)
    step3 = remove_small_components(step2, min_size=100)
    step4 = keep_top2_components(step3, min_area=150)
    step5 = line_fit_filter(step4, max_line_fit_error=5.0, min_angle_deg=20.0, max_angle_deg=160.0)
    return step5

def post_thinning_filter(thin_mask):
    s1 = remove_small_components(thin_mask, min_size=50)
    s2 = keep_top2_components(s1, min_area=50)
    return s2

def final_filter(bev_mask):
    f1 = morph_open(bev_mask, ksize=3)
    f2 = morph_close(f1, ksize=8)
    f3 = remove_small_components(f2, min_size=300)
    f4 = keep_top2_components(f3, min_area=50)
    f5 = line_fit_filter(f4, max_line_fit_error=5, min_angle_deg=15.0, max_angle_deg=165.0)
    return f5

# ---------------------------------------------------------------------------
# BEV 변환 함수 (bev_params.npz 기반)
# ---------------------------------------------------------------------------
def do_bev_transform(image, bev_param_file):
    params = np.load(bev_param_file)
    src_points = params['src_points']
    dst_points = params['dst_points']
    warp_w = int(params['warp_w'])
    warp_h = int(params['warp_h'])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    bev = cv2.warpPerspective(image, M, (warp_w, warp_h), flags=cv2.INTER_LINEAR)
    return bev

# ---------------------------------------------------------------------------
# argparse 설정
# ---------------------------------------------------------------------------
def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='/home/highsky/yolopv2.pt',
                        help='model.pt 경로')
    parser.add_argument('--source', type=str,
                        default='/home/highsky/Videos/Webcam/bev영상1.mp4',  # '2'
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

# ---------------------------------------------------------------------------
# 결과 영상을 저장하는 함수 (동기, 저장된 영상 또는 나중에 웹캠 결과 저장 시 사용)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# ★ 메인 처리 함수: detect_and_publish ★
# ---------------------------------------------------------------------------
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

    # 입력 소스 결정 (웹캠: 숫자, 영상/이미지: 파일 경로)
    if source.isdigit():
        rospy.loginfo("[INFO] 웹캠(장치=%s) 열기", source)
        dataset = LoadCamera(source, img_size=imgsz, stride=stride)
    else:
        rospy.loginfo("[INFO] 파일(영상/이미지): %s", source)
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # ────────────── 분기: 웹캠(실시간 스트림) vs. 저장된 영상/이미지 ──────────────
    if dataset.mode == 'stream':
        # 웹캠인 경우: 멀티스레딩(비동기 프레임 읽기) 적용
        record_frames = []
        start_time = None
        # GPU warm-up
        if device.type != 'cpu':
            _ = model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
        t0 = time.time()
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

            # 전처리: CLAHE 및 letterbox 적용
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

            t1 = time_synchronized()
            with torch.no_grad():
                [_, _], seg, ll = model(img_t)
            t2 = time_synchronized()
            inf_time.update(t2 - t1, img_t.size(0))

            # 차선 세그멘테이션 → 이진화
            binary_mask = lane_line_mask(ll, threshold=lane_threshold, method='fixed')
            # thinning 및 추가 필터
            thin_mask = ximgproc.thinning(binary_mask, thinningType=ximgproc.THINNING_ZHANGSUEN)
            if thin_mask is None or thin_mask.size == 0:
                rospy.logwarn("[WARNING] Thinning 결과 비어 있음 → binary_mask 사용")
                thin_mask = binary_mask
            bev_mask = do_bev_transform(thin_mask, bev_param_file)
            bevfilter_mask = final_filter(bev_mask)
            final_mask = ximgproc.thinning(bevfilter_mask, thinningType=ximgproc.THINNING_ZHANGSUEN)
            if final_mask is None or final_mask.size == 0:
                rospy.logwarn("[WARNING] Thinning 결과 비어 있음 → bevfilter_mask 사용")
                final_mask = bevfilter_mask

            # BEV 변환 컬러 영상 (옵션)
            bev_im = do_bev_transform(im0s, bev_param_file)

            # ROS 퍼블리시
            try:
                ros_mask = bridge.cv2_to_imgmsg(final_mask, encoding="mono8")
                pub_mask.publish(ros_mask)
            except CvBridgeError as e:
                rospy.logerr("[ERROR] CvBridge 변환 실패: %s", str(e))

            # 디스플레이
            cv2.imshow("BEV Image", bev_im)
            cv2.imshow("Thin Mask", final_mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                rospy.loginfo("[INFO] q 입력 → 종료")
                break

            # 결과 저장 (웹캠 스트림의 경우 나중에 모아서 저장)
            record_frames.append(cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR))

        # ─ 웹캠 스트림 종료 후 저장 (옵션)
        if save_img and len(record_frames) > 0:
            end_time = time.time()
            real_duration = end_time - start_time if start_time else 0
            make_video(record_frames, save_dir, 'webcam0', real_duration)
        rospy.loginfo("[INFO] 웹캠 추론 평균 시간: %.4fs/frame, 전체: %.3fs", inf_time.avg, time.time()-t0)

    else:
        # 저장된 영상 또는 이미지인 경우: 동기식 for 루프 사용
        rospy.loginfo("[DEBUG] 저장된 영상/이미지 동기 처리 시작")
        record_frames = []
        start_time = time.time()
        # 영상 파일인 경우 fps 정보를 가져와서 waitKey delay 결정 (없으면 기본 30ms)
        delay = 30
        if dataset.mode == 'video' and dataset.cap is not None:
            fps = dataset.cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                delay = int(1000 / fps)
        for frame_data in dataset:
            path_item, net_input_img, im0s, vid_cap = frame_data

            # 전처리: CLAHE 및 letterbox 적용
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

            t1 = time_synchronized()
            with torch.no_grad():
                [_, _], seg, ll = model(img_t)
            t2 = time_synchronized()
            inf_time.update(t2 - t1, img_t.size(0))

            # 차선 세그멘테이션 → 이진화 및 후처리
            binary_mask = lane_line_mask(ll, threshold=lane_threshold, method='fixed')
            thin_mask = ximgproc.thinning(binary_mask, thinningType=ximgproc.THINNING_ZHANGSUEN)
            if thin_mask is None or thin_mask.size == 0:
                rospy.logwarn("[WARNING] Thinning 결과 비어 있음 → binary_mask 사용")
                thin_mask = binary_mask
            bev_mask = do_bev_transform(thin_mask, bev_param_file)
            bevfilter_mask = final_filter(bev_mask)
            final_mask = ximgproc.thinning(bevfilter_mask, thinningType=ximgproc.THINNING_ZHANGSUEN)
            if final_mask is None or final_mask.size == 0:
                rospy.logwarn("[WARNING] Thinning 결과 비어 있음 → bevfilter_mask 사용")
                final_mask = bevfilter_mask

            bev_im = do_bev_transform(im0s, bev_param_file)

            # ROS 퍼블리시
            try:
                ros_mask = bridge.cv2_to_imgmsg(final_mask, encoding="mono8")
                pub_mask.publish(ros_mask)
            except CvBridgeError as e:
                rospy.logerr("[ERROR] CvBridge 변환 실패: %s", str(e))

            # 디스플레이 (영상 파일의 경우에도 결과 확인)
            cv2.imshow("BEV Image", bev_im)
            cv2.imshow("Thin Mask", final_mask)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                rospy.loginfo("[INFO] q 입력 → 종료")
                break

            # 영상 파일인 경우 바로 저장을 위해 vid_writer 사용
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
            # 동기식 처리에서도 디버깅용으로 프레임 저장
            record_frames.append(thin_bgr.copy())
        # for 루프 종료 후 영상 파일이면 vid_writer 해제
        if vid_writer is not None:
            vid_writer.release()
            rospy.loginfo("[INFO] 영상 저장 완료: %s", vid_path)
        end_time = time.time()
        real_duration = end_time - start_time if start_time else 0
        # 필요 시 저장된 프레임들을 모아서 추가 저장
        if dataset.mode == 'video' and save_img and len(record_frames) > 0:
            make_video(record_frames, save_dir, 'video_debug', real_duration)
        rospy.loginfo("[INFO] 동기 처리 추론 평균 시간: %.4fs/frame, 전체: %.3fs", inf_time.avg, time.time()-start_time)
    # ────────────── 마무리 ──────────────
    cv2.destroyAllWindows()
    rospy.loginfo("[INFO] 추론 완료.")

def ros_main():
    rospy.init_node('bev_lane_thinning_node', anonymous=True)
    parser = make_parser()
    opt, _ = parser.parse_known_args()
    pub_mask = rospy.Publisher('camera_lane_segmentation/lane_mask', Image, queue_size=1)
    detect_and_publish(opt, pub_mask)
    rospy.loginfo("[INFO] bev_lane_thinning_node 종료, spin() 호출")
    rospy.spin()

if __name__=='__main__':
    try:
        with torch.no_grad():
            ros_main()
    except rospy.ROSInterruptException:
        pass
