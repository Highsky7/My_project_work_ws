#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import time
from pathlib import Path
import cv2
import torch
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from utils.util import (
    time_synchronized,
    select_device,
    increment_path,
    scale_coords,
    xyxy2xywh,
    non_max_suppression,
    split_for_trace_model,
    driving_area_mask,
    lane_line_mask,
    plot_one_box,
    show_seg_result,
    AverageMeter,
    letterbox,
    LoadImages,
    LoadCamera
)

class ROSImageSubscriber:
    def __init__(self, topic, img_size=640, stride=32):
        """ROS 이미지 토픽을 구독하는 클래스 초기화"""
        self.bridge = CvBridge()
        self.img_size = img_size
        self.stride = stride
        self.sub = rospy.Subscriber(topic, Image, self.callback)
        self.img0 = None  # 원본 이미지
        self.img = None   # 모델 입력용 이미지
        self.new_frame = False

    def callback(self, data):
        """새로운 ROS 메시지가 도착했을 때 호출되는 콜백 함수"""
        try:
            self.img0 = self.bridge.imgmsg_to_cv2(data, "bgr8")
            img = letterbox(self.img0, self.img_size, stride=self.stride)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
            img = np.ascontiguousarray(img)
            self.img = img
            self.new_frame = True
        except CvBridgeError as e:
            print(f"[ERROR] CvBridge 오류: {e}")

    def get_frame(self):
        """최신 프레임을 반환"""
        if self.new_frame:
            self.new_frame = False
            return self.img, self.img0
        return None, None

def make_parser():
    """명령줄 인수를 파싱하는 함수"""
    parser = argparse.ArgumentParser(description="YOLOPv2 ROS, 동영상 및 카메라 감지")
    parser.add_argument('--weights', nargs='+', type=str,
                        default='./yolopv2.pt',
                        help='모델 가중치 파일 경로')
    parser.add_argument('--source', type=str,
                        default='/home/highsky/Videos/Webcam/직선.mp4',
                        help='입력 소스: ROS 토픽 (예: /camera/image), 동영상 파일 경로 (예: video.mp4), 또는 카메라 ID (예: 0)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='모델 입력 이미지 크기 (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3,
                        help='객체 신뢰도 임계값')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                        help='NMS IOU 임계값')
    parser.add_argument('--device', default='0',
                        help='CUDA 장치, 예: 0 또는 0,1,2,3 또는 cpu')
    parser.add_argument('--save-conf', action='store_true',
                        help='레이블에 신뢰도 저장')
    parser.add_argument('--save-txt', action='store_true',
                        help='결과를 *.txt 파일로 저장')
    parser.add_argument('--nosave', action='store_false',
                        help='이미지/비디오 저장 여부')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='필터링할 클래스: --class 0, 또는 --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='클래스 무관 NMS')
    parser.add_argument('--project', default='runs/detect',
                        help='결과 저장 경로')
    parser.add_argument('--name', default='exp',
                        help='실험 이름')
    parser.add_argument('--exist-ok', action='store_true',
                        help='기존 프로젝트/이름 덮어쓰기 허용')
    return parser

def detect():
    """YOLOPv2를 사용한 실시간, 동영상 및 카메라 감지 함수"""
    source, weights, save_txt, imgsz = opt.source, opt.weights, opt.save_txt, opt.img_size
    save_img = not opt.nosave

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    inf_time = AverageMeter()
    waste_time = AverageMeter()
    nms_time = AverageMeter()

    stride = 32
    model = torch.jit.load(weights)
    device = select_device(opt.device)
    half = device.type != 'cpu'
    model = model.to(device)
    if half:
        model.half()
    model.eval()

    # 입력 소스에 따라 데이터 로더 선택
    if source.startswith('/'):  # ROS 토픽인 경우
        dataset = ROSImageSubscriber(source, img_size=imgsz, stride=stride)
        is_ros = True
        is_camera = False
        print("[INFO] ROS 토픽 감지 시작")
    elif source.isdigit():  # 카메라 ID인 경우
        dataset = LoadCamera(source, img_size=imgsz, stride=stride)
        is_ros = False
        is_camera = True
        print(f"[INFO] 카메라 감지 시작: ID {source}")
    else:  # 동영상 파일인 경우
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        is_ros = False
        is_camera = False
        print(f"[INFO] 동영상 파일 감지 시작: {source}")
        print(f"[DEBUG] 동영상 총 프레임 수: {dataset.nf}")  # 동영상 프레임 수 확인

    bridge = CvBridge()
    ll_pub = rospy.Publisher('/lane_line_mask', Image, queue_size=1) if is_ros else None
    da_pub = rospy.Publisher('/driving_area_mask', Image, queue_size=1) if is_ros else None

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    t0 = time.time()

    # 입력 소스에 따라 루프 처리
    if is_ros:
        while not rospy.is_shutdown():
            img, im0 = dataset.get_frame()
            if img is None:
                continue
            process_frame(img, im0, model, device, half, inf_time, waste_time, nms_time,
                          save_img, save_txt, save_dir, bridge, ll_pub, da_pub, is_ros)
    elif is_camera:
        while True:
            try:
                path, img, im0, _ = next(dataset)
                process_frame(img, im0, model, device, half, inf_time, waste_time, nms_time,
                              save_img, save_txt, save_dir, bridge, ll_pub, da_pub, is_ros)
            except StopIteration:
                print("[INFO] 카메라 스트림 종료")
                break
    else:  # 동영상 파일 처리
        frame_count = 0
        for path, img, im0, cap in dataset:
            frame_count += 1
            print(f"[DEBUG] 처리 중인 프레임: {frame_count}/{dataset.nf}")
            process_frame(img, im0, model, device, half, inf_time, waste_time, nms_time,
                          save_img, save_txt, save_dir, bridge, ll_pub, da_pub, is_ros)
            if cap is None or not cap.isOpened():  # 비디오 캡처 객체가 닫혔는지 확인
                print("[INFO] 동영상 처리 완료 또는 오류")
                break

    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg, nms_time.avg))
    print(f'Done. ({time.time() - t0:.3f}s)')
    cv2.destroyAllWindows()

def process_frame(img, im0, model, device, half, inf_time, waste_time, nms_time,
                  save_img, save_txt, save_dir, bridge, ll_pub, da_pub, is_ros):
    """프레임 처리 로직을 별도 함수로 분리"""
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    t1 = time_synchronized()
    [pred, anchor_grid], seg, ll = model(img)
    t2 = time_synchronized()

    tw1 = time_synchronized()
    pred = split_for_trace_model(pred, anchor_grid)
    tw2 = time_synchronized()

    t3 = time_synchronized()
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                               classes=opt.classes, agnostic=opt.agnostic_nms)
    t4 = time_synchronized()

    da_seg_mask = driving_area_mask(seg)
    ll_seg_mask = lane_line_mask(ll)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ll_seg_mask, connectivity=8)
    if num_labels > 1:
        bottom_y = stats[1:, cv2.CC_STAT_TOP] + stats[1:, cv2.CC_STAT_HEIGHT]
        sorted_labels = sorted(range(1, num_labels), key=lambda x: bottom_y[x-1], reverse=True)
        top_labels = sorted_labels[:min(2, len(sorted_labels))]

        ll_seg_mask = np.zeros_like(labels, dtype=np.uint8)
        for label in top_labels:
            ll_seg_mask[labels == label] = 1

        if len(top_labels) == 2:
            centroid_x = centroids[top_labels, 0]
            label_left = top_labels[0] if centroid_x[0] < centroid_x[1] else top_labels[1]
            label_right = top_labels[1] if centroid_x[0] < centroid_x[1] else top_labels[0]

            left_lane_mask = (labels == label_left)
            right_lane_mask = (labels == label_right)

            da_seg_mask_new = np.zeros_like(da_seg_mask, dtype=np.uint8)
            for y in range(im0.shape[0]):
                left_x = np.where(left_lane_mask[y, :])[0]
                right_x = np.where(right_lane_mask[y, :])[0]
                if len(left_x) > 0 and len(right_x) > 0:
                    max_x_left = left_x.max()
                    min_x_right = right_x.min()
                    if max_x_left < min_x_right:
                        da_seg_mask_new[y, max_x_left+1:min_x_right] = 1
            da_seg_mask = da_seg_mask_new

        elif len(top_labels) == 1:
            label = top_labels[0]
            lane_mask = (labels == label)
            da_seg_mask_new = np.zeros_like(da_seg_mask, dtype=np.uint8)
            for y in range(im0.shape[0]):
                x_positions = np.where(lane_mask[y, :])[0]
                if len(x_positions) > 0:
                    center_x = int(np.mean(x_positions))
                    left_bound = max(0, center_x - 50)
                    right_bound = min(im0.shape[1], center_x + 50)
                    da_seg_mask_new[y, left_bound:right_bound] = 1
            da_seg_mask = da_seg_mask_new

        ll_seg_mask = (labels == 1).astype(np.uint8)
    else:
        da_seg_mask = np.zeros_like(da_seg_mask)
        ll_seg_mask = np.zeros_like(ll_seg_mask)

    for i, det in enumerate(pred):
        s = ''
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                if save_txt:
                    xywh = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
                    line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                    txt_path = str(save_dir / 'labels' / f'frame_{time.time()}')
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
                if save_img:
                    plot_one_box(xyxy, im0, line_thickness=3)

        print(f'{s}Done. ({t2 - t1:.3f}s)')

        # 결과 시각화
        show_seg_result(im0, (da_seg_mask, ll_seg_mask), is_demo=True)

        # OpenCV 창에 결과 표시
        cv2.imshow('YOLOPv2 Detection', im0)
        key = cv2.waitKey(1)  # 프레임 간 지연 (동영상 속도 조절 가능)
        if key & 0xFF == ord('q'):
            print("[INFO] 사용자가 'q'를 눌러 종료합니다.")
            if is_ros:
                rospy.signal_shutdown("User requested shutdown")
            raise SystemExit  # 루프 종료

    if is_ros:
        try:
            ll_mask_colored = cv2.cvtColor(ll_seg_mask * 255, cv2.COLOR_GRAY2BGR) #Converting ll_seg_mask_binary mask to BGR mask to publish as a ROS topic
            da_mask_colored = cv2.cvtColor(da_seg_mask * 255, cv2.COLOR_GRAY2BGR) #Converting ll_seg_mask_binary mask to BGR mask to publish as a ROS topic
            ll_pub.publish(bridge.cv2_to_imgmsg(ll_mask_colored, "bgr8"))
            da_pub.publish(bridge.cv2_to_imgmsg(da_mask_colored, "bgr8"))
        except CvBridgeError as e:
            print(f"[ERROR] CvBridge 오류: {e}")

    inf_time.update(t2 - t1, img.size(0))
    nms_time.update(t4 - t3, img.size(0))
    waste_time.update(tw2 - tw1, img.size(0))

if __name__ == '__main__':
    opt = make_parser().parse_args()
    print(opt)
    if opt.source.startswith('/'):  # ROS 토픽인 경우에만 노드 초기화
        rospy.init_node('yolopv2_ros', anonymous=True)
        print(f"[INFO] ROS 노드 초기화 완료, 소스: {opt.source}")
    with torch.no_grad():
        try:
            detect()
        except SystemExit:
            print("[INFO] 프로그램 종료")
            cv2.destroyAllWindows()