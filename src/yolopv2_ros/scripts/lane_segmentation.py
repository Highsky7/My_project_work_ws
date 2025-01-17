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
from pathlib import Path

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# YOLOPv2 유틸
from utils.utils import (
    time_synchronized,
    select_device,
    increment_path,
    lane_line_mask,
    AverageMeter,
    LoadCamera,
    LoadImages
)

"""
ROS 노드:
 1) Lane Line 세그멘테이션 (전체 이미지)
 2) 이진화 마스크 + Thinning (두꺼운 선 -> 얇은 선)
 3) Bird’s Eye View(BEV) 변환
 4) 결과를 ROS 토픽 (mono8) 퍼블리시
 5) 실시간 창 및 저장된 영상에서도 얇은 차선을 (BEV 형태로) 확인
"""

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='/home/highsky/yolopv2.pt',
                        help='model.pt path(s)')
    parser.add_argument('--source', type=str,
                        default='2',
                        help='source: 0(webcam) or path to video/image')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--device', default='0',
                        help='cuda device, i.e. 0 or cpu')

    # (수정) 보수적 세그멘테이션을 위한 Threshold (float in [0.0, 1.0])
    parser.add_argument('--lane-thres', type=float, default=0.8,
                        help='Threshold for lane segmentation mask (0.0 ~ 1.0). Higher => more conservative')

    # Detection용이지만, 우선은 남겨둠
    parser.add_argument('--conf-thres', type=float, default=0.3, help='(unused)')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='(unused)')

    # --nosave: True면 저장 O, False면 저장 X
    parser.add_argument('--nosave', action='store_false',
                        help='if true => do NOT save images/videos, else => save')

    parser.add_argument('--project', default='/home/highsky/My_project_work_ws/runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')

    # 프레임 스킵
    parser.add_argument('--frame-skip', type=int, default=0,
                        help='Skip N frames per read (0 for no skip)')

    return parser


def make_webcam_video(record_frames, save_dir: Path, stem_name: str, real_duration: float):
    """웹캠 모드에서 누적된 프레임(thin_mask -> BEV 변환)을 mp4로 저장."""
    if len(record_frames) == 0:
        rospy.loginfo("[INFO] 저장할 웹캠 프레임이 없습니다.")
        return

    num_frames = len(record_frames)
    if real_duration <= 0:
        real_duration = 1e-6
    real_fps = num_frames / real_duration

    rospy.loginfo("[INFO] 웹캠 녹화: 총 %d프레임, 소요 %.2f초 => FPS ~ %.2f",
                  num_frames, real_duration, real_fps)

    save_path = str(save_dir / f"{stem_name}_webcam.mp4")
    h, w = record_frames[0].shape[:2]
    out = cv2.VideoWriter(save_path,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          real_fps,
                          (w, h))  # frame size=(w,h)

    for f in record_frames:
        out.write(f)

    out.release()
    rospy.loginfo("[INFO] 웹캠 결과 영상 저장 완료: %s", save_path)


def detect_and_publish(opt, pub_mask):
    """
    1) Lane Line 세그멘테이션
    2) Thinning -> (얇은 차선)
    3) Bird’s Eye View(BEV) 변환
    4) ROS 퍼블리시, 실시간 표시, 저장
    """

    # OpenCV 최적화
    cv2.setUseOptimized(True)
    cv2.setNumThreads(0)

    # PyTorch cuDNN 최적화
    cudnn.benchmark = True

    bridge = CvBridge()

    source, weights = opt.source, opt.weights
    lane_thr = opt.lane_thres  # float in [0.0, 1.0]
    imgsz = opt.img_size

    # --nosave: true -> 저장 함(이중부정)
    save_img = not opt.nosave and (isinstance(source, str) and not source.endswith('.txt'))

    # 결과 저장 경로
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)

    inf_time = AverageMeter()

    vid_path = None
    vid_writer = None

    # 모델 로드
    stride = 32
    model = torch.jit.load(weights)
    device = select_device(opt.device)
    half = (device.type != 'cpu')
    model = model.to(device)
    if half:
        model.half()
    model.eval()

    # 데이터 로드
    if source.isdigit():
        rospy.loginfo("[INFO] 웹캠(장치 ID=%s)에서 영상을 받습니다.", source)
        dataset = LoadCamera(source, img_size=imgsz, stride=stride)
    else:
        rospy.loginfo("[INFO] 파일(이미지/동영상): %s 를 불러옵니다.", source)
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    record_frames = []
    start_time = None

    # GPU warmup
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    t0 = time.time()
    # 프레임 스킵(혹시모를 딜레이 발생 시 활성화, 현재는 스킵하는 프레임 = 0)
    # frame_skip = opt.frame_skip
    # frame_counter = 0

    # -------------------------------
    # 3-A) BEV를 위한 파라미터 설정 (예시)
    # -------------------------------
    # 아래 src/dst 좌표는 실험 환경에 맞춰 조정해야 합니다.
    # 예: 원본 이미지(640x480 가정)에서 사다리꼴 영역 4점
    # 주의: im0s.shape 등 실제 높이/너비와 맞추세요!
    # BEV 변환용 파라미터 (src_points는 카메라 영상에서 “BEV로 만들고 싶은 사다리꼴 영역”, dst_points는 “위에서 내려다본 결과가 차지할 직사각형(크기=bev_size)”.
    src_points = np.float32([
        [400, 300],  # 왼쪽 위
        [880, 300],  # 오른쪽 위
        [100,  720],  # 왼쪽 아래
        [1180, 720]   # 오른쪽 아래
    ])
    dst_points = np.float32([
        [200,   0],  # 왼쪽 위
        [1080,   0],  # 오른쪽 위
        [200, 1200],  # 왼쪽 아래
        [1080, 1200]   # 오른쪽 아래
    ])
    bev_size = (800, 1200)  # (width, height)

    # 변환 행렬
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # 메인 루프
    for path_item, img, im0s, vid_cap in dataset:
        # 프레임 스킵
        # if frame_skip > 0:
        #     if frame_counter % (frame_skip + 1) != 0:
        #         frame_counter += 1
        #         continue
        #     frame_counter += 1

        if dataset.mode == 'stream' and start_time is None:
            start_time = time.time()

        # 텐서 변환
        img_t = torch.from_numpy(img).to(device)
        img_t = img_t.half() if half else img_t.float()
        img_t /= 255.0
        if img_t.ndimension() == 3:
            img_t = img_t.unsqueeze(0)

        t1 = time_synchronized()
        with torch.no_grad():
            [_, _], seg, ll = model(img_t)
        t2 = time_synchronized()

        # (1) Lane Segmentation => (2) Threshold
        #    lane_line_mask(ll, threshold=lane_thr) 내부에서 (x>threshold).int()
        ll_seg = lane_line_mask(ll, threshold=lane_thr)

        # ll_seg는 이미 0/1 ndarray. => 255 스케일로 변환
        binary_mask = (ll_seg * 255).astype(np.uint8)

        # 2) Thinning 모폴로지 연산(두꺼운 선 -> 얇은 선)
        # 예: 단순 erode, 반복 횟수는 필요에 따라 조정(숫자가 클수록 선 얇아짐)
        # kernel = np.ones((5,5), np.uint8)
        # thin_mask = cv2.erode(binary_mask, kernel, iterations=3)

        # 2) Thinning 차선 스켈레톤화(Thinning, Zhang-Suen)
        thin_mask = ximgproc.thinning(binary_mask, thinningType=ximgproc.THINNING_ZHANGSUEN)

        # 3) BEV 변환
        bev_mask = cv2.warpPerspective(thin_mask, M, bev_size)

        inf_time.update(t2 - t1, img_t.size(0))

        # ROS 퍼블리시: (단일 채널, BEV 변환된 결과)
        try:
            ros_mask = bridge.cv2_to_imgmsg(bev_mask, encoding="mono8")
            pub_mask.publish(ros_mask)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", str(e))

        # 실시간 화면 표시: BEV 변환 결과
        cv2.imshow('YOLOPv2 Thin Mask (BEV)', bev_mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.loginfo("[INFO] q 키 입력 -> 종료")
            if isinstance(vid_writer, cv2.VideoWriter):
                vid_writer.release()
            if hasattr(dataset, 'cap') and dataset.cap:
                dataset.cap.release()
            cv2.destroyAllWindows()

            # 웹캠 녹화 -> mp4
            if dataset.mode == 'stream' and save_img:
                end_time = time.time()
                real_duration = end_time - start_time if start_time else 0
                stem_name = Path(path_item).stem if path_item else 'webcam0'
                make_webcam_video(record_frames, save_dir, stem_name, real_duration)
            return

        # ----------------------
        # 저장 로직 (BEV 마스크)
        # ----------------------
        if save_img:
            # 3채널 변환
            bev_bgr = cv2.cvtColor(bev_mask, cv2.COLOR_GRAY2BGR)

            if dataset.mode == 'image':
                save_path = str(save_dir / Path(path_item).name)
            else:
                save_path = str(save_dir / Path(path_item).stem) + '.mp4'

            if dataset.mode == 'image':
                sp = Path(save_path)
                if sp.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                    sp = sp.with_suffix('.jpg')

                # BEV 결과 이미지로 저장
                cv2.imwrite(str(sp), bev_bgr)

            elif dataset.mode == 'video':
                nonlocal_vid_path = vid_path
                nonlocal_vid_writer = vid_writer

                if nonlocal_vid_path != save_path:
                    nonlocal_vid_path = save_path
                    if isinstance(nonlocal_vid_writer, cv2.VideoWriter):
                        nonlocal_vid_writer.release()

                    if vid_cap:
                        fps = vid_cap.get(cv2.CAP_PROP_FPS) or 30
                    else:
                        fps = 30
                    wv, hv = bev_bgr.shape[1], bev_bgr.shape[0]

                    rospy.loginfo(f"[INFO] 비디오 저장 시작: {nonlocal_vid_path} (FPS={fps}, size=({wv},{hv}))")
                    nonlocal_vid_writer = cv2.VideoWriter(
                        nonlocal_vid_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps,
                        (wv, hv)
                    )

                # BEV 영상 저장
                nonlocal_vid_writer.write(bev_bgr)
                vid_path = nonlocal_vid_path
                vid_writer = nonlocal_vid_writer
            else:
                # webcam stream
                record_frames.append(bev_bgr.copy())

    # end for
    # 자원 정리
    if isinstance(vid_writer, cv2.VideoWriter):
        vid_writer.release()
    if hasattr(dataset, 'cap') and dataset.cap:
        dataset.cap.release()
    cv2.destroyAllWindows()

    # 웹캠 + save_img -> mp4
    if dataset.mode == 'stream' and save_img:
        end_time = time.time()
        real_duration = end_time - start_time if start_time else 0
        make_webcam_video(record_frames, save_dir, 'webcam0', real_duration)

    rospy.loginfo("inf : (%.4fs/frame)", inf_time.avg)
    rospy.loginfo("Done. (%.3fs)", (time.time() - t0))


def ros_main():
    rospy.init_node('yolopv2_laneline_node', anonymous=True)

    parser = make_parser()
    opt, _ = parser.parse_known_args()

    # 새 퍼블리셔: (BEV 변환된) 얇은 마스크
    pub_mask = rospy.Publisher('yolopv2/lane_mask', Image, queue_size=1)

    detect_and_publish(opt, pub_mask)

    rospy.loginfo("[INFO] YOLOPv2 LaneLine node finished. spin() for keepalive.")
    rospy.spin()


if __name__ == '__main__':
    try:
        with torch.no_grad():
            ros_main()
    except rospy.ROSInterruptException:
        pass
