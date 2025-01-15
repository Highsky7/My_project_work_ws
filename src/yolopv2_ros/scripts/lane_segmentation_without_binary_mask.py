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
 1) Lane Line 세그멘테이션(전체 이미지)
 2) 이진화 마스크 + Thinning (얇은 선)
 3) (A) 실제 카메라 컬러 영상과 오버레이
 4) (B) 오버레이된 컬러 결과를 Bird’s Eye View(BEV) 변환
 5) 결과를 ROS 토픽(컬러) 퍼블리시
 6) 실시간 창 및 저장된 영상에서도 컬러 BEV 형태로 얇은 차선을 확인
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

    # Detection용이지만, 우선 남겨둠
    parser.add_argument('--conf-thres', type=float, default=0.3, help='(unused)')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='(unused)')

    # --nosave: True => 저장 안 함, False => 저장
    parser.add_argument('--nosave', action='store_true',
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
    """웹캠 모드에서 누적된 프레임(컬러 + 오버레이 + BEV)을 mp4로 저장."""
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
                          (w, h))  # (width, height)

    for f in record_frames:
        out.write(f)

    out.release()
    rospy.loginfo("[INFO] 웹캠 결과 영상 저장 완료: %s", save_path)


def detect_and_publish(opt, pub_img):
    """
    1) Lane Line 세그멘테이션
    2) Thinning
    3) (A) 오버레이(컬러)
    4) (B) BEV 변환
    5) ROS 퍼블리시, 실시간 표시, 저장
    """

    # OpenCV 최적화
    cv2.setUseOptimized(True)
    cv2.setNumThreads(0)

    # PyTorch cuDNN 최적화
    cudnn.benchmark = True

    bridge = CvBridge()

    source, weights = opt.source, opt.weights
    imgsz = opt.img_size

    # --nosave: True => 저장 안 함 / False => 저장
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

    # (선택) 프레임 스킵
    frame_skip = opt.frame_skip
    frame_counter = 0

    # BEV 변환용 파라미터 (예시값)
    src_points = np.float32([
        [200, 300],  # 왼쪽 위
        [440, 300],  # 오른쪽 위
        [50,  479],  # 왼쪽 아래
        [590, 479]   # 오른쪽 아래
    ])
    dst_points = np.float32([
        [100,   0],  # 왼쪽 위
        [300,   0],  # 오른쪽 위
        [100, 599],  # 왼쪽 아래
        [300, 599]   # 오른쪽 아래
    ])
    bev_size = (400, 600)  # (width, height)

    # 변환 행렬
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # 메인 루프
    for path_item, img, im0s, vid_cap in dataset:
        # (선택) 프레임 스킵
        if frame_skip > 0:
            if frame_counter % (frame_skip + 1) != 0:
                frame_counter += 1
                continue
            frame_counter += 1

        if dataset.mode == 'stream' and start_time is None:
            start_time = time.time()

        # 텐서 변환
        img_t = torch.from_numpy(img).to(device)
        img_t = img_t.half() if half else img_t.float()
        img_t /= 255.0
        if img_t.ndimension() == 3:
            img_t = img_t.unsqueeze(0)

        # 추론
        t1 = time_synchronized()
        with torch.no_grad():
            [_, _], seg, ll = model(img_t)
        t2 = time_synchronized()

        # Lane Segmentation (이진화)
        ll_seg_mask = lane_line_mask(ll)
        binary_mask = (ll_seg_mask > 0).astype(np.uint8) * 255

        # Thinning (스켈레톤)
        thin_mask = ximgproc.thinning(binary_mask, thinningType=ximgproc.THINNING_ZHANGSUEN)

        # (A) 실제 컬러 영상 오버레이
        #  - 방법 1: 차선 픽셀에만 특정 색(BGR) 칠하기
        overlay_img = im0s.copy()  # BGR

        # thin_mask == 255 인 영역에 빨간색(예) 적용
        overlay_img[thin_mask == 255] = (0, 0, 255)  # 빨간색 BGR

        #  - 방법 2: addWeighted(반투명)으로도 가능
        # color_mask = np.zeros_like(im0s, dtype=np.uint8)
        # color_mask[thin_mask == 255] = (0, 0, 255)
        # alpha = 0.4
        # overlay_img = cv2.addWeighted(im0s, 1 - alpha, color_mask, alpha, 0)

        # (B) BEV 변환 (컬러 이미지)
        bev_img = cv2.warpPerspective(overlay_img, M, bev_size)

        inf_time.update(t2 - t1, img_t.size(0))

        # ROS 퍼블리시: 컬러를 보낼 때 bgr8로 인코딩
        try:
            ros_bev = bridge.cv2_to_imgmsg(bev_img, encoding="bgr8")
            pub_img.publish(ros_bev)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", str(e))

        # 실시간 화면 표시
        cv2.imshow('YOLOPv2 Lane + BEV (Color)', bev_img)
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
        # 저장 로직 (BEV 컬러 결과)
        # ----------------------
        if save_img:
            if dataset.mode == 'image':
                save_path = str(save_dir / Path(path_item).name)
            else:
                save_path = str(save_dir / Path(path_item).stem) + '.mp4'

            if dataset.mode == 'image':
                sp = Path(save_path)
                if sp.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                    sp = sp.with_suffix('.jpg')

                cv2.imwrite(str(sp), bev_img)  # 컬러 BEV 저장

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

                    wv, hv = bev_img.shape[1], bev_img.shape[0]
                    rospy.loginfo(f"[INFO] 비디오 저장 시작: {nonlocal_vid_path} (FPS={fps}, size=({wv},{hv}))")

                    nonlocal_vid_writer = cv2.VideoWriter(
                        nonlocal_vid_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps,
                        (wv, hv)
                    )

                # BEV 컬러 영상을 mp4에 저장
                nonlocal_vid_writer.write(bev_img)
                vid_path = nonlocal_vid_path
                vid_writer = nonlocal_vid_writer
            else:
                # webcam stream
                record_frames.append(bev_img.copy())

    # end for
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

    # 새 퍼블리셔: (BEV 변환된) 컬러 영상
    pub_img = rospy.Publisher('yolopv2/lane_image_bev', Image, queue_size=1)

    detect_and_publish(opt, pub_img)

    rospy.loginfo("[INFO] YOLOPv2 LaneLine node finished. spin() for keepalive.")
    rospy.spin()


if __name__ == '__main__':
    try:
        with torch.no_grad():
            ros_main()
    except rospy.ROSInterruptException:
        pass
