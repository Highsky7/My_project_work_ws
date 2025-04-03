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

# Assuming utils.util contains all necessary helper functions
# Make sure this file exists and is accessible
try:
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
        show_seg_result, # This function is crucial for visualization
        AverageMeter,
        letterbox,
        LoadImages,
        LoadCamera
    )
except ImportError:
    print("[ERROR] Could not import functions from 'utils.util'.")
    print("Please ensure 'utils/util.py' exists and is in the correct path.")
    exit()

class ROSImageSubscriber:
    def __init__(self, topic, img_size=640, stride=32):
        """ROS 이미지 토픽을 구독하는 클래스 초기화"""
        self.bridge = CvBridge()
        self.img_size = img_size
        self.stride = stride
        # Verify topic exists (optional, but good practice)
        # topics = rospy.get_published_topics()
        # if not any(t[0] == topic for t in topics):
        #    print(f"[WARN] ROS topic '{topic}' does not seem to be published.")
        self.sub = rospy.Subscriber(topic, Image, self.callback)
        self.img0 = None  # 원본 이미지 (for visualization)
        self.img = None   # 모델 입력용 리사이즈/정규화된 이미지
        self.new_frame = False
        self.last_msg_time = None
        print(f"[INFO] Subscribing to ROS topic: {topic}")

    def callback(self, data):
        """새로운 ROS 메시지가 도착했을 때 호출되는 콜백 함수"""
        try:
            # Use message timestamp if available
            self.last_msg_time = data.header.stamp.to_sec()
            # Convert ROS Image message to OpenCV image (BGR format)
            self.img0 = self.bridge.imgmsg_to_cv2(data, "bgr8")
            if self.img0 is None:
                print("[ERROR] CvBridge returned None image.")
                return

            # Preprocess for model input: letterbox resize, BGR->RGB, HWC->CHW
            img_letterboxed = letterbox(self.img0, self.img_size, stride=self.stride)[0]
            img_rgb_chw = img_letterboxed[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
            self.img = np.ascontiguousarray(img_rgb_chw) # Ensure contiguous array

            self.new_frame = True # Flag that a new frame is ready
        except CvBridgeError as e:
            print(f"[ERROR] CvBridge Error: {e}")
        except Exception as e:
            print(f"[ERROR] Error in ROS callback: {e}")

    def get_frame(self):
        """최신 프레임을 반환 (모델 입력용 이미지, 원본 이미지)"""
        if self.new_frame:
            self.new_frame = False
            return self.img, self.img0 # Return both processed and original image
        return None, None

def make_parser():
    """명령줄 인수를 파싱하는 함수"""
    parser = argparse.ArgumentParser(description="YOLOPv2 ROS, 동영상 및 카메라 감지")
    parser.add_argument('--weights', type=str,
                        default='./yolopv2.pt', # Default weight file name
                        help='모델 가중치 파일 경로 (e.g., ./yolopv2.pt)')
    parser.add_argument('--source', type=str,
                        default='/camera/image_raw', # Default to a common ROS topic
                        help='입력 소스: ROS 토픽 (예: /camera/image_raw), 동영상 파일 경로 (예: video.mp4), 또는 카메라 ID (예: 0)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='모델 입력 이미지 크기 (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3,
                        help='객체 신뢰도 임계값')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                        help='NMS IOU 임계값')
    parser.add_argument('--device', default='0',
                        help='CUDA 장치 (예: 0 또는 0,1,2,3) 또는 cpu')
    parser.add_argument('--save-conf', action='store_true',
                        help='레이블에 신뢰도 저장 (save-txt 활성화 시)')
    parser.add_argument('--save-txt', action='store_true',
                        help='결과를 *.txt 파일로 저장')
    parser.add_argument('--nosave', action='store_true', # Changed default: Don't save by default
                        help='이미지/비디오 저장 안함')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='필터링할 클래스 (예: --classes 0, 또는 --classes 0 2 3)')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='클래스 무관 NMS')
    parser.add_argument('--project', default='runs/detect',
                        help='결과 저장 경로')
    parser.add_argument('--name', default='exp',
                        help='실험 이름 (결과 저장 폴더 이름)')
    parser.add_argument('--exist-ok', action='store_true',
                        help='기존 프로젝트/이름 폴더 덮어쓰기 허용')
    return parser

def detect(opt):
    """YOLOPv2를 사용한 실시간, 동영상 및 카메라 감지 함수"""
    source, weights, save_txt, imgsz = opt.source, opt.weights, opt.save_txt, opt.img_size
    save_img = not opt.nosave # Determine if images/videos should be saved
    is_ros = False
    is_camera = False
    is_video = False

    # --- Output Directory Setup ---
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    if save_img or save_txt:
      (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
      print(f"[INFO] Saving results to {save_dir}")

    # --- Device Setup ---
    device = select_device(opt.device)
    half = device.type != 'cpu'  # Use half precision (FP16) only on GPU
    print(f"[INFO] Using device: {device}")

    # --- Model Loading ---
    try:
        print(f"[INFO] Loading model from {weights}...")
        # Assume weights are TorchScript (.pt)
        model = torch.jit.load(weights)
        model = model.to(device)
        if half:
            model.half()  # Convert model to FP16
        model.eval() # Set model to evaluation mode
        stride = 32 # Assuming stride from common YOLO models, adjust if needed
        print("[INFO] Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    # --- Input Source Setup ---
    vid_path, vid_writer = None, None
    if source.startswith('/'):  # ROS Topic
        if not rospy.core.is_initialized():
             print("[ERROR] ROS node not initialized. Make sure to run rospy.init_node() before calling detect().")
             return
        dataset = ROSImageSubscriber(source, img_size=imgsz, stride=stride)
        is_ros = True
        print("[INFO] ROS topic detection mode enabled.")
    elif source.isdigit():  # Camera ID
        try:
            dataset = LoadCamera(int(source), img_size=imgsz, stride=stride)
            is_camera = True
            print(f"[INFO] Camera detection mode enabled (ID: {source}).")
        except Exception as e:
            print(f"[ERROR] Failed to open camera {source}: {e}")
            return
    else:  # Video File
        try:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)
            is_video = True
            print(f"[INFO] Video file detection mode enabled: {source}")
            # print(f"[DEBUG] Video total frames: {dataset.nf}") # dataset.nf might not be available until iteration
        except Exception as e:
            print(f"[ERROR] Failed to load video {source}: {e}")
            return

    # --- ROS Publisher Setup (only if using ROS source) ---
    bridge = CvBridge()
    ll_pub = None
    da_pub = None
    if is_ros:
        ll_pub = rospy.Publisher('/yolopv2/lane_line_mask', Image, queue_size=1)
        da_pub = rospy.Publisher('/yolopv2/driving_area_mask', Image, queue_size=1)
        print("[INFO] ROS publishers initialized for segmentation masks.")

    # --- Warm-up Run ---
    if device.type != 'cpu':
        print("[INFO] Running model warm-up...")
        dummy_input = torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))
        try:
             _ = model(dummy_input) # Warm-up inference
             print("[INFO] Model warm-up complete.")
        except Exception as e:
             print(f"[WARN] Model warm-up failed (might be expected for some models): {e}")


    # --- Performance Timers ---
    inf_time = AverageMeter()
    nms_time = AverageMeter()
    # waste_time = AverageMeter() # Not used in the provided process_frame

    # --- Main Processing Loop ---
    t0 = time.time()
    frame_count = 0
    win_name = "YOLOPv2 Detection Result"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO) # Resizable window

    try:
        # --- ROS Loop ---
        if is_ros:
            print("[INFO] Starting ROS processing loop...")
            while not rospy.is_shutdown():
                img, im0 = dataset.get_frame() # Get preprocessed and original frames
                if img is None or im0 is None:
                    # print("[DEBUG] No new frame yet...")
                    time.sleep(0.01) # Avoid busy-waiting
                    continue

                frame_count += 1
                # print(f"[DEBUG] Processing ROS frame {frame_count}")
                process_frame(img, im0, model, device, half, inf_time, nms_time,
                              save_img, save_txt, save_dir, bridge, ll_pub, da_pub, is_ros,
                              opt, frame_count, win_name, vid_writer) # Pass necessary args

                if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1: # Check if window closed
                    print("[INFO] Output window closed by user.")
                    break
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[INFO] 'q' pressed, shutting down.")
                    break
            rospy.signal_shutdown("Processing finished or user quit.")

        # --- Camera/Video Loop ---
        else:
             print(f"[INFO] Starting {'Camera' if is_camera else 'Video'} processing loop...")
             for path, img, im0, cap in dataset: # path, img(processed), im0s(original), cap
                 if img is None or im0 is None:
                     print(f"[WARN] Skipping empty frame from source.")
                     continue

                 frame_count += 1
                 # print(f"[DEBUG] Processing {'Camera' if is_camera else 'Video'} frame {frame_count}")

                 # Setup video writer for saving output (if needed)
                 if is_video and save_img and vid_writer is None:
                     fps = cap.get(cv2.CAP_PROP_FPS)
                     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                     save_path = str(save_dir / Path(source).name) # Save with original video name
                     vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (im0.shape[1], im0.shape[0])) # Use shape of im0 for writer

                 process_frame(img, im0, model, device, half, inf_time, nms_time,
                               save_img, save_txt, save_dir, bridge, ll_pub, da_pub, is_ros,
                               opt, frame_count, win_name, vid_writer, path=path) # Pass path for saving txt

                 if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("[INFO] Output window closed by user.")
                    break
                 key = cv2.waitKey(1) & 0xFF
                 if key == ord('q'):
                     print("[INFO] 'q' pressed, stopping.")
                     break

             if vid_writer is not None:
                 vid_writer.release() # Release video writer if it was used

    except SystemExit:
        print("[INFO] SystemExit caught, shutting down.")
    except Exception as e:
        print(f"[ERROR] An error occurred during processing loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- Cleanup ---
        print(f'[INFO] Processed {frame_count} frames.')
        print(f'Average Inference time: {inf_time.avg:.4f}s/frame')
        print(f'Average NMS time: {nms_time.avg:.4f}s/frame')
        print(f'Total processing time: {time.time() - t0:.3f}s')
        cv2.destroyAllWindows()
        print("[INFO] OpenCV windows closed.")
        if is_ros and not rospy.is_shutdown():
             rospy.signal_shutdown("Processing finished")
             print("[INFO] ROS shutdown requested.")

def process_frame(img, im0, model, device, half, inf_time, nms_time,
                  save_img, save_txt, save_dir, bridge, ll_pub, da_pub, is_ros,
                  opt, frame_id, win_name, vid_writer=None, path="frame"):
    """프레임 처리, 감지, 분할 및 시각화 함수"""
    # img: preprocessed image for model (Tensor)
    # im0: original image (numpy array, BGR)

    # --- Preprocessing for Model ---
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0) # Add batch dimension if missing

    # --- Inference ---
    t1 = time_synchronized()
    try:
        # Model output structure might vary, adapt as needed
        # Assuming [detection_output, segmentation_output, lane_line_output]
        # Example: det_out, da_seg_out, ll_seg_out = model(img)
        # Your original code suggests: [pred, anchor_grid], seg, ll = model(img)
        # Let's stick to the original structure:
        outputs = model(img)
        # Check the type and structure of outputs
        # print(f"DEBUG: Model output type: {type(outputs)}, Len: {len(outputs) if isinstance(outputs, (list, tuple)) else 'N/A'}")
        # if isinstance(outputs, (list, tuple)) and len(outputs) == 3:
        #     (pred_raw, anchor_grid), da_seg_out, ll_seg_out = outputs # Original assumption
        # else:
        #      # Handle unexpected output format - maybe just tuple(det, seg, ll)?
        #      print(f"[WARN] Unexpected model output structure: {type(outputs)}")
        #      # Attempt to unpack if possible, otherwise raise error or adapt
        #      # Fallback/Assumption: Maybe it's (det, seg, ll) directly?
        #      if isinstance(outputs, (list, tuple)) and len(outputs) == 3:
        #         pred_raw, da_seg_out, ll_seg_out = outputs # Try direct unpacking
        #         anchor_grid = None # Indicate anchor_grid was not part of output
        #      else:
        #         # If still not matching, this is an error
        #         raise TypeError(f"Model output format error. Expected 3 elements (det, seg, ll) or similar, got {type(outputs)}")
        # Original line:
        [pred_raw, anchor_grid], da_seg_out, ll_seg_out = outputs # Make sure model returns this structure

    except Exception as e:
         print(f"[ERROR] Model inference failed: {e}")
         return # Skip processing this frame
    t2 = time_synchronized()
    inf_time.update(t2 - t1, img.size(0))

    # --- Detection Post-processing ---
    # The split_for_trace_model might be specific to traced models, adjust if using regular model
    # pred = split_for_trace_model(pred_raw, anchor_grid) # Apply if necessary
    pred = pred_raw # If split not needed or anchor_grid wasn't returned

    t3 = time_synchronized()
    # Apply Non-Maximum Suppression (NMS)
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                               classes=opt.classes, agnostic=opt.agnostic_nms)
    t4 = time_synchronized()
    nms_time.update(t4 - t3, img.size(0))

    # --- Segmentation Post-processing ---
    # Process drivable area segmentation output (da_seg_out)
    # This function should return a binary mask of the same size as the input image (im0)
    da_seg_mask = driving_area_mask(da_seg_out, im0.shape[:2]) # Ensure driving_area_mask resizes to im0 shape

    # Process lane line segmentation output (ll_seg_out)
    # This function should return a binary mask of the same size as the input image (im0)
    ll_seg_mask = lane_line_mask(ll_seg_out, im0.shape[:2]) # Ensure lane_line_mask resizes to im0 shape

    # --- Optional: Refine masks (like the connected components logic from original) ---
    # This refines masks based on connectivity and position, potentially improving results
    # Note: This part assumes single-channel binary masks from the functions above
    try:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ll_seg_mask, connectivity=8)
        if num_labels > 1: # Found components beyond background
            # Sort detected lanes by their bottom-most point
            bottom_y = stats[1:, cv2.CC_STAT_TOP] + stats[1:, cv2.CC_STAT_HEIGHT]
            sorted_indices = np.argsort(bottom_y)[::-1] # Sort descending (highest y = bottom)
            sorted_labels = 1 + sorted_indices # Get labels (1-based)

            # Keep only the bottom-most two lane lines (most likely the current lanes)
            top_labels = sorted_labels[:min(2, len(sorted_labels))]

            # Create a new mask with only the selected lanes
            refined_ll_mask = np.zeros_like(labels, dtype=np.uint8)
            for label in top_labels:
                refined_ll_mask[labels == label] = 255 # Use 255 for visualization

            # Further refine drivable area based on the two detected lanes
            if len(top_labels) == 2:
                centroid_x = centroids[top_labels, 0]
                # Determine left and right lanes based on x-centroid
                label_left_idx = np.argmin(centroid_x)
                label_right_idx = np.argmax(centroid_x)
                label_left = top_labels[label_left_idx]
                label_right = top_labels[label_right_idx]

                left_lane_pixels = (labels == label_left)
                right_lane_pixels = (labels == label_right)

                # Create new drivable area between the detected lanes
                refined_da_mask = np.zeros_like(da_seg_mask, dtype=np.uint8)
                for y in range(im0.shape[0]):
                    left_x_coords = np.where(left_lane_pixels[y, :])[0]
                    right_x_coords = np.where(right_lane_pixels[y, :])[0]

                    if left_x_coords.size > 0 and right_x_coords.size > 0:
                        max_x_left = left_x_coords.max() # Furthest right point of left lane
                        min_x_right = right_x_coords.min() # Furthest left point of right lane
                        if max_x_left < min_x_right: # Ensure lanes don't cross
                             refined_da_mask[y, max_x_left+1 : min_x_right] = 255 # Fill between

                da_seg_mask = refined_da_mask # Update DA mask

            # Update lane mask to the refined one
            ll_seg_mask = refined_ll_mask

        else: # No lane lines detected after connected components
            da_seg_mask = np.zeros_like(da_seg_mask) # Clear DA mask
            ll_seg_mask = np.zeros_like(ll_seg_mask) # Clear LL mask

        # Ensure masks are binary (0 or 255) or (0 or 1) depending on show_seg_result needs
        # Assuming show_seg_result handles 0/255 masks:
        da_seg_mask = (da_seg_mask > 0).astype(np.uint8) * 255
        ll_seg_mask = (ll_seg_mask > 0).astype(np.uint8) * 255

    except Exception as e:
        print(f"[WARN] Error during mask refinement: {e}")
        # Fallback to using the raw masks if refinement fails
        da_seg_mask = driving_area_mask(da_seg_out, im0.shape[:2]) # Re-calculate raw mask
        ll_seg_mask = lane_line_mask(ll_seg_out, im0.shape[:2])   # Re-calculate raw mask
        da_seg_mask = (da_seg_mask > 0).astype(np.uint8) * 255
        ll_seg_mask = (ll_seg_mask > 0).astype(np.uint8) * 255


    # --- Process Detections (Drawing Boxes & Saving Labels) ---
    # Process detections for the *first* image in the batch (index 0)
    det = pred[0] # Get detections for this image
    im0_display = im0.copy() # Create a copy for drawing, leave original im0 untouched if needed elsewhere

    if det is not None and len(det):
        # Rescale boxes from img_size (model input) back to im0 (original image) size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        # Draw boxes and save labels
        for *xyxy, conf, cls in reversed(det): # Iterate through detections
            # --- Draw Bounding Box ---
            # This function draws the box directly onto im0_display
            plot_one_box(xyxy, im0_display, label=f'{int(cls)} {conf:.2f}', color=(255,0,0), line_thickness=2)

            # --- Save Labels to File (if enabled) ---
            if save_txt:
                # Convert xyxy to xywh format (center_x, center_y, width, height) - normalized
                xywh = xyxy2xywh(torch.tensor(xyxy).view(1, 4) / torch.tensor(im0.shape)[[1, 0, 1, 0]]) # Normalize by W,H,W,H
                xywh = xywh.view(-1).tolist() # Flatten to list
                line = (int(cls), *xywh, conf) if opt.save_conf else (int(cls), *xywh) # Format line
                # Define txt filename based on source type
                if path != "frame": # Video or camera stream path
                    txt_file_name = f'{Path(path).stem}_{frame_id}'
                else: # ROS stream (no path)
                     txt_file_name = f'frame_{frame_id:05d}' # Use frame count for ROS
                txt_path = str(save_dir / 'labels' / txt_file_name)
                try:
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
                except Exception as e:
                    print(f"[ERROR] Failed to write labels to {txt_path}.txt: {e}")

    # Else (no detections) - do nothing for boxes/labels

    # --- Apply Segmentation Masks for Visualization ---
    # This function overlays da_seg_mask and ll_seg_mask onto im0_display
    # Assumes show_seg_result modifies im0_display in place or returns the modified image
    # Ensure da_seg_mask and ll_seg_mask are uint8 and have the same H,W as im0_display
    try:
        # Ensure masks have 3 channels if needed by show_seg_result, or handle inside it
        # Example: if needed:
        # da_3ch = cv2.cvtColor(da_seg_mask, cv2.COLOR_GRAY2BGR)
        # ll_3ch = cv2.cvtColor(ll_seg_mask, cv2.COLOR_GRAY2BGR)
        # show_seg_result(im0_display, (da_3ch, ll_3ch), ...)
        show_seg_result(im0_display, (da_seg_mask, ll_seg_mask), is_demo=True) # is_demo likely handles blending
    except Exception as e:
        print(f"[ERROR] Failed during show_seg_result visualization: {e}")
        # Continue without segmentation overlay if it fails

    # --- Display the Combined Result ---
    # im0_display now has detections boxes and segmentation overlays
    # Display the final image in the OpenCV window
    cv2.imshow(win_name, im0_display)

    # --- Save Output Frame (if enabled) ---
    if save_img:
        if vid_writer is not None: # Saving video
            vid_writer.write(im0_display)
        # else: # Saving individual images (e.g., from camera or ROS) - Less common for streams
        #    save_path = str(save_dir / f'frame_{frame_id:05d}.jpg')
        #    cv2.imwrite(save_path, im0_display)

    # --- Publish Masks to ROS (if enabled) ---
    if is_ros and ll_pub is not None and da_pub is not None:
        try:
            # Convert binary masks (0/255) to BGR for publishing as sensor_msgs/Image
            ll_mask_bgr = cv2.cvtColor(ll_seg_mask, cv2.COLOR_GRAY2BGR)
            da_mask_bgr = cv2.cvtColor(da_seg_mask, cv2.COLOR_GRAY2BGR)
            # Create ROS Image messages
            ll_msg = bridge.cv2_to_imgmsg(ll_mask_bgr, "bgr8")
            da_msg = bridge.cv2_to_imgmsg(da_mask_bgr, "bgr8")
            # Add timestamp (optional, use current time or original msg time if available)
            ll_msg.header.stamp = rospy.Time.now()
            da_msg.header.stamp = rospy.Time.now()
            # Publish the messages
            ll_pub.publish(ll_msg)
            da_pub.publish(da_msg)
        except CvBridgeError as e:
            print(f"[ERROR] CvBridge Error during mask publishing: {e}")
        except Exception as e:
            print(f"[ERROR] Failed to publish masks to ROS: {e}")

# --- Main Execution Block ---
if __name__ == '__main__':
    opt = make_parser().parse_args()
    print("[OPTIONS]", opt)

    # Initialize ROS node only if the source is a ROS topic
    if opt.source.startswith('/'):
        try:
            rospy.init_node('yolopv2_ros_detector', anonymous=True)
            print(f"[INFO] ROS node 'yolopv2_ros_detector' initialized.")
            print(f"[INFO] Waiting for ROS masters and topic ({opt.source})...")
            # Optional: Add a small delay or check for topic availability
            # rospy.wait_for_message(opt.source, Image, timeout=5.0) # Wait max 5s for first message
        except rospy.ROSInitException as e:
             print(f"[FATAL] Failed to initialize ROS node: {e}. Is ROS master running?")
             exit(1)
        except Exception as e:
             print(f"[WARN] Could not wait for message on {opt.source} (may not be published yet): {e}")


    # Run detection within a no_grad context for efficiency
    with torch.no_grad():
        try:
            detect(opt)
        except SystemExit:
            print("[INFO] Program terminated by user (SystemExit).")
        except KeyboardInterrupt:
            print("[INFO] Program interrupted by user (KeyboardInterrupt).")
        except Exception as e:
             print(f"[FATAL] An unhandled exception occurred in detect(): {e}")
             import traceback
             traceback.print_exc()

    print("[INFO] Program finished.")