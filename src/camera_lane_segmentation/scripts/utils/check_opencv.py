import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라 열기에 실패했습니다. /dev/video0 권한이나 연결 상태를 확인하세요.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Camera Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

