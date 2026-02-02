import cv2

# 替换为你的手机实际 IP 和端口
url = "http://192.168.110.189:8080/video"

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("无法打开视频流，请检查网络和URL")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取帧，可能是流中断")
        break

    # 显示画面
    cv2.imshow('手机摄像头', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()