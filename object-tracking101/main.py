from ultralytics import YOLO
import cv2
import pandas as pd
import time


def calculate_fps(curr_time, prev_time):
    """คำนวณค่า FPS จากเวลาปัจจุบันและเวลาของเฟรมก่อนหน้า"""
    if prev_time == 0:
        return 0, curr_time
    
    fps = 1 / (curr_time - prev_time)
    return fps, curr_time


def main():
    # 1. โหลดโมเดล YOLO26 (ตัวเลือกไซส์: yolo26n.pt, yolo26s.pt, yolo26m.pt)
    model = YOLO("yolo26m.pt")

    # 2. Export Classes ออกเป็น CSV
    classes_dict = model.names
    classes_df = pd.DataFrame(list(classes_dict.items()), columns=["ID", "Class"])
    classes_df.to_csv("yolo26_classes.csv", index=False)

    # 3. โหลดไฟล์วิดีโอ
    video_path = "./vdo/many-dog.mov"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: ไม่สามารถเปิดไฟล์วิดีโอได้ที่ {video_path}")
        return

    prev_time = 0

    # 4. ลูปอ่านและประมวลผลทีละเฟรม
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ทำ Object Tracking ด้วย YOLO26
        results = model.track(
            source=frame,
            persist=True,
            #classes=[0, 16, 32],      # กรองเฉพาะคลาสที่สนใจ (เช่น คน, สัตว์ หรือวัตถุอื่นๆ)
            #tracker="bytetrack.yaml",
            conf=0.3,
            iou=0.5,
            device="mps",             # ใช้ GPU บน Mac (ถ้าใช้ Windows/NVIDIA เปลี่ยนเป็น "0")
            verbose=False,
        )
        
        frame_ = results[0].plot()

        # คำนวณและแสดงผล FPS
        curr_time = time.time()
        fps, prev_time = calculate_fps(curr_time, prev_time)

        cv2.putText(
            frame_,
            f"YOLO26 FPS: {int(fps)}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        
        cv2.imshow("YOLO26 Tracking", frame_)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()