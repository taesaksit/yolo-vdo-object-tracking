from collections import defaultdict
import cv2
from ultralytics import YOLO
import time


class DiagonalLineCounter:
    def __init__(self, p1, p2):
        # p1 และ p2 คือจุดเริ่มต้นและจุดสิ้นสุดของเส้น เช่น (x1, y1), (x2, y2)
        self.p1 = p1
        self.p2 = p2
        self.car_count = 0
        self.counted_ids = set()

    def get_side(self, point):
        """คำนวณว่าจุด (x, y) อยู่ฝั่งไหนของเส้นตรง (p1 -> p2)

        คืนค่าเป็น บวก, ลบ, หรือ 0
        """
        x, y = point
        x1, y1 = self.p1
        x2, y2 = self.p2
        # ใช้สูตร Cross Product 2D เพื่อหาตำแหน่งเทียบกับเส้นตรง
        return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

    def check_crossing(self, track_id, prev_point, curr_point):
        if track_id in self.counted_ids:
            return False

        # เช็คฝั่งของจุดในเฟรมก่อนหน้าและเฟรมปัจจุบัน
        side_prev = self.get_side(prev_point)
        side_curr = self.get_side(curr_point)

        # ถ้าเครื่องหมายของฝั่งเปลี่ยนไป (ค่าบวกเปลี่ยนเป็นลบ หรือลบเปลี่ยนเป็นบวก)
        # แปลว่าจุดนั้นเคลื่อนที่ตัดผ่านเส้นตรงพอดี!
        if side_prev * side_curr < 0:
            self.car_count += 1
            self.counted_ids.add(track_id)
            return True

        return False


def calculate_fps(curr_time, prev_time):
    if prev_time == 0:
        return 0, curr_time
    fps = 1 / (curr_time - prev_time)
    return fps, curr_time


def main():
    model = YOLO("yolov8n.pt")
    video_path = "car.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: ไม่สามารถเปิดไฟล์วิดีโอได้ที่ {video_path}")
        return

    # Line ที่รถผ่าน
    line_start = (100, 1700)
    line_end = (1200, 1800)
    counter = DiagonalLineCounter(p1=line_start, p2=line_end)
    
    # เก็บประวัติพิกัด (x, y) ย้อนหลังของแต่ละ ID
    track_history = defaultdict(lambda: [])
    prev_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            source=frame,
            persist=True,
            classes=[2],  # คลาสรถยนต์
            tracker="bytetrack.yaml",
            conf=0.5,
            iou=0.5,
            verbose=False,
        )

        frame_ = results[0].plot()

        # วาดเส้นนับสีฟ้าเอียงๆ ลงบนจอ
        cv2.line(frame_, line_start, line_end, (255, 0, 0), 3)
        cv2.putText(
            frame_,
            "Diagonal Counting Line",
            (line_start[0], line_start[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                # ใช้จุดกึ่งกลางขอบล่างของกล่องเป็นจุดอ้างอิง
                cx = int((x1 + x2) / 2)
                cy = int(y2)
                current_point = (cx, cy)

                cv2.circle(frame_, current_point, 5, (0, 0, 255), -1)

                # บันทึกประวัติพิกัด (x, y) ย้อนหลัง 2 เฟรม
                track_history[track_id].append(current_point)
                if len(track_history[track_id]) > 2:
                    track_history[track_id].pop(0)

                # ถ้ามีข้อมูลครบ 2 เฟรม นำไปเช็คการข้ามเส้นเฉียง
                if len(track_history[track_id]) == 2:
                    prev_point = track_history[track_id][0]
                    curr_point = track_history[track_id][1]

                    counter.check_crossing(track_id, prev_point, curr_point)

        # คำนวณ FPS และแสดงผล
        curr_time = time.time()
        fps, prev_time = calculate_fps(curr_time, prev_time)

        cv2.putText(
            frame_,
            f"FPS: {int(fps)}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame_,
            f"Total Cars Counted: {counter.car_count}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        cv2.imshow("Diagonal Line Vehicle Counting", frame_)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
