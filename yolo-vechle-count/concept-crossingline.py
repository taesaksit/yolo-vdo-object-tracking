import cv2
import numpy as np


class LineCounter:
    def __init__(self, line_y):
        # ประกาศตัวแปรตำแหน่งเส้นแบบ Dynamic
        self.line_y = line_y
        self.car_count = 0
        self.counted_ids = set()

    def check_crossing(self, track_id, prev_y, curr_y):
        """ฟังก์ชันตรวจสอบการข้ามเส้น
        - track_id: รหัสประจำตัวรถ
        - prev_y: ตำแหน่ง Y เฟรมก่อนหน้า
        - curr_y: ตำแหน่ง Y เฟรมปัจจุบัน
        """
        # เงื่อนไข:
        # 1. เฟรมก่อนอยู่ก่อนเส้น (prev_y < line_y) และ เฟรมปัจจุบันข้ามเส้น (curr_y >= line_y)
        # 2. ID นี้ยังไม่เคยถูกนับมาก่อน
        if (
            prev_y < self.line_y
            and curr_y >= self.line_y
            and track_id not in self.counted_ids
        ):
            self.car_count += 1
            self.counted_ids.add(track_id)
            return True

        return False


def main():
    # 1. สร้างภาพจำลองพื้นหลังสีขาว
    img = np.ones((600, 600, 3), dtype=np.uint8) * 255

    # 2. ประกาศตัวแปรตำแหน่งเส้น (Dynamic Line Position)
    # คุณสามารถเปลี่ยนเลขตรงนี้ได้อิสระ เช่น อยากให้เส้นอยู่สูงหรือต่ำลงมา
    LINE_Y = 500
    line_start = (100, LINE_Y)
    line_end = (500, LINE_Y)

    # สร้างออบเจกต์ตัวนับ
    counter = LineCounter(line_y=LINE_Y)

    # 3. จำลองเหตุการณ์รถขับผ่าน (สมมติรถ ID #101)
    car_id = 101
    car_x = 250

    # สถานการณ์ทดลอง: ลองปรับเปลี่ยนค่า prev_y และ curr_y เพื่อดูผลลัพธ์
    prev_y = 100  # ยังอยู่เหนือเส้น (ไม่เกิน)
    curr_y = 499  # ข้ามลงมาใต้เส้นแล้ว (เกินเส้น -> ต้องนับ!)

    # เรียกใช้ฟังก์ชันตรวจสอบ
    is_counted = counter.check_crossing(car_id, prev_y, curr_y)

    # --- วาดภาพแสดงผล ---
    # วาดเส้นนับ
    cv2.line(img, line_start, line_end, (255, 0, 0), 3)
    cv2.putText(
        img,
        f"Dynamic Line (Y={LINE_Y})",
        (100, LINE_Y - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2,
    )

    # วาดรถจำลองในตำแหน่งปัจจุบัน
    cv2.rectangle(img, (car_x - 30, curr_y - 40), (car_x + 30, curr_y), (0, 255, 0), 2)
    cv2.circle(img, (car_x, curr_y), 5, (0, 0, 255), -1)

    # แสดงผลสถานะการนับบนภาพ
    status_msg = (
        f"Counted! Total: {counter.car_count}" if is_counted else "Not Counted (Yet)"
    )
    cv2.putText(
        img,
        f"Status: {status_msg}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 100, 0) if is_counted else (0, 0, 255),
        2,
    )

    print(f"Result -> {status_msg}")

    # แสดงหน้าต่างภาพนิ่ง
    cv2.imshow("Dynamic Line Crossing Function", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
