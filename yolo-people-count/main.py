import cv2
from ultralytics import solutions


def draw_count(frame, person_in, person_out):
    height, width, _ = frame.shape

    # ตำแหน่งกรอบด้านล่าง
    box_x1 = 20
    box_y1 = 20
    box_x2 = width - 20
    box_y2 = 80

    # วาดกรอบ
    cv2.rectangle(
        frame,
        (box_x1, box_y1),
        (box_x2, box_y2),
        (0, 0, 255),
        2,
    )

    # ข้อความ
    text = f"OUT: {person_in}    IN: {person_out}"

    cv2.putText(
        frame,
        text,
        (box_x1 + 20, box_y1 + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def main():
    vdo_path = "people_mall.mp4"
    cap = cv2.VideoCapture(vdo_path)
    assert cap.isOpened(), "Error reading video file"

    success, frame = cap.read()
    _, width, _ = frame.shape

    pt1 = (0, 270)
    pt2 = (width, 270)

    line_point = [pt1, pt2]

    counter = solutions.ObjectCounter(
        # --- พารามิเตอร์หลักสำหรับกำหนดค่าการนับ ---
        show=False,  # (bool) แสดงหน้าต่างวิดีโอผลลัพธ์ระหว่างประมวลผลหรือไม่
        region=line_point,  # (list) กำหนดพิกัดเส้นหรือพื้นที่สำหรับนับจำนวน (เช่น [(x1,y1), (x2,y2)])
        show_in=False,  # (bool) แสดงจำนวนวัตถุที่เข้ามา (In count) บนหน้าจอ
        show_out=False,  # (bool) แสดงจำนวนวัตถุที่ออกไป (Out count) บนหน้าจอ
        # --- พารามิเตอร์ตั้งค่าโมเดลและการแสดงผล ---
        model="yolo11n.pt",
        line_width=2,  # (int) ความหนาของเส้นกรอบและเส้นนับจำนวน
        device=0,
        classes=None,  # (list) กรองเฉพาะหมายเลขคลาสที่ต้องการนับ (เช่น [0] สำหรับคน หรือ [2] สำหรับรถ)
        conf=0.25,  #
        iou=0.45,  #
        show_conf=True,
        show_labels=True,
        verbose=False,
    )

    reverse_direction = True

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        result = counter(frame)

        person_in = result.classwise_count.get("person", {}).get("IN", 0)
        person_out = result.classwise_count.get("person", {}).get("OUT", 0)

        draw_count(
            result.plot_im,
            person_in,
            person_out,
        )

        cv2.imshow("Countig People", result.plot_im)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    main()
