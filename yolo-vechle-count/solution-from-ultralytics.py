import cv2
from ultralytics import solutions


cap = cv2.VideoCapture("yolo-vechle-count/car.mp4")
assert cap.isOpened(), "Error reading video file"

line_points = [(100, 1700), (1200, 1800)]

counter = solutions.ObjectCounter(
    show=False,
    region=line_points,
    model="yolo11n.pt",
    classes=[2],  # เฉพาะรถยนต์ (COCO class 2)
    line_width=2,
    device=0,
    show_in=True,
    show_out=False,
    iou=0.5,
    conf=0.5,
)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = counter(frame)  # tracking + counting + วาดเส้น/กล่องให้อัตโนมัติ

    cv2.imshow("Vehicle Counting", results.plot_im)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
