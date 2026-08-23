"""
Intelligent People Counting System - GUI version
--------------------------------------------------
Wraps the Ultralytics ObjectCounter (YOLO) logic from the original script
in a Tkinter dashboard styled like:

  [ PEOPLE COUNTING ]  (green header)
  [ video feed ]  [ IN  <count> ]  (gray box)
                  [ OUT <count> ]  (red box)
                  Status: Running
                  IP Address: x.x.x.x

Requirements:
    pip install opencv-python pillow ultralytics
"""

import time
import socket
import threading
import tkinter as tk

import cv2
from PIL import Image, ImageTk
from ultralytics import solutions


# ---------------------------------------------------------------------------
# Config - edit these for your setup
# ---------------------------------------------------------------------------
VIDEO_SOURCE = "people_mall.mp4"  # file path, RTSP URL, or camera index (0)
MODEL_PATH = "yolo11n.pt"
CONF_THRES = 0.25
IOU_THRES = 0.45
LINE_HEIGHT_RATIO = 0.5  # counting line at 50% of frame height
WINDOW_SIZE = "1400x760"


def get_local_ip() -> str:
    """Best-effort local IP address for display purposes."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except OSError:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


class PeopleCountingApp:
    def __init__(self, root: tk.Tk, video_source, model_path: str):
        self.root = root
        self.video_source = video_source
        self.model_path = model_path
        self.ip_address = get_local_ip()

        self.running = False
        self.cap = None
        self.counter = None
        self.thread = None

        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        self.root.title("Intelligent People Counting System")
        self.root.configure(bg="black")

        # ---- Header -----------------------------------------------------
        header = tk.Frame(self.root, bg="#0a7d34")
        header.pack(side="top", fill="x")
        tk.Label(
            header,
            text="Intelligent People Counting System: Managing Foot Traffic with AI and Python",
            font=("Segoe UI", 11, "bold"),
            bg="#0a7d34",
            fg="white",
            anchor="w",
        ).pack(fill="x", padx=15, pady=(6, 0))
        tk.Label(
            header,
            text="PEOPLE COUNTING",
            font=("Arial Black", 34, "bold"),
            bg="#0a7d34",
            fg="white",
        ).pack(pady=(0, 10))

        # ---- Body: video (left) + sidebar (right) ------------------------
        body = tk.Frame(self.root, bg="black")
        body.pack(side="top", fill="both", expand=True)

        self.video_label = tk.Label(body, bg="black")
        self.video_label.pack(side="left", fill="both", expand=True)

        sidebar = tk.Frame(body, bg="#d9d9d9", width=380)
        sidebar.pack(side="right", fill="y")
        sidebar.pack_propagate(False)

        # IN panel
        in_box = tk.Frame(sidebar, bg="#5a5a5a", height=210)
        in_box.pack(side="top", fill="x")
        in_box.pack_propagate(False)
        tk.Label(
            in_box, text="IN", font=("Arial", 34, "bold"), bg="#5a5a5a", fg="white"
        ).pack(side="left", padx=25)
        self.in_value = tk.Label(
            in_box, text="0", font=("Consolas", 64, "bold"), bg="#5a5a5a", fg="white"
        )
        self.in_value.pack(side="right", padx=25)

        # OUT panel
        out_box = tk.Frame(sidebar, bg="#8b1a1a", height=210)
        out_box.pack(side="top", fill="x")
        out_box.pack_propagate(False)
        tk.Label(
            out_box, text="OUT", font=("Arial", 34, "bold"), bg="#8b1a1a", fg="white"
        ).pack(side="left", padx=25)
        self.out_value = tk.Label(
            out_box, text="0", font=("Consolas", 64, "bold"), bg="#8b1a1a", fg="white"
        )
        self.out_value.pack(side="right", padx=25)

        # Status / IP
        info = tk.Frame(sidebar, bg="#d9d9d9")
        info.pack(side="top", fill="x", padx=25, pady=40)
        self.status_label = tk.Label(
            info,
            text="Status: Stopped",
            font=("Segoe UI", 16),
            bg="#d9d9d9",
            anchor="w",
        )
        self.status_label.pack(fill="x", pady=8)
        tk.Label(
            info,
            text=f"IP Address: {self.ip_address}",
            font=("Segoe UI", 16),
            bg="#d9d9d9",
            anchor="w",
        ).pack(fill="x", pady=8)

        # Controls
        controls = tk.Frame(sidebar, bg="#d9d9d9")
        controls.pack(side="bottom", fill="x", padx=25, pady=25)
        self.start_btn = tk.Button(controls, text="Start", width=10, command=self.start)
        self.start_btn.pack(side="left", padx=5)
        self.stop_btn = tk.Button(controls, text="Stop", width=10, command=self.stop)
        self.stop_btn.pack(side="left", padx=5)

    # ------------------------------------------------------------ counter
    def _init_counter(self, sample_frame):
        height, width = sample_frame.shape[:2]
        line_y = int(height * LINE_HEIGHT_RATIO)
        line_points = [(0, line_y), (width, line_y)]

        self.counter = solutions.ObjectCounter(
            show=False,
            region=line_points,
            show_in=False,
            show_out=False,
            model=self.model_path,
            line_width=2,
            device=0,
            classes=[0],  # person class only
            conf=CONF_THRES,
            iou=IOU_THRES,
            show_conf=True,
            show_labels=True,
            verbose=False,
        )

    # -------------------------------------------------------------- start
    def start(self):
        if self.running:
            return
        self.running = True
        self.status_label.config(text="Status: Running")
        self.thread = threading.Thread(target=self._video_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.status_label.config(text="Status: Stopped")

    # ---------------------------------------------------------- main loop
    def _video_loop(self):
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            self.root.after(
                0, lambda: self.status_label.config(text="Status: Cannot open source")
            )
            self.running = False
            return

        success, frame = cap.read()
        if not success:
            self.root.after(
                0, lambda: self.status_label.config(text="Status: Empty video")
            )
            cap.release()
            self.running = False
            return

        if self.counter is None:
            self._init_counter(frame)

        while self.running:
            success, frame = cap.read()
            if not success:
                break

            result = self.counter(frame)
            person_in = result.classwise_count.get("person", {}).get("OUT", 0)
            person_out = result.classwise_count.get("person", {}).get("IN", 0)

            rgb = cv2.cvtColor(result.plot_im, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            # fit to the left panel while keeping aspect ratio
            panel_w = max(self.video_label.winfo_width(), 640)
            panel_h = max(self.video_label.winfo_height(), 480)
            img.thumbnail((panel_w, panel_h))
            imgtk = ImageTk.PhotoImage(image=img)

            self.root.after(0, self._update_ui, imgtk, person_in, person_out)
            time.sleep(0.01)

        cap.release()
        self.running = False
        self.root.after(0, lambda: self.status_label.config(text="Status: Stopped"))

    def _update_ui(self, imgtk, person_in, person_out):
        self.video_label.imgtk = imgtk  # keep a reference, avoid GC
        self.video_label.configure(image=imgtk)
        self.in_value.config(text=str(person_in))
        self.out_value.config(text=str(person_out))


def main():
    root = tk.Tk()
    root.geometry(WINDOW_SIZE)
    app = PeopleCountingApp(root, VIDEO_SOURCE, MODEL_PATH)

    def on_close():
        app.stop()
        root.after(200, root.destroy)

    root.protocol("WM_DELETE_WINDOW", on_close)
    app.start()  # auto-start; remove this line if you prefer pressing "Start"
    root.mainloop()


if __name__ == "__main__":
    main()
