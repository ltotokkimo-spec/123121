import sys
import time
import threading
import numpy as np
import cv2
from PIL import Image, ImageQt
import pyautogui

from PyQt5.QtCore import Qt, QRect, pyqtSignal, QObject, QSize
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QLineEdit,
    QVBoxLayout, QHBoxLayout, QWidget, QMessageBox, QGroupBox, QRubberBand
)

# ---------- 螢幕選取小視窗（用來畫框選） ----------
class ScreenSelector(QWidget):
    region_selected = pyqtSignal(QRect, QImage)  # emit rect and QImage

    def __init__(self, screenshot_qimage):
        super().__init__(None, Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.screenshot_qimage = screenshot_qimage
        self.initUI()

    def initUI(self):
        self.setWindowState(Qt.WindowFullScreen)
        self.screen_label = QLabel(self)
        pix = QPixmap.fromImage(self.screenshot_qimage)
        self.screen_label.setPixmap(pix)
        self.screen_label.setGeometry(0, 0, pix.width(), pix.height())
        self.rubber = QRubberBand(QRubberBand.Rectangle, self)
        self.origin = None
        self.show()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.origin = event.pos()
            self.rubber.setGeometry(QRect(self.origin, QSize()))
            self.rubber.show()

    def mouseMoveEvent(self, event):
        if self.origin:
            self.rubber.setGeometry(QRect(self.origin, event.pos()).normalized())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.origin:
            rect = self.rubber.geometry()
            self.rubber.hide()
            # crop qimage
            cropped = self.screenshot_qimage.copy(rect)
            self.region_selected.emit(rect, cropped)
            self.close()

# ---------- 後台比對 Thread 的 signals ----------
class WorkerSignals(QObject):
    update_similarity = pyqtSignal(float)
    update_status = pyqtSignal(str)

# ---------- 比對與點擊的工作 Thread ----------
class MatchWorker(threading.Thread):
    def __init__(self, get_search_image_func, template_img_cv, threshold,
                 click_action, signals: WorkerSignals, stop_event):
        super().__init__()
        self.get_search_image = get_search_image_func  # function returns cv image (BGR)
        self.template = template_img_cv  # template in BGR or Gray
        self.threshold = threshold / 100.0
        self.click_action = click_action
        self.signals = signals
        self.stop_event = stop_event

    def run(self):
        self.signals.update_status.emit("Running")
        template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY) if len(self.template.shape) == 3 else self.template
        h0, w0 = template_gray.shape[:2]
        # scale pyramid settings
        scales = np.linspace(0.6, 1.4, 17)  # try different template scales
        while not self.stop_event.is_set():
            search_bgr = self.get_search_image()
            if search_bgr is None:
                self.signals.update_status.emit("No search area available")
                break
            search_gray = cv2.cvtColor(search_bgr, cv2.COLOR_BGR2GRAY)
            best_val = 0.0
            best_loc = None
            best_wh = (w0, h0)
            # Try multiple scales of the template
            for s in scales:
                tw = max(2, int(w0 * s))
                th = max(2, int(h0 * s))
                if tw >= search_gray.shape[1] or th >= search_gray.shape[0]:
                    continue
                tmpl_resized = cv2.resize(template_gray, (tw, th), interpolation=cv2.INTER_AREA)
                try:
                    res = cv2.matchTemplate(search_gray, tmpl_resized, cv2.TM_CCOEFF_NORMED)
                except Exception:
                    continue
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                if max_val > best_val:
                    best_val = max_val
                    best_loc = max_loc
                    best_wh = (tw, th)
            # send similarity percent
            sim_percent = float(best_val * 100.0)
            self.signals.update_similarity.emit(sim_percent)
            if best_val >= self.threshold and best_loc is not None:
                # compute click coordinates relative to full screen
                x_in_search = best_loc[0] + best_wh[0] // 2
                y_in_search = best_loc[1] + best_wh[1] // 2
                # call click action (should handle mapping to absolute screen coords)
                self.click_action(x_in_search, y_in_search)
                self.signals.update_status.emit(f"Matched ({sim_percent:.1f}%), clicked.")
                # after a successful click, wait a short cooldown
                for _ in range(10):
                    if self.stop_event.is_set(): break
                    time.sleep(0.1)
            else:
                self.signals.update_status.emit(f"Searching... {sim_percent:.1f}%")
            # polling delay
            for _ in range(8):
                if self.stop_event.is_set(): break
                time.sleep(0.05)
        self.signals.update_status.emit("Stopped")

# ---------- 主視窗 ----------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Screen Template Auto-Clicker")
        self.template_qimage = None
        self.search_qimage = None
        self.template_cv = None  # BGR numpy
        self.search_region_rect = None  # QRect in global screen coords
        self.search_cv_full = None
        self.worker = None
        self.stop_event = threading.Event()
        self.worker_signals = WorkerSignals()
        self.worker_signals.update_similarity.connect(self.on_update_similarity)
        self.worker_signals.update_status.connect(self.on_update_status)
        self.initUI()

    def initUI(self):
        w = QWidget()
        v = QVBoxLayout()

        # buttons
        hbtn = QHBoxLayout()
        self.btn_select_template = QPushButton("選擇範本 (Template)")
        self.btn_select_template.clicked.connect(self.select_template)
        self.btn_select_search = QPushButton("選擇搜尋區域 (Search Area)")
        self.btn_select_search.clicked.connect(self.select_search_area)
        hbtn.addWidget(self.btn_select_template)
        hbtn.addWidget(self.btn_select_search)
        v.addLayout(hbtn)

        # threshold input
        group = QGroupBox("比對設定")
        gh = QHBoxLayout()
        gh.addWidget(QLabel("相似度閾值 (%)："))
        self.thresh_input = QLineEdit("70")
        self.thresh_input.setMaximumWidth(80)
        gh.addWidget(self.thresh_input)
        group.setLayout(gh)
        v.addWidget(group)

        # start/stop
        h2 = QHBoxLayout()
        self.btn_start = QPushButton("啟動")
        self.btn_start.clicked.connect(self.start_worker)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.clicked.connect(self.stop_worker)
        self.btn_stop.setEnabled(False)
        h2.addWidget(self.btn_start)
        h2.addWidget(self.btn_stop)
        v.addLayout(h2)

        # previews and status
        ph = QHBoxLayout()
        self.lbl_template = QLabel("Template preview")
        self.lbl_template.setFixedSize(200, 120)
        self.lbl_search = QLabel("Search preview")
        self.lbl_search.setFixedSize(200, 120)
        ph.addWidget(self.lbl_template)
        ph.addWidget(self.lbl_search)
        v.addLayout(ph)

        self.status_label = QLabel("Status: Idle")
        self.sim_label = QLabel("Similarity: N/A")
        v.addWidget(self.status_label)
        v.addWidget(self.sim_label)

        w.setLayout(v)
        self.setCentralWidget(w)
        self.resize(460, 360)
        self.show()

    # ---------- helper: take full-screen screenshot as QImage ----------
    def grab_fullscreen_qimage(self):
        img = pyautogui.screenshot()  # PIL image
        qimg = ImageQt.ImageQt(img.convert("RGB"))
        return qimg

    def select_template(self):
        # hide main window, grab full screen, then show selector
        self.hide()
        qimg = self.grab_fullscreen_qimage()
        selector = ScreenSelector(qimg)
        selector.region_selected.connect(self.on_template_selected)
        selector.show()

    def on_template_selected(self, rect: QRect, qimg: QImage):
        self.template_qimage = qimg
        # convert to cv BGR
        w = qimg.width(); h = qimg.height()
        ptr = qimg.bits()
        ptr.setsize(qimg.byteCount())
        arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4))
        # RGBA -> BGR
        cv_img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        self.template_cv = cv_img
        # preview
        pix = QPixmap.fromImage(qimg).scaled(self.lbl_template.size(), Qt.KeepAspectRatio)
        self.lbl_template.setPixmap(pix)
        self.show()
        self.status_label.setText("Status: Template selected")

    def select_search_area(self):
        self.hide()
        qimg = self.grab_fullscreen_qimage()
        selector = ScreenSelector(qimg)
        selector.region_selected.connect(self.on_search_selected)
        selector.show()

    def on_search_selected(self, rect: QRect, qimg: QImage):
        # rect is relative to full screen; we need to store its global position
        self.search_qimage = qimg
        self.search_region_rect = rect  # QRect in screen coordinates
        # convert to cv BGR
        w = qimg.width(); h = qimg.height()
        ptr = qimg.bits()
        ptr.setsize(qimg.byteCount())
        arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4))
        search_cv = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        self.search_cv_full = search_cv
        pix = QPixmap.fromImage(qimg).scaled(self.lbl_search.size(), Qt.KeepAspectRatio)
        self.lbl_search.setPixmap(pix)
        self.show()
        self.status_label.setText("Status: Search area selected")

    def get_search_image_for_worker(self):
        # capture the search region fresh each loop (return BGR numpy)
        if self.search_region_rect is None:
            return None
        x = self.search_region_rect.x()
        y = self.search_region_rect.y()
        w = self.search_region_rect.width()
        h = self.search_region_rect.height()
        # use pyautogui screenshot region
        img = pyautogui.screenshot(region=(x, y, w, h))  # PIL
        arr = np.array(img.convert("RGB"))
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return bgr

    def click_action_factory(self):
        # returns a function that maps (x_in_search, y_in_search) -> absolute click
        def click_at(x_in_search, y_in_search):
            if self.search_region_rect is None:
                return
            abs_x = self.search_region_rect.x() + int(x_in_search)
            abs_y = self.search_region_rect.y() + int(y_in_search)
            # perform click - you can change to drag or double click as needed
            try:
                pyautogui.click(x=abs_x, y=abs_y)
            except Exception as e:
                print("Click failed:", e)
        return click_at

    def start_worker(self):
        if self.template_cv is None or self.search_region_rect is None:
            QMessageBox.warning(self, "Missing", "請先選擇範本與搜尋區域。")
            return
        try:
            threshold = float(self.thresh_input.text())
            if threshold < 1 or threshold > 100:
                raise ValueError()
        except:
            QMessageBox.warning(self, "Invalid", "請輸入 1~100 的相似度（%）。")
            return
        # disable/enable controls
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_select_template.setEnabled(False)
        self.btn_select_search.setEnabled(False)
        self.stop_event.clear()
        worker = MatchWorker(
            get_search_image_func=self.get_search_image_for_worker,
            template_img_cv=self.template_cv,
            threshold=threshold,
            click_action=self.click_action_factory(),
            signals=self.worker_signals,
            stop_event=self.stop_event
        )
        self.worker = worker
        worker.daemon = True
        worker.start()

    def stop_worker(self):
        self.stop_event.set()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_select_template.setEnabled(True)
        self.btn_select_search.setEnabled(True)
        self.status_label.setText("Status: Stopping...")

    # signals handlers
    def on_update_similarity(self, val):
        self.sim_label.setText(f"Similarity: {val:.1f}%")

    def on_update_status(self, txt):
        self.status_label.setText("Status: " + txt)

# ---------- 執行 ----------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mw = MainWindow()
    sys.exit(app.exec_())