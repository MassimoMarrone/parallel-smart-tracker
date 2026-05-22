import os, glob, re, time, sys, concurrent.futures
import numpy as np
import tifffile
import scipy.ndimage as ndi
import cv2
from skimage import exposure, filters, morphology, segmentation, measure
from skimage.util import img_as_ubyte

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QFrame,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QDoubleSpinBox, QSpinBox,
    QProgressBar, QFileDialog, QMessageBox, QScrollArea, QStackedWidget,
    QFormLayout, QGroupBox, QSizePolicy, QComboBox, QSlider)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QPixmap, QImage, QFont, QColor, QPainter, QPen, QBrush
from PySide6.QtCharts import QChart, QBarSeries, QBarSet, QChartView, QBarCategoryAxis, QValueAxis
# qt_material removed — using custom QSS below


# ---------------------------------------------------------------------------
# Algorithm code (tested, unchanged)
# ---------------------------------------------------------------------------

def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def load_tif_files(folder):
    patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    paths = []
    for p in patterns:
        paths.extend(glob.glob(os.path.join(folder, p)))
    paths = list(dict.fromkeys(paths))
    paths.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    return paths

def read_frame(path):
    img = tifffile.imread(path)
    if img.ndim == 3:
        img = img[0] if img.shape[0] < img.shape[1] else img[..., 0]
    img = img.astype(np.float64)
    lo, hi = img.min(), img.max()
    if hi > lo:
        img = (img - lo) / (hi - lo)
    return img

def preprocess(frame, clip_limit=0.10, smooth_sigma=3.0):
    filtered = ndi.median_filter(frame, size=3)
    selem = morphology.disk(25)
    tophat = morphology.white_tophat(filtered, footprint=selem)
    clahe = exposure.equalize_adapthist(tophat, clip_limit=clip_limit)
    smoothed = ndi.gaussian_filter(clahe, sigma=smooth_sigma)
    return smoothed

def get_initial_mask(preprocessed, min_cell_size=700, max_hole_size=1500):
    thresh = filters.threshold_local(preprocessed, block_size=31, method="gaussian")
    binary = preprocessed > thresh
    selem = morphology.disk(2)
    opened = morphology.opening(binary, footprint=selem)
    cleaned = morphology.remove_small_objects(opened, min_size=min_cell_size)
    filled = morphology.remove_small_holes(cleaned, area_threshold=max_hole_size)
    return filled.astype(bool)

def combine_masks(prev_mask, new_preprocessed, min_cell_size=700, max_hole_size=1500):
    new_mask = get_initial_mask(new_preprocessed, min_cell_size=min_cell_size, max_hole_size=max_hole_size)
    return np.logical_or(prev_mask, new_mask)

def run_chan_vese(preprocessed, init_mask, iterations=100, smooth_factor=3.0, contraction_bias=0.10):
    smoothing = max(3, int(round(smooth_factor * 2)))
    result = segmentation.morphological_chan_vese(
        preprocessed, num_iter=iterations,
        init_level_set=init_mask.astype(float),
        smoothing=smoothing, lambda1=1.0, lambda2=1.0 + contraction_bias,
    )
    return result.astype(bool)

def run_gac(preprocessed, init_mask, iterations=100, smoothing=2,
            alpha=300.0, sigma=3.0, threshold=0.55, balloon=-0.5):
    gimage = segmentation.inverse_gaussian_gradient(preprocessed, alpha=alpha, sigma=sigma)
    result = segmentation.morphological_geodesic_active_contour(
        gimage, num_iter=iterations, init_level_set=init_mask.astype(float),
        smoothing=smoothing, threshold=threshold, balloon=balloon,
    )
    return result.astype(bool)

def run_lbf(preprocessed, init_mask, iterations=100, sigma=8.0, mu=0.2, nu=0.05, dt=0.05):
    img = preprocessed.astype(np.float64)
    dist_in = ndi.distance_transform_edt(init_mask)
    dist_out = ndi.distance_transform_edt(~init_mask)
    phi = (dist_in - dist_out).astype(np.float64)
    epsilon = 1.5
    img2 = img ** 2
    for _ in range(iterations):
        H = 0.5 * (1.0 + (2.0 / np.pi) * np.arctan(phi / epsilon))
        delta = (epsilon / np.pi) / (epsilon ** 2 + phi ** 2)
        one_minus_H = 1.0 - H
        KH = ndi.gaussian_filter(H, sigma=sigma)
        K1mH = ndi.gaussian_filter(one_minus_H, sigma=sigma)
        KIH = ndi.gaussian_filter(img * H, sigma=sigma)
        KI1mH = ndi.gaussian_filter(img * one_minus_H, sigma=sigma)
        KI2H = ndi.gaussian_filter(img2 * H, sigma=sigma)
        KI21mH = ndi.gaussian_filter(img2 * one_minus_H, sigma=sigma)
        f1 = KIH / (KH + 1e-10)
        f2 = KI1mH / (K1mH + 1e-10)
        e1 = KI2H - 2.0 * f1 * KIH + f1 ** 2 * KH
        e2 = KI21mH - 2.0 * f2 * KI1mH + f2 ** 2 * K1mH
        gy, gx = np.gradient(phi)
        gmag = np.sqrt(gx ** 2 + gy ** 2 + 1e-10)
        kappa = (np.gradient(gx / gmag, axis=1) + np.gradient(gy / gmag, axis=0))
        dist_reg = ndi.laplace(phi) - kappa
        phi = phi + dt * (delta * (-e1 + e2 + nu * kappa) + mu * dist_reg)
    return (phi > 0).astype(bool)

def clean_mask(mask, min_cell_size=700, max_hole_size=1500):
    selem = morphology.disk(2)
    opened = morphology.opening(mask, footprint=selem)
    cleaned = morphology.remove_small_objects(opened, min_size=min_cell_size)
    filled = morphology.remove_small_holes(cleaned, area_threshold=max_hole_size)
    smoothed = morphology.binary_closing(filled, morphology.disk(3))
    # Remove blobs with very low circularity (4π·area/perimeter²)
    # Real cells are roughly round (circularity > 0.3); spuria artifacts are very irregular
    labeled = measure.label(smoothed)
    result = np.zeros_like(smoothed, dtype=bool)
    for region in measure.regionprops(labeled):
        if region.perimeter > 0:
            circularity = (4 * np.pi * region.area) / (region.perimeter ** 2)
            if circularity >= 0.3:
                result[labeled == region.label] = True
    return result

def overlay_boundaries(gray_frame, mask, boundary_color=(255, 255, 0)):
    gray_u8 = img_as_ubyte(np.clip(gray_frame, 0, 1))
    rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1)
    bounds = segmentation.find_boundaries(mask, mode="outer")
    rgb[bounds] = boundary_color
    return rgb

def count_cells(mask):
    return measure.label(mask).max()

def save_video(rgb_frames, output_path, fps=5):
    if not rgb_frames:
        raise ValueError("No frames to save.")
    h, w = rgb_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in rgb_frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

def _preprocess_one(args):
    path, clip_limit, smooth_sigma = args
    frame = read_frame(path)
    preprocess(frame, clip_limit=clip_limit, smooth_sigma=smooth_sigma)
    return path

def _load_and_preprocess(args):
    """Used by parallel TrackingWorker — returns (raw_frame, preprocessed)."""
    path, clip_limit, smooth_sigma = args
    frame = read_frame(path)
    prep = preprocess(frame, clip_limit=clip_limit, smooth_sigma=smooth_sigma)
    return frame, prep


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def apply_dark_chart_style(chart):
    """Apply consistent dark theme to a QChart object."""
    chart.setBackgroundBrush(QBrush(QColor("#1a1d2e")))
    chart.setBackgroundRoundness(8)
    chart.setPlotAreaBackgroundBrush(QBrush(QColor("#13162a")))
    chart.setPlotAreaBackgroundVisible(True)
    chart.setTitleBrush(QBrush(QColor("#dde4f0")))
    chart.legend().setColor(QColor("#8892a4"))
    chart.legend().setLabelColor(QColor("#8892a4"))


def style_axis(axis):
    axis.setLabelsBrush(QBrush(QColor("#64748b")))
    axis.setTitleBrush(QBrush(QColor("#8892a4")))
    axis.setLinePenColor(QColor("#252840"))
    axis.setGridLinePen(QPen(QColor("#1e2235"), 1))
    axis.setMinorGridLinePen(QPen(QColor("#191c2f"), 1))


def ndarray_to_pixmap(rgb: np.ndarray) -> QPixmap:
    h, w, ch = rgb.shape
    img = QImage(rgb.tobytes(), w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img)


def scan_local_datasets(base_dir: str) -> list:
    """Scan base_dir for subfolders that contain at least one .tif file."""
    results = []
    if not os.path.isdir(base_dir):
        return results
    try:
        entries = sorted(os.scandir(base_dir), key=lambda e: e.name.lower())
    except PermissionError:
        return results
    for entry in entries:
        if entry.is_dir():
            tif_files = load_tif_files(entry.path)
            if tif_files:
                results.append((entry.name, entry.path, len(tif_files)))
    return results


# ---------------------------------------------------------------------------
# Worker threads
# ---------------------------------------------------------------------------

class TrackingWorker(QThread):
    frame_ready = Signal(int, object, int)
    status = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, paths, params, parallel=False):
        super().__init__()
        self.paths = paths
        self.params = params
        self.parallel = parallel
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            if self.parallel:
                self._run_parallel()
            else:
                self._run_serial()
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()

    def _run_serial(self):
        paths = self.paths
        params = self.params
        min_cell_size = params.get("min_cell_size", 700)
        max_hole_size = params.get("max_hole_size", 1500)
        smooth_factor = params.get("smooth_factor", 3.0)
        contraction_bias = params.get("contraction_bias", 0.10)
        clip_limit = params.get("clip_limit", 0.10)

        prev_mask = None

        for i, path in enumerate(paths):
            if self._stop:
                break

            frame = read_frame(path)
            prep = preprocess(frame, clip_limit=clip_limit, smooth_sigma=smooth_factor)

            if i == 0:
                init_mask = get_initial_mask(prep, min_cell_size=min_cell_size, max_hole_size=max_hole_size)
                mask = run_chan_vese(prep, init_mask, iterations=100,
                                    smooth_factor=smooth_factor,
                                    contraction_bias=contraction_bias)
                mask = clean_mask(mask, min_cell_size=min_cell_size, max_hole_size=max_hole_size)
            else:
                combined = combine_masks(prev_mask, prep, min_cell_size=min_cell_size, max_hole_size=max_hole_size)
                mask = run_chan_vese(prep, combined, iterations=50,
                                    smooth_factor=smooth_factor,
                                    contraction_bias=contraction_bias)
                mask = clean_mask(mask, min_cell_size=min_cell_size, max_hole_size=max_hole_size)

            prev_mask = mask
            rgb = overlay_boundaries(frame, mask)
            n_cells = count_cells(mask)
            self.frame_ready.emit(i, rgb, n_cells)

    def _run_parallel(self):
        paths = self.paths
        params = self.params
        min_cell_size = params.get("min_cell_size", 700)
        max_hole_size = params.get("max_hole_size", 1500)
        smooth_factor = params.get("smooth_factor", 3.0)
        contraction_bias = params.get("contraction_bias", 0.10)
        clip_limit = params.get("clip_limit", 0.10)

        n = len(paths)
        self.status.emit(f"Preprocessing parallelo di {n} frame…")

        args_list = [(p, clip_limit, smooth_factor) for p in paths]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            loaded = list(executor.map(_load_and_preprocess, args_list))

        if self._stop:
            return

        self.status.emit("Level-set Chan-Vese in corso…")
        prev_mask = None

        for i, (frame, prep) in enumerate(loaded):
            if self._stop:
                break

            if i == 0:
                init_mask = get_initial_mask(prep, min_cell_size=min_cell_size, max_hole_size=max_hole_size)
                mask = run_chan_vese(prep, init_mask, iterations=100,
                                    smooth_factor=smooth_factor,
                                    contraction_bias=contraction_bias)
                mask = clean_mask(mask, min_cell_size=min_cell_size, max_hole_size=max_hole_size)
            else:
                combined = combine_masks(prev_mask, prep, min_cell_size=min_cell_size, max_hole_size=max_hole_size)
                mask = run_chan_vese(prep, combined, iterations=50,
                                    smooth_factor=smooth_factor,
                                    contraction_bias=contraction_bias)
                mask = clean_mask(mask, min_cell_size=min_cell_size, max_hole_size=max_hole_size)

            prev_mask = mask
            rgb = overlay_boundaries(frame, mask)
            n_cells = count_cells(mask)
            self.frame_ready.emit(i, rgb, n_cells)


class ComparisonWorker(QThread):
    frame_ready = Signal(str, int, object, int)  # algo, frame_idx, rgb, n_cells
    algo_done = Signal(str, float)               # algo, total_time
    finished = Signal()
    error = Signal(str)

    def __init__(self, paths, params):
        super().__init__()
        self.paths = paths
        self.params = params
        self._stop = False

    def stop(self):
        self._stop = True

    def _run_algo(self, algo_name, iter_first, iter_rest):
        params = self.params
        min_cell_size = params.get("min_cell_size", 700)
        max_hole_size = params.get("max_hole_size", 1500)
        smooth_factor = params.get("smooth_factor", 3.0)
        clip_limit = params.get("clip_limit", 0.10)

        prev_mask = None
        t_start = time.time()

        for i, path in enumerate(self.paths):
            if self._stop:
                break
            frame = read_frame(path)
            prep = preprocess(frame, clip_limit=clip_limit, smooth_sigma=smooth_factor)

            if i == 0:
                seed = get_initial_mask(prep, min_cell_size=min_cell_size, max_hole_size=max_hole_size)
                iters = iter_first
            else:
                seed = combine_masks(prev_mask, prep, min_cell_size=min_cell_size, max_hole_size=max_hole_size)
                iters = iter_rest

            if algo_name == "GAC":
                mask = run_gac(prep, seed, iterations=iters,
                               smoothing=params.get("gac_smoothing", 2),
                               alpha=params.get("gac_alpha", 300.0),
                               sigma=params.get("gac_sigma", 3.0),
                               threshold=params.get("gac_threshold", 0.55),
                               balloon=params.get("gac_balloon", -0.5))
            else:
                mask = run_lbf(prep, seed, iterations=iters,
                               sigma=params.get("lbf_sigma", 8.0),
                               mu=params.get("lbf_mu", 0.2),
                               nu=params.get("lbf_nu", 0.05),
                               dt=params.get("lbf_dt", 0.05))

            mask = clean_mask(mask, min_cell_size=min_cell_size, max_hole_size=max_hole_size)
            prev_mask = mask

            rgb = overlay_boundaries(frame, mask)
            self.frame_ready.emit(algo_name, i, rgb, count_cells(mask))

        self.algo_done.emit(algo_name, time.time() - t_start)

    def run(self):
        try:
            self._run_algo("GAC", iter_first=100, iter_rest=50)
            if not self._stop:
                self._run_algo("LBF", iter_first=30, iter_rest=20)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()


class SingleAlgoWorker(QThread):
    """Runs one comparison algorithm independently (for parallel comparison mode)."""
    frame_ready = Signal(str, int, object, int)
    algo_done = Signal(str, float)
    finished = Signal()
    error = Signal(str)

    def __init__(self, algo, paths, params, iter_first, iter_rest, parallel=False):
        super().__init__()
        self.algo = algo
        self.paths = paths
        self.params = params
        self.iter_first = iter_first
        self.iter_rest = iter_rest
        self.parallel = parallel
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            if self.parallel:
                self._run_parallel()
            else:
                self._run_serial()
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()

    def _run_serial(self):
        algo_name = self.algo
        params = self.params
        min_cell_size = params.get("min_cell_size", 700)
        max_hole_size = params.get("max_hole_size", 1500)
        smooth_factor = params.get("smooth_factor", 3.0)
        clip_limit = params.get("clip_limit", 0.10)

        prev_mask = None
        t_start = time.time()

        for i, path in enumerate(self.paths):
            if self._stop:
                break
            frame = read_frame(path)
            prep = preprocess(frame, clip_limit=clip_limit, smooth_sigma=smooth_factor)

            if i == 0:
                seed = get_initial_mask(prep, min_cell_size=min_cell_size, max_hole_size=max_hole_size)
                iters = self.iter_first
            else:
                seed = combine_masks(prev_mask, prep, min_cell_size=min_cell_size, max_hole_size=max_hole_size)
                iters = self.iter_rest

            mask = self._apply_algo(algo_name, prep, seed, iters)
            mask = clean_mask(mask, min_cell_size=min_cell_size, max_hole_size=max_hole_size)
            prev_mask = mask

            rgb = overlay_boundaries(frame, mask)
            self.frame_ready.emit(algo_name, i, rgb, count_cells(mask))

        self.algo_done.emit(algo_name, time.time() - t_start)

    def _run_parallel(self):
        algo_name = self.algo
        params = self.params
        min_cell_size = params.get("min_cell_size", 700)
        max_hole_size = params.get("max_hole_size", 1500)
        smooth_factor = params.get("smooth_factor", 3.0)
        clip_limit = params.get("clip_limit", 0.10)

        args_list = [(p, clip_limit, smooth_factor) for p in self.paths]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            loaded = list(executor.map(_load_and_preprocess, args_list))

        if self._stop:
            return

        prev_mask = None
        t_start = time.time()

        for i, (frame, prep) in enumerate(loaded):
            if self._stop:
                break

            if i == 0:
                seed = get_initial_mask(prep, min_cell_size=min_cell_size, max_hole_size=max_hole_size)
                iters = self.iter_first
            else:
                seed = combine_masks(prev_mask, prep, min_cell_size=min_cell_size, max_hole_size=max_hole_size)
                iters = self.iter_rest

            mask = self._apply_algo(algo_name, prep, seed, iters)
            mask = clean_mask(mask, min_cell_size=min_cell_size, max_hole_size=max_hole_size)
            prev_mask = mask

            rgb = overlay_boundaries(frame, mask)
            self.frame_ready.emit(algo_name, i, rgb, count_cells(mask))

        self.algo_done.emit(algo_name, time.time() - t_start)

    def _apply_algo(self, algo_name, prep, seed, iters):
        params = self.params
        if algo_name == "GAC":
            return run_gac(prep, seed, iterations=iters,
                           smoothing=params.get("gac_smoothing", 2),
                           alpha=params.get("gac_alpha", 300.0),
                           sigma=params.get("gac_sigma", 3.0),
                           threshold=params.get("gac_threshold", 0.55),
                           balloon=params.get("gac_balloon", -0.5))
        else:
            return run_lbf(prep, seed, iterations=iters,
                           sigma=params.get("lbf_sigma", 8.0),
                           mu=params.get("lbf_mu", 0.2),
                           nu=params.get("lbf_nu", 0.05),
                           dt=params.get("lbf_dt", 0.05))


class SpeedupWorker(QThread):
    result_ready = Signal(float, float, float)
    finished = Signal()
    error = Signal(str)

    def __init__(self, paths, clip_limit, smooth_sigma):
        super().__init__()
        self.paths = paths
        self.clip_limit = clip_limit
        self.smooth_sigma = smooth_sigma

    def run(self):
        try:
            paths = self.paths
            clip_limit = self.clip_limit
            smooth_sigma = self.smooth_sigma

            # Serial
            t0 = time.time()
            for path in paths:
                frame = read_frame(path)
                preprocess(frame, clip_limit=clip_limit, smooth_sigma=smooth_sigma)
            serial_time = time.time() - t0

            # Parallel
            args_list = [(p, clip_limit, smooth_sigma) for p in paths]
            t0 = time.time()
            with concurrent.futures.ProcessPoolExecutor() as executor:
                list(executor.map(_preprocess_one, args_list))
            parallel_time = time.time() - t0

            speedup = serial_time / parallel_time if parallel_time > 0 else float("inf")
            self.result_ready.emit(serial_time, parallel_time, speedup)

        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()


# ---------------------------------------------------------------------------
# Page 0: TrackingPage
# ---------------------------------------------------------------------------

class TrackingPage(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.worker = None
        self.tracked_frames = []
        self._total_frames = 0
        self._player_idx = 0
        self._player_timer = QTimer()
        self._player_timer.setInterval(200)  # 5 fps
        self._player_timer.timeout.connect(self._player_advance)
        self._local_datasets = []   # list of (name, path, n_files)
        self._build_ui()
        self._refresh_local_datasets()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---- Left panel ----
        left_container = QWidget()
        left_container.setStyleSheet("background-color: #0f1120;")
        left = QVBoxLayout(left_container)
        left.setContentsMargins(20, 16, 20, 16)
        left.setSpacing(10)

        accent = QFrame()
        accent.setFixedHeight(3)
        accent.setStyleSheet("background-color: #80cbc4; border-radius: 2px;")
        left.addWidget(accent)

        self.title_label = QLabel("TRACKING IN TEMPO REALE")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        f = QFont()
        f.setBold(True)
        f.setPointSize(14)
        self.title_label.setFont(f)
        self.title_label.setStyleSheet("color: #dde4f0; background: transparent; padding: 8px 0 4px 0;")
        left.addWidget(self.title_label)

        self.image_label = QLabel("Carica una cartella e avvia il tracking.")
        self.image_label.setMinimumSize(560, 420)
        self.image_label.setStyleSheet(
            "background-color: #141828; border: 1px solid #242840;"
            " border-radius: 8px; color: #4a5568; background: #141828;"
        )
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left.addWidget(self.image_label, stretch=1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        left.addWidget(self.progress_bar)

        # --- Video player controls (hidden until tracking completes) ---
        self.player_widget = QWidget()
        self.player_widget.setStyleSheet("background: transparent;")
        player_layout = QVBoxLayout(self.player_widget)
        player_layout.setContentsMargins(0, 4, 0, 0)
        player_layout.setSpacing(4)

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self._on_slider_moved)
        player_layout.addWidget(self.frame_slider)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self.btn_prev = QPushButton("⏮")
        self.btn_prev.setObjectName("btn_icon")
        self.btn_prev.setFixedWidth(44)
        self.btn_prev.setFixedHeight(34)
        self.btn_prev.clicked.connect(self._player_prev)
        btn_row.addWidget(self.btn_prev)

        self.btn_play_pause = QPushButton("▶")
        self.btn_play_pause.setObjectName("btn_icon")
        self.btn_play_pause.setFixedWidth(54)
        self.btn_play_pause.setFixedHeight(34)
        self.btn_play_pause.clicked.connect(self._player_toggle)
        btn_row.addWidget(self.btn_play_pause)

        self.btn_next = QPushButton("⏭")
        self.btn_next.setObjectName("btn_icon")
        self.btn_next.setFixedWidth(44)
        self.btn_next.setFixedHeight(34)
        self.btn_next.clicked.connect(self._player_next)
        btn_row.addWidget(self.btn_next)

        self.frame_counter_label = QLabel("0 / 0")
        self.frame_counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_counter_label.setStyleSheet("color: #80cbc4; font-size: 12px; background: transparent;")
        btn_row.addWidget(self.frame_counter_label, stretch=1)

        player_layout.addLayout(btn_row)
        self.player_widget.setVisible(False)
        left.addWidget(self.player_widget)

        self.status_label = QLabel("Pronto.")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #4a5568; font-size: 12px; background: transparent;")
        left.addWidget(self.status_label)

        root.addWidget(left_container, stretch=3)

        # ---- Right panel ----
        right_widget = QWidget()
        right_widget.setObjectName("right_panel")
        right_widget.setFixedWidth(280)
        right = QVBoxLayout(right_widget)
        right.setContentsMargins(20, 16, 20, 16)
        right.setSpacing(10)

        right.addStretch()  # push content toward vertical center

        # --- Dataset section (DATASET groupbox) ---
        local_group = QGroupBox("DATASET")
        local_layout = QVBoxLayout(local_group)
        local_layout.setSpacing(6)
        local_layout.setContentsMargins(10, 18, 10, 10)

        self.combo_datasets = QComboBox()
        self.combo_datasets.setPlaceholderText("(nessuna cartella trovata)")
        local_layout.addWidget(self.combo_datasets)

        local_btn_row = QHBoxLayout()
        local_btn_row.setSpacing(6)
        self.btn_load_local = QPushButton("✔ Carica")
        self.btn_load_local.setFixedHeight(30)
        self.btn_load_local.setEnabled(False)
        self.btn_load_local.clicked.connect(self._on_load_local)
        local_btn_row.addWidget(self.btn_load_local)

        btn_refresh = QPushButton("↺")
        btn_refresh.setObjectName("btn_icon")
        btn_refresh.setFixedWidth(32)
        btn_refresh.setFixedHeight(32)
        btn_refresh.setToolTip("Aggiorna lista")
        btn_refresh.clicked.connect(self._refresh_local_datasets)
        local_btn_row.addWidget(btn_refresh)
        local_layout.addLayout(local_btn_row)

        self.btn_load = QPushButton("📁 Sfoglia Cartella…")
        self.btn_load.setFixedHeight(34)
        self.btn_load.clicked.connect(self._on_load)
        local_layout.addWidget(self.btn_load)

        self.folder_label = QLabel("Nessuna cartella selezionata.")
        self.folder_label.setStyleSheet("color: #4a5568; font-size: 11px; background: transparent;")
        self.folder_label.setWordWrap(True)
        local_layout.addWidget(self.folder_label)

        right.addWidget(local_group)

        # --- Run row: Avvia | Stop | mode combo ---
        btn_row2 = QHBoxLayout()
        btn_row2.setSpacing(6)

        self.btn_start = QPushButton("▶  Avvia")
        self.btn_start.setObjectName("btn_primary")
        self.btn_start.setFixedHeight(40)
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self._on_start)
        btn_row2.addWidget(self.btn_start, stretch=1)

        self.btn_stop = QPushButton("■  Stop")
        self.btn_stop.setObjectName("btn_danger")
        self.btn_stop.setFixedHeight(40)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        btn_row2.addWidget(self.btn_stop, stretch=1)

        self.combo_mode = QComboBox()
        self.combo_mode.addItem("Seriale")
        self.combo_mode.addItem("Parallelo")
        self.combo_mode.setFixedHeight(40)
        self.combo_mode.setFixedWidth(90)
        btn_row2.addWidget(self.combo_mode)

        right.addLayout(btn_row2)

        # --- Collapsible parameters ---
        self._btn_params_toggle = QPushButton("⚙  Parametri")
        self._btn_params_toggle.setCheckable(True)
        self._btn_params_toggle.setFixedHeight(30)
        self._btn_params_toggle.setStyleSheet(
            "QPushButton { background: transparent; color: #4a5568; border: 1px solid #1c2035;"
            " border-radius: 8px; font-size: 11px; }"
            "QPushButton:hover { color: #80cbc4; border-color: #2a6b67; }"
            "QPushButton:checked { color: #80cbc4; border-color: #2a6b67; background: #0d2a28; }"
        )
        self._btn_params_toggle.clicked.connect(self._toggle_params)
        right.addWidget(self._btn_params_toggle)

        self._params_widget = QWidget()
        self._params_widget.setVisible(False)
        self._params_widget.setStyleSheet("background: transparent;")
        param_form = QFormLayout(self._params_widget)
        param_form.setSpacing(5)
        param_form.setContentsMargins(4, 4, 4, 4)
        param_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.sp_min_cell = QSpinBox()
        self.sp_min_cell.setRange(10, 10000)
        self.sp_min_cell.setValue(700)
        param_form.addRow("min_cell_size", self.sp_min_cell)

        self.sp_max_hole = QSpinBox()
        self.sp_max_hole.setRange(10, 10000)
        self.sp_max_hole.setValue(1500)
        param_form.addRow("max_hole_size", self.sp_max_hole)

        self.sp_smooth = QDoubleSpinBox()
        self.sp_smooth.setRange(0.1, 10.0)
        self.sp_smooth.setSingleStep(0.1)
        self.sp_smooth.setValue(3.0)
        param_form.addRow("smooth_factor", self.sp_smooth)

        self.sp_contraction = QDoubleSpinBox()
        self.sp_contraction.setRange(0.001, 1.0)
        self.sp_contraction.setSingleStep(0.01)
        self.sp_contraction.setDecimals(3)
        self.sp_contraction.setValue(0.100)
        param_form.addRow("contraction_bias", self.sp_contraction)

        self.sp_clip = QDoubleSpinBox()
        self.sp_clip.setRange(0.001, 1.0)
        self.sp_clip.setSingleStep(0.01)
        self.sp_clip.setDecimals(3)
        self.sp_clip.setValue(0.100)
        param_form.addRow("clip_limit", self.sp_clip)

        right.addWidget(self._params_widget)

        # --- Stats pills ---
        pills_row = QHBoxLayout()
        pills_row.setSpacing(6)

        def make_pill(value_text, key_text):
            pill = QWidget()
            pill.setObjectName("stat_pill")
            pill_layout = QVBoxLayout(pill)
            pill_layout.setContentsMargins(8, 10, 8, 10)
            pill_layout.setSpacing(2)
            val_lbl = QLabel(value_text)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val_f = QFont()
            val_f.setBold(True)
            val_f.setPointSize(18)
            val_lbl.setFont(val_f)
            val_lbl.setStyleSheet("color: #80cbc4; background: transparent;")
            key_lbl = QLabel(key_text)
            key_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            key_lbl.setStyleSheet("color: #4a5568; font-size: 10px; background: transparent;")
            pill_layout.addWidget(val_lbl)
            pill_layout.addWidget(key_lbl)
            return pill, val_lbl

        pill_total, self.lbl_total = make_pill("—", "TOTALI")
        pill_current, self.lbl_current = make_pill("—", "CORRENTE")
        pill_cells, self.lbl_cells = make_pill("—", "CELLULE")
        pills_row.addWidget(pill_total)
        pills_row.addWidget(pill_current)
        pills_row.addWidget(pill_cells)
        right.addLayout(pills_row)

        # --- Save button ---
        self.btn_save = QPushButton("💾 Salva Video")
        self.btn_save.setFixedHeight(36)
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._on_save)
        right.addWidget(self.btn_save)

        right.addStretch()
        root.addWidget(right_widget)

    def _toggle_params(self, checked):
        self._params_widget.setVisible(checked)
        self._btn_params_toggle.setText("✕  Chiudi" if checked else "⚙  Parametri")

    def get_params(self):
        return {
            "min_cell_size": self.sp_min_cell.value(),
            "max_hole_size": self.sp_max_hole.value(),
            "smooth_factor": self.sp_smooth.value(),
            "contraction_bias": self.sp_contraction.value(),
            "clip_limit": self.sp_clip.value(),
        }

    def _refresh_local_datasets(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self._local_datasets = scan_local_datasets(base_dir)
        self.combo_datasets.clear()
        if self._local_datasets:
            for name, path, n in self._local_datasets:
                self.combo_datasets.addItem(f"{name}  ({n} .tif)")
            self.btn_load_local.setEnabled(True)
        else:
            self.combo_datasets.setPlaceholderText("(nessuna cartella trovata)")
            self.btn_load_local.setEnabled(False)

    def _on_load_local(self):
        idx = self.combo_datasets.currentIndex()
        if idx < 0 or idx >= len(self._local_datasets):
            return
        _, folder_path, _ = self._local_datasets[idx]
        self._load_folder(folder_path)

    def _on_load(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleziona Cartella Immagini")
        if folder:
            self._load_folder(folder)

    def _load_folder(self, folder: str):
        paths = load_tif_files(folder)
        if not paths:
            QMessageBox.warning(self, "Nessun file", f"Nessun file .tif trovato in:\n{folder}")
            return
        self.state["paths"] = paths
        short = os.path.basename(folder)
        self.folder_label.setText(f"{short}  ({len(paths)} file .tif)")
        self.lbl_total.setText(str(len(paths)))
        self.lbl_current.setText("—")
        self.lbl_cells.setText("—")
        self.btn_start.setEnabled(True)
        self.tracked_frames.clear()
        self.btn_save.setEnabled(False)
        self.player_widget.setVisible(False)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"✔  {short}  —  {len(paths)} frame pronti.")

    def _on_start(self):
        paths = self.state.get("paths", [])
        if not paths:
            return
        self.tracked_frames.clear()
        self._total_frames = len(paths)
        self.progress_bar.setValue(0)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_save.setEnabled(False)
        self._player_timer.stop()
        self.player_widget.setVisible(False)
        self.title_label.setText("TRACKING IN TEMPO REALE")
        self.status_label.setText("Tracking in corso…")

        params = self.get_params()
        self.state["params"] = params

        parallel = self.combo_mode.currentIndex() == 1
        self.worker = TrackingWorker(paths, params, parallel=parallel)
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished.connect(self._on_finished)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_stop(self):
        if self.worker:
            self.worker.stop()

    def _on_frame_ready(self, idx, rgb, n_cells):
        self.tracked_frames.append(rgb)
        pct = int((idx + 1) / self._total_frames * 100)
        self.progress_bar.setValue(pct)
        self.lbl_current.setText(str(idx + 1))
        self.lbl_cells.setText(str(n_cells))
        self.title_label.setText(f"TRACKING IN TEMPO REALE  —  {n_cells} cellule")
        self.status_label.setText(f"Frame {idx + 1} / {self._total_frames}  —  {n_cells} cellule")

        pixmap = ndarray_to_pixmap(rgb)
        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def _on_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if self.tracked_frames:
            self.btn_save.setEnabled(True)
            self._player_init()
        n = len(self.tracked_frames)
        self.status_label.setText(f"Completato. {n} frame elaborati.")

    def _player_init(self):
        n = len(self.tracked_frames)
        self._player_idx = 0
        self.frame_slider.setMaximum(n - 1)
        self.frame_slider.setValue(0)
        self.frame_counter_label.setText(f"1 / {n}")
        self.btn_play_pause.setText("▶")
        self.player_widget.setVisible(True)
        self._player_show(0)

    def _player_show(self, idx):
        rgb = self.tracked_frames[idx]
        pixmap = ndarray_to_pixmap(rgb)
        self.image_label.setPixmap(
            pixmap.scaled(self.image_label.size(),
                          Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation)
        )
        n = len(self.tracked_frames)
        self.frame_counter_label.setText(f"{idx + 1} / {n}")

    def _player_toggle(self):
        if self._player_timer.isActive():
            self._player_timer.stop()
            self.btn_play_pause.setText("▶")
        else:
            self._player_timer.start()
            self.btn_play_pause.setText("⏸")

    def _player_advance(self):
        n = len(self.tracked_frames)
        if not n:
            self._player_timer.stop()
            return
        self._player_idx = (self._player_idx + 1) % n
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self._player_idx)
        self.frame_slider.blockSignals(False)
        self._player_show(self._player_idx)

    def _player_prev(self):
        self._player_timer.stop()
        self.btn_play_pause.setText("▶")
        n = len(self.tracked_frames)
        self._player_idx = (self._player_idx - 1) % n
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self._player_idx)
        self.frame_slider.blockSignals(False)
        self._player_show(self._player_idx)

    def _player_next(self):
        self._player_timer.stop()
        self.btn_play_pause.setText("▶")
        n = len(self.tracked_frames)
        self._player_idx = (self._player_idx + 1) % n
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self._player_idx)
        self.frame_slider.blockSignals(False)
        self._player_show(self._player_idx)

    def _on_slider_moved(self, value):
        self._player_timer.stop()
        self.btn_play_pause.setText("▶")
        self._player_idx = value
        self._player_show(value)

    def _on_error(self, msg):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        QMessageBox.critical(self, "Errore", msg)

    def _on_save(self):
        if not self.tracked_frames:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Salva Video", "", "MP4 Video (*.mp4)"
        )
        if not path:
            return
        try:
            save_video(self.tracked_frames, path, fps=5)
            QMessageBox.information(self, "Salvato", f"Video salvato in:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Errore salvataggio", str(exc))


# ---------------------------------------------------------------------------
# Page 1: ComparisonPage
# ---------------------------------------------------------------------------

class ComparisonPage(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.worker = None          # sequential worker
        self._gac_worker = None     # parallel mode: GAC worker
        self._lbf_worker = None     # parallel mode: LBF worker
        self._parallel_done_count = 0
        self._frames = {"GAC": [], "LBF": []}
        self._player_idx = {"GAC": 0, "LBF": 0}
        self._player_timers = {"GAC": QTimer(), "LBF": QTimer()}
        self._times = {}
        self._total_frames = 0
        for algo, timer in self._player_timers.items():
            timer.setInterval(200)
            timer.timeout.connect(lambda a=algo: self._player_advance(a))
        self._build_ui()

    def _is_running(self):
        if self.worker and self.worker.isRunning():
            return True
        if self._gac_worker and self._gac_worker.isRunning():
            return True
        if self._lbf_worker and self._lbf_worker.isRunning():
            return True
        return False

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(10)

        # ── Accent bar ─────────────────────────────────────────────────────
        acc = QFrame()
        acc.setFixedHeight(3)
        acc.setStyleSheet("background-color: #80cbc4; border-radius: 2px;")
        root.addWidget(acc)

        # ── Header: title left, params toggle right ────────────────────────
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 4, 0, 0)

        title = QLabel("CONFRONTO: GAC vs LBF")
        f = QFont()
        f.setBold(True)
        f.setPointSize(14)
        title.setFont(f)
        title.setStyleSheet("color: #dde4f0; background: transparent;")
        header_row.addWidget(title)
        header_row.addStretch()

        self._btn_params = QPushButton("⚙  Parametri")
        self._btn_params.setCheckable(True)
        self._btn_params.setFixedHeight(32)
        self._btn_params.setFixedWidth(130)
        self._btn_params.setStyleSheet(
            "QPushButton { background: transparent; color: #4a5568; border: 1px solid #1c2035;"
            " border-radius: 8px; font-size: 12px; padding: 0 10px; }"
            "QPushButton:hover { color: #80cbc4; border-color: #2a6b67; }"
            "QPushButton:checked { color: #80cbc4; border-color: #2a6b67; background: #0d2a28; }"
        )
        self._btn_params.clicked.connect(self._toggle_params)
        header_row.addWidget(self._btn_params)

        root.addLayout(header_row)

        # ── Control row ────────────────────────────────────────────────────
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)

        self.btn_run = QPushButton("▶  Avvia Confronto")
        self.btn_run.setObjectName("btn_primary")
        self.btn_run.setFixedHeight(40)
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._on_run)
        ctrl_row.addWidget(self.btn_run, stretch=2)

        self.btn_stop = QPushButton("■  Stop")
        self.btn_stop.setObjectName("btn_danger")
        self.btn_stop.setFixedHeight(40)
        self.btn_stop.setFixedWidth(120)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        ctrl_row.addWidget(self.btn_stop)

        self.combo_mode = QComboBox()
        self.combo_mode.addItem("Sequenziale")
        self.combo_mode.addItem("Parallelo")
        self.combo_mode.setFixedHeight(40)
        self.combo_mode.setFixedWidth(140)
        ctrl_row.addWidget(self.combo_mode)

        root.addLayout(ctrl_row)

        self.status_label = QLabel("Seleziona una cartella nella pagina Tracking.")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(
            "color: #4a5568; font-size: 11px; padding: 0; background: transparent;"
        )
        root.addWidget(self.status_label)

        # ── Main content: videos left, right panel (params + chart) ────────
        content_row = QHBoxLayout()
        content_row.setSpacing(12)
        content_row.setContentsMargins(0, 0, 0, 0)

        # ── Video cards (side by side) ─────────────────────────────────────
        self.image_labels = {}
        self.info_labels = {}
        self.progress_bars = {}
        self.player_widgets = {}
        self.frame_sliders = {}
        self.btn_play_pauses = {}
        self.frame_counters = {}
        self.btn_save_videos = {}

        algo_colors = {"GAC": "#e57373", "LBF": "#64b5f6"}

        cards_row = QHBoxLayout()
        cards_row.setSpacing(12)

        for algo in ("GAC", "LBF"):
            color = algo_colors[algo]
            card_widget = QWidget()
            card_widget.setStyleSheet(
                "QWidget { background-color: #141828; border-radius: 10px; border: 1px solid #242840; }"
            )
            card = QVBoxLayout(card_widget)
            card.setSpacing(6)
            card.setContentsMargins(12, 12, 12, 12)

            card_header = QHBoxLayout()
            name_lbl = QLabel(algo)
            nf = QFont()
            nf.setBold(True)
            nf.setPointSize(13)
            name_lbl.setFont(nf)
            name_lbl.setStyleSheet(f"color: {color}; background: transparent; border: none;")
            card_header.addWidget(name_lbl)
            card_header.addStretch()

            btn_save_v = QPushButton("💾 Salva")
            btn_save_v.setFixedHeight(26)
            btn_save_v.setFixedWidth(76)
            btn_save_v.setEnabled(False)
            btn_save_v.setStyleSheet(
                "QPushButton { background: transparent; color: #4a5568; border: 1px solid #242840;"
                " border-radius: 6px; font-size: 11px; }"
                "QPushButton:enabled { color: #80cbc4; border-color: #2a6b67; }"
                "QPushButton:enabled:hover { background: #0d2a28; }"
            )
            btn_save_v.clicked.connect(lambda _, a=algo: self._on_save_video(a))
            card_header.addWidget(btn_save_v)
            self.btn_save_videos[algo] = btn_save_v
            card.addLayout(card_header)

            img_lbl = QLabel("In attesa…")
            img_lbl.setMinimumSize(220, 160)
            img_lbl.setStyleSheet(
                "background-color: #0f1120; border: 1px solid #1c2035;"
                " border-radius: 6px; color: #4a5568;"
            )
            img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            card.addWidget(img_lbl, stretch=1)
            self.image_labels[algo] = img_lbl

            prog = QProgressBar()
            prog.setValue(0)
            card.addWidget(prog)
            self.progress_bars[algo] = prog

            player_w = QWidget()
            player_w.setStyleSheet("background: transparent;")
            player_l = QVBoxLayout(player_w)
            player_l.setContentsMargins(0, 0, 0, 0)
            player_l.setSpacing(3)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(0)
            slider.setValue(0)
            slider.valueChanged.connect(lambda v, a=algo: self._on_slider(a, v))
            player_l.addWidget(slider)
            self.frame_sliders[algo] = slider

            ctrl_row2 = QHBoxLayout()
            btn_prev = QPushButton("⏮")
            btn_prev.setObjectName("btn_icon")
            btn_prev.setFixedWidth(34)
            btn_prev.setFixedHeight(26)
            btn_prev.clicked.connect(lambda _, a=algo: self._player_prev(a))
            ctrl_row2.addWidget(btn_prev)

            btn_pp = QPushButton("▶")
            btn_pp.setObjectName("btn_icon")
            btn_pp.setFixedWidth(40)
            btn_pp.setFixedHeight(26)
            btn_pp.clicked.connect(lambda _, a=algo: self._player_toggle(a))
            ctrl_row2.addWidget(btn_pp)
            self.btn_play_pauses[algo] = btn_pp

            btn_next = QPushButton("⏭")
            btn_next.setObjectName("btn_icon")
            btn_next.setFixedWidth(34)
            btn_next.setFixedHeight(26)
            btn_next.clicked.connect(lambda _, a=algo: self._player_next(a))
            ctrl_row2.addWidget(btn_next)

            frame_ctr = QLabel("0 / 0")
            frame_ctr.setStyleSheet(f"color: {color}; font-size: 11px; background: transparent; border: none;")
            frame_ctr.setAlignment(Qt.AlignmentFlag.AlignCenter)
            ctrl_row2.addWidget(frame_ctr, stretch=1)
            self.frame_counters[algo] = frame_ctr

            player_l.addLayout(ctrl_row2)
            player_w.setVisible(False)
            card.addWidget(player_w)
            self.player_widgets[algo] = player_w

            info_lbl = QLabel("—")
            info_lbl.setStyleSheet(f"color: {color}; font-size: 11px; background: transparent; border: none;")
            info_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card.addWidget(info_lbl)
            self.info_labels[algo] = info_lbl

            cards_row.addWidget(card_widget)

        content_row.addLayout(cards_row, stretch=1)

        # ── Right panel: params + chart (collapsible) ──────────────────────
        self._right_panel = QWidget()
        self._right_panel.setFixedWidth(250)
        self._right_panel.setVisible(False)
        self._right_panel.setStyleSheet("background: transparent;")
        right_layout = QVBoxLayout(self._right_panel)
        right_layout.setContentsMargins(0, 14, 4, 8)
        right_layout.setSpacing(10)

        gac_group = QGroupBox("PARAMETRI GAC")
        gac_form = QFormLayout()
        gac_form.setSpacing(4)
        gac_form.setContentsMargins(8, 16, 8, 8)
        gac_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.sp_gac_smoothing = QSpinBox()
        self.sp_gac_smoothing.setRange(1, 10)
        self.sp_gac_smoothing.setValue(2)
        gac_form.addRow("smoothing", self.sp_gac_smoothing)

        self.sp_gac_alpha = QDoubleSpinBox()
        self.sp_gac_alpha.setRange(10.0, 1000.0)
        self.sp_gac_alpha.setSingleStep(50.0)
        self.sp_gac_alpha.setValue(300.0)
        gac_form.addRow("alpha", self.sp_gac_alpha)

        self.sp_gac_sigma = QDoubleSpinBox()
        self.sp_gac_sigma.setRange(0.5, 5.0)
        self.sp_gac_sigma.setSingleStep(0.5)
        self.sp_gac_sigma.setValue(3.0)
        gac_form.addRow("sigma", self.sp_gac_sigma)

        self.sp_gac_threshold = QDoubleSpinBox()
        self.sp_gac_threshold.setRange(0.1, 1.0)
        self.sp_gac_threshold.setSingleStep(0.05)
        self.sp_gac_threshold.setDecimals(2)
        self.sp_gac_threshold.setValue(0.55)
        gac_form.addRow("threshold", self.sp_gac_threshold)

        self.sp_gac_balloon = QDoubleSpinBox()
        self.sp_gac_balloon.setRange(-1.0, 1.0)
        self.sp_gac_balloon.setSingleStep(0.1)
        self.sp_gac_balloon.setDecimals(2)
        self.sp_gac_balloon.setValue(-0.5)
        gac_form.addRow("balloon", self.sp_gac_balloon)

        gac_group.setLayout(gac_form)
        right_layout.addWidget(gac_group)

        lbf_group = QGroupBox("PARAMETRI LBF")
        lbf_form = QFormLayout()
        lbf_form.setSpacing(4)
        lbf_form.setContentsMargins(8, 16, 8, 8)
        lbf_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.sp_lbf_sigma = QDoubleSpinBox()
        self.sp_lbf_sigma.setRange(1.0, 50.0)
        self.sp_lbf_sigma.setSingleStep(1.0)
        self.sp_lbf_sigma.setValue(8.0)
        lbf_form.addRow("sigma", self.sp_lbf_sigma)

        self.sp_lbf_mu = QDoubleSpinBox()
        self.sp_lbf_mu.setRange(0.01, 5.0)
        self.sp_lbf_mu.setSingleStep(0.1)
        self.sp_lbf_mu.setDecimals(2)
        self.sp_lbf_mu.setValue(0.2)
        lbf_form.addRow("mu", self.sp_lbf_mu)

        self.sp_lbf_nu = QDoubleSpinBox()
        self.sp_lbf_nu.setRange(0.001, 1.0)
        self.sp_lbf_nu.setSingleStep(0.01)
        self.sp_lbf_nu.setDecimals(3)
        self.sp_lbf_nu.setValue(0.05)
        lbf_form.addRow("nu", self.sp_lbf_nu)

        self.sp_lbf_dt = QDoubleSpinBox()
        self.sp_lbf_dt.setRange(0.01, 0.5)
        self.sp_lbf_dt.setSingleStep(0.01)
        self.sp_lbf_dt.setDecimals(2)
        self.sp_lbf_dt.setValue(0.05)
        lbf_form.addRow("dt", self.sp_lbf_dt)

        lbf_group.setLayout(lbf_form)
        right_layout.addWidget(lbf_group)

        self.chart_view = QChartView()
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        _ph = QChart()
        apply_dark_chart_style(_ph)
        self.chart_view.setChart(_ph)
        right_layout.addWidget(self.chart_view, stretch=1)

        content_row.addWidget(self._right_panel)
        root.addLayout(content_row, stretch=1)

    def get_comparison_params(self):
        base = dict(self.state.get("params", {}))
        base.update({
            "gac_smoothing": self.sp_gac_smoothing.value(),
            "gac_alpha":     self.sp_gac_alpha.value(),
            "gac_sigma":     self.sp_gac_sigma.value(),
            "gac_threshold": self.sp_gac_threshold.value(),
            "gac_balloon":   self.sp_gac_balloon.value(),
            "lbf_sigma":     self.sp_lbf_sigma.value(),
            "lbf_mu":        self.sp_lbf_mu.value(),
            "lbf_nu":        self.sp_lbf_nu.value(),
            "lbf_dt":        self.sp_lbf_dt.value(),
        })
        return base

    def _toggle_params(self, checked):
        self._right_panel.setVisible(checked)
        self._btn_params.setText("✕  Chiudi" if checked else "⚙  Parametri")

    def activate(self):
        if self.state.get("paths") and not self._is_running():
            n = len(self.state["paths"])
            self.btn_run.setEnabled(True)
            self.status_label.setText(
                f"Pronto · {n} frame · GAC: 100/50 iter · LBF: 30/20 iter"
            )

    def _on_run(self):
        paths = self.state.get("paths", [])
        if not paths:
            return
        self._frames = {"GAC": [], "LBF": []}
        self._times = {}
        self._total_frames = len(paths)
        self._parallel_done_count = 0

        for algo in ("GAC", "LBF"):
            self.progress_bars[algo].setValue(0)
            self.player_widgets[algo].setVisible(False)
            self._player_timers[algo].stop()
            self.btn_play_pauses[algo].setText("▶")
            self.image_labels[algo].clear()
            self.image_labels[algo].setText("In elaborazione…")
            self.info_labels[algo].setText("—")
            self.btn_save_videos[algo].setEnabled(False)

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)

        params = self.get_comparison_params()
        parallel = self.combo_mode.currentIndex() == 1

        if parallel:
            self.status_label.setText(f"GAC + LBF in parallelo… (0 / {self._total_frames} frame)")
            self._gac_worker = SingleAlgoWorker("GAC", paths, params, iter_first=100, iter_rest=50, parallel=True)
            self._gac_worker.frame_ready.connect(self._on_frame_ready)
            self._gac_worker.algo_done.connect(self._on_algo_done)
            self._gac_worker.finished.connect(self._on_one_parallel_finished)
            self._gac_worker.finished.connect(self._gac_worker.deleteLater)
            self._gac_worker.error.connect(self._on_error)

            self._lbf_worker = SingleAlgoWorker("LBF", paths, params, iter_first=30, iter_rest=20, parallel=True)
            self._lbf_worker.frame_ready.connect(self._on_frame_ready)
            self._lbf_worker.algo_done.connect(self._on_algo_done)
            self._lbf_worker.finished.connect(self._on_one_parallel_finished)
            self._lbf_worker.finished.connect(self._lbf_worker.deleteLater)
            self._lbf_worker.error.connect(self._on_error)

            self._gac_worker.start()
            self._lbf_worker.start()
        else:
            self.status_label.setText(f"GAC in corso… (0 / {self._total_frames} frame)")
            self.worker = ComparisonWorker(paths, params)
            self.worker.frame_ready.connect(self._on_frame_ready)
            self.worker.algo_done.connect(self._on_algo_done)
            self.worker.finished.connect(self._on_finished)
            self.worker.finished.connect(self.worker.deleteLater)
            self.worker.error.connect(self._on_error)
            self.worker.start()

    def _on_stop(self):
        if self.worker:
            self.worker.stop()
        if self._gac_worker:
            self._gac_worker.stop()
        if self._lbf_worker:
            self._lbf_worker.stop()

    def _on_one_parallel_finished(self):
        self._parallel_done_count += 1
        if self._parallel_done_count >= 2:
            self._on_finished()

    def _on_frame_ready(self, algo, idx, rgb, n_cells):
        self._frames[algo].append(rgb)
        n = self._total_frames
        self.progress_bars[algo].setValue(int((idx + 1) / n * 100))
        self.status_label.setText(f"{algo} in corso… ({idx+1} / {n}) · {n_cells} cellule")

        pixmap = ndarray_to_pixmap(rgb)
        img_lbl = self.image_labels[algo]
        img_lbl.setPixmap(pixmap.scaled(img_lbl.size(),
                                        Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation))

    def _on_algo_done(self, algo, total_time):
        self._times[algo] = total_time
        frames = self._frames[algo]
        n = len(frames)

        self.frame_sliders[algo].setMaximum(max(0, n - 1))
        self.frame_sliders[algo].setValue(0)
        self._player_idx[algo] = 0
        self.frame_counters[algo].setText(f"1 / {n}")
        self.player_widgets[algo].setVisible(True)
        self.info_labels[algo].setText(f"{n} frame · {total_time:.1f}s totale")
        self.btn_save_videos[algo].setEnabled(n > 0)

        if len(self._times) == 2:
            self._update_chart()

    def _on_finished(self):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if len(self._times) == 2:
            self.status_label.setText("Confronto completato.")
        else:
            self.status_label.setText("Interrotto.")

    def _on_error(self, msg):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_label.setText("Errore durante il confronto.")
        QMessageBox.critical(self, "Errore", msg)

    def _on_save_video(self, algo):
        frames = self._frames.get(algo, [])
        if not frames:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, f"Salva Video {algo}", f"confronto_{algo.lower()}.mp4", "MP4 Video (*.mp4)"
        )
        if not path:
            return
        try:
            save_video(frames, path, fps=5)
            QMessageBox.information(self, "Salvato", f"Video {algo} salvato in:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Errore salvataggio", str(exc))

    def _player_show(self, algo, idx):
        frames = self._frames[algo]
        if not frames or idx >= len(frames):
            return
        pixmap = ndarray_to_pixmap(frames[idx])
        img_lbl = self.image_labels[algo]
        img_lbl.setPixmap(pixmap.scaled(img_lbl.size(),
                                        Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation))
        self.frame_counters[algo].setText(f"{idx+1} / {len(frames)}")

    def _player_toggle(self, algo):
        timer = self._player_timers[algo]
        if timer.isActive():
            timer.stop()
            self.btn_play_pauses[algo].setText("▶")
        else:
            timer.start()
            self.btn_play_pauses[algo].setText("⏸")

    def _player_advance(self, algo):
        n = len(self._frames[algo])
        if n == 0:
            return
        self._player_idx[algo] = (self._player_idx[algo] + 1) % n
        idx = self._player_idx[algo]
        self.frame_sliders[algo].blockSignals(True)
        self.frame_sliders[algo].setValue(idx)
        self.frame_sliders[algo].blockSignals(False)
        self._player_show(algo, idx)

    def _player_prev(self, algo):
        self._player_timers[algo].stop()
        self.btn_play_pauses[algo].setText("▶")
        n = len(self._frames[algo])
        if n == 0:
            return
        self._player_idx[algo] = (self._player_idx[algo] - 1) % n
        idx = self._player_idx[algo]
        self.frame_sliders[algo].blockSignals(True)
        self.frame_sliders[algo].setValue(idx)
        self.frame_sliders[algo].blockSignals(False)
        self._player_show(algo, idx)

    def _player_next(self, algo):
        self._player_timers[algo].stop()
        self.btn_play_pauses[algo].setText("▶")
        n = len(self._frames[algo])
        if n == 0:
            return
        self._player_idx[algo] = (self._player_idx[algo] + 1) % n
        idx = self._player_idx[algo]
        self.frame_sliders[algo].blockSignals(True)
        self.frame_sliders[algo].setValue(idx)
        self.frame_sliders[algo].blockSignals(False)
        self._player_show(algo, idx)

    def _on_slider(self, algo, value):
        self._player_timers[algo].stop()
        self.btn_play_pauses[algo].setText("▶")
        self._player_idx[algo] = value
        self._player_show(algo, value)

    def _update_chart(self):
        chart = QChart()
        chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)
        chart.setTitle("Tempo totale di tracking")
        apply_dark_chart_style(chart)

        colors = {"GAC": QColor("#e57373"), "LBF": QColor("#64b5f6")}
        series = QBarSeries()
        for algo in ("GAC", "LBF"):
            if algo in self._times:
                bar_set = QBarSet(algo)
                bar_set.append(self._times[algo])
                bar_set.setColor(colors[algo])
                series.append(bar_set)

        chart.addSeries(series)

        axis_x = QBarCategoryAxis()
        axis_x.append([""])
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        series.attachAxis(axis_x)
        style_axis(axis_x)

        axis_y = QValueAxis()
        axis_y.setTitleText("Secondi")
        axis_y.setMin(0)
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(axis_y)
        style_axis(axis_y)

        self.chart_view.setChart(chart)


# ---------------------------------------------------------------------------
# Page 2: SpeedupPage
# ---------------------------------------------------------------------------

class SpeedupPage(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.worker = None
        self._build_ui()

    def _is_running(self):
        return self.worker is not None and self.worker.isRunning()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(12)

        acc = QFrame()
        acc.setFixedHeight(3)
        acc.setStyleSheet("background-color: #80cbc4; border-radius: 2px;")
        root.addWidget(acc)

        title = QLabel("SPEEDUP CALCOLO PARALLELO")
        f = QFont()
        f.setBold(True)
        f.setPointSize(14)
        title.setFont(f)
        title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        title.setStyleSheet("color: #dde4f0; background: transparent; padding: 4px 0 0 0;")
        root.addWidget(title)

        subtitle = QLabel(
            "Confronto preprocessing seriale vs. parallelo con ProcessPoolExecutor"
        )
        subtitle.setStyleSheet("color: #4a5568; font-size: 12px; background: transparent;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        root.addWidget(subtitle)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        self.btn_run = QPushButton("⚡  Avvia Test Speedup")
        self.btn_run.setObjectName("btn_primary")
        self.btn_run.setFixedHeight(44)
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._on_run)
        btn_row.addWidget(self.btn_run)

        self.btn_save_chart = QPushButton("💾  Salva Grafico")
        self.btn_save_chart.setFixedHeight(44)
        self.btn_save_chart.setEnabled(False)
        self.btn_save_chart.clicked.connect(self._on_save_chart)
        btn_row.addWidget(self.btn_save_chart)
        btn_row.addStretch()
        root.addLayout(btn_row)

        self.status_label = QLabel("Seleziona una cartella nella pagina Tracking.")
        self.status_label.setStyleSheet("color: #4a5568; font-size: 12px; background: transparent;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.status_label)

        self.result_label = QLabel()
        self.result_label.setStyleSheet(
            "color: #80cbc4; background-color: #141828; border: 1px solid #242840;"
            " border-radius: 10px; padding: 16px;"
        )
        mono_f = QFont("Courier")
        mono_f.setPointSize(12)
        self.result_label.setFont(mono_f)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.result_label.hide()
        root.addWidget(self.result_label)

        self.chart_view = QChartView()
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        _placeholder = QChart()
        apply_dark_chart_style(_placeholder)
        self.chart_view.setChart(_placeholder)
        root.addWidget(self.chart_view, stretch=1)

    def activate(self):
        if self.state.get("paths") and not self._is_running():
            self.btn_run.setEnabled(True)
            self.status_label.setText("Pronto per il test speedup.")

    def _on_run(self):
        paths = self.state.get("paths", [])
        if not paths:
            return
        params = self.state.get("params", {})
        clip_limit = params.get("clip_limit", 0.10)
        smooth_sigma = params.get("smooth_factor", 3.0)

        self.btn_run.setEnabled(False)
        self.btn_save_chart.setEnabled(False)
        self.status_label.setText("Test in corso…")
        self.result_label.hide()

        self.worker = SpeedupWorker(paths, clip_limit, smooth_sigma)
        self.worker.result_ready.connect(self._on_results)
        self.worker.finished.connect(lambda: self.btn_run.setEnabled(True))
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_results(self, serial, parallel, speedup):
        n = len(self.state.get("paths", []))
        text = (
            f"Frame elaborati   : {n}\n"
            f"Tempo seriale     : {serial:.3f} s\n"
            f"Tempo parallelo   : {parallel:.3f} s\n"
            f"Speedup           : {speedup:.2f}×"
        )
        self.result_label.setText(text)
        self.result_label.show()
        self.status_label.setText("Test completato.")
        self._update_chart(serial, parallel, speedup)
        self.btn_save_chart.setEnabled(True)

    def _on_save_chart(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Salva Grafico Speedup", "speedup_chart.png", "PNG Image (*.png)"
        )
        if not path:
            return
        try:
            pixmap = self.chart_view.grab()
            if pixmap.save(path, "PNG"):
                QMessageBox.information(self, "Salvato", f"Grafico salvato in:\n{path}")
            else:
                QMessageBox.critical(self, "Errore", "Impossibile salvare l'immagine.")
        except Exception as exc:
            QMessageBox.critical(self, "Errore salvataggio", str(exc))

    def _update_chart(self, serial, parallel, speedup):
        chart = QChart()
        chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)
        chart.setTitle(f"Speedup: {speedup:.2f}×")
        apply_dark_chart_style(chart)

        set_serial = QBarSet("Seriale")
        set_serial.append(serial)
        set_serial.setColor(QColor("#e57373"))

        set_parallel = QBarSet("Parallelo")
        set_parallel.append(parallel)
        set_parallel.setColor(QColor("#80cbc4"))

        series = QBarSeries()
        series.append(set_serial)
        series.append(set_parallel)
        chart.addSeries(series)

        axis_x = QBarCategoryAxis()
        axis_x.append([""])
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        series.attachAxis(axis_x)
        style_axis(axis_x)

        axis_y = QValueAxis()
        axis_y.setTitleText("Secondi")
        axis_y.setMin(0)
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(axis_y)
        style_axis(axis_y)

        self.chart_view.setChart(chart)

    def _on_error(self, msg):
        self.btn_run.setEnabled(True)
        self.status_label.setText("Errore durante il test.")
        QMessageBox.critical(self, "Errore", msg)


# ---------------------------------------------------------------------------
# Page 3: InfoPage
# ---------------------------------------------------------------------------

class InfoPage(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 24, 32, 24)
        root.setSpacing(10)

        acc = QFrame()
        acc.setFixedHeight(3)
        acc.setStyleSheet("background-color: #80cbc4; border-radius: 2px;")
        root.addWidget(acc)

        title = QLabel("PARALLEL SMART TRACKER")
        f = QFont()
        f.setBold(True)
        f.setPointSize(22)
        title.setFont(f)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #dde4f0; background: transparent; padding: 12px 0 4px 0;")
        root.addWidget(title)

        subtitle = QLabel("Analisi e tracking di cellule in microscopia a fluorescenza")
        subtitle.setStyleSheet("color: #80cbc4; background: transparent;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sf = QFont()
        sf.setPointSize(12)
        subtitle.setFont(sf)
        root.addWidget(subtitle)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none; background: transparent;")

        content = QLabel()
        content.setTextFormat(Qt.TextFormat.RichText)
        content.setWordWrap(True)
        content.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        content.setStyleSheet("padding: 12px 8px; color: #94a3b8; background: transparent;")
        content.setText("""
<h2 style="color:#80cbc4;">Descrizione del Progetto</h2>
<p>
<b>Smart Tracker</b> è un'applicazione avanzata per il tracking automatico di cellule
in sequenze di immagini di microscopia a fluorescenza. Combina algoritmi di segmentazione
basati su <i>level set</i> con un'interfaccia grafica moderna e supporto al calcolo parallelo.
</p>

<h2 style="color:#80cbc4;">Algoritmi di Segmentazione</h2>

<h3 style="color:#dde4f0;">Chan-Vese (Region global)</h3>
<p>
Il metodo Chan-Vese suddivide l'immagine in due regioni (interno/esterno) minimizzando
un'energia basata sulla variazione totale e sull'omogeneità delle intensità. È robusto
al rumore e funziona anche su contorni deboli o discontinui. Viene usato come algoritmo
principale nel tracking frame-per-frame.
</p>

<h3 style="color:#dde4f0;">GAC — Geodesic Active Contours (Edge-based)</h3>
<p>
Il modello GAC guida il contorno attivo lungo i gradienti dell'immagine tramite una
funzione di arresto basata sul gradiente inverso gaussiano. È particolarmente efficace
quando i bordi cellulari sono nitidi e ben definiti.
</p>

<h3 style="color:#dde4f0;">LBF — Local Binary Fitting (Region local)</h3>
<p>
LBF adatta il contorno usando informazioni di intensità locali piuttosto che globali,
rendendolo adatto a immagini con illuminazione non uniforme. La funzione di energia
incorpora kernel gaussiani per stimare le medie locali interne ed esterne al contorno.
</p>

<h2 style="color:#80cbc4;">Pipeline di Elaborazione</h2>
<ol>
  <li><b>Lettura</b>: caricamento e normalizzazione dei frame TIFF.</li>
  <li><b>Preprocessing</b>: filtro mediano → top-hat morfologico → CLAHE → smoothing gaussiano.</li>
  <li><b>Maschera iniziale</b>: sogliatura locale adattiva + apertura morfologica.</li>
  <li><b>Level set</b>: raffinamento della maschera con Chan-Vese (100 iter. frame 1, 50 iter. successivi).</li>
  <li><b>Pulizia</b>: apertura, rimozione oggetti piccoli, chiusura binaria.</li>
  <li><b>Overlay</b>: visualizzazione dei contorni in giallo sull'immagine originale in scala di grigi.</li>
  <li><b>Conteggio</b>: labeling connesso per contare le cellule distinte.</li>
</ol>

<h2 style="color:#80cbc4;">Calcolo Parallelo</h2>
<p>
La fase di preprocessing è computazionalmente intensiva. La pagina <b>Speedup</b> misura
il guadagno in prestazioni ottenuto eseguendo il preprocessing in parallelo su tutti i
core disponibili tramite <code>concurrent.futures.ProcessPoolExecutor</code>, confrontandolo
con l'esecuzione seriale.
</p>

<h2 style="color:#80cbc4;">Dominio Applicativo</h2>
<p>
L'applicazione è progettata per immagini di <b>microscopia a fluorescenza</b>, dove le cellule
appaiono come regioni luminose su sfondo scuro. Parametri come <i>min_cell_size</i>,
<i>smooth_factor</i> e <i>clip_limit</i> possono essere regolati per adattarsi a diversi tipi
di preparato istologico e protocolli di imaging.
</p>

<p style="color:#4a5568; font-size:11px; margin-top:20px;">
Smart Tracker v2.0 — PySide6 · Qt Charts · scikit-image · tifffile · OpenCV
</p>
""")
        scroll.setWidget(content)
        root.addWidget(scroll, stretch=1)


# ---------------------------------------------------------------------------
# Global stylesheet
# ---------------------------------------------------------------------------

_APP_STYLE = """
/* ── Base ── */
QMainWindow, QDialog {
    background-color: #0f1120;
    color: #dde4f0;
}

QWidget {
    background-color: transparent;
    color: #dde4f0;
    font-family: "Inter", "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}

QStackedWidget { background-color: #0f1120; }
QScrollArea { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }

/* ── Page background ── */
QWidget#page_bg { background-color: #0f1120; }

/* ── Right panel (Tracking) ── */
QWidget#right_panel {
    background-color: #0a0c14;
    border-left: 1px solid #1c2035;
}

/* ── Stat pills ── */
QWidget#stat_pill {
    background-color: #141828;
    border: 1px solid #242840;
    border-radius: 8px;
}

/* ── Group Box — dark card ── */
QGroupBox {
    background-color: #141828;
    border: 1px solid #242840;
    border-radius: 10px;
    margin-top: 20px;
    padding: 18px 12px 10px 12px;
    font-size: 10px;
    font-weight: bold;
    letter-spacing: 1px;
    color: #4a5568;
    text-transform: uppercase;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    top: 4px;
    color: #4a5568;
    font-size: 10px;
    font-weight: bold;
    letter-spacing: 1px;
    background-color: #141828;
    padding: 2px 8px;
    border-radius: 4px;
}
QGroupBox QLabel {
    color: #94a3b8;
    font-size: 12px;
    background: transparent;
}

/* ── Buttons — base ── */
QPushButton {
    background-color: #1c2035;
    color: #94a3b8;
    border: 1px solid #242840;
    border-radius: 6px;
    padding: 5px 14px;
    font-size: 13px;
    font-weight: normal;
    min-height: 26px;
}
QPushButton:hover {
    background-color: #242840;
    border-color: #363d65;
    color: #dde4f0;
}
QPushButton:pressed  { background-color: #141828; }
QPushButton:disabled {
    background-color: #141828;
    color: #363d65;
    border-color: #1c2035;
}

/* ── Buttons — primary ── */
QPushButton#btn_primary {
    background-color: #0d6e66;
    border-color: #0d6e66;
    color: #dde4f0;
    font-weight: 600;
}
QPushButton#btn_primary:hover {
    background-color: #80cbc4;
    border-color: #80cbc4;
    color: #0a0c14;
}
QPushButton#btn_primary:pressed { background-color: #5fb3ac; border-color: #5fb3ac; }
QPushButton#btn_primary:disabled {
    background-color: #0a2e2b;
    border-color: #0a2e2b;
    color: #2a5552;
}

/* ── Buttons — danger ── */
QPushButton#btn_danger {
    background-color: #5c1a1a;
    border-color: #5c1a1a;
    color: #e57373;
    font-weight: 600;
}
QPushButton#btn_danger:hover {
    background-color: #7a2020;
    border-color: #7a2020;
    color: #ffcdd2;
}
QPushButton#btn_danger:pressed { background-color: #3d1010; }
QPushButton#btn_danger:disabled {
    background-color: #2a1010;
    border-color: #2a1010;
    color: #5c2a2a;
}

/* ── Buttons — icon/minimal ── */
QPushButton#btn_icon {
    background-color: #141828;
    border: 1px solid #242840;
    border-radius: 5px;
    color: #4a5568;
    padding: 3px 6px;
    font-size: 15px;
    min-height: 0;
}
QPushButton#btn_icon:hover {
    background-color: #1c2035;
    color: #9dd5cf;
    border-color: #363d65;
}
QPushButton#btn_icon:pressed { background-color: #0f1120; }

/* ── QLabel ── */
QLabel { background: transparent; }

/* ── SpinBox / DoubleSpinBox ── */
QSpinBox, QDoubleSpinBox {
    background-color: #0f1120;
    color: #dde4f0;
    border: 1px solid #242840;
    border-radius: 5px;
    padding: 4px 6px;
    selection-background-color: #80cbc4;
    selection-color: #0a0c14;
    min-height: 22px;
}
QSpinBox:hover, QDoubleSpinBox:hover { border-color: #363d65; }
QSpinBox:focus, QDoubleSpinBox:focus { border-color: #80cbc4; outline: none; }
QSpinBox::up-button, QDoubleSpinBox::up-button {
    background-color: #1c2035; border: none; border-top-right-radius: 5px; width: 18px;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background-color: #1c2035; border: none; border-bottom-right-radius: 5px; width: 18px;
}
QSpinBox::up-button:hover,   QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #242840;
}

/* ── ComboBox ── */
QComboBox {
    background-color: #0f1120;
    color: #dde4f0;
    border: 1px solid #242840;
    border-radius: 5px;
    padding: 4px 8px;
    min-height: 22px;
}
QComboBox:hover { border-color: #363d65; }
QComboBox:focus { border-color: #80cbc4; outline: none; }
QComboBox::drop-down {
    border: none;
    border-left: 1px solid #1c2035;
    width: 28px;
    background-color: #141828;
    border-radius: 0px 5px 5px 0px;
}
QComboBox::down-arrow { width: 10px; height: 10px; }
QComboBox QAbstractItemView {
    background-color: #1c2035;
    border: 1px solid #242840;
    border-radius: 6px;
    color: #dde4f0;
    selection-background-color: #242840;
    selection-color: #80cbc4;
    outline: none;
    padding: 4px;
}

/* ── Progress bar ── */
QProgressBar {
    background-color: #1c2035;
    border: none;
    border-radius: 3px;
    max-height: 5px;
    color: transparent;
    font-size: 1px;
}
QProgressBar::chunk { background-color: #80cbc4; border-radius: 3px; }

/* ── Slider ── */
QSlider::groove:horizontal {
    background-color: #1c2035; height: 4px; border-radius: 2px;
}
QSlider::handle:horizontal {
    background-color: #80cbc4; width: 13px; height: 13px;
    border-radius: 7px; margin: -5px 0; border: 2px solid #0f1120;
}
QSlider::handle:horizontal:hover { background-color: #9dd5cf; }
QSlider::sub-page:horizontal { background-color: #5fb3ac; border-radius: 2px; }

/* ── Scrollbars — 5px, subtle ── */
QScrollBar:vertical   { background: transparent; width: 5px; margin: 0; }
QScrollBar:horizontal { background: transparent; height: 5px; }
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background-color: #242840; border-radius: 3px; min-height: 30px; min-width: 30px;
}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
    background-color: #363d65;
}
QScrollBar::add-line:vertical,  QScrollBar::sub-line:vertical  { height: 0; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

/* ── Chart view ── */
QChartView {
    background-color: #141828;
    border: 1px solid #242840;
    border-radius: 8px;
}

/* ── Tooltips ── */
QToolTip {
    background-color: #1c2035;
    color: #dde4f0;
    border: 1px solid #363d65;
    border-radius: 5px;
    padding: 5px 8px;
}

/* ── Message boxes ── */
QMessageBox { background-color: #0f1120; }
QMessageBox QPushButton { min-width: 80px; }
"""

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

_SIDEBAR_STYLE = """
QWidget#sidebar {
    background-color: #0a0c14;
    border-right: 1px solid #1c2035;
}

/* Expanded nav button — rounded card style */
QPushButton#nav_btn {
    background-color: transparent;
    color: #4a5568;
    border: 1px solid transparent;
    text-align: left;
    padding-left: 16px;
    font-size: 13px;
    font-weight: normal;
    border-radius: 10px;
    margin: 3px 10px;
}
QPushButton#nav_btn:hover {
    background-color: #141828;
    color: #8892a4;
    border: 1px solid #1c2035;
    border-radius: 10px;
}
QPushButton#nav_btn[active="true"] {
    background-color: #0d2a28;
    color: #80cbc4;
    border: 1.5px solid #2a6b67;
    border-radius: 10px;
    font-weight: bold;
}

/* Collapsed nav button — icon only, centered, large, rounded */
QPushButton#nav_btn[collapsed="true"] {
    background-color: transparent;
    color: #4a5568;
    border: 1px solid transparent;
    text-align: center;
    padding: 0px;
    font-size: 24px;
    border-radius: 14px;
    margin: 8px 10px;
}
QPushButton#nav_btn[collapsed="true"]:hover {
    background-color: #141828;
    color: #8892a4;
    border: 1px solid #1c2035;
}
QPushButton#nav_btn[active="true"][collapsed="true"] {
    background-color: #0d2a28;
    color: #80cbc4;
    border: 2px solid #2a6b67;
    border-radius: 14px;
    font-weight: bold;
}

/* Toggle button */
QPushButton#nav_toggle {
    background-color: transparent;
    color: #4a5568;
    border: 1px solid #1c2035;
    border-radius: 8px;
    font-size: 14px;
    padding: 0px;
}
QPushButton#nav_toggle:hover {
    background-color: #141828;
    color: #80cbc4;
    border-color: #80cbc4;
}
"""

_W_EXPANDED = 220
_W_COLLAPSED = 64
_ICONS   = ["🎯", "📊", "⚡", "ℹ️"]
_LABELS  = ["🎯  Tracking", "📊  Confronto", "⚡  Speedup", "ℹ️  Info"]


class Sidebar(QWidget):
    page_selected = Signal(int)

    def __init__(self):
        super().__init__()
        self.setObjectName("sidebar")
        self.setFixedWidth(_W_EXPANDED)
        self.setStyleSheet(_SIDEBAR_STYLE)
        self._collapsed = False

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 16)
        self._layout.setSpacing(0)
        layout = self._layout

        # ── Logo area ──────────────────────────────────────────────────────
        self._logo_widget = QWidget()
        self._logo_widget.setStyleSheet("background: transparent;")
        logo_layout = QVBoxLayout(self._logo_widget)
        logo_layout.setContentsMargins(0, 16, 0, 0)
        logo_layout.setSpacing(3)

        title = QLabel("Smart Tracker")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tf = QFont()
        tf.setBold(True)
        tf.setPointSize(15)
        title.setFont(tf)
        title.setStyleSheet("color: #dde4f0; background: transparent;")
        logo_layout.addWidget(title)

        tagline = QLabel("Cell Analysis")
        tagline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tagline.setStyleSheet(
            "color: #80cbc4; font-size: 10px; background: transparent;"
            " letter-spacing: 2px; font-variant: small-caps;"
        )
        logo_layout.addWidget(tagline)

        sep_container = QWidget()
        sep_container.setStyleSheet("background: transparent;")
        sep_layout = QHBoxLayout(sep_container)
        sep_layout.setContentsMargins(16, 10, 16, 10)
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFixedHeight(1)
        separator.setStyleSheet("background-color: #1c2035; border: none;")
        sep_layout.addWidget(separator)
        logo_layout.addWidget(sep_container)

        layout.addWidget(self._logo_widget)

        # ── Toggle button row ──────────────────────────────────────────────
        self._toggle_row = QWidget()
        self._toggle_row.setStyleSheet("background: transparent;")
        toggle_row = self._toggle_row
        self._tr_layout = QHBoxLayout(toggle_row)
        self._tr_layout.setContentsMargins(8, 0, 8, 4)

        self._toggle_btn = QPushButton("‹")
        self._toggle_btn.setObjectName("nav_toggle")
        self._toggle_btn.setFixedSize(28, 24)
        self._toggle_btn.clicked.connect(self.toggle)

        # Expanded: toggle sits at the right edge
        self._tr_layout.addStretch()
        self._tr_layout.addWidget(self._toggle_btn)
        layout.addWidget(toggle_row)

        layout.addStretch(1)  # 1/4 of available space above buttons

        # ── Nav buttons ────────────────────────────────────────────────────
        self._buttons = []
        for i, label in enumerate(_LABELS):
            btn = QPushButton(label)
            btn.setObjectName("nav_btn")
            btn.setFixedHeight(46)
            btn.setProperty("collapsed", "false")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, idx=i: self.page_selected.emit(idx))
            layout.addWidget(btn)
            self._buttons.append(btn)

        layout.addStretch(3)  # 3/4 of available space below buttons

        self._version = QLabel("v2.0 · PySide6")
        self._version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._version.setStyleSheet(
            "color: #242840; font-size: 10px; padding: 8px; background: transparent;"
        )
        layout.addWidget(self._version)

        self._select(0)

    # ── Animation ──────────────────────────────────────────────────────────
    def _set_toggle_layout(self, centered: bool):
        # Clear the toggle row layout and rebuild alignment
        while self._tr_layout.count():
            item = self._tr_layout.takeAt(0)
            # don't delete the button widget — only spacers are heap items
        if centered:
            self._tr_layout.addStretch()
            self._tr_layout.addWidget(self._toggle_btn)
            self._tr_layout.addStretch()
        else:
            self._tr_layout.addStretch()
            self._tr_layout.addWidget(self._toggle_btn)

    def toggle(self):
        if self._collapsed:
            # Expanding: move toggle_row back above the stretch (index 1)
            self._layout.removeWidget(self._toggle_row)
            self._layout.insertWidget(1, self._toggle_row)
            self._animate_to(_W_EXPANDED)
            self._collapsed = False
            self._toggle_btn.setText("‹")
            self._set_toggle_layout(centered=False)
            self._logo_widget.show()
            self._version.show()
            for btn in self._buttons:
                idx = self._buttons.index(btn)
                btn.setText(_LABELS[idx])
                btn.setFixedHeight(46)
                btn.setProperty("collapsed", "false")
                btn.style().unpolish(btn)
                btn.style().polish(btn)
        else:
            # Collapsing: move toggle_row just above the first button (index 2, after stretch)
            self._layout.removeWidget(self._toggle_row)
            self._layout.insertWidget(2, self._toggle_row)
            self._animate_to(_W_COLLAPSED)
            self._collapsed = True
            self._toggle_btn.setText("›")
            self._set_toggle_layout(centered=True)
            self._logo_widget.hide()
            self._version.hide()
            for btn in self._buttons:
                idx = self._buttons.index(btn)
                btn.setText(_ICONS[idx])
                btn.setFixedHeight(44)
                btn.setProperty("collapsed", "true")
                btn.style().unpolish(btn)
                btn.style().polish(btn)

    def _animate_to(self, target_width):
        anim = QPropertyAnimation(self, b"minimumWidth", self)
        anim.setDuration(220)
        anim.setStartValue(self.width())
        anim.setEndValue(target_width)
        anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        anim2 = QPropertyAnimation(self, b"maximumWidth", self)
        anim2.setDuration(220)
        anim2.setStartValue(self.width())
        anim2.setEndValue(target_width)
        anim2.setEasingCurve(QEasingCurve.Type.InOutQuad)
        anim.start()
        anim2.start()
        # keep references so they aren't GC'd mid-animation
        self._anim = anim
        self._anim2 = anim2

    # ── Selection ──────────────────────────────────────────────────────────
    def _select(self, index):
        for i, btn in enumerate(self._buttons):
            active = i == index
            btn.setProperty("active", "true" if active else "false")
            btn.setChecked(active)
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def select_page(self, index):
        self._select(index)


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Tracker — PySide6")
        self.resize(1280, 800)

        self.state = {"paths": [], "params": {}}

        central = QWidget()
        central.setStyleSheet("background-color: #0f1120;")
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.sidebar = Sidebar()
        self.sidebar.page_selected.connect(self._select_page)
        root.addWidget(self.sidebar)

        self.stack = QStackedWidget()
        self.stack.setStyleSheet("background-color: #0f1120;")
        root.addWidget(self.stack)

        self.pages = [
            TrackingPage(self.state),
            ComparisonPage(self.state),
            SpeedupPage(self.state),
            InfoPage(self.state),
        ]
        for page in self.pages:
            self.stack.addWidget(page)

        # Timer to sync params from TrackingPage into shared state
        self._sync_timer = QTimer(self)
        self._sync_timer.setInterval(500)
        self._sync_timer.timeout.connect(self._sync_params)
        self._sync_timer.start()

        self._select_page(0)

    def _sync_params(self):
        tracking_page = self.pages[0]
        if callable(getattr(tracking_page, "get_params", None)):
            self.state["params"] = tracking_page.get_params()

    def _select_page(self, index):
        self.sidebar.select_page(index)
        self.stack.setCurrentIndex(index)
        page = self.pages[index]
        if callable(getattr(page, "activate", None)):
            page.activate()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # required for ProcessPoolExecutor in frozen builds
    app = QApplication(sys.argv)
    app.setStyleSheet(_APP_STYLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
