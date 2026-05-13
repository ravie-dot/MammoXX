import sys
import os
import json
import csv
import cv2
import numpy as np
import platform

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QPushButton,
    QFileDialog,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QMessageBox,
    QLineEdit,
    QListWidgetItem,
    QScrollArea,
    QFrame,
    QGroupBox,
    QScrollBar,
    QTextBrowser,
)

from PyQt5.QtGui import (
    QPixmap,
    QPainter,
    QColor,
    QPen,
    QFont,
    QImage
)

from PyQt5.QtCore import Qt, QPoint, QTimer

from mammo_paths import COORDINATES, LABEL_CSV, MASKS, POLYGONS, ensure_data_dirs


# =========================================
# FOLDERS (aligned with Flask web app: data/)
# =========================================

from mammo_paths import COORDINATES, LABEL_CSV, MASKS, POLYGONS, ensure_data_dirs

# Use the base directory from mammo_paths
from mammo_paths import _BASE_DIR
if platform.system() == "Darwin":  # macOS
    from pathlib import Path
    USER_FILE = str(Path.home() / "Library" / "Application Support" / "MammoX" / "users.json")
    os.makedirs(os.path.dirname(USER_FILE), exist_ok=True)
else:
    USER_FILE = os.path.join(_BASE_DIR, "users.json")


ensure_data_dirs()


def _cv2_imread(path):
    """OpenCV imread fails on some Windows paths (Unicode, spaces); use imdecode."""
    path = os.path.normpath(path)
    img = cv2.imread(path)
    if img is not None:
        return img
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _cv2_imwrite(path, image, ext=".png"):
    """Write image reliably on Windows (Unicode paths)."""
    path = os.path.normpath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    buf.tofile(path)
    return True

if not os.path.exists(USER_FILE):
    with open(USER_FILE, "w") as f:
        json.dump([], f)


# =========================================
# CANVAS
# =========================================

class ScrollableCanvas(QWidget):

    def __init__(self, coord_list, status_callback, image_name_label):

        super().__init__()

        self.coord_list = coord_list
        self.status_callback = status_callback
        self.image_name_label = image_name_label

        self.image = QPixmap()
        self.image_path = ""
        self.current_image_name = ""

        self.zoom_level = 1.0
        self.zoom_min = 0.05
        self.zoom_max = 10.0

        self.draw_mode = False
        self.erase_mode = False

        self.polygons = []
        self.current_polygon = []

        self.undo_stack = []
        self.redo_stack = []

        self.selected_point = None
        self.dragging = False

        self.point_radius = 6

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        self.image_label.setMouseTracking(True)

        self.image_label.mousePressEvent = self.mouse_press_event
        self.image_label.mouseMoveEvent = self.mouse_move_event
        self.image_label.mouseReleaseEvent = self.mouse_release_event
        self.image_label.wheelEvent = self.wheel_event

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(False)

        self.scroll_area.setStyleSheet("""
            QScrollArea{
                background:#111;
                border:2px solid cyan;
            }
        """)

        layout = QVBoxLayout()
        layout.addWidget(self.scroll_area)

        self.setLayout(layout)

        self.polygon_colors = [
            (0,255,255),
            (255,255,0),
            (255,0,255),
            (0,255,0),
            (255,128,0),
            (255,0,0)
        ]

    # =========================================
    # DRAW MODE
    # =========================================

    def set_draw_mode(self, enabled):

        self.draw_mode = enabled

        if enabled:
            self.image_label.setCursor(Qt.CrossCursor)
            self.status_callback("Draw on — add or drag vertices")
        else:
            self.image_label.setCursor(Qt.ArrowCursor)
            self.status_callback("Draw off")

    def set_erase_mode(self, enabled):

        self.erase_mode = enabled
        if enabled:
            self.image_label.setCursor(Qt.PointingHandCursor)
            self.status_callback("Erase on — click a vertex to remove it")
        elif not self.draw_mode:
            self.image_label.setCursor(Qt.ArrowCursor)
            self.status_callback("Erase off")

    # =========================================
    # LOAD IMAGE AND PREVIOUS ANNOTATIONS
    # =========================================

    def load_existing_annotations(self):
        """Load previously saved polygons for the current image."""
        if not self.current_image_name:
            return
        
        base_name = os.path.splitext(self.current_image_name)[0]
        
        # Look for JSON files associated with this image
        json_files = list(COORDINATES.glob(f"{base_name}_poly*.json"))
        
        self.polygons = []
        
        for json_file in sorted(json_files):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    polygon = json.load(f)
                    if len(polygon) >= 3:
                        self.polygons.append(polygon)
                        self.status_callback(f"Loaded saved region from {json_file.name}")
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        if self.polygons:
            self.status_callback(f"Loaded {len(self.polygons)} saved region(s)")
        else:
            self.status_callback("No saved regions found for this image")
        
        self.update_display()

    def load_image(self, path):

        self.image_path = os.path.normpath(os.path.abspath(path))
        self.current_image_name = os.path.basename(self.image_path)

        # Update the image name label
        if self.image_name_label:
            self.image_name_label.setText(f"📷 Current Image: {self.current_image_name}")
            self.image_name_label.setStyleSheet("""
                QLabel {
                    color: #00ffcc;
                    background-color: #0d1117;
                    padding: 8px;
                    border-radius: 5px;
                    font-weight: bold;
                    font-size: 12px;
                }
            """)

        self.image = QPixmap(self.image_path)

        # Clear current polygons
        self.polygons.clear()
        self.current_polygon.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()

        self.zoom_level = 1.0

        self.update_display()

        self.status_callback(f"Loaded {self.current_image_name}")
        
        # Load existing annotations for this image
        QTimer.singleShot(100, self.load_existing_annotations)
        
        QTimer.singleShot(200, self.fit_zoom_to_viewport)

    def fit_zoom_to_viewport(self):

        if self.image.isNull():
            return
        vp = self.scroll_area.viewport()
        vw, vh = vp.width(), vp.height()
        if vw < 80 or vh < 80:
            return
        iw, ih = self.image.width(), self.image.height()
        if iw < 1 or ih < 1:
            return
        z = min(vw / iw, vh / ih) * 0.98
        self.zoom_level = max(self.zoom_min, min(self.zoom_max, z))
        self.update_display()
        self.status_callback(
            f"Fit view — zoom {self.zoom_level:.2f}× ({len(self.polygons)} saved region(s))"
        )

    # =========================================
    # ZOOM
    # =========================================

    def wheel_event(self, event):

        if self.image.isNull():
            return

        if event.angleDelta().y() > 0:
            self.zoom_level *= 1.1
        else:
            self.zoom_level /= 1.1

        self.zoom_level = max(self.zoom_min, min(self.zoom_max, self.zoom_level))

        self.update_display()
        self.status_callback(f"Zoom {self.zoom_level:.2f}× (max {self.zoom_max}×)")

    # =========================================
    # COORDINATE CONVERSION
    # =========================================

    def get_image_coordinates(self, mouse_x, mouse_y):

        if self.image.isNull():
            return None, None

        img_x = mouse_x / self.zoom_level
        img_y = mouse_y / self.zoom_level

        if (
            img_x < 0 or
            img_y < 0 or
            img_x >= self.image.width() or
            img_y >= self.image.height()
        ):
            return None, None

        return round(float(img_x), 1), round(float(img_y), 1)

    # =========================================
    # DISPLAY
    # =========================================

    def update_display(self):

        if self.image.isNull():
            return

        scaled_width = int(self.image.width() * self.zoom_level)
        scaled_height = int(self.image.height() * self.zoom_level)

        scaled_pixmap = self.image.scaled(
            scaled_width,
            scaled_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        canvas = QPixmap(scaled_width, scaled_height)

        canvas.fill(QColor("#111"))

        painter = QPainter(canvas)

        painter.drawPixmap(0,0,scaled_pixmap)

        scale_x = scaled_width / self.image.width()
        scale_y = scaled_height / self.image.height()

        painter.setRenderHint(QPainter.Antialiasing)

        # Thinner line width - base width 1.5 pixels, scaled with zoom
        lw = max(1, int(1.5 / max(self.zoom_level, 0.1)))
        # Ensure line is always thin enough
        lw = min(lw, 2)

        # Draw saved polygons
        for idx, polygon in enumerate(self.polygons):

            color = self.polygon_colors[idx % len(self.polygon_colors)]

            self.draw_polygon(
                painter,
                polygon,
                scale_x,
                scale_y,
                color,
                lw,
            )

        # Draw current polygon
        self.draw_polygon(
            painter,
            self.current_polygon,
            scale_x,
            scale_y,
            (0, 255, 255),
            lw,
        )

        painter.end()

        self.image_label.setPixmap(canvas)

        self.image_label.setMinimumSize(
            scaled_width,
            scaled_height
        )

        self.update_coords()

    # =========================================
    # DRAW POLYGON
    # =========================================

    def draw_polygon(
        self,
        painter,
        polygon,
        scale_x,
        scale_y,
        color,
        line_width=1,  # Changed from 3 to 1 for thinner lines
    ):

        painter.setPen(QPen(QColor(*color), line_width))

        # draw lines
        for i in range(len(polygon)-1):

            x1,y1 = polygon[i]
            x2,y2 = polygon[i+1]

            painter.drawLine(
                int(x1*scale_x),
                int(y1*scale_y),
                int(x2*scale_x),
                int(y2*scale_y)
            )

        # close polygon
        if len(polygon) >= 3:

            x1,y1 = polygon[-1]
            x2,y2 = polygon[0]

            painter.drawLine(
                int(x1*scale_x),
                int(y1*scale_y),
                int(x2*scale_x),
                int(y2*scale_y)
            )

        # draw points (smaller radius for thinner appearance)
        pr = max(3, int(self.point_radius * max(0.6, self.zoom_level ** 0.45)))
        for x, y in polygon:
            painter.setBrush(QColor(255, 0, 0))
            painter.drawEllipse(
                QPoint(int(x * scale_x), int(y * scale_y)),
                pr,
                pr,
            )

    # =========================================
    # UPDATE COORDS
    # =========================================

    def update_coords(self):

        self.coord_list.clear()

        for idx, polygon in enumerate(self.polygons):
            item = QListWidgetItem(f"Saved outline {idx+1}")
            item.setForeground(Qt.yellow)
            self.coord_list.addItem(item)

            for x, y in polygon:
                self.coord_list.addItem(f"   ({float(x):.1f}, {float(y):.1f})")

        if self.current_polygon:

            item = QListWidgetItem("Editing")
            item.setForeground(Qt.cyan)
            self.coord_list.addItem(item)

            for x, y in self.current_polygon:
                self.coord_list.addItem(f"   ({float(x):.1f}, {float(y):.1f})")

    # =========================================
    # MOUSE EVENTS
    # =========================================

    def mouse_press_event(self, event):

        if self.image.isNull():
            return

        if not self.draw_mode and not self.erase_mode:
            return

        if event.button() != Qt.LeftButton:
            return

        img_x, img_y = self.get_image_coordinates(
            event.pos().x(),
            event.pos().y()
        )

        if img_x is None:
            return

        thresh = 14.0 / self.zoom_level

        def hit_vertex(poly):
            for i, (px, py) in enumerate(poly):
                if ((img_x - px) ** 2 + (img_y - py) ** 2) ** 0.5 < thresh:
                    return i
            return None

        if self.erase_mode:
            # Check both current polygon and saved polygons
            hi = hit_vertex(self.current_polygon)
            if hi is not None:
                self.undo_stack.append([p[:] for p in self.current_polygon])
                self.current_polygon.pop(hi)
                self.redo_stack.clear()
                self.update_display()
                self.status_callback(f"Vertex removed near ({img_x:.1f}, {img_y:.1f})")
            else:
                # Check in saved polygons
                found = False
                for idx, polygon in enumerate(self.polygons):
                    hi = hit_vertex(polygon)
                    if hi is not None:
                        # Remove the entire saved polygon if vertex is clicked
                        del self.polygons[idx]
                        self.update_display()
                        self.status_callback(f"Removed saved region {idx+1}")
                        found = True
                        break
                if not found:
                    self.status_callback("No vertex here — zoom in or switch to Draw")
            return

        if self.draw_mode:
            hi = hit_vertex(self.current_polygon)
            if hi is not None:
                self.selected_point = hi
                self.dragging = True
                return

            self.undo_stack.append([p[:] for p in self.current_polygon])
            self.current_polygon.append([img_x, img_y])
            self.redo_stack.clear()
            self.update_display()
            self.status_callback(f"Added ({img_x:.1f}, {img_y:.1f})")

    def mouse_move_event(self, event):

        if self.dragging and self.selected_point is not None:

            img_x, img_y = self.get_image_coordinates(
                event.pos().x(),
                event.pos().y()
            )

            if img_x is None:
                return

            self.current_polygon[self.selected_point] = [img_x,img_y]

            self.update_display()

    def mouse_release_event(self, event):

        self.dragging = False
        self.selected_point = None

    # =========================================
    # FUNCTIONS
    # =========================================

    def undo(self):

        if self.undo_stack:

            self.redo_stack.append(
                self.current_polygon.copy()
            )

            self.current_polygon = self.undo_stack.pop()

            self.update_display()

    def redo(self):

        if self.redo_stack:

            self.undo_stack.append(
                self.current_polygon.copy()
            )

            self.current_polygon = self.redo_stack.pop()

            self.update_display()

    def clear_all(self):
        """Clear only the current polygon, not saved ones"""
        if self.current_polygon:
            self.current_polygon.clear()
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.update_display()
            self.status_callback("Current polygon cleared")

    def delete_all_regions(self):
        """Delete all saved regions for the current image"""
        if self.polygons and QMessageBox.question(
            self,
            "Delete All Regions",
            f"Delete all {len(self.polygons)} saved region(s) for this image?",
            QMessageBox.Yes | QMessageBox.No
        ) == QMessageBox.Yes:
            # Delete the JSON files
            base_name = os.path.splitext(self.current_image_name)[0]
            for json_file in COORDINATES.glob(f"{base_name}_poly*.json"):
                try:
                    os.remove(json_file)
                    print(f"Deleted {json_file}")
                except Exception as e:
                    print(f"Error deleting {json_file}: {e}")
            
            # Clear in-memory polygons
            self.polygons.clear()
            self.update_display()
            self.status_callback("All saved regions deleted")

    # =========================================
    # SAVE
    # =========================================

    def save_annotation(self):

        try:

            if len(self.current_polygon) < 3:

                QMessageBox.warning(
                    self,
                    "Error",
                    "At least three vertices are required to save a closed region."
                )

                return

            img = _cv2_imread(self.image_path)

            if img is None:

                QMessageBox.critical(
                    self,
                    "Error",
                    "Failed to load image (file missing or path not readable).\n"
                    + self.image_path
                )

                return

            base_name = os.path.splitext(
                self.current_image_name
            )[0]

            # Find the next available polygon ID
            existing_polys = [int(f.stem.split('_poly')[1]) 
                             for f in COORDINATES.glob(f"{base_name}_poly*.json")]
            poly_id = max(existing_polys) + 1 if existing_polys else len(self.polygons)

            pts = np.array(
                self.current_polygon,
                np.int32
            )

            ensure_data_dirs()

            mask_name = f"{base_name}_poly{poly_id}.png"
            viz_name = f"{base_name}_viz{poly_id}.png"
            json_name = f"{base_name}_poly{poly_id}.json"

            mask_path = MASKS / mask_name
            viz_path = POLYGONS / viz_name
            json_path = COORDINATES / json_name

            # MASK
            mask = np.zeros(img.shape[:2], np.uint8)

            cv2.fillPoly(mask, [pts], 255)

            if not _cv2_imwrite(str(mask_path), mask, ".png"):
                raise IOError("Could not write mask PNG")

            # VISUALIZATION
            viz = img.copy()

            cv2.polylines(
                viz,
                [pts],
                True,
                (0,255,255),
                1  # Thinner line in saved visualization
            )

            for x,y in self.current_polygon:

                cv2.circle(
                    viz,
                    (int(x), int(y)),
                    3,  # Smaller points in saved visualization
                    (0,0,255),
                    -1
                )

            if not _cv2_imwrite(str(viz_path), viz, ".png"):
                raise IOError("Could not write visualization PNG")

            # JSON
            with open(json_path, "w", encoding="utf-8") as f:

                json.dump(
                    self.current_polygon,
                    f,
                    indent=4
                )

            # CSV
            file_exists = LABEL_CSV.is_file()

            with open(
                LABEL_CSV,
                "a",
                newline="",
                encoding="utf-8",
            ) as f:

                writer = csv.writer(f)

                if not file_exists:

                    writer.writerow([
                        "image",
                        "polygon",
                        "point",
                        "x",
                        "y"
                    ])

                for i,(x,y) in enumerate(self.current_polygon):

                    writer.writerow([
                        self.current_image_name,
                        poly_id,
                        i + 1,
                        int(round(x)),
                        int(round(y)),
                    ])

            self.polygons.append(
                self.current_polygon.copy()
            )

            self.current_polygon.clear()

            self.update_display()

            QMessageBox.information(
                self,
                "Saved",
                f"Region saved successfully as polygon {poly_id}.\n\n"
                f"Mask:\n{mask_path}\n\n"
                f"Visualization:\n{viz_path}\n\n"
                f"Coordinates (JSON):\n{json_path}\n\n"
                f"CSV append:\n{LABEL_CSV}"
            )

        except Exception as e:

            QMessageBox.critical(
                self,
                "Error",
                str(e)
            )


# =========================================
# MAIN WINDOW
# =========================================

class MainWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "Medical Annotation Tool - MammoX"
        )

        self.setGeometry(100,100,1400,800)

        self.setStyleSheet("""
            QMainWindow{
                background:#0d1117;
            }

            QPushButton{
                background:cyan;
                color:black;
                padding:10px;
                border-radius:8px;
                font-weight:bold;
            }

            QPushButton:hover{
                background:#00b8ff;
            }

            QPushButton:pressed{
                background:#0088cc;
            }

            QListWidget{
                background:#161b22;
                color:white;
                border:1px solid cyan;
            }

            QListWidget::item:hover{
                background:#1f242e;
            }

            QLabel{
                color:white;
            }

            QGroupBox{
                color:cyan;
                border:2px solid cyan;
                margin-top:10px;
                font-weight:bold;
            }
        """)

        central = QWidget()

        self.setCentralWidget(central)

        layout = QHBoxLayout()

        # LEFT PANEL
        left = QWidget()

        left_layout = QVBoxLayout()

        # Image name display label at the top
        self.image_name_label = QLabel("📷 No image loaded")
        self.image_name_label.setStyleSheet("""
            QLabel {
                color: #ff9966;
                background-color: #0d1117;
                padding: 8px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 12px;
                border: 1px solid #ff9966;
            }
        """)
        self.image_name_label.setWordWrap(True)
        
        self.open_btn = QPushButton("📂 OPEN IMAGE")
        self.fit_btn = QPushButton("🔍 FIT TO VIEW")
        self.draw_btn = QPushButton("✏️ DRAW: OFF")
        self.draw_btn.setCheckable(True)
        self.erase_btn = QPushButton("🗑️ ERASE: OFF")
        self.erase_btn.setCheckable(True)

        self.undo_btn = QPushButton("↩️ UNDO")
        self.redo_btn = QPushButton("↪️ REDO")
        self.clear_btn = QPushButton("🧹 CLEAR CURRENT")
        self.save_btn = QPushButton("💾 SAVE REGION")
        self.delete_all_btn = QPushButton("❌ DELETE ALL REGIONS")

        self.coord_list = QListWidget()

        # Add widgets to left panel with image name at the top
        left_layout.addWidget(self.image_name_label)
        left_layout.addWidget(self.open_btn)
        left_layout.addWidget(self.fit_btn)
        left_layout.addWidget(self.draw_btn)
        left_layout.addWidget(self.erase_btn)
        left_layout.addWidget(self.undo_btn)
        left_layout.addWidget(self.redo_btn)
        left_layout.addWidget(self.clear_btn)
        left_layout.addWidget(self.save_btn)
        left_layout.addWidget(self.delete_all_btn)
        left_layout.addWidget(self.coord_list)

        left.setLayout(left_layout)

        left.setMaximumWidth(300)

        # CANVAS
        self.canvas = ScrollableCanvas(
            self.coord_list,
            self.update_status,
            self.image_name_label  # Pass the label to canvas
        )

        # ABOUT BREAST CANCER PANEL
        self.info_panel = QTextBrowser()
        self.info_panel.setOpenExternalLinks(True)
        self.info_panel.setMaximumWidth(320)
        self.info_panel.setStyleSheet(
            """
            QTextBrowser {
                background: #161b22;
                color: #e6edf3;
                border: 1px solid #30363d;
                border-radius: 8px;
                padding: 10px;
                font-size: 13px;
            }
            """
        )
        self.info_panel.setHtml(
            "<h3>🎗️ About Breast Cancer</h3>"
            "<br>"
            "<b>📊 Key Facts:</b>"
            "<ul>"
            "<li>Most common cancer in women worldwide</li>"
            "<li>Early detection saves lives</li>"
            "<li>Can occur in both men and women</li>"
            "</ul>"
            "<br>"
            "<b>⚠️ Common Signs & Symptoms:</b>"
            "<ul>"
            "<li>New lump or mass in breast</li>"
            "<li>Swelling of part of the breast</li>"
            "<li>Skin dimpling or irritation</li>"
            "<li>Nipple pain or retraction</li>"
            "<li>Nipple discharge (not breast milk)</li>"
            "<li>Redness or flaky skin on breast</li>"
            "</ul>"
            "<br>"
            "<b>🔬 Risk Factors:</b>"
            "<ul>"
            "<li>Being female (primary risk factor)</li>"
            "<li>Increasing age</li>"
            "<li>Family history of breast cancer</li>"
            "<li>Genetic mutations (BRCA1, BRCA2)</li>"
            "<li>Personal history of breast cancer</li>"
            "<li>Dense breast tissue</li>"
            "<li>Early menstruation / late menopause</li>"
            "</ul>"
            "<br>"
            "<b>✅ Prevention & Screening:</b>"
            "<ul>"
            "<li>Regular mammograms (age 40-45+)</li>"
            "<li>Monthly breast self-exams</li>"
            "<li>Clinical breast exams by doctor</li>"
            "<li>Maintain healthy weight</li>"
            "<li>Limit alcohol consumption</li>"
            "<li>Regular exercise (150 min/week)</li>"
            "<li>Breastfeeding reduces risk</li>"
            "</ul>"
            "<br>"
            "<b>🏥 Treatment Options:</b>"
            "<ul>"
            "<li>Surgery (lumpectomy, mastectomy)</li>"
            "<li>Radiation therapy</li>"
            "<li>Chemotherapy</li>"
            "<li>Hormone therapy</li>"
            "<li>Targeted therapy</li>"
            "<li>Immunotherapy</li>"
            "</ul>"
            "<br>"
            "<b>📈 Survival Statistics:</b>"
            "<ul>"
            "<li>5-year survival rate: 90% (all stages)</li>"
            "<li>Localized stage survival: 99%</li>"
            "<li>Early detection saves 98% of lives</li>"
            "</ul>"
            "<br>"
            "<b>💡 What This App Does:</b>"
            "<ul>"
            "<li>Annotates regions of interest in mammograms</li>"
            "<li>Helps train AI for breast cancer detection</li>"
            "<li><i>Does NOT provide diagnosis</i></li>"
            "<li><i>Consult a doctor for medical advice</i></li>"
            "</ul>"
            "<br>"
            "<p style='color:#8a9ba8; font-size:11px;'>"
            "<i>Educational information only. Not medical advice.</i>"
            "</p>"
        )

        layout.addWidget(left)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.info_panel)

        central.setLayout(layout)

        # BUTTON EVENTS
        self.open_btn.clicked.connect(self.open_image)

        self.fit_btn.clicked.connect(self.canvas.fit_zoom_to_viewport)

        self.draw_btn.clicked.connect(self.toggle_draw)

        self.erase_btn.clicked.connect(self.toggle_erase)

        self.undo_btn.clicked.connect(
            self.canvas.undo
        )

        self.redo_btn.clicked.connect(
            self.canvas.redo
        )

        self.clear_btn.clicked.connect(
            self.canvas.clear_all
        )

        self.save_btn.clicked.connect(
            self.canvas.save_annotation
        )
        
        self.delete_all_btn.clicked.connect(
            self.canvas.delete_all_regions
        )

        self.statusBar().showMessage("Ready")

    def update_status(self, msg):

        self.statusBar().showMessage(msg)

    def toggle_draw(self):

        if self.draw_btn.isChecked():

            self.draw_btn.setText("✏️ DRAW: ON")
            self.erase_btn.setChecked(False)
            self.erase_btn.setText("🗑️ ERASE: OFF")
            self.canvas.set_erase_mode(False)
            self.canvas.set_draw_mode(True)

        else:

            self.draw_btn.setText("✏️ DRAW: OFF")

            self.canvas.set_draw_mode(False)

    def toggle_erase(self):

        if self.erase_btn.isChecked():

            self.erase_btn.setText("🗑️ ERASE: ON")
            self.draw_btn.setChecked(False)
            self.draw_btn.setText("✏️ DRAW: OFF")
            self.canvas.set_draw_mode(False)
            self.canvas.set_erase_mode(True)

        else:

            self.erase_btn.setText("🗑️ ERASE: OFF")
            self.canvas.set_erase_mode(False)

    def open_image(self):

        path,_ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )

        if path:

            self.canvas.load_image(path)


# =========================================
# LOGIN WINDOW
# =========================================

class LoginWindow(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("MammoAnnotation Login")

        self.setFixedSize(400,500)

        self.setStyleSheet("""
            QWidget{
                background:#f5f7fa;
                font-family:Segoe UI;
            }

            QLineEdit{
                padding:10px;
                border:2px solid #ccc;
                border-radius:10px;
                background:white;
            }

            QLineEdit:focus{
                border-color:#0095f6;
            }

            QPushButton{
                background:#0095f6;
                color:white;
                padding:10px;
                border-radius:10px;
                font-weight:bold;
            }

            QPushButton:hover{
                background:#0077cc;
            }
        """)

        layout = QVBoxLayout()

        layout.setContentsMargins(40,40,40,40)

        logo = QLabel("MammoX")

        logo.setAlignment(Qt.AlignCenter)

        logo.setStyleSheet("""
            font-size:40px;
            color:#0095f6;
            font-weight:bold;
        """)

        self.username = QLineEdit()
        self.username.setPlaceholderText("Username")

        self.email = QLineEdit()
        self.email.setPlaceholderText("Email")

        self.password = QLineEdit()
        self.password.setPlaceholderText("Password")

        self.password.setEchoMode(QLineEdit.Password)

        login_btn = QPushButton("Login")
        register_btn = QPushButton("Register")

        login_btn.clicked.connect(self.login)
        register_btn.clicked.connect(self.register)

        layout.addStretch()

        layout.addWidget(logo)
        layout.addWidget(self.username)
        layout.addWidget(self.email)
        layout.addWidget(self.password)
        layout.addWidget(login_btn)
        layout.addWidget(register_btn)

        layout.addStretch()

        self.setLayout(layout)

    def register(self):

        username = self.username.text().strip()
        email = self.email.text().strip()
        password = self.password.text().strip()

        if not username or not email or not password:

            QMessageBox.warning(
                self,
                "Error",
                "All fields required"
            )

            return

        with open(USER_FILE,"r") as f:

            users = json.load(f)

        for user in users:

            if user["username"] == username:

                QMessageBox.warning(
                    self,
                    "Error",
                    "Username exists"
                )

                return

        users.append({
            "username":username,
            "email":email,
            "password":password
        })

        with open(USER_FILE,"w") as f:

            json.dump(users,f,indent=4)

        QMessageBox.information(
            self,
            "Success",
            "Registration Successful"
        )

    def login(self):

        username = self.username.text().strip()
        email = self.email.text().strip()
        password = self.password.text().strip()

        if not username or not email or not password:

            QMessageBox.warning(
                self,
                "Error",
                "Fill all fields"
            )

            return

        with open(USER_FILE,"r") as f:

            users = json.load(f)

        for user in users:

            if (
                user["username"] == username and
                user["email"] == email and
                user["password"] == password
            ):

                self.main = MainWindow()

                self.main.show()

                self.close()

                return

        QMessageBox.warning(
            self,
            "Failed",
            "Invalid Credentials"
        )


# =========================================
# RUN
# =========================================

if __name__ == "__main__":

    try:

        app = QApplication(sys.argv)

        app.setStyle("Fusion")

        login = LoginWindow()

        login.show()

        sys.exit(app.exec_())

    except Exception as e:

        print("APPLICATION ERROR")
        print(str(e))

        input("Press Enter To Exit")