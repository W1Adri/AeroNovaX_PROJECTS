import sys
import math
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout,
    QHBoxLayout, QLineEdit, QLabel, QMessageBox
)
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor
from PyQt5.QtCore import Qt, QPoint, QTimer

class Drone:
    """Drone class representing each drone in the simulation."""
    def __init__(self, drone_id, path, speed=5, radius=50):
        self.id = drone_id
        self.path = path
        self.speed = speed
        self.radius = radius
        self.current_index = 0
        self.x, self.y = self.path[0]
        self.finished = False
    
    def move(self):
        if self.finished:
            return
        
        target_x, target_y = self.path[self.current_index]
        dx = target_x - self.x
        dy = target_y - self.y
        distance = math.hypot(dx, dy)
        if distance == 0:
            self.current_index += 1
            if self.current_index >= len(self.path):
                self.finished = True
            return
        step = min(self.speed, distance)
        self.x += step * dx / distance
        self.y += step * dy / distance
        if distance <= self.speed:
            self.current_index += 1
            if self.current_index >= len(self.path):
                self.finished = True

class Canvas(QWidget):
    """Canvas class for drawing areas and visualizing drones."""
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: white;")
        self.areas = []  # List of restricted areas as lists of (x, y) tuples
        self.current_drawing = []
        self.drawing_enabled = False
        self.area_color = QColor(255, 0, 0, 100)  # Semi-transparent red for restricted areas
        self.drones = []  # List of Drone instances

    def enable_drawing(self):
        self.drawing_enabled = True
        self.current_drawing = []
        QMessageBox.information(self, "Drawing Mode", "Left-click to add points. Right-click to finish.")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw canvas boundaries
        pen = QPen(Qt.black, 3)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(0, 0, self.width()-1, self.height()-1)

        # Draw restricted areas
        for area in self.areas:
            brush = QBrush(self.area_color)
            pen = QPen(Qt.black, 2)
            painter.setPen(pen)
            painter.setBrush(brush)
            polygon = [QPoint(x, y) for x, y in area]
            painter.drawPolygon(*polygon)

        # Draw current drawing
        if self.current_drawing:
            pen = QPen(Qt.blue, 2, Qt.DashLine)
            painter.setPen(pen)
            for i in range(len(self.current_drawing) - 1):
                painter.drawLine(self.current_drawing[i], self.current_drawing[i + 1])

        # Draw drones and their vision radius
        for drone in self.drones:
            # Draw vision radius
            pen = QPen(QColor(0, 255, 0, 100), 1, Qt.DashLine)  # Green dashed line
            brush = QBrush(QColor(0, 255, 0, 50))  # Semi-transparent green
            painter.setPen(pen)
            painter.setBrush(brush)
            painter.drawEllipse(QPoint(int(drone.x), int(drone.y)), drone.radius, drone.radius)

            # Draw drone
            pen = QPen(Qt.black, 1)
            brush = QBrush(QColor(0, 0, 255))  # Blue drone
            painter.setPen(pen)
            painter.setBrush(brush)
            painter.drawEllipse(QPoint(int(drone.x), int(drone.y)), 5, 5)

    def mousePressEvent(self, event):
        if self.drawing_enabled:
            if event.button() == Qt.LeftButton:
                self.current_drawing.append(event.pos())
                self.update()
            elif event.button() == Qt.RightButton:
                if len(self.current_drawing) > 2:
                    points = [(point.x(), point.y()) for point in self.current_drawing]
                    self.areas.append(points)
                self.current_drawing = []
                self.drawing_enabled = False
                self.update()

    def add_area_by_coords(self, coords):
        self.areas.append(coords)
        self.update()

    def update_drones(self, drones):
        self.drones = drones
        self.update()

class MainWindow(QMainWindow):
    """Main window of the Drone Swarm Simulation application."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone Swarm Simulation")
        self.setGeometry(100, 100, 1000, 700)

        # Central Widget and Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Left Panel for Controls
        control_panel = QVBoxLayout()

        # Draw Area Button
        self.draw_button = QPushButton("Draw Restricted Area")
        self.draw_button.clicked.connect(self.canvas_enable_drawing)
        control_panel.addWidget(self.draw_button)

        # Coordinate Input
        self.coord_input = QLineEdit()
        self.coord_input.setPlaceholderText("x1,y1;x2,y2;...")
        control_panel.addWidget(self.coord_input)

        # Add Area Button
        self.add_area_button = QPushButton("Add Restricted Area by Coordinates")
        self.add_area_button.clicked.connect(self.add_area_via_coords)
        control_panel.addWidget(self.add_area_button)

        # Time Interval Input
        self.time_interval_input = QLineEdit()
        self.time_interval_input.setPlaceholderText("Time Interval (seconds)")
        control_panel.addWidget(self.time_interval_input)

        # Drone Speed Input
        self.drone_speed_input = QLineEdit()
        self.drone_speed_input.setPlaceholderText("Drone Speed (pixels/second)")
        control_panel.addWidget(self.drone_speed_input)

        # Drone Radius Input
        self.drone_radius_input = QLineEdit()
        self.drone_radius_input.setPlaceholderText("Drone Vision Radius (pixels)")
        control_panel.addWidget(self.drone_radius_input)

        # Optimize Button
        self.optimize_button = QPushButton("Optimize Drone Coverage")
        self.optimize_button.clicked.connect(self.optimize_drone_coverage)
        control_panel.addWidget(self.optimize_button)

        # Spacer
        control_panel.addStretch()

        # Drone Information Label
        self.drone_info_label = QLabel("Drone Information:")
        control_panel.addWidget(self.drone_info_label)

        # Drone List Label
        self.drone_list_label = QLabel("")
        control_panel.addWidget(self.drone_list_label)

        # Add control panel to main layout
        main_layout.addLayout(control_panel, 1)  # Stretch factor 1

        # Canvas
        self.canvas = Canvas()
        main_layout.addWidget(self.canvas, 4)  # Stretch factor 4

        # Drone Simulation Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.simulate_drones)
        self.timer.start(30)  # Update every 30 milliseconds for smoother animation

        # Initialize drones
        self.drones = []
        self.canvas.update_drones(self.drones)
        self.update_drone_info()

    def canvas_enable_drawing(self):
        self.canvas.enable_drawing()

    def add_area_via_coords(self):
        coord_text = self.coord_input.text()
        try:
            coords = []
            points = coord_text.split(';')
            for point in points:
                x, y = map(float, point.strip().split(','))
                coords.append((x, y))
            self.canvas.add_area_by_coords(coords)
            self.coord_input.clear()
        except Exception as e:
            QMessageBox.warning(self, "Input Error", f"Invalid format: {e}")

    def simulate_drones(self):
        for drone in self.drones:
            drone.move()
        self.canvas.update_drones(self.drones)
        self.update_drone_info()

    def update_drone_info(self):
        info = ""
        for drone in self.drones:
            info += (f"Drone {drone.id}: Position=({int(drone.x)}, {int(drone.y)}), "
                     f"Radius={drone.radius}, Speed={drone.speed}\n")
        self.drone_list_label.setText(info)

    def optimize_drone_coverage(self):
        try:
            time_interval = float(self.time_interval_input.text())
            drone_speed = float(self.drone_speed_input.text())
            drone_radius = float(self.drone_radius_input.text())
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid numerical values.")
            return

        # Clear existing drones
        self.drones = []

        # Exclude restricted areas
        restricted_areas = self.canvas.areas

        # Define the area to cover (entire canvas minus restricted areas)
        area_width = self.canvas.width()
        area_height = self.canvas.height()

        # Create grid
        cell_size = drone_radius / math.sqrt(2)  # Ensure overlap
        rows = int(math.ceil(area_height / cell_size))
        cols = int(math.ceil(area_width / cell_size))

        # Generate grid cells and mark obstacles
        grid = []
        for row in range(rows):
            grid_row = []
            for col in range(cols):
                x = col * cell_size + cell_size / 2
                y = row * cell_size + cell_size / 2
                if x >= area_width or y >= area_height:
                    grid_row.append(1)  # Mark as obstacle
                elif self.is_point_in_restricted_areas((x, y), restricted_areas):
                    grid_row.append(1)  # Mark as obstacle
                else:
                    grid_row.append(0)  # Free cell
            grid.append(grid_row)

        # Generate coverage path using Wavefront Algorithm
        start_cell = (0, 0)
        while grid[start_cell[1]][start_cell[0]] != 0:
            if start_cell[0] < cols - 1:
                start_cell = (start_cell[0] + 1, start_cell[1])
            elif start_cell[1] < rows - 1:
                start_cell = (0, start_cell[1] + 1)
            else:
                QMessageBox.warning(self, "No Starting Point", "No free starting point found.")
                return

        coverage_path = self.wavefront_coverage(grid, start_cell)

        if not coverage_path:
            QMessageBox.warning(self, "No Coverage Path", "Could not generate a coverage path.")
            return

        # Convert cell indices back to positions
        coverage_path_positions = []
        for col, row in coverage_path:
            x = col * cell_size + cell_size / 2
            y = row * cell_size + cell_size / 2
            coverage_path_positions.append((x, y))

        # Calculate total path length
        total_path_length = self.calculate_path_length(coverage_path_positions)

        # Calculate maximum path length per drone
        max_path_length = drone_speed * time_interval

        if max_path_length <= 0:
            QMessageBox.warning(self, "Invalid Parameters", "Drone speed and time interval must be positive.")
            return

        # Calculate number of drones needed
        num_drones = int(math.ceil(total_path_length / max_path_length))

        if num_drones == 0:
            num_drones = 1  # At least one drone is needed

        # Split the coverage path among drones
        paths_per_drone = []
        path_length = 0
        current_path = []
        for i in range(len(coverage_path_positions)-1):
            current_path.append(coverage_path_positions[i])
            path_length += math.hypot(
                coverage_path_positions[i+1][0]-coverage_path_positions[i][0],
                coverage_path_positions[i+1][1]-coverage_path_positions[i][1])
            if path_length >= max_path_length:
                paths_per_drone.append(current_path)
                current_path = []
                path_length = 0
        if current_path:
            current_path.append(coverage_path_positions[-1])
            paths_per_drone.append(current_path)

        # Create drones with assigned paths
        drone_id = 1
        for path in paths_per_drone:
            drone = Drone(drone_id=drone_id, path=path, speed=drone_speed, radius=drone_radius)
            self.drones.append(drone)
            drone_id += 1

        self.canvas.update_drones(self.drones)
        self.update_drone_info()

    def wavefront_coverage(self, grid, start_cell):
        """
        Perform Wavefront Coverage to generate a path covering all free cells without crossing obstacles.

        Parameters:
            grid (list of lists): Grid representation with 0 for free cells and 1 for obstacles.
            start_cell (tuple): Starting cell coordinates (col, row).

        Returns:
            list: List of cell coordinates forming the coverage path.
        """
        rows = len(grid)
        cols = len(grid[0])
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        path = []

        def is_valid(cell):
            col, row = cell
            return 0 <= col < cols and 0 <= row < rows and grid[row][col] == 0 and not visited[row][col]

        stack = [start_cell]
        while stack:
            cell = stack.pop()
            col, row = cell
            if not is_valid(cell):
                continue
            visited[row][col] = True
            path.append(cell)
            # Add neighboring cells in order (up, right, down, left)
            neighbors = [(col, row-1), (col+1, row), (col, row+1), (col-1, row)]
            for neighbor in neighbors:
                if is_valid(neighbor):
                    stack.append(neighbor)
        return path

    def is_point_in_restricted_areas(self, point, restricted_areas):
        x, y = point
        for area in restricted_areas:
            if self.point_in_polygon(x, y, area):
                return True
        return False

    def point_in_polygon(self, x, y, polygon):
        num = len(polygon)
        j = num - 1
        c = False
        for i in range(num):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and \
                    (x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi):
                c = not c
            j = i
        return c

    def calculate_path_length(self, path):
        length = 0
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            length += math.hypot(x2 - x1, y2 - y1)
        return length

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
