from collections import deque
from dataclasses import dataclass
from itertools import cycle
from typing import Optional, Tuple, Union

import numpy as np
from vispy import color, scene


@dataclass
class MoveCurveAction:
    curve_indices: list[int]
    offset: tuple[float, float]


class plot:
    """
    Fast plot of many large traces as a single object, using vispy.
    """

    MAX_HL = 12
    BG_DARK = "#222"
    LBL_POS_DEFAULTX = 170
    LBL_POS_DEFAULTY = 40
    LBL_SPACING = 16
    ACTION_HISTORY_SIZE = 20

    def __init__(
        self,
        curves: Optional[Union[list[np.ndarray], np.ndarray]] = None,
        labels: Optional[list[str]] = None,
        bgcolor: str = BG_DARK,
        parent=None,
        dontrun: bool = False,
    ):
        """
        :param icurves: input curve or list of curves
        :param clrmap: (optional) what colormap name from vispy.colormap to use
        """
        self.canvas = scene.SceneCanvas(
            size=(1280, 900),
            position=(200, 200),
            keys="interactive",
            bgcolor=bgcolor,
            parent=parent,
        )

        self.line = None

        self.grid = self.canvas.central_widget.add_grid(spacing=0)

        self.x_axis = scene.AxisWidget(orientation="bottom")
        self.y_axis = scene.AxisWidget(orientation="left")
        self.x_axis.stretch = (1, 0.05)
        self.y_axis.stretch = (0.05, 1)
        self.grid.add_widget(self.x_axis, row=1, col=1)
        self.grid.add_widget(self.y_axis, row=0, col=0)
        self.view = self.grid.add_view(row=0, col=1, camera="panzoom")
        self.x_axis.link_view(self.view)
        self.y_axis.link_view(self.view)

        self.ctrl_pressed = False
        self.shift_pressed = False
        self.alt_pressed = False

        self.canvas.connect(self.on_key_press)
        self.canvas.connect(self.on_key_release)
        self.canvas.connect(self.on_mouse_press)
        self.canvas.connect(self.on_mouse_release)

        # This value stores the mouse position between calls of mouse move as a tuple
        self._init_pos: Optional[Tuple[int, int]] = None
        self.canvas.connect(self.on_mouse_move)

        if curves is not None:
            self.draw_curves(curves, labels)

        self.canvas.show()
        if parent is None and dontrun is False:
            self.canvas.app.run()

        self.action_history = deque([], self.ACTION_HISTORY_SIZE)
        self.move_curve_in_progress = None

    def clear(self):
        if self.line is not None:
            self.line.parent = None
        self.selected_lines = []
        self.hl_labels = []

    def draw_curves(
        self,
        curves: Union[list[np.ndarray], np.ndarray],
        labels: Optional[list[str]] = None,
        clrmap: str = "husl",
    ):
        # cases where a single array needs to be drawn
        if isinstance(curves, list):
            if not (isinstance(curves[0], list) or isinstance(curves[0], np.ndarray)):
                curves = [curves]
        elif isinstance(curves, np.ndarray) and curves.ndim == 1:
            curves = [curves]

        # keep an array of lengths
        self.shapes = [len(curve) for curve in curves]

        # the Line visual requires a vector of X,Y coordinates
        flat_curves = np.empty((0, 2))
        min_len = len(curves[0])
        max_len = 0
        for curve_py in curves:
            curve = np.dstack((np.arange(len(curve_py)), np.array(curve_py)))[0]
            length = len(curve)
            if length < min_len:
                min_len = length
            if length > max_len:
                max_len = length
            flat_curves = np.concatenate((flat_curves, curve))

        if labels is not None:
            assert len(labels) == len(curves)
            self.labels = labels
        else:
            self.labels = [f"0x{i:x}" for i in range(len(curves))]

        # Specify which points are connected
        # Start by connecting each point to its successor
        connect = np.empty((flat_curves.shape[0] - 1, 2), np.int32)
        connect[:, 0] = np.arange(flat_curves.shape[0] - 1)
        connect[:, 1] = connect[:, 0] + 1

        # Prevent vispy from drawing a line between the last point
        # of a curve and the first point of the next curve
        cur_x = len(curves[0])
        for curve in curves[1:]:
            connect[cur_x - 1, 1] = cur_x - 1
            cur_x += len(curve)
        connect[-1, 1] = flat_curves.shape[0] - 1

        nb_traces = len(curves)
        total_size = len(flat_curves)
        self.colors = np.ones((total_size, 3), dtype=np.float32)
        self.backup_colors = np.ones((nb_traces, 3), dtype=np.float32)

        R_p = np.linspace(0.4, 0.4, num=nb_traces)
        G_p = np.linspace(0.5, 0.3, num=nb_traces)
        B_p = np.linspace(0.5, 0.3, num=nb_traces)

        cur_x = 0
        for i, size in enumerate(self.shapes):
            cslice = slice(cur_x, cur_x + size)
            self.colors[cslice, 0] = R_p[i]
            self.colors[cslice, 1] = G_p[i]
            self.colors[cslice, 2] = B_p[i]

            self.backup_colors[i, 0] = R_p[i]
            self.backup_colors[i, 1] = G_p[i]
            self.backup_colors[i, 2] = B_p[i]

            cur_x += size

        self.line = scene.Line(
            pos=flat_curves, color=self.colors, parent=self.view.scene, connect=connect
        )

        self.selected_lines = []
        # To store the lines offsets applied with the "shift"
        self.lines_offset = {}
        self.hl_labels = []
        self.hl_colorset = cycle(
            color.get_colormap(clrmap)[np.linspace(0.0, 1.0, self.MAX_HL)]
        )

        self.view.camera.set_range(
            x=(-1, max_len), y=(flat_curves[:, 1].min(), flat_curves[:, 1].max())
        )

    def run(self):
        self.canvas.app.run()

    def find_closest_line(self, x: int, y: int) -> int:
        # set bounding box where the points will be searched to be
        # at most 1/ratio of the visible area
        # XXX: cons: behaviour changes depending on the window size
        ratio = 20
        camera_state = self.view.camera.get_state()["rect"]
        bounding_x = max(1., camera_state.width / ratio)
        bounding_y = camera_state.height / ratio

        tr = self.canvas.scene.node_transform(self.view.scene)
        # Canvas coordinates of clicked point
        x1, y1, _, _ = tr.map((x, y))
        rx = int(round(x1))
        rbx = int(bounding_x)
        ref_point = np.array([x1, y1], dtype=np.float32)

        def normf(p):
            return np.linalg.norm(ref_point - p)

        # Find closest point, filtering out points whose
        # y-coordinate is outside the bounding box around
        # the clicked point
        cur_x = 0
        found_min = (None, np.uint64(2**64 - 1))
        for i, curvesize in enumerate(self.shapes):
            cur_view = self.line.pos[cur_x : cur_x + curvesize][rx - rbx : rx + rbx]
            cur_view = cur_view[cur_view[:, 1] > (y1 - bounding_y)]
            cur_view = cur_view[cur_view[:, 1] < (y1 + bounding_y)]
            if cur_view.size != 0:
                norms = np.apply_along_axis(normf, 1, cur_view)
                if norms.size != 0:
                    min_norm = norms.min()
                    if min_norm < found_min[1]:
                        found_min = i, min_norm
            cur_x += curvesize

        # this is the index of the closest line
        # or None if there are no point in the
        # defined area
        return found_min[0]

    def on_key_press(self, event):
        if event.key == "Control":
            self.ctrl_pressed = True
        if event.key == "Shift":
            self.shift_pressed = True
            self._init_pos = None
        if event.key == "Alt":
            self.alt_pressed = True
        if event.key == "z":
            if self.ctrl_pressed and len(self.action_history) != 0:
                # For now there is a single type of action
                move_action = self.action_history.pop()
                for curve_index in move_action.curve_indices:
                    self.apply_offset(
                        curve_index, (-move_action.offset[0], -move_action.offset[1])
                    )

    def on_key_release(self, event):
        if event.key == "Control":
            self.ctrl_pressed = False
        if event.key == "Shift":
            self.shift_pressed = False
            if self.move_curve_in_progress is not None:
                self.action_history.append(self.move_curve_in_progress)
                self.move_curve_in_progress = None
        if event.key == "Alt":
            self.alt_pressed = False

    def on_mouse_press(self, event):
        self._init_pos = event.pos

    def restore_offset(self, curves: Optional[int] = None):
        """
        Replace the curves to their initial place, i.e. removes the cumulative offset previously
        applied on them.
        :param curves: The curves' number to reset the offset. If None, uses the selected curves. If
        no curves are selected, restores all the offsets for all curves.
        """
        if curves is None:
            curves = self.selected_lines
            if len(curves) == 0:
                curves = list(self.lines_offset.keys())
        for line_no in curves:
            if line_no in self.lines_offset:
                offset = self.lines_offset[line_no]
                self.apply_offset(line_no, (-offset[0], -offset[1]))
                del self.lines_offset[line_no]

    def apply_offset(self, curve_no: int, offset: Tuple[float, float]):
        """
        Moves the curve for a given (x, y) offset.
        The cumulative offset is stored in internal dictionary in order to be possibly restored.
        :param curve_no: The curve identifier to apply the offset
        :param offset: The displacement to apply to the curve.
        """
        curve_offs = self._find_nth_curve_start(curve_no)
        size = self.shapes[curve_no]
        self.line.pos[curve_offs : curve_offs + size][:, 0] += offset[0]
        self.line.pos[curve_offs : curve_offs + size][:, 1] += offset[1]
        self.line.set_data(pos=self.line.pos)
        curve_offset = self.lines_offset.get(curve_no, [0.0, 0.0])
        self.lines_offset[curve_no] = [
            offset[0] + curve_offset[0],
            offset[1] + curve_offset[1],
        ]

    def on_mouse_move(self, event):
        if self.shift_pressed:
            if len(self.selected_lines) > 0:
                if self._init_pos is None:
                    self._init_pos = event.pos
                # map to screen displacement
                tr = self.canvas.scene.node_transform(self.view.scene)
                x, y, _, _ = tr.map(event.pos)
                init_x, init_y, _, _ = tr.map(self._init_pos)
                delta_x = x - init_x
                delta_y = y - init_y

                offset = (0.0, delta_y) if self.alt_pressed else (delta_x, 0.0)
                for curve_no in self.selected_lines:
                    self.apply_offset(curve_no, offset)

                if self.move_curve_in_progress is None:
                    self.move_curve_in_progress = MoveCurveAction(
                        self.selected_lines, (0.0, 0.0)
                    )
                self.move_curve_in_progress.offset = (
                    self.move_curve_in_progress.offset[0] + offset[0],
                    self.move_curve_in_progress.offset[1] + offset[1],
                )

                self._init_pos = event.pos
                self.canvas.update()

    def on_mouse_release(self, event):
        ## ignore release when moving traces
        if self.shift_pressed:
            return

        x, y = event.pos

        # if released more than 3 pixels away from click (i.e. dragging), ignore
        if not (abs(x - self._init_pos[0]) < 3 and abs(y - self._init_pos[1]) < 3):
            return

        closest_line = self.find_closest_line(x, y)
        if closest_line is None:
            return

        if self.ctrl_pressed:
            self.multiple_select(closest_line)
        else:
            self.single_select(closest_line)

    def _add_label(self, curve_index: int, new_color: str):
        new_label = scene.Text(
            f"{self.labels[curve_index]}",
            color=new_color,
            anchor_x="left",
            parent=self.canvas.scene,
        )
        new_label.pos = (
            self.LBL_POS_DEFAULTX,
            self.LBL_POS_DEFAULTY + self.LBL_SPACING * len(self.hl_labels),
        )
        self.hl_labels.append((curve_index, new_label))

    def _del_label_from_curve_index(self, curve_index: int):
        idx = self._find_label_from_curve_index(curve_index)
        self.hl_labels[idx][1].parent = None
        del self.hl_labels[idx]

        ## redraw text items
        for i, lbl in enumerate(self.hl_labels[idx:]):
            lbl[1].pos = (
                self.LBL_POS_DEFAULTX,
                self.LBL_POS_DEFAULTY + self.LBL_SPACING * (idx + i),
            )

    def _find_label_from_curve_index(self, curve_index: int) -> int:
        return list(map(lambda x: x[0], self.hl_labels)).index(curve_index)

    def _find_nth_curve_start(self, n: int) -> int:
        return sum(self.shapes[:n])

    def _set_curve_color(self, n: int, new_color):
        size = self.shapes[n]
        x = self._find_nth_curve_start(n)
        self.colors[x : x + size] = np.repeat(new_color.rgb, size, axis=0)

    def _restore_nth_curve_color(self, n: int):
        size = self.shapes[n]
        x = self._find_nth_curve_start(n)
        self.colors[x : x + size] = np.repeat([self.backup_colors[n]], size, axis=0)

    def single_select(self, curve_index: int):
        # Unselect previously highlighted curves
        for line in self.selected_lines:
            self._restore_nth_curve_color(line)

        # Delete labels
        for lbl in self.hl_labels:
            lbl[1].parent = None

        self.hl_labels = []

        # Display its index/label
        new_color = next(self.hl_colorset)  # Pick a new color
        self._add_label(curve_index, new_color)
        self.selected_lines = [curve_index]  # Add this curve to the selected batch
        self._set_curve_color(curve_index, new_color)  # Set its new color

        self.line.set_data(color=self.colors)  # Update colors

    def multiple_select(self, curve_index: int):
        if curve_index in self.selected_lines:
            # Clicked on already selected curve
            # so we cancel selection
            # - erase corresponding text
            # - restore original color of previously selected line
            self._del_label_from_curve_index(curve_index)
            self._restore_nth_curve_color(curve_index)
            self.selected_lines.remove(curve_index)
        else:
            new_color = next(self.hl_colorset)
            self._add_label(curve_index, new_color)
            self.selected_lines.append(curve_index)
            self._set_curve_color(curve_index, new_color)

        self.line.set_data(color=self.colors)

    def add_horizontal_ruler(self, y: float) -> scene.visuals.InfiniteLine:
        """Add a single light grey horizontal line at 'y' on the canvas."""
        return scene.visuals.InfiniteLine(
            pos=float(y),
            color=color.Color("#ddd", alpha=0.8).rgba,
            parent=self.view.scene,
            vertical=False,
        )

    def add_horizontal_band(self, y0: float, y1: float) -> scene.visuals.Polygon:
        """Add a horizontal band (rectangle) covering 'y0' to 'y1' on the canvas."""
        size = max(self.shapes)
        coords = [(0, y0), (0, y1), (size, y1), (size, y0)]
        return scene.visuals.Polygon(
            coords, color=color.Color("#ddd", alpha=0.1), parent=self.view.scene
        )

    def add_vertical_ruler(self, x: float) -> scene.visuals.InfiniteLine:
        """Add a single light grey vertical line at position 'x' on the canvas."""
        return scene.visuals.InfiniteLine(
            pos=float(x),
            color=color.Color("#ddd", alpha=0.8).rgba,
            parent=self.view.scene,
            vertical=True,
        )


if __name__ == "__main__":
    N = 50
    a = [
        i / 10 * np.sin(np.linspace(0.0 + i / 10, 10.0 + i / 10, num=i * 1000))
        for i in range(1, N)
    ]
    v = plot(a, dontrun=True)
    v.multiple_select(4)
    v.multiple_select(7)
    v.add_horizontal_ruler(2.2)
    v.add_horizontal_band(1.0, -1.0)
    v.add_vertical_ruler(10)
    v.run()
