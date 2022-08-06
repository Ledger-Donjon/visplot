
from itertools import cycle
from typing import Optional, Tuple
import numpy as np
from vispy import scene
from vispy import color
from vispy import util 


class plot:
    """
    Fast plot of many large traces as a single object, using vispy. 
    """
    MAX_HL = 12 
    BG_DARK = "#222"
    LBL_POS_DEFAULTX = 170
    LBL_POS_DEFAULTY = 40
    LBL_SPACING = 16 

    def __init__(
        self, curves=None, labels=None, bgcolor=BG_DARK, parent=None, dontrun=False
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
        self.view = self.grid.add_view(row=0, col=1, camera="panzoom")

        self.x_axis = scene.AxisWidget(orientation="bottom")
        self.y_axis = scene.AxisWidget(orientation="left")
        self.x_axis.stretch = (1, 0.05)
        self.y_axis.stretch = (0.05, 1)
        self.grid.add_widget(self.x_axis, row=1, col=1)
        self.grid.add_widget(self.y_axis, row=0, col=0)
        self.x_axis.link_view(self.view)
        self.y_axis.link_view(self.view)

        self.ctrl_pressed = False 
        self.shift_pressed = False 
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

    def clear(self):
        if self.line is not None:
            self.line.parent = None
        self.selected_lines = [] 
        self.hl_labels = []

    def draw_curves(self, curves_, labels=None, clrmap="husl"):
        curves = np.array(curves_)

        self.shape_ = curves.shape

        if labels is not None:
            assert len(labels) == self.shape_[0]
            self.labels = labels
        else:
            self.labels = [f"0x{i:x}" for i in range(self.shape_[0])]

        if len(curves.shape) == 1:
            ## Single curve
            curves = np.array([curves])

        nb_traces, size = curves.shape

        # the Line visual requires a vector of X,Y coordinates
        xy_curves = np.dstack((np.tile(np.arange(size), (nb_traces, 1)), curves))

        # Specify which points are connected
        # Start by connecting each point to its successor
        connect = np.empty((nb_traces * size - 1, 2), np.int32)
        connect[:, 0] = np.arange(nb_traces * size - 1)
        connect[:, 1] = connect[:, 0] + 1

        # Prevent vispy from drawing a line between the last point
        # of a curve and the first point of the next curve
        for i in range(size, nb_traces * size, size):
            connect[i - 1, 1] = i - 1

        self.colors = np.ones((nb_traces*size,3),dtype=np.float32)
        self.backup_colors = np.ones((nb_traces,3),dtype=np.float32)

        R_p = np.linspace(0.4,0.4,num=nb_traces)
        G_p = np.linspace(0.5,0.3,num=nb_traces)
        B_p = np.linspace(0.5,0.3,num=nb_traces)

        self.colors[:,0] = np.repeat(R_p,size)
        self.colors[:,1] = np.repeat(G_p,size)
        self.colors[:,2] = np.repeat(B_p,size)

        self.backup_colors[:,0] = R_p
        self.backup_colors[:,1] = G_p
        self.backup_colors[:,2] = B_p

        self.line = scene.Line(pos=xy_curves, color=self.colors, parent=self.view.scene, connect=connect)

        self.selected_lines = [] 
        self.hl_labels = []
        self.hl_colorset = cycle(color.get_colormap(clrmap)[np.linspace(0.0, 1.0, self.MAX_HL)])

        self.view.camera.set_range(x=(-1, size), y=(curves.min(), curves.max()))

    def run(self):
        self.canvas.app.run()

    def find_closest_line(self, x, y):
        radius = 50

        tr = self.canvas.scene.node_transform(self.view.scene)
        # Canvas coordinates of clicked point
        x1,y1,_,_ = tr.map((x,y))
        # Canvas coordinates of upper right corner of bounding box
        # containing clicked point
        x2,max_y,_,_ = tr.map((x+radius,y+radius))

        _,min_y,_,_ = tr.map((x, y-radius))
        min_y, max_y = min(min_y, max_y), max(min_y, max_y)

        # Gather all segments left and right of the clicked point
        rx = int(round(x1))
        tab = self.line.pos[:,rx-radius:rx+radius]

        # Find closest point, filtering out points whose 
        # y-coordinate is outside the bounding box around
        # the clicked point
        max_norm = 1000000 
        imin = None 
        f = np.array([x1,y1], dtype=np.float32)
        for i,s in enumerate(tab):
            for p in s:
                if min_y<p[1]<max_y:
                    t = np.linalg.norm(f-p)
                    if t < max_norm:
                        max_norm = t
                        imin = i

        # this is the index of the closest line
        # or None if there are no point in the
        # defined area
        return imin 

    def on_key_press(self, event):
        if event.key == 'Control':
            self.ctrl_pressed = True
        if event.key == 'Shift':
            self.shift_pressed = True
            self._init_pos = None

    def on_key_release(self, event):
        if event.key == 'Control':
            self.ctrl_pressed = False 
        if event.key == 'Shift':
            self.shift_pressed = False 

    def on_mouse_press(self, event):
        self._init_pos = event.pos

    def on_mouse_move(self,event):
        if self.shift_pressed == True:
            if len(self.selected_lines) > 0:
                if self._init_pos is None:
                    self._init_pos = event.pos
                # map to screen displacement
                tr = self.canvas.scene.node_transform(self.view.scene)
                x,y,_,_ = tr.map(event.pos)
                init_x,init_y,_,_ = tr.map(self._init_pos)
                delta_x = int(x - init_x)
                for l in self.selected_lines:
                    self.line.pos[l][:,1] = np.roll(self.line.pos[l][:,1],delta_x)
                self._init_pos = event.pos
                self.canvas.update()

    def on_mouse_release(self, event):
        ## ignore release when moving traces
        if self.shift_pressed:
            return

        x,y = event.pos

        # if released more than 3 pixels away from click (i.e. dragging), ignore
        if not (abs(x-self._init_pos[0])<3 and abs(y-self._init_pos[1])<3):
            return

        closest_line = self.find_closest_line(x,y)
        if closest_line is None:
            return

        if self.ctrl_pressed:
            self.multiple_select(closest_line)
        else:
            self.single_select(closest_line)
                
    def _add_label(self, curve_index, new_color):
        new_label = scene.Text(f"{self.labels[curve_index]}", color=new_color, anchor_x = 'left', parent=self.canvas.scene)
        new_label.pos = self.LBL_POS_DEFAULTX, self.LBL_POS_DEFAULTY + self.LBL_SPACING * len(self.hl_labels)
        self.hl_labels.append((curve_index, new_label))

    def _del_label_from_curve_index(self, curve_index):
        idx = self._find_label_from_curve_index(curve_index)
        self.hl_labels[idx][1].parent = None
        del self.hl_labels[idx]

        ## redraw text items
        for i, lbl in enumerate(self.hl_labels[idx:]):
            lbl[1].pos = self.LBL_POS_DEFAULTX, self.LBL_POS_DEFAULTY + self.LBL_SPACING * (idx+i)

    def _find_label_from_curve_index(self, curve_index):
        return list(map(lambda x:x[0], self.hl_labels)).index(curve_index)

    def _set_curve_color(self,n, new_color):
        _, S = self.shape_
        a = n * S
        self.colors[a:a+S] = np.repeat(new_color.rgb, S, axis=0)

    def _restore_nth_curve_color(self,n):
        _, S = self.shape_
        nnx = n * S
        self.colors[nnx:nnx+S] = np.repeat([self.backup_colors[n]], S, axis=0)

    def single_select(self, curve_index):
        # Unselect previously highlighted curves
        for line in self.selected_lines:
            self._restore_nth_curve_color(line)

        # Delete labels
        for lbl in self.hl_labels:
            lbl[1].parent = None

        self.hl_labels = []

        # Display its index/label
        new_color = next(self.hl_colorset)               # Pick a new color
        self._add_label(curve_index, new_color)
        self.selected_lines = [curve_index]              # Add this curve to the selected batch
        self._set_curve_color(curve_index, new_color)    # Set its new color

        self.line.set_data(color=self.colors)            # Update colors

    def multiple_select(self, curve_index):
        if curve_index in self.selected_lines:
            # Clicked on already selected curve
            # so we cancel selection
            # - erase corresponding text
            # - restore original color of previously selected line
            self._del_label_from_curve_index(curve_index)
            self._restore_nth_curve_color(curve_index)
            self.selected_lines.remove(curve_index)
        else:
            N,S = self.shape_
            new_color = next(self.hl_colorset)
            self._add_label(curve_index, new_color)
            self.selected_lines.append(curve_index)
            self._set_curve_color(curve_index, new_color)
        
        self.line.set_data(color=self.colors)

if __name__ == "__main__":
    N = 50
    a = [i/10*np.sin(np.linspace(0.0+i/10,10.0+i/10,num=2000)) for i in range(N)]
    v = plot(a, dontrun=True)
    v.multiple_select(4)
    v.multiple_select(7)
    v.run()
