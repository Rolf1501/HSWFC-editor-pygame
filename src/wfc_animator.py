from animator import Animator
from communicator import Communicator
from coord import Coord
from grid import Grid
from toy_examples import ToyExamples as Toy, Example
from util_data import Cardinals as C
from wfc import WFC
from direct.gui.DirectGui import *
import tkinter as tk
from tkinter import ttk
from json_parser import JSONParser as J

from panda3d.core import (
    NodePath,
    load_prc_file,
    WindowProperties,
    Camera,
    OrthographicLens,
    PerspectiveLens,
    MouseWatcher,
    MouseWatcherRegion,
    DisplayRegion,
    LQuaternionf,
    LVecBase3,
)
from util_data import Colour
from queue import Queue as Q
from numpy.random import random
from time import time
from panda3d.core import (
    CollisionHandlerQueue,
    CollisionTraverser,
    CollisionNode,
    CollisionRay,
    GeomNode,
)

comm = Communicator()


class WFCAnimator(Animator):
    def __init__(self, grid_w=20, grid_h=20, grid_d=5, unit_dims=Coord(1, 1, 1)):
        self.init_camera_params(grid_w, grid_h, grid_d, unit_dims)

        # Keep reference from canvas to model.
        # self.canvas = Grid(grid_w, grid_h, grid_d, default_fill_value=-1)
        self.colours = Grid(grid_w, grid_h, grid_d, default_fill_value=Q())
        self.info_grid = Grid(grid_w, grid_h, grid_d, default_fill_value=None)

        self.window_width, self.window_height = 800, 800
        self.props = WindowProperties()
        self.props.setParentWindow(self.frame.winfo_id())
        self.props.setOrigin(0, 0)
        self.props.setSize(self.window_width, self.window_height)
        self.frame.geometry(f"{self.window_width}x{self.window_height}")
        self.win.request_properties(self.props)

        self.mouse_watchers: list[MouseWatcher] = []

        self.aspect_ratio = self.get_aspect_ratio()

        self.manual()
        self.reset_models()

        self.delta_acc = 0
        self.delta_collapse = 0

        self.hover_mode = False
        self.examples = J().read_examples()
        self.terminals = J().read_terminals()
        self.active_example = next(iter(self.examples))
        self.grid_extent = Coord(grid_w, grid_h, grid_d)

        self.button_width, self.button_height = 15, 1

        display_region = self.camNode.get_display_region(0)
        display_region.set_active(0)

        dr1, self.cam1 = self.create_display_region((0, 1, 0, 0.2))
        dr2, self.cam2 = self.create_display_region((0, 1, 0.2, 1))

        self.render2 = NodePath("render2")
        self.render2_lookat_pos = Coord(0, 0, 0)
        self.cam1.reparent_to(self.render2)

        self.cam2.reparent_to(self.render)
        dummy_node = NodePath("dummy")
        dummy_node.reparent_to(self.render)
        self.cam2.reparent_to(dummy_node)
        model = self.loader.loadModel("parts/cube.egg")
        self.cam2.set_pos(10, 10, 10)
        self.cam2.look_at((0, 0, 0), (0, 1, 0))

        self.attach_display_region_mouse_watcher(dr1)
        self.attach_display_region_mouse_watcher(dr2)

        self.disable_mouse()

        model.reparent_to(self.render2)

        self.init_wfc()

        # self.init_info_grid()
        self.init_collider()
        self.init_key_events()
        self.init_mouse_events()
        self.init_tasks()
        self.init_axes()
        self.init_gui()

        self.frame.mainloop()

    def attach_display_region_mouse_watcher(self, display_region: DisplayRegion):
        mw = MouseWatcher(name=f"{len(self.mouse_watchers)}")
        mw.set_display_region(display_region)
        self.mouseWatcher.get_parent().attach_new_node(mw)
        self.mouse_watchers.append(mw)

    def create_display_region(self, dims):
        self.make_camera(self.win, displayRegion=dims)
        dr = self.win.get_display_region(len(self.win.get_display_regions()) - 1)
        print(len(self.camList))
        cam: NodePath = self.camList[len(self.camList) - 1]
        cam.node().get_lens().set_aspect_ratio(
            float(dr.get_pixel_width()) / float(dr.get_pixel_height())
        )
        return dr, cam

    def align_right(self, element_width, frame_width):
        return frame_width * 0.5 - element_width

    def align_bottom(self, element_height, frame_height):
        return frame_height * 0.5 - element_height

    def init_wfc(self):
        example: Example = self.examples[self.active_example]
        example_terminals = {t: self.terminals[t] for t in example.terminal_ids}
        self.wfc = WFC(
            example_terminals,
            example.adjacencies,
            self.grid_extent,
            default_weights=example.default_weights,
        )

    @staticmethod
    def toggle_widget_visibility(widget, position, *args):
        if widget.winfo_ismapped():
            widget.forget()
            return False
        else:
            position(*args)
            return True

    def init_gui(self):
        self.style = ttk.Style()
        self.style.configure(
            "Menu.TFrame", background="red", padx=5, pady=5, fill="both"
        )
        self.init_hide_menu()

        self.menu_frame = ttk.Frame(self.frame)
        self.position_menu()
        self.menu_tabs = ttk.Notebook(self.menu_frame)
        self.menu_tabs.pack()
        self.adjacency_menu_frame = ttk.Frame(
            master=self.menu_tabs, width=200, style="Menu.TFrame"
        )
        self.general_menu_frame = ttk.Frame(
            master=self.menu_tabs, width=200, style="Menu.TFrame"
        )
        self.menu_tabs.add(self.general_menu_frame, text="General")
        self.menu_tabs.add(self.adjacency_menu_frame, text="Constraints")

        self.init_gui_example_options()
        self.init_gui_extent()
        self.init_gui_adjacency_viewer()

        self.init_camera_controls()

    def init_camera_controls(self):
        self.prev_mouse_x, self.prev_mouse_y = 0, 0
        self.disable_mouse()
        self.accept("mouse3", self.start_panning)
        self.accept("mouse3-up", self.stop_panning)
        self.accept("mouse1", self.start_rotating)
        self.accept("mouse1-up", self.stop_rotating)

    def get_delta_mouse_pos(self, mouse_x, mouse_y):
        return self.prev_mouse_x - mouse_x, self.prev_mouse_y - mouse_y

    def prep_camera_transformation(self):
        for i in self.mouse_watchers:
            if i.has_mouse():
                self.prev_mouse_x, self.prev_mouse_y = i.get_mouse()
                self.active_cam = i.get_display_region().get_camera()
                break

    def start_panning(self):
        self.prep_camera_transformation()
        self.panning_task = self.task_mgr.add(self.pan_camera, "pan_camera")

    def stop_panning(self):
        self.task_mgr.remove(self.panning_task)

    def pan_camera(self, task):
        mw: MouseWatcher = self.mouseWatcherNode
        strength = 20
        if mw.has_mouse():
            mouse_pos = mw.get_mouse()
            delta_x, delta_y = self.get_delta_mouse_pos(*mouse_pos)
            self.active_cam.set_pos(
                self.active_cam, delta_x * strength, delta_y * strength, 0
            )

            self.prev_mouse_x, self.prev_mouse_y = mouse_pos
        return task.cont

    def start_rotating(self):
        self.prep_camera_transformation()
        self.rotating_task = self.task_mgr.add(self.rotate_camera, "rotate_camera")

    def stop_rotating(self):
        self.task_mgr.remove(self.rotating_task)

    def rotate_camera(self, task):
        mw: MouseWatcher = self.mouseWatcherNode
        strength = 20
        if mw.has_mouse():
            mouse_pos = mw.get_mouse()
            delta_x, delta_y = self.get_delta_mouse_pos(*mouse_pos)

            prev_pos = self.active_cam.get_pos()
            self.active_cam.set_pos(
                self.active_cam, delta_x * strength, 0, delta_y * strength
            )

            cam_pos = self.active_cam.get_pos()
            up = LVecBase3(0, 1, 0)

            eps_z = 0.001
            dotp = cam_pos.dot(up)
            if abs(dotp) >= eps_z:
                self.prev_mouse_x, self.prev_mouse_y = mouse_pos
                self.active_cam.look_at(self.render2_lookat_pos, up)
            else:
                self.active_cam.set_pos(prev_pos)
        return task.cont

    def init_gui_adjacency_viewer(self):
        self.adj_canvas = tk.Canvas(master=self.adjacency_menu_frame)
        self.adj_canvas.configure(scrollregion=self.adj_canvas.bbox("all"))
        self.style.configure("Constraint.Vertical.TScrollbar", fill="y")
        self.adj_scroll = ttk.Scrollbar(
            master=self.adj_canvas,
            orient=tk.VERTICAL,
            command=self.adj_canvas.yview,
            style="Constraint.Vertical.TScrollbar",
        )
        self.adj_scroll.grid(row=0, column=1, sticky=tk.NSEW)

        self.adj_canvas.configure(yscrollcommand=self.adj_scroll.set)
        self.adj_canvas.pack()

        part_adjacencies = self.wfc.adj_matrix.get_all_part_adjacencies_as_id()
        i = 0

        self.adj_listing = ttk.Frame(self.adj_canvas)
        self.adj_listing.grid(row=0, column=0, sticky=tk.EW)
        label = ttk.Label(
            self.adj_listing,
            text=f"Showing adjacency constraints in \n{self.active_example}",
        )
        label.grid(row=i, column=0, sticky=tk.EW)
        i += 1
        for part_index in part_adjacencies:
            part = part_adjacencies[part_index]
            label = ttk.Label(
                self.adj_listing,
                text=f"Adjacencies of Part: {part_index}",
            )
            label.grid(row=i, column=0, sticky=tk.EW)
            i += 1
            for offset in part:
                adjs = part[offset]
                lab = ttk.Label(
                    self.adj_listing, text=f"\t{C(offset)}: {str(adjs)[1:-1]}"
                )
                lab.grid(row=i, column=0, sticky=tk.EW)
                i += 1

    def position_menu(self):
        self.menu_frame.pack(anchor=tk.E, expand=1, fill="y")

    def init_hide_menu(self):
        def toggle_menu():
            self.toggle_menu.config(
                text=f"{'Hide' if self.toggle_widget_visibility(self.menu_frame, self.position_menu) else 'Show'} menu"
            )

        self.toggle_menu = ttk.Button(
            master=self.frame, text="Hide menu", command=toggle_menu
        )
        self.toggle_menu.pack(anchor=tk.NE)

    def init_gui_example_options(self):
        self.option_frame = ttk.Frame(master=self.general_menu_frame)
        self.option_frame.pack(anchor=tk.W)
        options: list[ttk.Radiobutton] = []
        var = tk.StringVar()
        i = 0
        for example in self.examples.keys():
            button = ttk.Radiobutton(
                master=self.option_frame,
                text=str(example),
                variable=var,
                value=str(example),
            )
            i += 1

            options.append(button)
            button.pack(anchor=tk.W)
        options[0].invoke()
        self.style.configure("TButton", height=self.button_height)
        button = ttk.Button(
            master=self.option_frame,
            text="Apply selection",
            command=lambda: self.apply_example_select(var.get()),
        )
        button.pack()

    def init_gui_extent(self):
        def make_scale(text):
            s = tk.Scale(
                master=self.extent_frame,
                from_=1,
                to=50,
                length=200,
                orient=tk.HORIZONTAL,
                label=text,
            )
            return s

        self.extent_frame = ttk.Frame(master=self.general_menu_frame)
        self.extent_w = make_scale("Grid width")
        self.extent_h = make_scale("Grid height")
        self.extent_d = make_scale("Grid depth")

        self.extent_w.set(self.grid_extent.x)
        self.extent_h.set(self.grid_extent.y)
        self.extent_d.set(self.grid_extent.z)

        self.extent_button = ttk.Button(
            master=self.extent_frame,
            text="Apply grid changes",
            command=lambda: self.apply_extent(
                Coord(
                    int(self.extent_w.get()),
                    int(self.extent_h.get()),
                    int(self.extent_d.get()),
                )
            ),
        )
        self.extent_w.pack()
        self.extent_h.pack()
        self.extent_d.pack()

        self.extent_button.pack()
        self.extent_frame.pack()

    def apply_extent(self, extent: Coord):
        self.grid_extent = extent
        self.reset()

    def reset(self):
        self.reset_models()
        self.init_wfc()

    def apply_example_select(self, name):
        if self.active_example != name:
            self.active_example = name
            print(f"Selected {name}")
            self.reset()

    def reset_models(self):
        self.clear_models()
        self.shown_model_index = 0
        self.pending_models = 0

    def init_camera_params(self, grid_w, grid_h, grid_d, unit_dims):
        self.unit_dims = unit_dims
        lookat_point = Coord(grid_w, grid_h, -grid_d).scaled(0.5)
        default_cam_pos = lookat_point + Coord(0, grid_h + 50, 0)
        self.lookat_point = lookat_point
        self.default_camera_pos = default_cam_pos
        super().__init__(
            lookat_point=lookat_point,
            default_camera_pos=default_cam_pos,
            unit_dims=unit_dims,
        )

    def init_collider(self):
        self.collision_traverser = CollisionTraverser()
        self.collision_handler_queue = CollisionHandlerQueue()

    def init_axes(self):
        self.axis_system = self.create_axes()

    def init_mouse_events(self):
        self.picker_node = CollisionNode("mouse_ray")
        pickerNP = self.camera.attach_new_node(self.picker_node)
        self.picker_node.set_from_collide_mask(GeomNode.get_default_collide_mask())
        self.picker_ray = CollisionRay()
        self.picker_node.addSolid(self.picker_ray)
        self.collision_traverser.add_collider(pickerNP, self.collision_handler_queue)
        self.currently_picked_coord = None
        self.accept("mouse1", self.mouse_select)

    def init_tasks(self):
        self.task_mgr.add(self.play, "play")
        self.task_mgr.add(self.wfc_collapse_once, "collapse")
        self.task_mgr.add(self.prop_info_hover, "prop_info")
        self.task_mgr.add(self.enable_collapse_repeat, "enable_collapse_repeat")
        self.task_mgr.add(self.window_listener, "window_listener")

    def window_listener(self, task):
        f_width, f_height = self.frame.winfo_width(), self.frame.winfo_height()
        if f_width != self.window_width or f_height != self.window_height:
            self.window_width, self.window_height = f_width, f_height
            self.aspect_ratio = self.window_width * 1.0 / self.window_height

            self.props.setSize(self.window_width, self.window_height)
            self.win.request_properties(self.props)
            print("Ratio changed!", self.aspect_ratio)

        return task.cont

    def init_key_events(self):
        self.accept("p", self.toggle_paused)
        self.accept("arrow_right-up", self.show_next)
        self.accept("arrow_left-up", self.hide_previous)
        self.accept("z", self.toggle_axes)
        self.accept("c", self.hide_all)
        self.accept("v", self.show_all)
        self.accept("space", self.enable_collapse_once)
        self.accept("space-repeat", self.enable_collapse_once, [0.05])
        self.accept("1", self.toggle_hover)
        self.accept("enter", self.full_throttle)
        self.accept("control-space", self.toggle_collapse_repeat)
        self.accept("m", self.toggle_communicator)
        self.accept("escape", exit)

    def init_info_grid(self):
        self.create_grid()

    def full_throttle(self):
        start = time()
        comm.silence()
        print(f"START: {start}")
        self.wfc.collapse_automatic()
        print(f"END: {time() - start}")
        self.paused = False
        self.collapse_repeat = True
        self.collapse_once = False
        self.step_size = 0

    def manual(self):
        self.paused = False
        self.collapse_repeat = False
        self.collapse_once = False
        self.step_size = 0.05

    def toggle_paused(self):
        self.paused = not self.paused

    def toggle_collapse_once(self):
        self.collapse_once = not self.collapse_once
        self.collapse_repeat = False
        comm.communicate(
            f"Collapse once turned {'On' if self.collapse_once else 'Off'}."
        )

    def toggle_collapse_repeat(self):
        self.collapse_repeat = not self.collapse_repeat
        comm.communicate(
            f"Collapse repeat turned {'On' if self.collapse_repeat else 'Off'}."
        )

    def toggle_hover(self):
        self.hover_mode = not self.hover_mode
        comm.communicate(f"Turned hover mode {'On' if self.hover_mode else 'Off'}")

    def toggle_communicator(self):
        if comm.is_silent():
            comm.restore()
            comm.communicate("Turned communicator On")
        else:
            comm.silence()
            comm.communicate("Turned communicator Off")

    def enable_collapse_repeat(self, task):
        if self.collapse_repeat and not self.wfc.is_done():
            self.enable_collapse_once(step_size=self.step_size)
        return task.cont

    def enable_collapse_once(self, step_size=0):
        if not self.wfc.is_done():
            if not self.collapse_once:
                if self.delta_collapse >= step_size:
                    self.collapse_once = True
                    self.delta_collapse = 0
                else:
                    dt = self.clock.get_dt()
                    self.delta_collapse += dt
        else:
            comm.communicate("WFC is already done")

    def mouse_select(self):
        if self.mouseWatcherNode.has_mouse():
            mouse_pos = self.mouseWatcherNode.get_mouse()
            self.picker_ray.set_from_lens(self.camNode, mouse_pos.x, mouse_pos.y)
            self.collision_traverser.traverse(self.render)
            if self.collision_handler_queue.get_num_entries() > 0:
                self.collision_handler_queue.sort_entries()
                picked_cell = self.collision_handler_queue.get_entry(
                    0
                ).get_into_node_path()
                coord = Coord.from_string(picked_cell.get_net_tag("cell"))
                if coord != self.currently_picked_coord:
                    comm.communicate(
                        f"selected model: {picked_cell} at position: {coord}"
                    )
                    self.currently_picked_coord = coord
                    try:
                        comm.communicate(
                            f"Prop status of {coord}: {self.wfc.get_prop_status(coord)}."
                        )
                    except:
                        pass
                    return coord
        return None

    def prop_info_hover(self, task):
        if self.hover_mode:
            coord = self.mouse_select()
            if coord:
                comm.communicate(
                    f"Picked coord: {coord}; prop status: {self.wfc.get_prop_status(coord)}"
                )
        return task.cont

    def create_grid(self, path="parts/cube.egg"):
        for x in range(self.info_grid.width):
            for y in range(self.info_grid.height):
                for z in range(self.info_grid.depth):
                    cell: NodePath = self.loader.loadModel(path)
                    to_unit_scale = self.scale_to_unit(cell)
                    pos = Coord(x, y, z)
                    self.position_in_grid(cell, pos, Coord(1, 1, 1))

                    cell.set_scale(
                        Coord(to_unit_scale.x, to_unit_scale.y * 0.8, to_unit_scale.z)
                    )
                    cell.set_tag("cell", pos.to_coord_string())

                    colour = Colour(0, 0, 1, 0.5)
                    cell.set_material(self.make_material(colour))
                    if colour.a < 1:
                        cell.set_transparency(True)
                    cell.reparent_to(self.render)
                    self.info_grid.set(x, y, z, cell)

    def create_axes(self, path="parts/cube.egg"):
        model_x: NodePath = self.loader.loadModel(path)
        model_y: NodePath = self.loader.loadModel(path)
        model_z: NodePath = self.loader.loadModel(path)

        axis_diam = 0.1 * self.scale_to_unit(model_x).x
        axis_length = 2 * self.scale_to_unit(model_x).x
        model_x.set_pos(axis_length * 0.5, 0, 0)
        model_y.set_pos(0, axis_length * 0.5, 0)
        model_z.set_pos(0, 0, -axis_length * 0.5)
        model_x.set_scale(axis_length, axis_diam, axis_diam)
        model_y.set_scale(axis_diam, axis_length, axis_diam)
        model_z.set_scale(axis_diam, axis_diam, axis_length)
        model_x.set_material(self.make_material(Colour(1, 0, 0, 1)))
        model_y.set_material(self.make_material(Colour(0, 1, 0, 1)))
        model_z.set_material(self.make_material(Colour(0, 0, 1, 1)))
        model_x.reparent_to(self.render)
        model_y.reparent_to(self.render)
        model_z.reparent_to(self.render)

        return [model_x, model_y, model_z]

    def toggle_axes(self):
        for m in self.axis_system:
            m.show() if m.is_hidden() else m.hide()

    def hide_all(self):
        for k in self.models.keys():
            self.hide_model(k)

        self.shown_model_index = 0

    def show_all(self):
        comm.communicate(f"Showing all models: in {self.models.keys()}")
        for k in self.models.keys():
            self.show_model(k)
            self.shown_model_index = k

    def add_model(
        self,
        origin_coord: Coord,
        extent: Coord = Coord(1, 1, 1),
        path="parts/1x1x1.glb",
        colour: Colour = Colour(1, 1, 0, 1),
        is_hidden=True,
    ):
        """
        Adds a model placed on the corresponding grid/canvas cells.
        The model is added to a dictionary, so it can be modified later.
        """

        model, key = self.make_model(
            origin_coord, extent, path, colour, is_hidden=is_hidden
        )
        self.models[key] = model

    def clear_models(self):
        for m in self.models:
            self.models[m].remove_node()
        self.models = {}

    def colour_variation(
        self, colour: Colour, colour_variation=Colour(0.1, 0.1, 0.1, 0)
    ):
        def rand_sign():
            return 1 if random() < 0.5 else -1

        # Add some variation to make the parts distinguishable.
        return Colour(
            colour.r + random() * colour_variation.r * rand_sign(),
            colour.g + random() * colour_variation.g * rand_sign(),
            colour.b + random() * colour_variation.b * rand_sign(),
            colour.a + random() * colour_variation.a * rand_sign(),
        )

    def show_next(self, pause=True):
        if pause:
            self.paused = True
        if (
            self.show_model(self.shown_model_index)
            and self.shown_model_index < len(self.models.keys()) - 1
        ):
            self.shown_model_index += 1

    def hide_previous(self, pause=True):
        if pause:
            self.paused = True
        if self.hide_model(self.shown_model_index) and self.shown_model_index > 0:
            self.shown_model_index -= 1

    def play(self, task):
        if self.delta_acc >= self.step_size or self.pending_models > 0:
            if not self.paused:
                self.show_next(pause=False)
                self.delta_acc = 0
                self.pending_models = max(self.pending_models - 1, 0)
        else:
            dt = self.clock.get_dt()
            self.delta_acc += dt

        return task.cont

    def wfc_collapse_once(self, task):
        if self.collapse_once:
            origin_coord, terminal_id, t_o, _ = self.wfc.collapse_once()

            comm.communicate(f"Placed: {terminal_id} at {origin_coord}")
            self.inform_animator_choice(terminal_id, origin_coord, t_o)

            self.collapse_once = False
        return task.cont

    def inform_animator_choice(self, choice, coord, orientation):
        terminal = self.wfc.terminals[choice]
        comm.communicate(f"Model {choice} added at {coord}")
        if terminal.colour:
            colour_v = self.colour_variation(terminal.colour)
            self.pending_models += len(terminal.oriented_indices[orientation])
            for atom_index in terminal.oriented_indices[orientation]:
                model_path = terminal.atom_index_to_id_mapping[atom_index].path
                self.add_model(coord + atom_index, path=model_path, colour=colour_v)


comm.silence()

# grid_extent = Coord(50, 1, 50)
# grid_extent = Coord(5, 1, 5)
grid_extent = Coord(5, 5, 5)
# grid_extent = Coord(16, 16, 16)


anim = WFCAnimator(*grid_extent, unit_dims=Coord(1, 1, 1))

anim.run()
