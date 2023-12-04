from panda3d.core import load_prc_file, NodePath, Material, PointLight
from direct.showbase.ShowBase import ShowBase
from queue import Queue as Q
from collections import namedtuple
from grid import Grid
from coord import Coord
from util_data import Colour
from model import Part

class Animator(ShowBase):
    def __init__(self, unit_dims: Coord=Coord(1,1,1)):
        ShowBase.__init__(self)
        # Loading a config is required in order for the models in relative paths to be found.
        load_prc_file("./Config.prc")

        self.init_default_material()
        self.init_lights()
        self.init_camera()
        self.init_camera_key_events()
        self.disable_mouse()
        self.models: dict[int, NodePath]  = {}

        self.unit_dims = unit_dims # Specifies the dimensions of a single cell.

    
    def init_camera_key_events(self):
        # Camera translation
        self.accept("a", self.translate_camera, [Coord(-1,0,0)])
        self.accept("q", self.translate_camera, [Coord(1,0,0)])
        self.accept("s", self.translate_camera, [Coord(0,-1,0)])
        self.accept("w", self.translate_camera, [Coord(0,1,0)])
        self.accept("d", self.translate_camera, [Coord(0,0,-1)])
        self.accept("e", self.translate_camera, [Coord(0,0,1)])

        # Camera rotation
        self.accept("r", self.rotate_camera, [Coord(1,0,0)])
        self.accept("f", self.rotate_camera, [Coord(-1,0,0)])
        self.accept("t", self.rotate_camera, [Coord(0,1,0)])
        self.accept("g", self.rotate_camera, [Coord(0,-1,0)])
        self.accept("y", self.rotate_camera, [Coord(0,0,1)])
        self.accept("h", self.rotate_camera, [Coord(0,0,-1)])

        # Camera lookat
        self.accept("l", self.camera_lookat)

    def camera_lookat(self, point: Coord=Coord(0,0,0), up: Coord=Coord(0,0,1)):
        self.camera.look_at(point, up)
        print(self.camera.get_pos())
        print(self.camera.get_hpr())

    def translate_camera(self, coord: Coord):
        self.camera.set_pos(Coord(*self.camera.get_pos()) + coord)

    def rotate_camera(self, coord: Coord):
        Coord(*self.camera.get_hpr()) + coord
        self.camera.set_hpr(Coord(*self.camera.get_hpr()) + coord)

    def init_lights(self):
        p_light = PointLight("p_light")
        p_light.attenuation = (0.5,0,0)
        p_lnp = self.render.attach_new_node(p_light)
        p_lnp.set_pos(50,20,50)
        self.render.set_light(p_lnp)

        p_light2 = PointLight("p_light2")
        p_light2.attenuation = (0.8,0,0)
        p_lnp2 = self.render.attach_new_node(p_light2)
        p_lnp2.set_pos(-50,-20,-50)
        self.render.set_light(p_lnp2)

    def init_camera(self):
        self.default_camera_pos = Coord(1,11,3)
        self.camera.set_pos(self.default_camera_pos)
        self.camera_lookat()

    def init_default_material(self):
        self.default_material = Material(name="default")
        self.default_material.set_ambient((0.1,0.1,0.1,1))
        self.default_material.set_diffuse((0,0.5,0.5,1))
        self.default_material.set_shininess(10)

    def make_material(self, colour: Colour):
        material = Material(name="default")
        material.set_ambient((0.1,0.1,0.1,colour.a))
        material.set_diffuse(colour)
        material.set_shininess(2)
        return material
    
    def make_model(self,  origin_coord: Coord, extent: Coord=Coord(1,1,1), path="parts/cube.egg", colour: Colour=Colour(1,1,0,1)):
        model: NodePath = self.loader.loadModel(path)
        
        scalar = self.scale(model, extent)
        model.set_scale(scalar)
        model.set_pos(origin_coord)

        model.set_material(self.make_material(colour))

        model.reparentTo(self.render)
        model.hide()
        if colour.a < 1:
            model.set_transparency(True)
        new_key = len(self.models.keys())
        self.models[new_key] = model
        return model, new_key

    def scale(self, model: NodePath, extent: Coord):
        """
        Calculates the scalar required for the model such that the model covers grid/canvas cells equal to the extent.
        """
        scalar_grid = self.scale_to_unit(model)
        scalar_extent = scalar_grid * extent
        return scalar_extent
    
    def scale_to_unit(self, model: NodePath) -> Coord:
        """
        Calculates the scalar needed to fit the model in a single grid/canvas cell.
        """
        min_b, max_b = model.get_tight_bounds()
        model_extent = max_b - min_b
        scaled = Coord(self.unit_dims.x / (1.0 * model_extent.x), self.unit_dims.y / (1.0 * model_extent.y), self.unit_dims.z / (1.0 * model_extent.z))
        return scaled
    
    def get_model(self, key):
        if key in self.models.keys():
            return self.models[key]
        else:
            return None
    
    def toggle_hidden(self, key: int):
        model = self.get_model(key)
        if model:
            if model.is_hidden():
                model.show()
            else:
                model.hide()

    def hide_model(self, key: int):
        model = self.get_model(key)
        if model:
            self.models[key].hide()
            return True
        return False

    def show_model(self, key: int):
        model = self.get_model(key)
        if model:
            self.models[key].show()
            return True
        return False


class GSAnimator(Animator):
    def __init__(self, parts: dict[int, Part], unit_dims: Coord = Coord(1, 1, 1)):
        super().__init__(unit_dims)
        self.parts = parts
        self.add_parts()
        self.init_key_events()

    def add_parts(self):
        for p in self.parts.values():
            p_origin = p.extent.center()
            p_extent = p.extent.whd()
            self.make_model(p_origin, p_extent, colour=p.colour)
            print(f"Part: {p}")

    def show_all(self):
        for m in self.models.values():
            m.show()
            print(f"showing model {m} at {m.get_pos()}") 

    def init_key_events(self):
        self.accept("m", self.show_all)
        self.accept("x", self.enable_mouse)

class GridAnimator(Animator):

    def __init__(self, grid_w=20, grid_h=20, grid_d=5, unit_dims=Coord(1,1,1)):
        self.unit_dims = unit_dims
        super().__init__(unit_dims=unit_dims)

        # Keep reference from canvas to model.
        self.canvas = Grid(grid_w, grid_h, grid_d, default_fill_value=-1)

        self.shown_model_index = 0

        self.paused = True
        self.delta_acc = 0
        self.step_size = 0.1 # in seconds

        self.init_key_events()
        self.init_tasks()

    def init_tasks(self):
        self.task_mgr.add(self.play, "play")

    def init_key_events(self):
        self.accept("p", self.toggle_pause)
        self.accept("arrow_right-up", self.show_next)
        self.accept("arrow_left-up", self.hide_previous)

    def hide_all(self):
        for k in self.models.keys():
            self.hide_model(k)

        self.shown_model_index = 0
    
    def add_model(self,  origin_coord: Coord, extent: Coord=Coord(1,1,1), path="parts/cube.egg", colour: Colour=Colour(1,1,0,1)):
        """
        Adds a model placed on the corresponding grid/canvas cells.
        The model is added to a dictionary, so it can be modified later.
        The grid/canvas keeps a references the model via the model's key.
        """
        # TODO: account for rotation of parts. Currently not applicable since the model's rotation is irrelevant.
        _, new_key = self.make_model(origin_coord, extent, path, colour)

        self.update_canvas(origin_coord, extent, new_key)

    def show_next(self):
        if self.show_model(self.shown_model_index) and self.shown_model_index < len(self.models.keys()) - 1:
            self.shown_model_index += 1

    def hide_previous(self):
        if self.hide_model(self.shown_model_index) and self.shown_model_index > 0:
            self.shown_model_index -= 1
    
    def toggle_pause(self):
        self.paused = not self.paused

    def play(self, task):
        if self.delta_acc >= self.step_size:
            if not self.paused:
                self.show_next()
                self.delta_acc = 0
        else:
            dt = self.clock.get_dt()
            self.delta_acc += dt

        return task.cont

    def update_canvas(self, origin_coord: Coord, extent: Coord, key):
        c_x, c_y, c_z = origin_coord
        for x in range(extent.x):
            for y in range(extent.y):
                for z in range(extent.z):
                    self.canvas.set(c_x + x, c_y + y, c_z + z, key)
    


    
