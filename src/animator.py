from panda3d.core import load_prc_file, NodePath, Material, PointLight
from direct.showbase.ShowBase import ShowBase
from grid import Grid
from coord import Coord
from util_data import Colour
from model import Part
from queue import Queue as Q
from numpy.random import random

class Animator(ShowBase):
    def __init__(self, lookat_point = Coord(0,0,0), default_camera_pos=Coord(10,10,10), unit_dims: Coord=Coord(1,1,1)):
        ShowBase.__init__(self)
        # Loading a config is required in order for the models in relative paths to be found.
        load_prc_file("./Config.prc")

        self.init_default_material()
        self.init_lights()
        self.init_camera()
        self.init_camera_key_events()
        self.init_camera_mouse_events()
        self.disable_mouse()
        self.mouse_enabled = False
        self.models: dict[int, NodePath]  = {}
        self.lookat_point = lookat_point

        self.unit_dims = unit_dims # Specifies the dimensions of a single cell.

    def init_camera_mouse_events(self):
        self.accept("x", self.toggle_mouse_controls)

    def toggle_mouse_controls(self):
        if self.mouse_enabled:
            self.disable_mouse()
        else:
            self.enable_mouse()
        
        self.mouse_enabled = not self.mouse_enabled
    
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

    def camera_lookat(self, up: Coord=Coord(0,1,0)):
        self.camera.look_at(self.lookat_point, up)
        # print(self.camera.get_pos())
        # print(self.camera.get_hpr())

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
        p_light2.attenuation = (0.9,0,0)
        p_lnp2 = self.render.attach_new_node(p_light2)
        p_lnp2.set_pos(-50,-20,-50)
        self.render.set_light(p_lnp2)

    def init_camera(self):
        self.default_camera_pos = self.default_camera_pos
        self.camera.set_pos(self.default_camera_pos)
        self.camera.set_hpr(0,0,0)
        self.camera_lookat()

    def init_default_material(self):
        self.default_material = Material(name="default")
        self.default_material.set_ambient((0.1,0.1,0.1,1))
        self.default_material.set_diffuse((0,0.5,0.5,1))
        self.default_material.set_shininess(1)

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
        model_center = origin_coord + extent.scaled(0.5)
        # In panda3d, z+ faces the camera. in numpy, z+ faces away from the camera. Make them both uniform, the z-direction is negated.
        translation = model_center * Coord(1,1,-1)
        
        model.set_pos(translation)

        # Override all existing materials to force the new material to take effect.
        for mat in model.find_all_materials():
            model.replace_material(mat, self.make_material(colour))

        model.reparentTo(self.render)
        model.hide()

        if colour.a < 1:
            model.set_transparency(True)
        
        new_key = len(self.models.keys())
        self.models[new_key] = model
        return model, new_key

    def scale(self, model: NodePath, extent: Coord):
        """
        Calculates the scalar required such that the model covers grid/canvas cells equal to the extent.
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

class GridAnimator(Animator):
    def __init__(self, grid_w=20, grid_h=20, grid_d=5, unit_dims=Coord(1,1,1)):
        self.unit_dims = unit_dims
        lookat_point=Coord(grid_w, grid_h, grid_d).scaled(0.5)
        default_cam_pos = lookat_point + Coord(0,grid_h,0)
        self.lookat_point = lookat_point
        self.default_camera_pos = default_cam_pos
        super().__init__(lookat_point=lookat_point, default_camera_pos=default_cam_pos, unit_dims=unit_dims)

        # Keep reference from canvas to model.
        self.canvas = Grid(grid_w, grid_h, grid_d, default_fill_value=-1)
        self.colours = Grid(grid_w, grid_h, grid_d, default_fill_value=Q())
        self.shown_model_index = 0

        self.paused = True
        self.delta_acc = 0
        self.step_size = 0.05 # in seconds

        self.axis_system = self.create_axes()

        self.init_key_events()
        self.init_tasks()

    def init_tasks(self):
        self.task_mgr.add(self.play, "play")

    def init_key_events(self):
        self.accept("p", self.toggle_pause)
        self.accept("arrow_right-up", self.show_next)
        self.accept("arrow_left-up", self.hide_previous)
        self.accept("z", self.toggle_axes)
        self.accept("c", self.hide_all)
        self.accept("v", self.show_all)
    
    def create_axes(self, path="parts/cube.egg"):
        model_x: NodePath = self.loader.loadModel(path)
        model_y: NodePath = self.loader.loadModel(path)
        model_z: NodePath = self.loader.loadModel(path)

        axis_diam = 1 * self.scale_to_unit(model_x).x
        axis_length = 2 * self.scale_to_unit(model_x).x
        model_x.set_pos(0,0,0)
        model_y.set_pos(0,0,0)
        model_z.set_pos(0,0,0)
        model_x.set_scale(axis_length, axis_diam, axis_diam)
        model_y.set_scale(axis_diam, axis_length, axis_diam)
        model_z.set_scale(axis_diam, axis_diam, axis_length)
        model_x.set_material(self.make_material(Colour(1,0,0,1)))
        model_y.set_material(self.make_material(Colour(0,1,0,1)))
        model_z.set_material(self.make_material(Colour(0,0,1,1)))

        return [model_x, model_y, model_z]

    def toggle_axes(self):
        for m in self.axis_system:
            if m.is_hidden():
                print("toggled model on")
                m.show()
            else:
                m.hide()

    def hide_all(self):
        for k in self.models.keys():
            self.hide_model(k)

        self.shown_model_index = 0

    def show_all(self):
        for k in self.models.keys():
            self.show_model(k)
            self.shown_model_index = k
    
    def add_model(self,  origin_coord: Coord, extent: Coord=Coord(1,1,1), path="parts/1x1x1.glb", colour: Colour=Colour(1,1,0,1), colour_variation: Colour=Colour(0.1,0.1,0.1,0)):
        """
        Adds a model placed on the corresponding grid/canvas cells.
        The model is added to a dictionary, so it can be modified later.
        The grid/canvas keeps a references the model via the model's key.
        """

        # Add some variation to make the parts distinguishable.
        colour_v = Colour(colour.r + random() * colour_variation.r * self.rand_sign(), 
                          colour.g + random() * colour_variation.g * self.rand_sign(), 
                          colour.b + random() * colour_variation.b * self.rand_sign(),
                          colour.a + random() * colour_variation.a * self.rand_sign())
        _, new_key = self.make_model(origin_coord, extent, path, colour_v)

        # self.add_colour_mode(*origin_coord, colour)
        self.update_canvas(origin_coord, extent, new_key)

    def rand_sign(self):
        return 1 if random() < 0.5 else -1
    def show_next(self, pause=True):
        if pause:
            self.paused = True
        if self.show_model(self.shown_model_index) and self.shown_model_index < len(self.models.keys()) - 1:
            self.shown_model_index += 1

    def hide_previous(self, pause=True):
        if pause:
            self.paused = True
        if self.hide_model(self.shown_model_index) and self.shown_model_index > 0:
            self.shown_model_index -= 1

    def add_colour_mode(self, x, y, z, new_colour: Colour):
        self.colours.get(x, y, z).put(self.make_material(new_colour))

    def to_next_colour_mode(self, x, y, z):
        key = self.canvas.get(x,y,z)
        model = self.models[key]

        # Save current material colour
        old_colour = model.get_material()
        
        new_colour = self.colours.get(x, y, z).get()
        model.set_material(new_colour)
        self.colours.get(x, y, z).put(old_colour)

    def toggle_next_colour_mode(self):
        for x in range(self.colours.width):
            for y in range(self.colours.height):
                for z in range(self.colours.depth):
                    self.to_next_colour_mode(x,y,z)
    
    def toggle_pause(self):
        self.paused = not self.paused

    def play(self, task):
        if self.delta_acc >= self.step_size:
            if not self.paused:
                self.show_next(pause=False)
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
