from panda3d.core import load_prc_file, NodePath, Material, PointLight
from direct.showbase.ShowBase import ShowBase
from queue import Queue as Q
from collections import namedtuple
from grid import Grid
from coord import Coord
from util_data import Colour

class Job(namedtuple("Job", ["dims", "orig"])):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()
    

class Animator(ShowBase):

    def __init__(self, grid_w=20, grid_h=20, grid_d=5, unit_dims=Coord(1,1,1)):
        # Keep reference from canvas to model.
        self.canvas = Grid(grid_w, grid_h, grid_d, default_fill_value=-1)
        ShowBase.__init__(self)

        # Loading a config is required in order for the models in relative paths to be found.
        load_prc_file("./Config.prc")

        self.pending: Q[Job] = Q()
        self.unit_dims = unit_dims # Specifies the dimensions of a single cell.

        self.models: dict[int, NodePath] = {}
        self.shown_model_index = 0

        self.paused = True
        self.delta_acc = 0
        self.step_size = 0.1 # in seconds


        self.init_default_material()
        self.init_lights()
        self.init_key_events()
        self.init_tasks()

    def init_tasks(self):
        self.task_mgr.add(self.play, "play")
    
    def init_lights(self):
        p_light = PointLight("p_light")
        p_light.attenuation = (0.5,0,0)
        p_lnp = self.render.attach_new_node(p_light)
        p_lnp.set_pos(50,50,50)
        self.render.set_light(p_lnp)

        p_light2 = PointLight("p_light2")
        p_light2.attenuation = (0.5,0,0)
        p_lnp2 = self.render.attach_new_node(p_light2)
        p_lnp2.set_pos(-50,-50,-50)
        self.render.set_light(p_lnp2)

    def init_key_events(self):
        self.accept("a-up", self.show_next_model)
        self.accept("h-up", self.toggle_hidden, [0])
        self.accept("p", self.toggle_pause)

    def init_default_material(self):
        self.default_material = Material()
        self.default_material.set_ambient((0.2,0.2,0.2,1))
        self.default_material.set_diffuse((0.5,0.5,0.5,1))
        self.default_material.set_shininess(10)

    def add_model(self,  origin_coord: Coord, extent: Coord=Coord(1,1,1), path="parts/cube.egg", colour: Colour=Colour(1,1,0,1)):
        """
        Adds a model placed on the corresponding grid/canvas cells.
        The model is added to a dictionary, so it can be modified later.
        The grid/canvas keeps a references the model via the model's key.
        """
        # TODO: account for rotation of parts. Currently not applicable since the model's rotation is irrelevant.

        model: NodePath = self.loader.loadModel(path)
        
        scalar = self.scale(model, extent)
        model.set_scale(scalar)
        model.set_pos(origin_coord)

        model.set_material(self.default_material)
        # mat = model.get_material()
        # mat.set_diffuse(colour)
        new_key = len(self.models.keys())

        self.models[new_key] = model
        self.update_canvas(origin_coord, extent, new_key)

    def show_next_model(self):
        if self.shown_model_index in self.models.keys():
            self.models[self.shown_model_index].reparentTo(self.render)
            self.shown_model_index += 1

    def toggle_pause(self):
        self.paused = not self.paused

    def play(self, task):
        if self.delta_acc >= self.step_size:
            if not self.paused:
                self.show_next_model()
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


    def scale(self, model: NodePath, extent: Coord):
        """
        Calculates the scalar required for the model such that the model covers grid/canvas cells equal to the extent.
        """
        scalar_grid = self.scale_to_grid(model)
        scalar_extent = scalar_grid * extent
        return scalar_extent
    
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

    def show_model(self, key: int):
        model = self.get_model(key)
        if model:
            self.models[key].show()

    def scale_to_grid(self, model: NodePath) -> Coord:
        """
        Calculates the scalar needed to fit the model in a single grid/canvas cell.
        """
        min_b, max_b = model.get_tight_bounds()
        model_extent = max_b - min_b
        scaled = Coord(self.unit_dims.x / (1.0 * model_extent.x), self.unit_dims.y / (1.0 * model_extent.y), self.unit_dims.z / (1.0 * model_extent.z))
        return scaled

