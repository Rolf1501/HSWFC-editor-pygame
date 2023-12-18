from panda3d.core import load_prc_file, NodePath, Material, PointLight
from direct.showbase.ShowBase import ShowBase
from coord import Coord
from util_data import Colour
from model import Part

class Animator(ShowBase):
    def __init__(self, lookat_point = Coord(0,0,0), default_camera_pos=Coord(10,10,10), unit_dims: Coord=Coord(1,1,1)):
        ShowBase.__init__(self)
        # Loading a config is required in order for the models in relative paths to be found.
        load_prc_file("./Config.prc")

        self.init_default_material()
        self.init_lights()
        self.init_camera(default_camera_pos)
        self.init_camera_key_events()
        self.init_camera_mouse_events()
        self.disable_mouse()
        self.mouse_enabled = False
        self.models: dict[int, NodePath]  = {}
        self.lookat_point = lookat_point
        self.default_camera_pos = default_camera_pos

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
        self.accept("a-repeat", self.translate_camera, [Coord(-1,0,0)])
        self.accept("a-up", self.translate_camera, [Coord(-1,0,0)])
        self.accept("q-repeat", self.translate_camera, [Coord(1,0,0)])
        self.accept("q-up", self.translate_camera, [Coord(1,0,0)])
        self.accept("s-repeat", self.translate_camera, [Coord(0,-1,0)])
        self.accept("s-up", self.translate_camera, [Coord(0,-1,0)])
        self.accept("w-repeat", self.translate_camera, [Coord(0,1,0)])
        self.accept("w-up", self.translate_camera, [Coord(0,1,0)])
        self.accept("d-repeat", self.translate_camera, [Coord(0,0,-1)])
        self.accept("d-up", self.translate_camera, [Coord(0,0,-1)])
        self.accept("e-repeat", self.translate_camera, [Coord(0,0,1)])
        self.accept("e-up", self.translate_camera, [Coord(0,0,1)])

        # Camera rotation
        self.accept("r-repeat", self.rotate_camera, [Coord(1,0,0)])
        self.accept("r-up", self.rotate_camera, [Coord(1,0,0)])
        self.accept("f-repeat", self.rotate_camera, [Coord(-1,0,0)])
        self.accept("f-up", self.rotate_camera, [Coord(-1,0,0)])
        self.accept("t-repeat", self.rotate_camera, [Coord(0,1,0)])
        self.accept("t-up", self.rotate_camera, [Coord(0,1,0)])
        self.accept("g-repeat", self.rotate_camera, [Coord(0,-1,0)])
        self.accept("g-up", self.rotate_camera, [Coord(0,-1,0)])
        self.accept("y-repeat", self.rotate_camera, [Coord(0,0,1)])
        self.accept("y-up", self.rotate_camera, [Coord(0,0,1)])
        self.accept("h-repeat", self.rotate_camera, [Coord(0,0,-1)])
        self.accept("h-up", self.rotate_camera, [Coord(0,0,-1)])

        # Camera lookat
        self.accept("l", self.camera_lookat)

    def camera_lookat(self, up: Coord=Coord(0,1,0)):
        self.camera.look_at(self.lookat_point, up)

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

    def init_camera(self, camera_pos):
        self.camera.set_pos(camera_pos)
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
    
    def position_in_grid(self, model: NodePath, pos: Coord, extent: Coord):
        # In panda3d, z+ faces the camera. in numpy, z+ faces away from the camera. Make them both uniform, the z-direction is negated.
        pos_center = pos + extent.scaled(0.5)
        pos_neg_z = pos_center * Coord(1,1,-1)
        model.set_pos(pos_neg_z)
    
    def make_model(self,  origin_coord: Coord, extent: Coord=Coord(1,1,1), path="parts/cube.egg", colour: Colour=Colour(1,1,0,1)):
        model: NodePath = self.loader.loadModel(path)
        to_unit_scale = self.scale_to_unit(model)

        model.set_scale(to_unit_scale * extent)
        self.position_in_grid(model, origin_coord, extent)

        material = self.make_material(colour)
        
        # Override all existing materials to force the new material to take effect.
        if not model.find_all_materials():
            model.set_material(material)
        else:
            for mat in model.find_all_materials():
                model.replace_material(mat, self.make_material(colour))

        model.reparent_to(self.render)
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
    def __init__(self, parts: dict[int, Part], unit_dims: Coord = Coord(1, 1, 1), lookat_point = Coord(0,0,0)):
        self.parts = parts
        self.lookat_point = lookat_point
        super().__init__(unit_dims=unit_dims, lookat_point=lookat_point)
        self.add_parts()
        self.init_key_events()

    def add_parts(self):
        for p in self.parts.values():
            p_origin = p.extent.center()
            p_extent = p.extent.whd()
            self.make_model(p_origin, p_extent, colour=p.colour, compensate_extent=False)
            print(f"Part: {p}")

    def show_all(self):
        for m in self.models.values():
            m.show()
            print(f"showing model {m} at {m.get_pos()}") 

    def init_key_events(self):
        self.accept("m", self.show_all)


