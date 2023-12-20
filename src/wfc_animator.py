from animator import Animator
from communicator import Communicator
from coord import Coord
from grid import Grid
from terminal import Void
from toy_examples import ToyExamples as Toy
from wfc import WFC, Propagation

from panda3d.core import NodePath
from util_data import Colour, Dimensions
from queue import Queue as Q
from numpy.random import random
from time import time
from panda3d.core import CollisionHandlerQueue, CollisionTraverser, CollisionNode, CollisionRay, GeomNode

comm = Communicator()

class WFCAnimator(Animator):
    def __init__(self, wfc: WFC, grid_w=20, grid_h=20, grid_d=5, unit_dims=Coord(1,1,1)):
        self.wfc = wfc
        self.unit_dims = unit_dims
        lookat_point=Coord(grid_w, grid_h, -grid_d).scaled(0.5)
        default_cam_pos = lookat_point + Coord(0,grid_h+50,0)
        self.lookat_point = lookat_point
        self.default_camera_pos = default_cam_pos
        super().__init__(lookat_point=lookat_point, default_camera_pos=default_cam_pos, unit_dims=unit_dims)

        # Keep reference from canvas to model.
        self.canvas = Grid(grid_w, grid_h, grid_d, default_fill_value=-1)
        self.colours = Grid(grid_w, grid_h, grid_d, default_fill_value=Q())
        self.info_grid = Grid(grid_w, grid_h, grid_d, default_fill_value=None)

        self.shown_model_index = 0

        self.manual()
        
        self.delta_acc = 0
        self.delta_collapse = 0

        self.hover_mode = False

        self.axis_system = self.create_axes()

        self.collision_traverser = CollisionTraverser()
        self.collision_handler_queue = CollisionHandlerQueue()

        self.init_info_grid()
        self.init_key_events()
        self.init_mouse_events()
        self.init_tasks()

    def init_mouse_events(self):
        self.picker_node = CollisionNode('mouse_ray')
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

    def init_info_grid(self):
        self.create_grid()

    def full_throttle(self):
        self.paused = False
        self.collapse_repeat = True
        self.collapse_once = False
        self.step_size = 0

    def manual(self):
        self.paused = True
        self.collapse_repeat = False
        self.collapse_once = True
        self.step_size = 0.05

    def toggle_paused(self):
        self.paused = not self.paused
    
    def toggle_collapse_once(self):
        self.collapse_once = not self.collapse_once
        self.collapse_repeat = False
        comm.communicate(f"Collapse once turned {'Off' if not self.collapse_repeat else 'On'}.")
    
    def toggle_hover(self):
        self.hover_mode = not self.hover_mode

    def enable_collapse_once(self, step_size=0):
        if not self.wfc.is_done():
            if not self.collapse_once: 
                if self.delta_collapse >= step_size:
                    self.collapse_once = True
                    self.collapse_repeat = False
                    comm.communicate(f"Collapse once turned {'Off' if not self.collapse_repeat else 'On'}.")
                    self.delta_collapse = 0
                else:
                    dt = self.clock.get_dt()
                    self.delta_collapse += dt
        else:
            comm.communicate("WFC is already done")

            
    def mouse_select(self):
        if base.mouseWatcherNode.has_mouse():
            mouse_pos = base.mouseWatcherNode.get_mouse()
            self.picker_ray.set_from_lens(base.camNode, mouse_pos.x, mouse_pos.y)
            self.collision_traverser.traverse(self.render)
            if self.collision_handler_queue.get_num_entries() > 0:
                self.collision_handler_queue.sort_entries()
                picked_cell = self.collision_handler_queue.get_entry(0).get_into_node_path()
                coord = Coord.from_string(picked_cell.get_net_tag("cell"))
                if coord != self.currently_picked_coord:
                    comm.communicate(f"selected model: {picked_cell} at position: {coord}")
                    self.currently_picked_coord = coord
                    return coord
        return None

    def prop_info_hover(self, task):
        if self.hover_mode:
            coord = self.mouse_select()
            if coord:
                comm.communicate(f"Picked coord: {coord}; prop status: {self.wfc.get_prop_status(coord)}")
        return task.cont
    
    

    def create_grid(self, path="parts/cube.egg"):
        for x in range(self.info_grid.width):
            for y in range(self.info_grid.height):
                for z in range(self.info_grid.depth):
                    cell: NodePath = self.loader.loadModel(path)
                    to_unit_scale = self.scale_to_unit(cell)
                    pos = Coord(x,y,z)
                    self.position_in_grid(cell, pos, Coord(1,1,1))

                    cell.set_scale(Coord(to_unit_scale.x, to_unit_scale.y * 0.8, to_unit_scale.z))
                    cell.set_tag("cell", pos.to_coord_string())

                    colour = Colour(0,0,1,0.5)
                    cell.set_material(self.make_material(colour))
                    if colour.a < 1:
                        cell.set_transparency(True)
                    cell.reparent_to(self.render)
                    self.info_grid.set(x,y,z, cell)
    
    def create_axes(self, path="parts/cube.egg"):
        # def create_axis(diameter_scalar, length_scalar, dimension: Dimensions, path, colour):
        #     model: NodePath = self.loader.loadModel(path)
        #     model.set_pos()
            
        model_x: NodePath = self.loader.loadModel(path)
        model_y: NodePath = self.loader.loadModel(path)
        model_z: NodePath = self.loader.loadModel(path)

        axis_diam = 0.1 * self.scale_to_unit(model_x).x
        axis_length = 2 * self.scale_to_unit(model_x).x
        model_x.set_pos(axis_length * 0.5,0,0)
        model_y.set_pos(0,axis_length * 0.5,0)
        model_z.set_pos(0,0,-axis_length * 0.5)
        model_x.set_scale(axis_length, axis_diam, axis_diam)
        model_y.set_scale(axis_diam, axis_length, axis_diam)
        model_z.set_scale(axis_diam, axis_diam, axis_length)
        model_x.set_material(self.make_material(Colour(1,0,0,1)))
        model_y.set_material(self.make_material(Colour(0,1,0,1)))
        model_z.set_material(self.make_material(Colour(0,0,1,1)))
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
    
    def add_model(self,  origin_coord: Coord, extent: Coord=Coord(1,1,1), path="parts/1x1x1.glb", colour: Colour=Colour(1,1,0,1), colour_variation: Colour=Colour(0.1,0.1,0.1,0)):
        """
        Adds a model placed on the corresponding grid/canvas cells.
        The model is added to a dictionary, so it can be modified later.
        The grid/canvas keeps a references the model via the model's key.
        """
        def rand_sign():
            return 1 if random() < 0.5 else -1
        
        # Add some variation to make the parts distinguishable.
        colour_v = Colour(colour.r + random() * colour_variation.r * rand_sign(), 
                          colour.g + random() * colour_variation.g * rand_sign(), 
                          colour.b + random() * colour_variation.b * rand_sign(),
                          colour.a + random() * colour_variation.a * rand_sign())
        _, new_key = self.make_model(origin_coord, extent, path, colour_v)

        # self.add_colour_mode(*origin_coord, colour)
        self.update_canvas(origin_coord, extent, new_key)

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

    def wfc_collapse_once(self, task):
        if self.collapse_once:
            id, _, origin = self.wfc.collapse_once()
            if id is not None and origin is not None:
                self.inform_animator_choice(id, origin)
            self.collapse_once = False
        return task.cont
    
    def inform_animator_choice(self, choice, coord):
        terminal = self.wfc.terminals[choice]
        comm.communicate(f"Model {choice} added at {coord}")
        if terminal.colour:
            self.add_model(coord, extent=terminal.extent.whd(), colour=terminal.colour)

# comm.silence()


# terminals, adjs, def_w = Toy().example_slanted()
# terminals, adjs, def_w = Toy().example_zebra_horizontal()
# terminals, adjs, def_w = Toy().example_zebra_vertical()
# terminals, adjs, def_w = Toy().example_zebra_horizontal_3()
# terminals, adjs, def_w = Toy().example_zebra_vertical_3()
# terminals, adjs, def_w = Toy().example_big_tiles()
terminals, adjs, def_w = Toy().example_meta_tiles_fit_area()
# terminals, adjs, def_w = Toy().example_meta_tiles_2()
# terminals, adjs, def_w = Toy().example_meta_tiles()
# terminals, adjs, def_w = Toy().example_meta_tiles_zebra_horizontal()

grid_extent = Coord(20,1,20)
# grid_extent = Coord(20,20,20)
# grid_extent = Coord(6,5,6)
start_coord = grid_extent * Coord(0.5,0,0.5)
start_coord = Coord(int(start_coord.x), int(start_coord.y), int(start_coord.z))

start_time = time()
wfc = WFC(terminals, adjs, grid_extent=grid_extent, start_coord=start_coord, default_weights=def_w)
wfc_init_time = time() - start_time
print(f"WFC init: {wfc_init_time}")

anim = WFCAnimator(wfc, grid_extent.x, grid_extent.y, grid_extent.z, unit_dims=Coord(1,1,1))
# anim.full_throttle()
anim_init_time = time() - start_time - wfc_init_time
print(f"Anim init time: {anim_init_time}")

print("Running WFC")

# run_time = time() - anim_init_time - wfc_init_time - start_time
# print(f"Running time: {run_time}")

# print(f"Total elapsed time: {time() - start_time}")
# wfc.grid_man.grid.print_xz()

anim.run()