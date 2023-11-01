from boundingbox import BoundingBox as BB
from model_hierarchy_tree import Properties
from util_data import Cardinals, Dimensions, Operations

class Part:
    def __init__(self, size: BB, rotation: int) -> None:
        self.size = size
        self.rotation = rotation
    
    def absolute_north(self) -> Cardinals:
        # Assumes rotation expressed in degrees
        return Cardinals((self.rotation / 90.0) % len(Cardinals))

def handle_adjacency(source: Part, attachment: Part, adjacency: Cardinals, relative_adjacency=True):
    """
    Aligns the attachments on the face of source indicated by the adjacency.
    """

    # The links are relative, so need to take the rotation into account.
    if relative_adjacency:
        adjacency = Cardinals((adjacency.value + source.absolute_north().value) % len(Cardinals))

    if adjacency == Cardinals.EAST:
        attachment.size.maxx = source.size.maxx + attachment.size.width()
        attachment.size.minx = source.size.maxx

    elif adjacency == Cardinals.WEST:
        attachment.size.minx = source.size.minx - attachment.size.width()
        attachment.size.maxx = source.size.minx

    elif adjacency == Cardinals.NORTH:
        attachment.size.miny = source.size.miny - attachment.size.height()
        attachment.size.maxy = source.size.miny

    elif adjacency == Cardinals.SOUTH:
        attachment.size.maxy = source.size.maxy + attachment.size.height()
        attachment.size.miny = source.size.maxy
    
    return source, attachment

def center(source: Part, attachment: Part, dimension: Dimensions) -> (BB, BB):
    """
    Centers attachment relative to source.
    """
    bb1 = source.size
    bb2 = attachment.size

    bb1_center = bb1.center()

    # In case the part has been rotated such that the relative axes do not align with the global x and y axes, switch the dimension.
    if source.absolute_north() == Cardinals.EAST or source.absolute_north() == Cardinals.WEST:
        dimension = Dimensions((dimension.value + 1) % len(Dimensions))
    if dimension == Dimensions.X:
        bb2_width = bb2.width()
        bb2.minx = bb1_center.x - 0.5 * bb2_width
        bb2.maxx = bb2.minx + bb2_width
    elif dimension == Dimensions.Y:
        bb2_height = bb2.height()
        bb2.miny = bb1_center.y - 0.5 * bb2_height
        bb2.maxy = bb2.miny + bb2_height

    return bb1, bb2

def handle_properties(source: Part, attachment: Part, properties: list[Properties]):
    source_n = source
    attachment_n = attachment

    for prop in properties:
        if prop.operation == Operations.CENTER:
            source_n.size, attachment_n.size = center(source, attachment, prop.dimension)
            
    return source_n, attachment_n