import numpy as np
import param

from shapely import geometry, Polygon

from .. import aux

from .nested_parameter_group import NestedConf
from .custom import PositiveNumber, RandomizedPhase, XYLine, \
    PositiveIntegerRange, PositiveRange, NumericTuple2DRobust, IntegerTuple2DRobust


__all__ = [
    'Pos2D',
    'Pos2DPixel',
    'RadiallyExtended',
    'OrientedPoint',
    'MobilePoint',
    'MobileVector',
    'LineExtended',
    'LineClosed',
    'Area2D',
    'Area2DPixel',
    'Area',
    'BoundedArea',
    'PosPixelRel2Point',
    'PosPixelRel2Area',
]

__displayname__ = 'Spatial elements'


# class Pos2D(param.Parameterized):
class Pos2D(NestedConf):
    pos = NumericTuple2DRobust(doc='The xy spatial position coordinates')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initial_pos = self.pos
        self.last_pos = self.get_position()

    def get_position(self):
        return tuple(self.pos)

    def set_position(self, pos):
        if not isinstance(pos, tuple):
            pos = tuple(pos)
        self.last_pos = self.get_position()
        self.pos = pos

    @property
    def x(self):
        return self.pos[0]

    @property
    def y(self):
        return self.pos[1]

    @property
    def last_delta_pos(self):
        x0, y0 = self.last_pos
        x1, y1 = self.get_position()
        return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** (1 / 2)


class Pos2DPixel(Pos2D):
    pos = IntegerTuple2DRobust(doc='The xy spatial position coordinates')


class RadiallyExtended(Pos2D):
    radius = PositiveNumber(0.003, softmax=0.1, step=0.001, doc='The spatial radius of the source in meters')

    def __init__(self, **kwargs):

        super().__init__(**kwargs)


    def get_shape(self, scale=1):
        p = self.get_position()
        return geometry.Point(p).buffer(self.radius * scale) if not np.isnan(p).all() else None

    def contained(self, point):
        return geometry.Point(self.get_position()).distance(geometry.Point(point)) <= self.radius


class OrientedPoint(Pos2D):
    orientation = RandomizedPhase(label='orientation', doc='The absolute orientation in space.')

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.initial_orientation = self.orientation
        self.last_orientation = self.get_orientation()

    @property
    def rotationMatrix(self):
        a = -self.orientation
        return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

    def translate(self, point):
        p = np.array(self.pos) + np.array(point) @ self.rotationMatrix
        if isinstance(point, tuple):
            return tuple(p)
        else:
            return aux.np2Dtotuples(p)

    def set_orientation(self, orientation):
        self.last_orientation = self.get_orientation()
        self.orientation = orientation

    def get_orientation(self):
        return self.orientation

    def get_pose(self):
        return np.array(self.pos), self.orientation

    def update_pose(self, pos, orientation):
        self.set_position(pos)
        self.set_orientation(orientation % (np.pi * 2))

    @property
    def last_delta_orientation(self):
        a0 = self.last_orientation
        a1 = self.get_orientation()
        return a1 - a0


class MobilePoint(OrientedPoint):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lin_vel = 0.0
        self.ang_vel = 0.0
        self.ang_acc = 0.0
        self.cum_dst = 0.0
        self.dst = 0.0

    def get_angularvelocity(self):
        return self.ang_vel

    def get_linearvelocity(self):
        return self.lin_vel

    def set_linearvelocity(self, lin_vel):
        self.lin_vel = lin_vel

    def set_angularvelocity(self, ang_vel):
        self.ang_vel = ang_vel

    def update_all(self, pos, orientation, lin_vel, ang_vel):
        self.set_position(pos)
        self.set_orientation(orientation % (np.pi * 2))
        self.set_linearvelocity(lin_vel)
        self.set_angularvelocity(ang_vel)


class MobileVector(MobilePoint):
    length = PositiveNumber(1, doc='The initial length of the body in meters')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def front_end(self):
        return self.translate((self.length / 2, 0))

    @property
    def rear_end(self):
        return self.translate((-self.length / 2, 0))

    def drag_to_front(self, fp, d_or=0):
        o = self.get_orientation() + d_or
        k = np.array([np.cos(o), np.sin(o)])
        p = fp - k * self.length / 2
        self.update_pose(p, o)



class LineExtended(NestedConf):
    width = PositiveNumber(0.001, softmax=10.0, doc='The width of the line vertices')
    vertices = XYLine(doc='The list of 2d points')
    closed = param.Boolean(False, doc='Whether the line is closed')

    @property
    def Nvertices(self):
        return len(self.vertices)

    @property
    def _edges(self):
        vs = self.vertices
        edges = [[vs[i], vs[i + 1]] for i in range(self.Nvertices - 1)]
        if self.closed:
            edges.append([vs[self.Nvertices], vs[0]])
        return edges


class LineClosed(LineExtended):
    closed = param.Boolean(True)


class Area2D(NestedConf):
    dims = PositiveRange(doc='The arena dimensions')
    centered = param.Boolean(True, doc='Whether area is centered to (0,0)')

    @property
    def w(self):
        return self.dims[0]

    @property
    def h(self):
        return self.dims[1]

    @property
    def range(self):
        X, Y = self.dims
        return np.array([-X / 2, X / 2, -Y / 2, Y / 2])


class Area2DPixel(Area2D):
    dims = PositiveIntegerRange((100, 100), softmax=10000, step=1, doc='The arena dimensions in pixels')

    def get_rect_at_pos(self, pos=(0, 0), **kwargs):
        import pygame
        if pos is not None and not any(np.isnan(pos)):
            if self.centered:
                return pygame.Rect(pos[0] - self.w / 2, pos[1] - self.h / 2, self.w, self.h)
            else:
                return pygame.Rect(pos[0], pos[1], self.w, self.h)
        else:
            return None


class Area(Area2D):
    dims = PositiveRange((0.1, 0.1), softmax=1.0, step=0.01, doc='The arena dimensions in meters')
    geometry = param.Selector(objects=['circular', 'rectangular'], doc='The arena shape')
    torus = param.Boolean(False, doc='Whether to allow a toroidal space')


class PosPixelRel2Point(Pos2DPixel):
    reference_point = param.ClassSelector(Pos2DPixel, doc='The reference position instance', is_instance=False)
    pos_scale = PositiveRange((0.5, 0.5), softmax=1.0, step=0.01,
                              doc='The position relative to reference position')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update_pos()

    @param.depends('pos_scale', 'reference_point', watch=True)
    def update_pos(self):
        w, h = self.pos_scale
        x_pos = int(self.reference_point.x * w)
        y_pos = int(self.reference_point.y * h)
        self.pos = (x_pos, y_pos)


class PosPixelRel2Area(Pos2DPixel):
    reference_area = param.ClassSelector(Area2DPixel, doc='The reference position instance', is_instance=True)
    pos_scale = PositiveRange((0.5, 0.5), softmax=1.0, step=0.01,
                              doc='The position relative to reference position')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update_pos()

    @param.depends('pos_scale', 'reference_area', watch=True)
    def update_pos(self):
        w, h = self.pos_scale
        x_pos = int(self.reference_area.w * w)
        y_pos = int(self.reference_area.h * h)
        self.pos = (x_pos, y_pos)


class BoundedArea(Area, LineClosed):
    boundary_margin = param.Magnitude(1.0, doc='The boundary margin width relative to the area vertices')

    def __init__(self, vertices=None, **kwargs):

        Area.__init__(self, **kwargs)
        X, Y = self.dims
        if vertices is None:
            if self.geometry == 'circular':
                # This is a circle_to_polygon shape from the function
                vertices = aux.circle_to_polygon(60, X / 2)
            elif self.geometry == 'rectangular':
                # This is a rectangular shape
                vertices = [(-X / 2, -Y / 2),
                            (-X / 2, Y / 2),
                            (X / 2, Y / 2),
                            (X / 2, -Y / 2)]
        LineClosed.__init__(self, vertices=vertices, **kwargs)

    @property
    def polygon(self):
        return Polygon(np.array(self.vertices) * self.boundary_margin)

    def in_area(self, p):
        return self.polygon.contains(geometry.Point(p))
