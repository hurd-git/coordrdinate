from abc import ABC, abstractmethod, ABCMeta
from typing import Union, Tuple, Dict
from enum import Enum, unique, EnumMeta

import numpy as np
from numpy import ndarray


def map_angle(angle):
    """
    map angle to [0, 2*pi)
    """
    return angle % (2 * np.pi)


# class CooEnumMeta(type):
#     def __new__(mcls, name, bases, namespace):
#         super().__new__(mcls, name, bases, namespace)


# class Coo2D(Enum):
#     __dim__ = 2
#
#     Cartesian2D = 1
#     Polar = 2
#
#
# class Coo3D(Enum):
#     __dim__ = 3
#
#     Cartesian3D = 1
#     Cylindrical = 2
#     Spherical = 3


def _get_str(var: Union[str, "Coo2D"]):
    if isinstance(var, Coo2D):
        return var.name
    elif isinstance(var, str):
        return var
    else:
        raise TypeError("Unsupported input type")


class Space(ABC):
    _coo: EnumMeta
    _axes: Dict[str, tuple]
    _axes_lock: Dict[str, bool]

    # def __init__(self):
    #     for axis in self._coo.__members__:
    #         self._axes[axis] = tuple(0.0 for _ in range(self._coo.__dim__))


class Coordinate(ABC):
    def __new__(cls, *args, **kwargs):
        if cls in {Coordinate, Coordinate2D, Coordinate3D}:
            raise TypeError("Can't instantiate abstract class")
        return object.__new__(cls)


class Space2D(Space):

    _axes: Dict[str, Tuple[float, float]] = {
        'Cartesian2D': (0., 0.),
        'Polar': (0., 0.),
    }
    _axes_lock: Dict[str, bool] = {
        'Cartesian2D': False,
        'Polar': False,
    }

    def __init__(self):
        self._coo = Coo2D
        # super().__init__()

    @property
    def axes(self):
        return self._axes

    def get_axis(self, coordinate_name: Union[str, "Coo2D"]) -> Tuple[float, float]:
        name = _get_str(coordinate_name)
        return self._axes[name]

    def set_axis(self, coordinate_name: Union[str, "Coo2D"], x: float, y: float):
        name = _get_str(coordinate_name)
        if name in self._coo.__members__:
            if self._axes_lock[name] is False:
                self._axes[name] = (float(x), float(y))
            else:
                raise AttributeError("Cannot set a new position on the locked axis")
        else:
            raise TypeError("Unsupported coordinate")

    def lock_axis(self, coordinate_name: Union[str, "Coo2D"]):
        name = _get_str(coordinate_name)
        if name in self._coo.__members__:
            self._axes_lock[name] = True
        else:
            raise TypeError("Unsupported coordinate")

    def unlock_axis(self, coordinate_name: Union[str, "Coo2D"]):
        name = _get_str(coordinate_name)
        if name in self._coo.__members__:
            self._axes_lock[name] = False
        else:
            raise TypeError("Unsupported coordinate")

    def lock_all_axes(self):
        for name in self._coo.__members__:
            self._axes_lock[name] = True

    def unlock_all_axes(self):
        for name in self._coo.__members__:
            self._axes_lock[name] = False

    def newCartesian2D(self, x: Union[float, ndarray], y: Union[float, ndarray]):
        return Cartesian2D(x, y, self)

    def newPolar(self, x: Union[float, ndarray], y: Union[float, ndarray]):
        return Polar(x, y, self)


class Space3D(Space):
    pass


class Coordinate2D(Coordinate, ABC):
    def __init__(self, space: Space2D = None):
        if space is None:
            space = Space2D()
        self._space = space

    @property
    def axis(self):
        return self.space.get_axis(self.__class__.__name__)

    def set_axis(self, x: float, y: float):
        self.space.set_axis(self.__class__.__name__, x, y)

    @property
    def space(self):
        return self._space

    @space.setter
    def space(self, space: Space2D):
        self.set_space(space)

    def set_space(self, space: Space2D):
        if space.__class__ is Space2D:
            self.space = space
        else:
            raise TypeError("Wrong space type")

    @abstractmethod
    def toCartesian(self) -> "Cartesian2D":
        pass

    @abstractmethod
    def toPolar(self) -> "Polar":
        pass


class Cartesian2D(Coordinate2D):
    def __init__(self, x: Union[float, ndarray], y: Union[float, ndarray], space: Space2D = None):
        super().__init__(space)
        self.x = np.array(x, dtype=float)
        self.y = np.array(y, dtype=float)

    def toCartesian(self):
        return self

    def toPolar(self):
        radius = np.sqrt(np.square(self.x - self.axis[0]) +
                         np.square(self.y - self.axis[1]))
        angle = map_angle(
            np.arctan2(self.y - self.axis[1],
                       self.x - self.axis[0])
        )
        return Polar(radius, angle, space=self.space)

    def __str__(self) -> str:
        return ""

    def __getitem__(self, item):
        return Cartesian2D(self.x[item], self.y[item], space=self.space)


class Polar(Coordinate2D):
    def __init__(self, radius: Union[float, ndarray], angle: Union[float, ndarray], space: Space2D = None):
        super().__init__(space)
        self.radius = np.array(radius, dtype=float)
        self.angle = np.array(angle, dtype=float)

    def toCartesian(self):
        r = self.radius * np.cos(self.angle) + self.axis[0]
        z = self.radius * np.sin(self.angle) + self.axis[1]
        return Cartesian2D(r, z, space=self.space)

    def toPolar(self):
        return self

    def __str__(self) -> str:
        return ""

    def __getitem__(self, item):
        return Polar(self.radius[item], self.angle[item], space=self.space)


class Coordinate3D(Coordinate, ABC):
    @abstractmethod
    def toCartesian(self):
        pass

    @abstractmethod
    def toCylindrical(self):
        pass

    @abstractmethod
    def toSpherical(self):
        pass


class Cartesian3D:
    pass


class Cylindrical:
    pass


class Spherical:
    pass


class Coo2D(Enum):
    __dim__ = 2

    Cartesian2D = Cartesian2D
    Polar = Polar


class Coo3D(Enum):
    __dim__ = 3

    Cartesian3D = Cartesian3D
    Cylindrical = Cylindrical
    Spherical = Spherical


# TODO: 不同坐标轴之间的投影、2D、3D、2D+3D

myspace = Space2D()
myspace.set_axis(Coo2D.Polar, 1, 1)

p = Cartesian2D(1, 2, myspace)

p1 = p.toCartesian()
p2 = p1.toPolar()
p3 = p2.toCartesian()
