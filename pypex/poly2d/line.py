from pypex.base import shape
from pypex.poly2d.intersection import linter
from pypex.base.conf import ROUND_PRECISION as PRECISION


class Line(shape.Shape2D):
    __intersect__ = ['INTERSECT']
    __overlapping__ = ['OVERLAP']

    def __str__(self):
        return "Line: [{}]".format(", ".join([str(v) for v in self.hull]))

    def __repr__(self):
        return "Line: [{}]".format(", ".join([str(v) for v in self.hull]))

    def intersects(self, line, _full=False, in_touch=False, tol=PRECISION):
        """
        Figure out whether two line are in intersection or not

        :param tol: int; consider as same up to 'tol' decimal numbers
        :param in_touch: bool
        :param line: pypex.poly2d.line.Line
        :param _full: bool; define whether return full output or not
        :return: bool or tuple
        """

        # fixme: return dual type is probably not a good idea
        intersection = linter.intersection(self.hull[0], self.hull[1], line.hull[0], line.hull[1], in_touch)
        if _full:
            return intersection
        return intersection[1] and (intersection[4] in "INTERSECT")

    def intersection(self, line, in_touch=False, tol=PRECISION):
        """
        Find intersection point of two lines if exists.

        :param tol: int
        :param in_touch: bool
        :param line: pypex.poly2d.line.Line
        :return: pypex.poly2d.point.Point or None
        """

        intersection = self.intersects(line, _full=True, in_touch=in_touch, tol=tol)
        intersect = intersection[1] and (intersection[4] in "INTERSECT")
        if not intersect:
            return None
        return intersection[2]

    def sort_clockwise(self, *args, **kwargs):
        return self.hull
