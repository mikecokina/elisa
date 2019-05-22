from pypex.base import shape
from pypex.poly2d.intersection import linter
from pypex.base.conf import ROUND_PRECISION


class Line(shape.Shape2D):
    __intersect__ = ['INTERSECT']
    __overlapping__ = ['OVERLAP']

    def __str__(self):
        return "Line: [{}]".format(", ".join([str(v) for v in self.hull]))

    def __repr__(self):
        return "Line: [{}]".format(", ".join([str(v) for v in self.hull]))

    def intersects(self, line, _full=False, in_touch=False, tol=ROUND_PRECISION):
        """
        Figure out whether two line are in intersection or not

        :param tol: int; consider as same up to 'tol' decimal numbers
        :param in_touch: bool
        :param line: pypex.poly2d.line.Line
        :param _full: bool; define whether return full output or not
        :return: bool or tuple
        """

        # fixme: return dual type is probably not a good idea
        intersection = linter.intersection(self.hull[0], self.hull[1], line.hull[0], line.hull[1], in_touch, tol)
        if _full:
            return intersection
        return intersection[1] and (intersection[4] in "INTERSECT")

    def full_intersects(self, line, in_touch=False, tol=ROUND_PRECISION):
        """
        Figure out whether two line are in intersection or not.
        Method is here to avoid repeating evaluation of if condition statement in case of _full=True in intersects()
        method in case when is necessary to find intersection of huge amount of lines in loop

        :param tol: int; consider as same up to 'tol' decimal numbers
        :param in_touch: bool
        :param line: pypex.poly2d.line.Line
        :return: tuple
        """
        return linter.intersection(self.hull[0], self.hull[1], line.hull[0], line.hull[1], in_touch, tol)

    def intersection(self, line, in_touch=False, tol=ROUND_PRECISION):
        """
        Find intersection point of two lines if exists.

        :param tol: int
        :param in_touch: bool
        :param line: pypex.poly2d.line.Line
        :return: pypex.poly2d.point.Point or None
        """

        intersection = self.full_intersects(line, in_touch=in_touch, tol=tol)
        intersect = intersection[1] and (intersection[4] in "INTERSECT")
        if not intersect:
            return None
        return intersection[2]

    def sort_clockwise(self, *args, **kwargs):
        return self.hull
