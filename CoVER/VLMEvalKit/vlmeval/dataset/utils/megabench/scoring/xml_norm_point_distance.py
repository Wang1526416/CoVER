
from .common.conversions import parse_point_2d_from_xml
from .common.metrics import point_distance


class XmlNormPointDistance:

    @classmethod
    def parse_2d_point(cls, point) -> tuple[float, float]:
        if not isinstance(point, (tuple | list)):
            point = parse_point_2d_from_xml(point)
            if not point:
                raise ValueError("Point could not be parsed from XML string.")
        elif len(point) != 2:
            raise ValueError("Point is not 2D.")
        if not all(0 <= comp <= 1 for comp in point):
            raise ValueError("Point is not normalized.")
        return tuple(point)

    @classmethod
    def match(cls, responses, targets) -> float:
        try:
            responses = cls.parse_2d_point(responses)
            targets = cls.parse_2d_point(targets)
        except ValueError:
            return 0


        return max(0, 1 - point_distance(responses, targets))
