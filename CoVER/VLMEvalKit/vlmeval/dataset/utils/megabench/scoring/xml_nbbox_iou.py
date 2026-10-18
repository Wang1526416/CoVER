import logging
from .common.metrics import calculate_iou
from .common.conversions import parse_bboxes_from_xml
from numbers import Number


class XmlNbboxIouSingle:

    @classmethod
    def match(cls, responses, targets) -> float:

        logging.debug(f"{responses=}, {targets=}")
        if not isinstance(responses, (tuple | list)):
            responses = parse_bboxes_from_xml(responses)
        if not isinstance(targets, (tuple | list)):
            targets = parse_bboxes_from_xml(targets)

        if len(responses) == 0:
            return 0
        elif isinstance(responses[0], Number) and len(responses) == 4:
            responses = [responses]

        iou_scores = calculate_iou(responses, targets)
        if not iou_scores:
            return 0


        return sum(iou_scores) / len(iou_scores)
