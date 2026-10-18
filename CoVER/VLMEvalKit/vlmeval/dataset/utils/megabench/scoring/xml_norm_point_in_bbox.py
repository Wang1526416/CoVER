from .common.conversions import parse_point_2d_from_xml, str_to_bboxes


class XmlNormPointInBbox:

    @classmethod
    def match(cls, responses, eval_context) -> int:
        bounding_box_has_match = {
            bbox: False for bbox in eval_context["bounding_boxes"]
        }
        bounding_boxes = [
            str_to_bboxes(bbox_str)[0] for bbox_str in eval_context["bounding_boxes"]
        ]
        assert bounding_boxes

        if not isinstance(responses, (tuple | list)):
            responses = parse_point_2d_from_xml(responses)
            if not responses:
                return 0, bounding_box_has_match
        elif len(responses) != 2:
            return 0, bounding_box_has_match

        x, y = responses
        for min_x, min_y, max_x, max_y in bounding_boxes:
            if min_x <= x <= max_x and min_y <= y <= max_y:
                bounding_box_has_match[str((min_x, min_y, max_x, max_y))] = True
                return 1, bounding_box_has_match
        return 0, bounding_box_has_match
