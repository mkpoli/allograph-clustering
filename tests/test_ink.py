import numpy as np

from allograph_clustering.ink import FRAME, frame_one, keep_mask


def test_keep_mask_drops_specks_and_edge_fragments():
    ink = np.zeros((40, 40), bool)
    ink[10:30, 18:22] = True     # the stroke
    ink[0:6, 0:3] = True         # a fragment of the neighbour at the edge
    ink[35, 35] = True           # a speck
    keep = keep_mask(ink)
    assert keep[10:30, 18:22].all()
    assert not keep[0:6, 0:3].any()
    assert not keep[35, 35]


def test_frame_keeps_aspect_and_marks_faint_crops_empty():
    crop = np.full((160, 160), 230, np.uint8)
    crop[20:140, 30:40] = 20
    grey, mask, quality = frame_one((crop, (160, 80)))
    assert grey.shape == mask.shape == (FRAME, FRAME)
    assert mask[:, :24].max() == 0 and mask.max() == 255
    assert quality[0] > 0.5
    _, mask, quality = frame_one((np.full((160, 160), 200, np.uint8), (160, 160)))
    assert mask.max() == 0 and quality[0] == 0
