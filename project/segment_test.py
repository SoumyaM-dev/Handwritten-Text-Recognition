# segment_test.py
from itertools import combinations

# 1) Sample bounding boxes: (x, y, w, h, label)
#    Pretend these came from a word‑detector on an image.
boxes = [
    (10,  20, 50, 20, "Hello"),
    (70,  18, 48, 22, "world"),
    (130, 22, 55, 18, "this"),
    (192, 19, 40, 20, "is"),
    (235, 21, 60, 19, "test"),
    # second “line”
    (12,  60, 45, 18, "Another"),
    (60,  58, 50, 20, "line"),
]

def center(box):
    x, y, w, h, _ = box
    return (x + w/2, y + h/2)

def group_by_gap(boxes, axis=1, max_gap=15):
    """
    Group boxes along axis=1 (y) into rows, or axis=0 (x) into columns.
    Boxes whose centers differ by <= max_gap are in the same group.
    Returns list of groups; each group is a list of boxes.
    """
    # start each box in its own group
    groups = [[b] for b in boxes]

    changed = True
    while changed:
        changed = False
        # attempt to merge any two groups whose members overlap in center-gap
        for g1, g2 in combinations(groups, 2):
            # check if any box in g1 and any in g2 are “close”
            for b1 in g1:
                for b2 in g2:
                    c1 = center(b1)[axis]
                    c2 = center(b2)[axis]
                    if abs(c1 - c2) <= max_gap:
                        # merge g2 into g1
                        g1.extend(g2)
                        groups.remove(g2)
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break
    return groups

# 2) Group into horizontal rows (by y-center)
rows = group_by_gap(boxes, axis=1, max_gap=15)
print("Rows detected:")
for row in rows:
    # sort left→right by x
    sorted_row = sorted(row, key=lambda b: center(b)[0])
    print("  ", [b[4] for b in sorted_row])

# 3) Group into vertical columns (by x-center)
cols = group_by_gap(boxes, axis=0, max_gap=30)
print("\nColumns detected:")
for col in cols:
    # sort top→bottom by y
    sorted_col = sorted(col, key=lambda b: center(b)[1])
    print("  ", [b[4] for b in sorted_col])
