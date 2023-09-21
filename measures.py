

# Calculates (box1 & box2) / (box1 | box2) 
def intersection_over_union(box1, box2):
    _, _, w1, h1 = box1
    _, _, w2, h2 = box2
    
    area1 = w1 * h1
    area2 = w2 * h2
    intersection_area = intersection(box1, box2)

    return intersection_area / (area1 + area2 - intersection_area) 


# Calculates (box2 & box1) / box2
def intersection_over_area(box1, box2):
    _, _, w2, h2 = box2
    area2 = w2 * h2
    intersection_area = intersection(box1, box2)


    return intersection_area / area2


# Calculates box2 & box1
def intersection(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate the coordinates of the intersection rectangle
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    w_intersection = min(x1 + w1, x2 + w2) - x_intersection
    h_intersection = min(y1 + h1, y2 + h2) - y_intersection

    # Check if there is an intersection
    if w_intersection > 0 and h_intersection > 0:
        # There is an overlapping area 
        overlapping_area = w_intersection * h_intersection
        return overlapping_area
    else:
        return 0