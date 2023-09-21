def opencv_to_yolo(x, y, w, h, image_height, image_width):
    x /= image_width
    y /= image_height
    w /= image_width
    h /= image_height
    x += w/2
    y += h/2
    return x, y, w, h


def yolo_to_opencv(x, y, w, h, image_height, image_width):
    x, y, w, h = map(float, (x, y, w, h))
    x *= image_width
    y *= image_height
    w *= image_width
    h *= image_height
    x -= w/2
    y -= h/2
    x, y, w, h = map(int, (x, y, w, h))
    return x, y, w, h


# 30 fps
def frame_to_time(frame_nr):
    # get the time in hours
    return frame_nr / (30 * 60 * 60)


# def yolo_to_detections():
#     pass