FRAME_PER_SEGMENT = 50

# BGRA
ROI_COLOR = (169, 169, 0, 96)
ROI_COLOR_BGR = (169, 169, 0)

def getColorMOI(moi_id):
    color = MOI_COLOR[moi_id]
    return color

def getColorMOI_BGRA(moi_id):
    color = getColorMOI(moi_id)
    return (color[2], color[1], color[0], color[3])

def getColorMOI_BGR(moi_id):
    color = getColorMOI(moi_id)
    return (color[2], color[1], color[0])

# RGBA color
MOI_COLOR = [(0, 0, 0, 0),
             (204, 51, 153, 255),  # pink
             (75, 0, 130, 255),  # violet
             (139, 69, 19, 255),  # brown
             (255, 153, 51, 255),  # orange
             (65, 105, 225, 255),  # light blue
             (0, 206, 209, 255),  # cyan
             (50, 205, 50, 255),  # light green
             (128, 128, 0, 255),  # dark yellow
             (220, 20, 60, 255),  # red
             (255, 215, 0, 255),  # light yellow
             (0, 100, 0, 255),  # dark green
             (25, 25, 112, 255),  # dark blue
             ]
