import numpy, math
def rotate(rotatepam):
    a, b, c, x, y = rotatepam
    t = numpy.identity(4)
    t[0, 0] = math.cos(a)*math.cos(c)-math.cos(b)*math.sin(a)*math.sin(c)
    t[0, 1] = -math.cos(b)*math.cos(a)*math.sin(c)-math.cos(a)*math.sin(c)
    t[0, 2] = math.sin(a)*math.sin(b)

    t[1, 0] = math.cos(c)*math.sin(a)+math.cos(a)*math.cos(b)*math.sin(c)
    t[1, 1] = math.cos(a)*math.cos(b)*math.cos(c)-math.sin(a)*math.sin(c)
    t[1, 2] = -math.cos(a)*math.sin(b)
 
    t[2, 0] = math.sin(b)*math.sin(c)
    t[2, 1] = math.cos(a)*math.sin(b)
    t[2, 2] = math.cos(b)
    return t


def translation(displacement):
    t = numpy.identity(4)
    t[0, 3] = displacement[0]
    t[1, 3] = displacement[1]
    t[2, 3] = displacement[2]
    return t


def scaling(scale):
    s = numpy.identity(4)
    s[0, 0] = scale[0]
    s[1, 1] = scale[1]
    s[2, 2] = scale[2]
    s[3, 3] = 1
    return s
