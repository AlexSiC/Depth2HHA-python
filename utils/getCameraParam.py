# --*-- coding:utf-8 --*--
import numpy as np

'''
getCameraParam: get the camera matrix
colOrZ: color or depth
'''
def getCameraParam(colorOrZ='color'):
    if colorOrZ == 'color':
        fx_rgb = 6.92683e+02
        fy_rgb = 6.92738e+02
        cx_rgb = 4.08508e+02
        cy_rgb = 2.95696e+02
        C = np.array([[fx_rgb, 0, cx_rgb], [0, fy_rgb, cy_rgb], [0, 0, 1]])
    else:
        fx_d = 7.12929e+02
        fy_d = 7.12808e+02
        cx_d = 3.96903e+02
        cy_d = 2.95821e+02
        C = np.array([[fx_d, 0, cx_d], [0, fy_d, cy_d], [0, 0, 1]])
    return C