import numpy as np
from models.functions_file import get_runtime
beam_angle_ls = []
no_beams = 2000
fan_angle = 30
@get_runtime
def linspace_fuck_you():
    np.linspace(-fan_angle/2, fan_angle/2, no_beams)


# general varable/list/dict setup yk how it be
@get_runtime
def im_better():
    if no_beams > 0:
        if no_beams == 1:
            beam_angle_ls.append(0)
        elif no_beams - 2 >= 0:
            if no_beams % 2 != 0:
                beam_angle_ls.append(0)
            if no_beams % 2 != 0:
                for i in range(2,no_beams-1,2):
                    angle = fan_angle*(i)/((no_beams-1)*2)
                    beam_angle_ls.append(angle)
                    beam_angle_ls.insert(0,-angle)
            else:        
                for i in range(2,no_beams-1,2):
                    angle = fan_angle*(i)/((no_beams)*2)
                    beam_angle_ls.append(angle)
                    beam_angle_ls.insert(0,-angle)
            beam_angle_ls.insert(0,-fan_angle/2)
            beam_angle_ls.append(fan_angle/2)
    beam_angle_ls
linspace_fuck_you()
im_better()