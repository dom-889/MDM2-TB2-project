from model2 import ring_thing, fan_setup, video_capture
from functions_file import optimise_function_inputs
import numpy as np
from pick import pick
import os

# main function, you should know how these are used (i think)

def main():
    ipt = input("do you want to take a picture? [y/n] ")

    while True:

        '''if ipt == "y":
            video_capture()
            grid_setup(fan_angle=np.pi/4, no_beams=50, ring_subdivisions=100, beam_subdivisions=100, image_string="test_image.png",show_plot=True)
            break'''

        if ipt == "n":
            options = [i for i in os.listdir(f"{os.getcwd()}/test_images")]
            selected = pick(options, title="Choose an image to reconstruct")
            print(f'You selected: {selected[0]}, reconstructing now...')
            fan_sd, ring_sd = optimise_function_inputs(selected[0])
            fan_ls = fan_setup(np.pi/4, 50)
            ring_thing(fan_list=fan_ls, ring_subdivisions=100, beam_subdivisions=100, aperture=1, image_string=selected[0])
            break

        else:
            ipt = input("incorrect input :( \ndo you want to take a picture? [y/n] ")
        
if __name__ == '__main__':
    main()
