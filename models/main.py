from model1 import grid_setup, video_capture
import numpy as np
from pick import pick
import os

# main function, you should know how these are used (i think)

def main():
    ipt = input("do you want to take a picture? [y/n] ")

    while True:

        if ipt == "y":
            video_capture()
            grid_setup(fan_angle=np.pi/4, no_beams=50, ring_subdivisions=100, beam_subdivisions=100, image_string="test_image.png",show_plot=True)
            break

        elif ipt == "n":
            options = [i for i in os.listdir(f"{os.getcwd()}/test_images")]
            selected = pick(options, title="Choose an image to reconstruct")
            print(f'You selected: {selected[0]}, reconstructing now...')
            grid_setup(fan_angle=np.pi/4, no_beams=200, ring_subdivisions=200, beam_subdivisions=1000, image_string=selected[0],show_plot=True)
            break

        else:
            ipt = input("incorrect input :( \ndo you want to take a picture? [y/n] ")
        
if __name__ == '__main__':
    main()
