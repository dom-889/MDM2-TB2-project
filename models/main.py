from model1 import grid_setup, video_capture
import numpy as np
import csv

# main function, you should know how these are used (i think)

def main():
    ipt = input("do you want to take a picture? [y/n] ")

    while True:

        if ipt == "y":
            video_capture()
            grid_setup(np.pi/4, 1000, 1, show_plot=True, beam_subdivisions=1000)
            break

        elif ipt == "n":
            grid_setup(np.pi/4, 200, 500, show_plot=True, beam_subdivisions=100)
            break

        else:
            ipt = input("incorrect input :( \ndo you want to take a picture? [y/n] ")
        
if __name__ == '__main__':
    main()
