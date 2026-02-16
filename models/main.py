from model1 import grid_setup, video_capture
import numpy as np


# main function, you should know how these are used (i think)

def main():

    ipt = input("do you want to take a picture? [y/n] ")

    while True:

        if ipt == "y":
            video_capture()
            grid_setup(np.pi/4, 100, 500, False)
            break

        elif ipt == "n":
            grid_setup(np.pi/4, 10, 20, show_plot=True)
            break

        else:
            ipt = input("incorrect input :( \ndo you want to take a picture? [y/n] ")
        
if __name__ == '__main__':
    main()
