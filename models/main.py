from model1 import grid_setup, video_capture


# main function folder, you should know how these are used

def main():

    ipt = input("do you want to take a picture? [y/n] ")

    while True:

        if ipt == "y":
            video_capture()
            break

        elif ipt == "n":
            grid_setup()
            break

        else:
            ipt = input("incorrect input :( \ndo you want to take a picture? [y/n] ")
        
if __name__ == '__main__':
    main()
