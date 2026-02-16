fan_subdivisions = 3.2

fan_angle = 10
fan_sd_ls = []

if fan_subdivisions > 0:
    print("this works")
    if fan_subdivisions == 1:
        fan_sd_ls.append(0)
    elif fan_subdivisions - 2 >= 0:
        if fan_subdivisions % 2 != 0:
            print("added 0 to list")
            fan_sd_ls.append(0)

        if fan_subdivisions % 2 != 0:
            for i in range(2,fan_subdivisions-1,2):
                angle = fan_angle*(i)/((fan_subdivisions-1)*2)
                fan_sd_ls.append(angle)
                fan_sd_ls.insert(0,-angle)
        else:        
            for i in range(2,fan_subdivisions-1,2):
                angle = fan_angle*(i)/((fan_subdivisions)*2)
                fan_sd_ls.append(angle)
                fan_sd_ls.insert(0,-angle)

        fan_sd_ls.insert(0,-fan_angle/2)
        fan_sd_ls.append(fan_angle/2)
else:
    raise TypeError(f"variable fan_subdivisions = {fan_subdivisions} is either 0 or not int \nplease check the value/input is correct before proceeding")

print(f"{fan_sd_ls}, \n{len(fan_sd_ls)}")