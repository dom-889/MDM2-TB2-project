from pick import pick
import os
import subprocess


options = [i for i in os.listdir(f"{os.getcwd()}/test_images")]
selected = pick(options, title="Choose an option")
print(f'You selected: {selected[0]}')