import sys
import subprocess
import csv

# make sure you run this before you try using any of the code cause you want to make sure you have all of the modules
with open('main/environment.csv', 'r') as file:
    csvFile = csv.reader(file)
    for lines in csvFile:
            for i in lines:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', f'{i}'])