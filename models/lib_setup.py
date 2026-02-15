import sys
import subprocess
import csv

with open('models\environment.csv', 'r') as file:
    csvFile = csv.reader(file)
    for lines in csvFile:
            for i in lines:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', f'{i}'])