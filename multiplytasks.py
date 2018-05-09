import subprocess
import numpy
import re
import os
from shutil import copyfile

completedFile = "logs/completed.log"
lines = numpy.loadtxt(completedFile, delimiter="\n", unpack=False, dtype=numpy.str)

tasks = os.listdir("tasks")
copiesNum = 4

for task in tasks:
    task = 'ae-sparc-10352_8_cut.task'
    taskFile = re.match(r'(.+)\.task', task)
    if not taskFile:
        continue
    taskName = taskFile.group(1)
    if taskName[-5:-1] == '-run':
        continue
    for num in range(copiesNum):
        copyfile("tasks/{}.task".format(taskName), "tasks/{}-run{}.task".format(taskName, num+1))
    break