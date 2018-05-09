import subprocess
import numpy
import re
import os

completedFile = "logs/completed.log"
lines = numpy.loadtxt(completedFile, delimiter="\n", unpack=False, dtype=numpy.str)

tasks = os.listdir("tasks")

for task in tasks:
    taskFile = re.match(r'(.+)\.task', task)
    if not taskFile:
        continue
    taskName = taskFile.group(1)
    if taskName in lines:
        print("skipping {} task".format(taskName))
        continue
    bashCommand = "python3 cnn_feature_cluster.py --task={}".format(taskName, taskName)
    print("Starting process {}...".format(taskName))
    process = subprocess.run(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Process {} done...".format(taskName))
    lines = numpy.append(lines,[taskName])
    numpy.savetxt(completedFile, lines, delimiter="\n", fmt="%s")
    output, error = process.stdout, process.stderr
    print(error)
    pass