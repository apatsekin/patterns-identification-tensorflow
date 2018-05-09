import time
timeContainer = {}
timersCount = {}
def measure_time(tag='default', comment='Time', times=5):
    if tag in timeContainer:
        if timersCount[tag] > times:
            return
        print("-- {}: {} seconds".format(comment, time.time() - timeContainer[tag]))
        timersCount[tag] += 1


def start_timer(tag='default'):
    timeContainer[tag] = time.time()
    if tag not in timersCount:
        timersCount[tag] = 1