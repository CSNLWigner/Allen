import datetime
import pickle as pkl


now = datetime.datetime.now()
timestep = now.strftime('%Y-%m-%d_%H:%M:%S')
print(timestep)

with open('timestamp','wb') as file:
    pkl.dump(timestep,file)

"""
Usage:
with open('timestamp','rb') as file: timestamp = pkl.load(file)
"""