import btk
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as anim
import mpl_toolkits.mplot3d.axes3d as p3

reader = btk.btkAcquisitionFileReader()
reader.SetFilename("Mocap Data/816_0190.c3d")
reader.Update()
acq = reader.GetOutput()

num_markers = acq.GetPoints().GetItemNumber()
fps = acq.GetPointFrequency()
num_frames = acq.GetPointFrameNumber()

xlist = []
ylist = []
zlist = []

for j in range(0, num_frames, 10):
    x = []
    y = []
    z = []
    for i in range(0, num_markers):
        x.append(acq.GetPoint(i).GetValue(j,0)) 
        y.append(acq.GetPoint(i).GetValue(j,1))
        z.append(acq.GetPoint(i).GetValue(j,2))
    xlist.append(x)
    ylist.append(y)
    zlist.append(z)
    
def disp_c3d(i):
    graph._offsets3d = (xlist[i], ylist[i], zlist[i])
    return 0
    
fig = plt.figure()
ax = p3.Axes3D(fig)
graph = ax.scatter(xlist[0], ylist[0], zlist[0], c='blue')
ax.set_xlabel('x')
ax.set_xlim(-500,2000)
ax.set_ylabel('y')
ax.set_ylim(-500,2000)
ax.set_zlabel('z')
ax.axis("off")
ani = anim.FuncAnimation(fig, disp_c3d, num_frames, interval=0, repeat=True, blit=False)

plt.show()