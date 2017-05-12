import btk
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as anim
import mpl_toolkits.mplot3d.axes3d as p3
from scipy.spatial import distance
from sklearn.cluster import SpectralClustering
from scipy.optimize import minimize

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

def RBC(affinity_matrix): #Didn't work!
    '''
    Carries out Rigid Body Clustering on affinity matrix to group markers. 
    Should input empty matrix i.e. zeros(num_markers,num_markers).
    '''
    for i in range(0, num_markers):
        for j in range(0, num_markers):
            if i == j:
                continue
            mut_dists = []
            for k in range(0, num_frames, 10):
                dist = distance.euclidean(acq.GetPoint(i).GetValues()[k], acq.GetPoint(j).GetValues()[k])
                mut_dists.append(dist)
            rho = np.std(mut_dists)
            sig = (1.0/num_frames) * np.sum(mut_dists)
            affinity_matrix[i][j] = np.exp((-1.0*rho)/(2.0*sig*sig))
    c1 = SpectralClustering(12, affinity='precomputed')
    return c1.fit_predict(affinity_matrix)    
    
for i in range(0, num_frames, 10):
    x = []
    y = []
    z = []

    hip_origin = acq.GetPoint(0).GetValues()[i]
    head_top = acq.GetPoint(16).GetValues()[i]
    right_shoulder = acq.GetPoint(20).GetValues()[i]
    left_shoulder = acq.GetPoint(21).GetValues()[i]
    right_hip = np.mean([acq.GetPoint(1).GetValues()[i], acq.GetPoint(3).GetValues()[i], acq.GetPoint(5).GetValues()[i], acq.GetPoint(38).GetValues()[i]], axis=0)
    left_hip = np.mean([acq.GetPoint(2).GetValues()[i], acq.GetPoint(4).GetValues()[i], acq.GetPoint(6).GetValues()[i], acq.GetPoint(42).GetValues()[i]], axis=0)
    top_spine = np.mean([acq.GetPoint(12).GetValues()[i], acq.GetPoint(13).GetValues()[i], acq.GetPoint(15).GetValues()[i]], axis=0)
    mid_spine = np.mean([acq.GetPoint(10).GetValues()[i], acq.GetPoint(11).GetValues()[i], acq.GetPoint(14).GetValues()[i]], axis=0)
    head_base = np.mean([acq.GetPoint(17).GetValues()[i], acq.GetPoint(18).GetValues()[i], acq.GetPoint(19).GetValues()[i]], axis=0)
    right_elbow = np.mean([acq.GetPoint(22).GetValues()[i], acq.GetPoint(23).GetValues()[i], acq.GetPoint(28).GetValues()[i]], axis=0)
    left_elbow = np.mean([acq.GetPoint(24).GetValues()[i], acq.GetPoint(25).GetValues()[i], acq.GetPoint(31).GetValues()[i]], axis=0)
    right_knee = np.mean([acq.GetPoint(39).GetValues()[i], acq.GetPoint(40).GetValues()[i], acq.GetPoint(41).GetValues()[i], acq.GetPoint(48).GetValues()[i]], axis=0)
    left_knee = np.mean([acq.GetPoint(43).GetValues()[i], acq.GetPoint(44).GetValues()[i], acq.GetPoint(45).GetValues()[i], acq.GetPoint(51).GetValues()[i]], axis=0)
    right_wrist = np.mean([acq.GetPoint(26).GetValues()[i], acq.GetPoint(27).GetValues()[i], acq.GetPoint(34).GetValues()[i]], axis=0)
    left_wrist = np.mean([acq.GetPoint(29).GetValues()[i], acq.GetPoint(30).GetValues()[i], acq.GetPoint(37).GetValues()[i]], axis=0)
    right_heel = np.mean([acq.GetPoint(46).GetValues()[i], acq.GetPoint(47).GetValues()[i], acq.GetPoint(52).GetValues()[i], acq.GetPoint(56).GetValues()[i]], axis=0)
    left_heel = np.mean([acq.GetPoint(49).GetValues()[i], acq.GetPoint(50).GetValues()[i], acq.GetPoint(58).GetValues()[i], acq.GetPoint(62).GetValues()[i]], axis=0)
    right_foot = np.mean([acq.GetPoint(53).GetValues()[i], acq.GetPoint(54).GetValues()[i], acq.GetPoint(55).GetValues()[i]], axis=0)
    left_foot = np.mean([acq.GetPoint(59).GetValues()[i], acq.GetPoint(60).GetValues()[i], acq.GetPoint(61).GetValues()[i]], axis=0)
    
    x.extend([hip_origin[0], head_top[0], right_shoulder[0], left_shoulder[0], right_hip[0], left_hip[0], mid_spine[0], top_spine[0], head_base[0], right_elbow[0], left_elbow[0], right_knee[0], left_knee[0], right_wrist[0], left_wrist[0], right_heel[0], left_heel[0], right_foot[0], left_foot[0]])
    y.extend([hip_origin[1], head_top[1], right_shoulder[1], left_shoulder[1], right_hip[1], left_hip[1], mid_spine[1], top_spine[1], head_base[1], right_elbow[1], left_elbow[1], right_knee[1], left_knee[1], right_wrist[1], left_wrist[1], right_heel[1], left_heel[1], right_foot[1], left_foot[1]])
    z.extend([hip_origin[2], head_top[2], right_shoulder[2], left_shoulder[2], right_hip[2], left_hip[2], mid_spine[2], top_spine[2], head_base[2], right_elbow[2], left_elbow[2], right_knee[2], left_knee[2], right_wrist[2], left_wrist[2], right_heel[2], left_heel[2], right_foot[2], left_foot[2]])
    xlist.append(x)
    ylist.append(y)
    zlist.append(z)

def len_var(joint_index, parent_index):
    '''
    Plots length between two connected joints each frame to observe variation in length.
    '''
    plt.figure()
    dist_list = []
    for i in range(0, len(xlist)):
        dist = distance.euclidean(np.array([xlist[i][joint_index],ylist[i][joint_index],zlist[i][joint_index]]),np.array([xlist[i][parent_index],ylist[i][parent_index],zlist[i][parent_index]]))
        dist_list.append(dist)
    plt.plot(list(np.arange(len(xlist))),dist_list)
    plt.xlabel('Frame')
    plt.ylabel('Bone length')
    return dist_list
    
#len_var(13,9) 
            
def cost_func(length):
    '''
    Cost function to enforce constant bone lengths.
    '''
    total_cost = 0.0
    for i in range(0,num_frames/10):
        joint = np.array([xlist[i][13],ylist[i][13],zlist[i][13]])
        parent = np.array([xlist[i][9],ylist[i][9],zlist[i][9]])
        e = joint - parent
        cost = np.linalg.norm(joint - (parent + (length * e/np.linalg.norm(e))))
        total_cost += cost
    return total_cost
    
res = minimize(cost_func, 260.0, options={'disp':True})
print res.x

def const_bone(joint_index, parent_index):
    '''
    Changes position of joint to ensure constant bone lengths.
    '''
    for i in range(0,len(xlist)):
        disp = np.array([xlist[i][joint_index],ylist[i][joint_index],zlist[i][joint_index]]) - np.array([xlist[i][parent_index],ylist[i][parent_index],zlist[i][parent_index]])
        dispN = disp/np.linalg.norm(disp)
        new_joint = (res.x * dispN) + np.array([xlist[i][parent_index],ylist[i][parent_index],zlist[i][parent_index]])
        xlist[i][joint_index] = new_joint[0]
        ylist[i][joint_index] = new_joint[1]
        zlist[i][joint_index] = new_joint[2]
    return 0
    
const_bone(13,9)
        
def disp_joints(i):
    '''
    Updates joints and bones each frame.
    '''
    graph._offsets3d = (xlist[i], ylist[i], zlist[i])
    hips.set_data([xlist[i][5],xlist[i][0],xlist[i][4]], [ylist[i][5],ylist[i][0],ylist[i][4]])
    hips.set_3d_properties([zlist[i][5],zlist[i][0],zlist[i][4]])
    spine.set_data([xlist[i][0],xlist[i][6],xlist[i][7],xlist[i][8],xlist[i][1]], [ylist[i][0],ylist[i][6],ylist[i][7],ylist[i][8],ylist[i][1]])
    spine.set_3d_properties([zlist[i][0],zlist[i][6],zlist[i][7],zlist[i][8],zlist[i][1]])
    shoulders.set_data([xlist[i][2],xlist[i][7],xlist[i][3]], [ylist[i][2],ylist[i][7],ylist[i][3]])
    shoulders.set_3d_properties([zlist[i][2],zlist[i][7],zlist[i][3]])
    Rarm.set_data([xlist[i][2],xlist[i][9],xlist[i][13]], [ylist[i][2],ylist[i][9],ylist[i][13]])
    Rarm.set_3d_properties([zlist[i][2],zlist[i][9],zlist[i][13]])
    Larm.set_data([xlist[i][3],xlist[i][10],xlist[i][14]], [ylist[i][3],ylist[i][10],ylist[i][14]])
    Larm.set_3d_properties([zlist[i][3],zlist[i][10],zlist[i][14]])
    Rleg.set_data([xlist[i][4],xlist[i][11],xlist[i][15],xlist[i][17]], [ylist[i][4],ylist[i][11],ylist[i][15],ylist[i][17]])
    Rleg.set_3d_properties([zlist[i][4],zlist[i][11],zlist[i][15],zlist[i][17]])
    Lleg.set_data([xlist[i][5],xlist[i][12],xlist[i][16],xlist[i][18]], [ylist[i][5],ylist[i][12],ylist[i][16],ylist[i][18]])
    Lleg.set_3d_properties([zlist[i][5],zlist[i][12],zlist[i][16],zlist[i][18]])
    return 0

# Create animation
fig = plt.figure()
ax = p3.Axes3D(fig)
#ax = fig.add_subplot(111, projection='3d')
graph = ax.scatter(xlist[0], ylist[0], zlist[0], s=20, c='black')
hips, = ax.plot([xlist[0][5],xlist[0][0],xlist[0][4]], [ylist[0][5],ylist[0][0],ylist[0][4]], [zlist[0][5],zlist[0][0],zlist[0][4]], 'r')
spine, = ax.plot([xlist[0][0],xlist[0][6],xlist[0][7],xlist[0][8],xlist[0][1]], [ylist[0][0],ylist[0][6],ylist[0][7],ylist[0][8],ylist[0][1]], [zlist[0][0],zlist[0][6],zlist[0][7],zlist[0][8],zlist[0][1]], 'r')
shoulders, = ax.plot([xlist[0][2],xlist[0][7],xlist[0][3]], [ylist[0][2],ylist[0][7],ylist[0][3]], [zlist[0][2],zlist[0][7],zlist[0][3]], 'r')
Rarm, = ax.plot([xlist[0][2],xlist[0][9],xlist[0][13]], [ylist[0][2],ylist[0][9],ylist[0][13]], [zlist[0][2],zlist[0][9],zlist[0][13]], 'r')
Larm, = ax.plot([xlist[0][3],xlist[0][10],xlist[0][14]], [ylist[0][3],ylist[0][10],ylist[0][14]], [zlist[0][3],zlist[0][10],zlist[0][14]], 'r')
Rleg, = ax.plot([xlist[0][4],xlist[0][11],xlist[0][15],xlist[0][17]], [ylist[0][4],ylist[0][11],ylist[0][15],ylist[0][17]], [zlist[0][4],zlist[0][11],zlist[0][15],zlist[0][17]], 'r')
Lleg, = ax.plot([xlist[0][5],xlist[0][12],xlist[0][16],xlist[0][18]], [ylist[0][5],ylist[0][12],ylist[0][16],ylist[0][18]], [zlist[0][5],zlist[0][12],zlist[0][16],zlist[0][18]], 'r')
ax.set_xlabel('x')
ax.set_xlim(-500,2000)
ax.set_ylabel('y')
ax.set_ylim(-500,2000)
ax.set_zlabel('z')
ax.axis("off")
ani = anim.FuncAnimation(fig, disp_joints, num_frames, interval=0, repeat=True, blit=False)

plt.show()



#print('Marker labels:')
#for i in range(0, num_markers):
#    print acq.GetPoint(i).GetLabel() 
#    print acq.GetPoint(i).GetValues()
#    x.append(acq.GetPoint(i).GetValue(0,0)) 
#    y.append(acq.GetPoint(i).GetValue(0,1))
#    z.append(acq.GetPoint(i).GetValue(0,2))
    #print acq.GetPoint(i).GetValue(0,0) #GetValue(frame,coord)
    #print acq.GetPoint(i).GetValue(0,1)
    #print acq.GetPoint(i).GetValue(0,2)

#print('\n\nAnalog channels:')
#for i in range(0, acq.GetAnalogs().GetItemNumber()):
#    print acq.GetAnalog(i).GetLabel()

#plt.plot(x,y,z,'orange',marker='o',linestyle='None')
#plt.show()
    
print acq.GetPointFrequency() #Sampling rate (fps)
print acq.GetPointFrameNumber() #No of frames
print acq.GetPoints().GetItemNumber() #No. of markers

