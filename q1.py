import sys
import os
import numpy as np
import csv
# import plotly.plotly as py
# import plotly.graph_objs as go
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pl
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# for plotting the contours and the mesh
fig = pl.figure()
fig2=pl.figure(2)

ax = fig.gca(projection='3d')

BATCH_SIZE=500
LEARN_RATE=0.009


# reding the x file

def load_x_lr():
	global rownum
	file_x=open('linearX.csv',"rU")
	reader=csv.reader(file_x,delimiter=",")
	
	rownum_x=0
	for row in reader:
		# [1,row]
		X_dat.append([1,float(row[0])])
		rownum_x+=1

	file_x.close()

# Reading the Y file


def load_y_lr():
	file_y=open('linearY.csv',"rU")
	reader=csv.reader(file_y,delimiter=",")
	rownum_y=0

	for row in reader:
		# [1,row]
		Y_dat.append(float(row[0]))
		rownum_y+=1

	file_y.close()

#  Oj =Oj+ (yi-hxi)/m * xj 
# end-start= batch BATCH_SIZE

# performs normal gradient descent, by finding iteration loss 

def desc(X_,Y_):

	Ttrans_x=np.matmul(T,np.transpose(X_))
	matri=np.matmul((Y_-Ttrans_x),X_)
	temp=np.matmul(X_,np.transpose(T))-Y_
	loss=np.matmul(np.transpose(temp),temp)
	return matri[0],matri[1],loss

# forms the mesh and the contours is called by gard_desc repeatedly

def plot_3d():
	theta1 = np.arange(-0.5,2,0.01)
	theta2 = np.arange(-0.5,0.5,0.01)
	theta1, theta2 = np.meshgrid(theta1, theta2)

	Z=0.0
	i=0

	for item in X_data:
		Z += (theta1 + theta2 * item[1] - Y_data[i])**2
		# print(Z)
		i+=1
	
	Z = Z /2.0

	surf = ax.plot_wireframe(theta1, theta2, Z,linewidth=0.5)
	surf2=pl.contour(theta1, theta2, Z)

	pl.ion()

# not used yet

def desc_with_batch(start, end):
	i=start
	summ0=0
	summ1=0
	loss=0
	while(i<end):
		Ttrans_x=np.dot(T,X_data[i])
		diff_loss=Y_data[i]-Ttrans_x
		loss+=(diff_loss*diff_loss)/2
		loss0=diff_loss*X_data[i][0]
		loss1=diff_loss*X_data[i][1]

		summ0+=loss0
		summ1+=loss1
		i+=1

	loss=loss/(end-start)
	summ0=summ0/(end-start)
	summ1=summ1/(end-start)
	return summ0,summ1,loss

def plot_data(theta):
	tempx=[]
	for item in X_data:
		tempx.append(item[1])

	pl.plot(tempx,Y_data,'ro')
	x = np.linspace(-2,5,1200)
	pl.plot(x,theta[1]*x+theta[0])
	pl.show()


# runs epochs till the loss obtained is very less , calls the method desc that computes the loss
def grad_desc(X_,Y_):
	temp=0
	plot_3d()
	while(1):
		i=0
		loss=0

		des0,des1,loss=desc(X_,Y_)
		
		T[0]=T[0]+LEARN_RATE * des0
		T[1]=T[1]+LEARN_RATE * des1

		X3.append(T[0])
		Y3.append(T[1])
		Z3.append(loss)
		# print("loss= ",loss)
		# plot_3d()
		# ax.scatter(T[0], T[1], loss, color='r')
		# pl.scatter(T[0],T[1], loss, color='r')
		# pl.pause(0.2)
		# plot_point_forJ(T)
		# pl.show()
		if abs(loss-temp)<0.0000000001:
			break
		temp=loss
	# print("loss=",loss)
	return loss
		# print(T)

	# print("jjjjjj\n\n\n")

# finds the value of theta using the normal equation and uses it to plot the curve
def desc_normaleq(X_,Y_):
	XtX=np.matmul(np.transpose(X_),X_)
	print(XtX)
	ans=np.matmul(np.matmul(np.linalg.inv(XtX),np.transpose(X_)),Y_)
	return ans

# normalise the X_data
def normalise_x(X):
	mu=np.mean(X, axis=0)
	var=np.var(X,axis=0)
	sd=np.sqrt(var)
	# print(mu)
	nor_X=[]

	for item in X:
		nor_X.append([1,(item[1]-mu[1])/sd[1]])

	return np.array(nor_X)

# kuch hagga hai
def normalise_y(X):
	mu=np.mean(X, axis=0)
	var=np.var(X,axis=0)
	sd=np.sqrt(var)
	# print(mu, var)
	nor_X=[]
	for item in X:
		nor_X.append((item-mu)/sd)

	return np.array(nor_X)


# T=theta
T=np.array([0.0,0.0])

X_dat=[]

rownum_x=0
Y_dat=[]


load_x_lr()
load_y_lr()


# convert to numpy arrays

X_data=normalise_x(np.array(X_dat))
Y_data=(np.array(Y_dat))

# print(X_data)

X3=[]
Y3=[]
Z3=[]

grad_desc(X_data, Y_data)
print(T)

# plot_3d()
# plt_contours()
# plot_data(T)

# lss=grad_desc(X_data,Y_data)

# tt=desc_normaleq(weightX_data,weightY_data)




		






































