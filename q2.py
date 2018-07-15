import sys
import os
import numpy as np
import csv

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pl
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

#  Oj =Oj+ (yi-hxi)/m * xj 
# end-start= batch size

# load data into arrays X_weighted
def load_weigh_x():
	file_x=open('weightedX.csv',"rU")
	reader=csv.reader(file_x,delimiter=",")
	
	row_x=0
	for row in reader:
		# [1,row]
		weightX_dat.append([1,float(row[0])])
		row_x+=1

	file_x.close()

# load Y_weighted data into arrays
def load_weigh_y():
	file_y=open('weightedY.csv',"rU")
	reader=csv.reader(file_y,delimiter=",")
	rownum_y=0



	for row in reader:
		weightY_dat.append(float(row[0]))
		rownum_y+=1

	file_y.close()	

# plots the straight line on data
def plot_data(theta):
	tempx=[]
	for item in weightX_data:
		tempx.append(item[1])

	maxx=np.max(np.array(tempx))
	minn=np.min(np.array(tempx))
	pl.plot(tempx,weightY_data,'ro')
	x = np.linspace(minn,maxx,1200)
	pl.plot(x,theta[1]*x+theta[0])
	pl.show()

# plots the weighted line for the data

def plot_weighted(theta):

	tempx=[]
	tempy=[]
	for item in theta:
		tempx.append(item[0])
		tempy.append(item[1])

	# print(theta)

	pl.plot(weightX_data[:,1],weightY_data,'ro')
	# x = np.linspace(0,20,1200)
	pl.plot(tempx,tempy)
	pl.show()

# creates the W atrix for each x
def create_w(x,X_):
	# dev_arr stores the values of wi
	dev_arr=[]
	count=0
	# calculate value of each element
	for item in X_:
		count+=1
		temp=np.exp((-1)*np.power((x-item),2)/(2*10*10))
		dev_arr.append(temp)

	w=np.zeros((count,count), float)
	n_cou=0
	# put in diagonal elements as weights
	for item in dev_arr:
		w[n_cou][n_cou]=item
		n_cou+=1

	return w

def normalise_x(X):
	mu=np.mean(X, axis=0)
	var=np.var(X,axis=0)
	sd=np.sqrt(var)
	# print(mu)
	nor_X=[]
	for item in X:
		nor_X.append([1,(item[1]-mu[1])/sd[1]])

	return np.array(nor_X)

def normalise_y(X):
	mu=np.mean(X, axis=0)
	var=np.var(X,axis=0)
	sd=np.sqrt(var)
	# print(mu, var)
	nor_X=[]
	for item in X:
		nor_X.append((item-mu)/sd)

	return np.array(nor_X)

def weighted_lr(X_, Y_):
	xs=[]
	# removes the zeros appended by the load_weigh_x() function, xs contains only pure values
	for item in X_:
		xs.append(item[1])
	xs_=np.array(xs)

	# print(xs_)
	maxi=np.amax(xs_, axis=0)
	mini=np.amin(xs_, axis=0)

	# print("\n\n",maxi)
	# generates random points on which we evaluate the weighted lr function
	points=np.linspace(mini,maxi,num=100)

	curve=[]

	for item in points:
		W=create_w(item,xs_)
		
		The=np.linalg.inv(np.matmul(np.matmul(np.transpose(X_),W),X_))
		Oth=np.matmul(np.matmul(np.transpose(X_),W),Y_)
		Theta=np.matmul(The,Oth)
		y=Theta[1]*item + Theta[0]
		# the corresponding values of x and y are stored as tuples
		curve.append([item,y])

	return curve

def desc_normaleq(X_,Y_):
	XtX=np.matmul(np.transpose(X_),X_)
	print(XtX)
	ans=np.matmul(np.matmul(np.linalg.inv(XtX),np.transpose(X_)),Y_)
	return ans


weightX_dat=[]
weightY_dat=[]

load_weigh_y()
load_weigh_x()

weightX_data=normalise_x(np.array(weightX_dat))
weightY_data=normalise_y(np.array(weightY_dat))


# curve=weighted_lr(weightX_data,weightY_data)

lin=desc_normaleq(weightX_data,weightY_data)
print(lin)

# plot_data(lin)
# plot_weighted(curve)










