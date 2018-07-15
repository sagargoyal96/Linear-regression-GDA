import sys
import os
import numpy as np
import csv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pl
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# load data in the arrays of gda

def load_gda_x():
	file_x=open('q4x.dat',"rU")
	datContent = [i.strip().split() for i in file_x.readlines()]

	row_x=0
	for row in datContent:
		# [1,row]
		gdaX_dat.append([float(row[0]),float(row[1])])
		row_x+=1

	# print(gdaX_dat)
	file_x.close()

def load_gda_y():
	file_y=open('q4y.dat',"rU")
	datContent = [i.strip().split() for i in file_y.readlines()]

	row_y=0
	for row in datContent:
		if row[0]=='Alaska':
			gdaY_dat.append(0)
		else:
			gdaY_dat.append(1)
		row_y+=1
	file_y.close()


# plots the basic data of the entire gda
def plot_data_gda(X_, Y_):
	norm_X=normalise(X_)

	tempx=[]
	tempy=[]
	for item in norm_X:
		tempx.append(item[0])
		tempy.append(item[1])
	i=0
	for item in tempx:
		pl.plot(tempx[i],tempy[i],'ro',c= ('C2' if Y_[i] == 0 else 'C1'))
		i+=1
	
	# x = np.linspace(0,20,1200)
	# # y=mx+c
	# pl.plot(x,theta[0,1]*(-1)/theta[0,2]*x+theta[0,0]*(-1)/theta[0,2])
	pl.show()


# plots the data along with the line

def plot_gda_boun(theta,X,Y):

	X=normalise(X)
	maxx=np.max(X, axis=0)
	minn=np.min(X,axis=0)

	tempx=[]
	tempy=[]
	for item in X:
		tempx.append(item[0])
		tempy.append(item[1])
	i=0
	for item in tempx:
		pl.plot(tempx[i],tempy[i],'ro',c= ('C2' if Y[i] == 0 else 'C1'))
		i+=1
	
	x = np.linspace(minn[0],maxx[0],1200)
	pl.plot(x,theta[1]*(-1)/theta[2]*x+theta[0]*(-1)/theta[2])
	pl.show()

def normalise(X):
	mu=np.mean(X, axis=0)
	var=np.var(X,axis=0)
	sd=np.sqrt(var)

	nor_X=[]
	for item in X:
		nor_X.append([(item[0]-mu[0])/sd[0], (item[1]-mu[1])/sd[1]])

	return np.array(nor_X)

# finds all the parameters in theta by applying gda
def gda_theta(X_,Y_):
	# Alaska has been referred to as 0 and canada as 1

	i=0

	norm_X=normalise(X_)

	av_x0_0=0
	av_x1_0=0
	av_x0_1=0
	av_x1_1=0

	# lists of data separated by type (y=1, x0,x1 and y=0 x0,x1: x0_1--> y=0,x1)
	x0_0=[]
	x0_1=[]
	x1_0=[]
	x1_1=[]

	x0=[]
	x1=[]

	count_x0=0
	count_x1=0
	i=0
	for item in Y_:
		if item==0:
			count_x0+=1
			av_x0_0+=norm_X[i,0]
			av_x0_1+=norm_X[i,1]
			x0_0.append(norm_X[i,0])
			x0_1.append(norm_X[i,1])
			x0.append([norm_X[i,0],norm_X[i,1]])
		else:
			count_x1+=1
			av_x1_0+=norm_X[i,0]
			av_x1_1+=norm_X[i,1]
			x1_0.append(norm_X[i,0])
			x1_1.append(norm_X[i,1])
			x1.append([norm_X[i,0],norm_X[i,1]])

		i+=1

	x0a=np.array(x0)
	x1a=np.array(x1)

	av_x0_0/=count_x0
	av_x1_0/=count_x1
	av_x0_1/=count_x0
	av_x1_1/=count_x1

	# av matrix for y=0 and y=1
	u0=np.array([av_x0_0,av_x0_1])
	u1=np.array([av_x1_0,av_x1_1])

	print(u0, u1)

	phi=count_x1/(count_x1+count_x0)

	i=0

	#covariance matrices for u0 and u1
	S0=np.zeros((2,2))
	S1=np.zeros((2,2))
	Seq=np.zeros((2,2))

	for item in x0:
		S0+=np.matmul(np.transpose(np.matrix(x0a[i]-u0)),np.matrix(x0a[i]-u0))
		# print(np.matrix(x0[i]))
		# print(np.matmul(np.transpose(np.matrix(x0[i]-u0)),np.matrix(x0[i]-u0)))
		i+=1

	i=0
	for item in x1:
		S1+=np.matmul(np.transpose(np.matrix(x1a[i]-u1)),np.matrix(x1a[i]-u1))
		i+=1

	Seq=np.add(S0,S1)
	Seq=np.divide(Seq,(count_x0+count_x1))
	# print(Seq)

	S0=np.divide(S0,count_x0)
	S1=np.divide(S1,count_x1)
	


	return phi, u0, u1, S0, S1 , Seq


def gda_equ_sigma(X_, Y_):
	phi, u0 , u1, S0, S1, Seq= gda_theta(X_,Y_)
	print("u0= ", u0, ", u1= ", u1, ", sigma= ", Seq)

# finds the linear boundary assuming that the 2 sigmas are equal and plots it

def gda_linboundary(X_, Y_):
	phi, u0 , u1, S0, S1, Seq= gda_theta(X_,Y_)
	print(u0, u1, Seq)
	# with the u1 term and sigma term
	cons1=np.matmul(np.matmul(np.matrix(u1), np.linalg.inv(np.matrix(Seq))),np.transpose(np.matrix(u1)))
	# with the u0 and sigma terms
	cons0=np.matmul(np.matmul((np.matrix(u0)), np.linalg.inv(np.matrix(Seq))),np.transpose(np.matrix(u0)))
	# log phi-log 1-phi
	log_phi=np.log(phi/(1-phi))
	# const infront of xt
	cons_T=np.matmul(np.linalg.inv(np.matrix(Seq)), np.transpose(np.subtract(np.matrix(u0),np.matrix(u1))))
	# cons in front of x
	cons= np.subtract(np.matmul((np.matrix(u0)), np.linalg.inv(Seq)) , np.matmul((np.matrix(u1)), np.linalg.inv(Seq)))

	c=cons1[0,0]-cons0[0,0]-log_phi

	ans=[c, cons_T[0,0]+cons[0,0], cons_T[1,0]+cons[0,1]]

	plot_gda_boun(ans, X_, Y_)



# finds and plots the quadratic boundary along with the input data
def gda_quadboundary(X_, Y_):
	phi, u0 , u1, S0, S1, Seq= gda_theta(X_,Y_)
	print(S0, S1, Seq)
	# with the u1 and sigma terms
	cons1=np.matmul(np.matmul(np.matrix(u1), np.linalg.inv(np.matrix(S1))),np.transpose(np.matrix(u1)))
	# with the u0 and sigma terms
	cons0=np.matmul(np.matmul((np.matrix(u0)), np.linalg.inv(np.matrix(S0))),np.transpose(np.matrix(u0)))
	log_phi=np.log(phi/(1-phi))

	# const infront of xt
	cons_T=np.subtract(np.matmul(np.linalg.inv(np.matrix(S1)), np.transpose(np.matrix(u1))) , np.matmul(np.linalg.inv(np.matrix(S0)), np.transpose(np.matrix(u0))))
	# const in front of xtx
	consxtx= np.subtract((np.linalg.inv(S1)) , np.linalg.inv(S0))


	# print(consxtx[1,1])
	c=cons1[0,0]-cons0[0,0]-log_phi

	# plotting functions below this
	# -------------------------------------------------------------------------
	y = np.arange(-10,10,0.01)
	x = np.arange(-10,10,0.01)

	x,y = np.meshgrid(x,y)
	# quadratic function that will be plotted

	Z = consxtx[0,0]*(x**2) + consxtx[1,1]*(y**2) + ((consxtx[0,1]+consxtx[1,0])*x*y) -2*x*cons_T[0,0] - 2*y*cons_T[1,0] + c 
	
	# makes contours
	pl.contour(x,y,Z,[0])

	pl.xlim([-6,6])
	pl.ylim([-6,6])

	X=normalise(X_)
	maxx=np.max(X, axis=0)
	minn=np.min(X, axis=0)

	tempx=[]
	tempy=[]
	for item in X:
		tempx.append(item[0])
		tempy.append(item[1])
	i=0
	for item in tempx:
		pl.plot(tempx[i],tempy[i],'ro',c= ('C2' if Y_[i] == 0 else 'C1'))
		i+=1


	# plot linear with quadratic
	# -------------------------------------------------------------------------

	cons1lin=np.matmul(np.matmul(np.matrix(u1), np.linalg.inv(np.matrix(Seq))),np.transpose(np.matrix(u1)))
	# with the u0 and sigma terms
	cons0lin=np.matmul(np.matmul((np.matrix(u0)), np.linalg.inv(np.matrix(Seq))),np.transpose(np.matrix(u0)))
	cons_Tlin=np.matmul(np.linalg.inv(np.matrix(Seq)), np.transpose(np.subtract(np.matrix(u0),np.matrix(u1))))
	# cons in front of x
	conslin= np.subtract(np.matmul((np.matrix(u0)), np.linalg.inv(Seq)) , np.matmul((np.matrix(u1)), np.linalg.inv(Seq)))

	c=cons1lin[0,0]-cons0lin[0,0]-log_phi

	ans=[c, cons_Tlin[0,0]+conslin[0,0], cons_Tlin[1,0]+conslin[0,1]]

	plot_gda_boun(ans, X_, Y_)


	pl.show()


gdaX_dat=[]
gdaY_dat=[]

load_gda_y()
load_gda_x()

gdaX_data=np.array(gdaX_dat)
gdaY_data=np.array(gdaY_dat)

gda_quadboundary(gdaX_data, gdaY_data)

# gda_linboundary(gdaX_data, gdaY_data)

# gda_equ_sigma(gdaX_data, gdaY_data)

# plot_data_gda(gdaX_data, gdaY_data)








































