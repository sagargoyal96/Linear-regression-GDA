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



BATCH_SIZE=500
LEARN_RATE=0.009

# load the data into arrays

def load_log_x():
	file_x=open('logisticX.csv',"rU")
	reader=csv.reader(file_x,delimiter=",")
	
	row_x=0
	for row in reader:
		# [1,row]
		logX_dat.append([1,float(row[0]),float(row[1])])
		row_x+=1

	file_x.close()

def load_log_y():
	file_y=open('logisticY.csv',"rU")
	reader=csv.reader(file_y,delimiter=",")
	rownum_y=0



	for row in reader:
		logY_dat.append(float(row[0]))
		rownum_y+=1

	file_y.close()	



def normalise_x(X):
	mu=np.mean(X, axis=0)
	var=np.var(X,axis=0)
	sd=np.sqrt(var)
	# print(mu)
	nor_X=[]
	for item in X:
		nor_X.append([1,(item[1]-mu[1])/sd[1], (item[2]-mu[2])/sd[2]])

	return np.array(nor_X)

# kuch hagga hai
def normalise_y(X):
	mu=np.mean(X, axis=0)
	var=np.var(X,axis=0)
	sd=np.sqrt(var)
	print(mu, var)
	nor_X=[]
	for item in X:
		nor_X.append((item-mu)/sd)

	return np.array(nor_X)


# plot the logistic data
def plot_data_logistic(theta):

	tempx=[]
	tempy=[]

	# temps are used to remove the starting 1's

	for item in logX_data:
		tempx.append(item[1])
		tempy.append(item[2])

	maxx=np.max(np.array(tempx))
	minn=np.min(np.array(tempx))
	# these are used to produce the data for plotting

	i=0
	for item in tempx:
		pl.plot(tempx[i],tempy[i],'ro',c= ('C2' if logY_data[i] == 0 else 'C1'))
		i+=1
	
	x = np.linspace(maxx,minn,1200)
	# y=mx+c
	pl.plot(x,theta[0,1]*(-1)/theta[0,2]*x+theta[0,0]*(-1)/theta[0,2])
	pl.show()


# computes the hx for logistic regression (after applying the functionover the entire matrix and returns the value

def calc_log_hx(X_):
	ans1=np.matmul(log_T,np.transpose(X_))
	ans=np.apply_along_axis(my_func,0,ans1)
	return ans

# this is called by calc_log_hx to compute the value of hx
def my_func(a):
	return 1/(1+np.exp(-1*a))

# computes the Hessian and the Grad J(o) and uses it to calc the loss in logistic regression
def log_reg(X_, Y_):
	global log_T
	grad_J=np.matmul((np.matrix((Y_-calc_log_hx(X_)))),np.matrix(X_))
	
	# print(np.shape(X_))
	# print(np.shape(grad_J))
	i=0

	H=[[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]
	for item in X_:
		ith=np.matmul(np.transpose(np.matrix(X_[i])),np.matrix(X_[i]))

		hx=my_func(np.matmul(np.matrix(log_T),np.transpose(np.matrix((X_[i])))))
		# print("hx= ",hx)
		fin_ith=np.multiply((hx*(1-hx)),ith)
		# hessian
		H=np.add(H,fin_ith)
		i+=1

	Hinv=np.linalg.inv(H)
	# the error that will be subtracted
	change=np.matmul(Hinv,np.transpose(grad_J))
	# print(np.shape(change))
	change=np.multiply(LEARN_RATE,np.transpose(change))
	log_T=np.add(log_T,change)	

	return change


# runs epochs till the loss reduces beyond a certain value for logistic regression
def log_reg_loop(X_,Y_):
	count=0
	while(1):
		chng=log_reg(X_,Y_)
		# print(chng)
		count+=1
		if abs(chng[0,0])<0.00001 and abs(chng[0,0])<0.00001:
			break;

log_T=np.array([0.0,0.0,0.0])

logX_dat=[]
logY_dat=[]

load_log_y()
load_log_x()

logY_data=(np.array(logY_dat))
logX_data=normalise_x(np.array(logX_dat))

log_reg_loop(logX_data, logY_data)
print(log_T)

plot_data_logistic(log_T)











