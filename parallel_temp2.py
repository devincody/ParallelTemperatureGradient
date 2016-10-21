
from mpi4py import MPI
comm = MPI.COMM_WORLD
import numpy as np

rank = comm.rank
size = comm.size

#print "my rank is: ", rank

def BC_Mask(x_size, y_size, x_start, y_start, x_max, y_max): #boundary condition mask
	mask = np.ones((y_size,x_size)) #starts with a mask of all 1s, so nothing would be changed
	for i in range(y_start, y_start + y_size): #iterates through every point
		for j in range(x_start, x_start + x_size): #go through x points
			if i == 0 or i == y_max-1 or j == 0 or j == x_max-1: #if the point is on any bondary, then
				mask[i-y_start][j-x_start] = 0 #set the point to 0
	return mask #returns the array

def BC_Off(x_size, y_size, x_start, y_start, x_max, y_max): #boundary condition mask
	mask = np.zeros((y_size,x_size)) #starts with a mask of all 1s, so nothing would be changed
	for i in range(y_start, y_start + y_size): #iterates through every point
		for j in range(x_start, x_start + x_size): #go through x points
			if j == 0 or j == x_max-1: #if the point is on any bondary, then
				if i >=30:
					mask[i-y_start][j-x_start] = float(60-i)/30 #set the point to 0
				else:
					mask[i-y_start][j-x_start] = float(i)/30
	return mask #returns the array

if rank == 0:
	x_max = 60 # number of columns (numb of points in the X direction)
	y_max = 60 # number of rows (number of points in the Y direction)
	t = 3000 # number of total iterations
	parameters = np.empty(3, dtype=int)

	parameters[0] = x_max
	parameters[1] = y_max
	parameters[2] = t

	#parameters[0:3] = [x_max,y_max,t] #parameters to define simulation
	for i in range(1,size): #sends the parameters and masking function to the slaves
		comm.Send(parameters, dest = int(i), tag = 1)	
		comm.send(BC_Mask, dest = int(i), tag = 2)



if rank != 0:
	parameters = np.empty(3, dtype=int)
	comm.Recv(parameters,source = 0, tag = 1)
#	BC_Mask = comm.recv(source = 0, tag = 2) #now all nodes have the Mask and the parameters
	x_max = parameters[0]
	y_max = parameters[1]
	t = parameters[2]


x_size = int(x_max/float(size)) #PREVIOUS KNOWLEDGE: split data based on columns

y_size = y_max #the data processed by a given node is 60 units high by 10 units

y_start = 0
x_start = int(rank/float(size)*x_max) #PREVIOUS KNOWLEDGE: Starting X position, derived from rank and size
#print x_start, y_start, x_size, y_size
##### INITIALIZE DATA #####
my_data_prev = np.empty((y_size,x_size))
my_data_next = np.zeros((y_size,x_size))
my_mask = BC_Mask(x_size,y_size,x_start,y_start,x_max, y_max)
my_off = BC_Off(x_size,y_size,x_start,y_start,x_max, y_max)
my_data_prev = np.multiply(my_data_next,my_mask) #apply element by element multiplication for mask
my_data_prev = np.add(my_data_prev,my_off)
# if rank == 2:
# 	print "first",len(my_data_prev),np.size(my_data_prev)
# 	print "second", len(my_mask), np.size(my_mask)
# 	print "third", len(my_data_prev), np.size(my_data_prev)

def averaging(matrix,x_size,y_size):
	out = np.zeros((y_size,x_size))
	if rank == 0:
		for i in range(1,y_size-1):
			for j in range(1,x_size):
				x,y = j,i
				out[i][j] = .25*(matrix[y+1][x]+matrix[y-1][x]+matrix[y][x+1]+matrix[y][x-1])
	elif rank == size-1:
		for i in range(1,y_size-1):
			for j in range(0,x_size-1):
				x,y = j+1,i
				out[i][j] = .25*(matrix[y+1][x]+matrix[y-1][x]+matrix[y][x+1]+matrix[y][x-1])
	else:
		for i in range(1,y_size-1):
			for j in range(0,x_size):
				x,y = j+1,i
				out[i][j] = .25*(matrix[y+1][x]+matrix[y-1][x]+matrix[y][x+1]+matrix[y][x-1])
	return out
# RANK 0:
# 	000
# 	0xx
# 	000
# RANK 1-4:
# 	000
# 	xxx
# 	000
# RANK 5:
# 	000
# 	xx0
#	000

c = 0
for time in range(t):
	print "time =", time
	combined = my_data_prev[:,:]
	for inter in range(0,size-1):
		#mesg = "inter: " + str(inter)+", rank: " + str(rank) + ", time: " + str(time)
		#print mesg
		
		if rank == inter: #left pannel
			sendataL = np.array(my_data_prev[:,x_size-1])
			recvdataR = np.empty(y_size, dtype =np.float64)
			comm.Sendrecv(sendbuf = sendataL, dest = (inter + 1), source = (inter + 1), recvbuf = recvdataR, sendtag = inter, recvtag = (10+inter))
			#print str(len(combined)) + " <len, len rcv> " + str(len(recvdataR))
			combined = np.concatenate((combined, np.expand_dims(recvdataR,axis = 1)),axis = 1)
		if rank == inter + 1: #right pannel
			sendataR = np.array(my_data_prev[:,0])
			recvdataL = np.zeros(y_size, dtype =np.float64)
			#msg = "rank = inter+1, send right, recieve left", sendataR.dtype, recvdataL.dtype
			#print msg
			comm.Sendrecv(sendbuf = sendataR, dest = inter, source = inter,recvbuf = recvdataL, sendtag = (10+inter), recvtag = inter)
			combined = np.concatenate((np.expand_dims(recvdataL, axis = 1),combined),axis = 1)
			# if rank ==2:
			# 	print "rank2 combined: " ,combined
			

	### Send DATA to the right, recieve from left
	# empty = np.zeros((y_size,0))
	# if rank in range(0,size-1):
	# 	sendata = my_data_prev[:,x_size-1]
	# 	comm.Send(sendata,dest = rank + 1, tag = 4)
	# 	recvdataL = np.zeros(y_size, dtype =np.float64) #this data are the values to the left
	# 	comm.Recv(recvdataL, source = rank + 1, tag = 5)
	# 	empty = np.concatenate((recvdataL,empty),axis = 1)
	# empty = np.concatenate((empty,my_data_prev),axis = 1)
	# ###Send Data to the Left, recieve from right
	# empty2 = np.zeros((y_size,0))
	# if rank in range(1,size):
	# 	sendata = my_data_prev[:,0]
	# 	comm.Send(sendata,dest = rank - 1, tag = 5)
	# 	recvdataR = np.empty(y_size, dtype =np.float64) #data from the right
	# 	comm.Recv(recvdataR, source = rank + 1, tag = 4)
	# 	empty2 = np.concatenate((empty2,recvdataR),axis = 1)
	# combined = np.concatenate((empty,empty2),axis = 1)	

	my_data_next = averaging(combined,x_size,y_size)
	my_data_prev = np.multiply(my_data_next,my_mask)
	my_data_prev = np.add(my_data_prev,my_off)
	##combine all data:
	### Averaging function:


	if time%40 == 0:
		c += 1
		if rank != 0 :
			comm.Send(my_data_prev,0)

		if rank == 0:
			for r in range(1,size):
				new_data = np.empty((y_size,x_size), dtype=np.float64)
				comm.Recv(new_data, source = r)
				my_data_prev = np.concatenate((my_data_prev,new_data), axis = 1)
			name = 'foo' + str(c)
			np.save(name,my_data_prev)












