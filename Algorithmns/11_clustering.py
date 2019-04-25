import matplotlib.pyplot as plt
import math  
    
def plots(cluster1_x1,cluster1_x2,cluster2_x1,cluster2_x2):
    plt.figure()
    plt.plot(cluster1_x1,cluster1_x2,'.')
    plt.plot(cluster2_x1,cluster2_x2,'*')
    plt.grid(True)
    plt.show()
    
def round1(c1_x1,c1_x2,c2_x1,c2_x2):
    cluster1_x1 = []
    cluster1_x2 = []
    cluster2_x1 = []
    cluster2_x2 = []
    
    for i,j in zip(x1,x2):
    	a = math.sqrt(((i-c1_x1)**2 + (j-c1_x2)**2))
    	b = math.sqrt(((i-c2_x1)**2 + (j-c2_x2)**2))
    	if a < b:
    		cluster1_x1.append(i)
    		cluster1_x2.append(j)
    	else: 
    		cluster2_x1.append(i)
    		cluster2_x2.append(j)
		
    plots(cluster1_x1,cluster1_x2,cluster2_x1,cluster2_x2)
	
    c1_x1 = sum(cluster1_x1)/len(cluster1_x1)
    c1_x2 = sum(cluster1_x2)/len(cluster1_x2)
    c2_x1 = sum(cluster2_x1)/len(cluster2_x1)
    c2_x2 = sum(cluster2_x2)/len(cluster2_x2)
	
    round2 (c1_x1,c1_x2,c2_x1,c2_x2)
    print ((c1_x1,c1_x2,c2_x1,c2_x2))

def round2(c1_x1,c1_x2,c2_x1,c2_x2):    
    cluster1_x1 = []
    cluster1_x2 = []
    cluster2_x1 = []
    cluster2_x2 = []
    
    for i,j in zip(x1,x2):
    	c = math.sqrt(((i-c1_x1)**2 + (j-c1_x2)**2))
    	d = math.sqrt(((i-c2_x1)**2 + (j-c2_x2)**2))
    	if c < d:
    		cluster1_x1.append(i)
    		cluster1_x2.append(j)
    	else: 
    		cluster2_x1.append(i)
    		cluster2_x2.append(j)
    
    plots(cluster1_x1,cluster1_x2,cluster2_x1,cluster2_x2)

x1 = [15, 19, 15, 5, 13, 17, 15, 12, 8, 6, 9, 13]
x2 = [13, 16, 17, 6, 17, 14, 15, 13, 7, 6, 10, 12]

plots(x1,x2,[],[])
round1(x1[4],x2[4],x1[10],x2[10])

