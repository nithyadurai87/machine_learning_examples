import matplotlib.pyplot as plt
import math  
    
def plots(a1_cluster,a2_cluster,b1_cluster,b2_cluster):
    plt.figure()
    plt.plot(a1_cluster,a2_cluster,'.')
    plt.plot(b1_cluster,b2_cluster,'*')
    plt.grid(True)
    plt.show()
    
def cluster1(c1_x1,c1_x2,c2_x1,c2_x2):
    cluster1_x1 = []
    cluster1_x2 = []
    cluster2_x1 = []
    cluster2_x2 = []
    
    for i,j in zip(x1,x2):
    	a = math.sqrt(((i-c1_x1)**2 + (j-c1_x2)**2))
    	b = math.sqrt(((i-c2_x1)**2 + (j-c2_x1)**2))
    	if a < b:
    		cluster1_x1.append(i)
    		cluster1_x2.append(j)
    	else: 
    		cluster2_x1.append(i)
    		cluster2_x2.append(j)
		
    plots(cluster1_x1,cluster1_x2,cluster2_x1,cluster2_x2)
	
    c3_x1 = sum(cluster1_x1)/len(cluster1_x1)
    c3_x2 = sum(cluster1_x2)/len(cluster1_x2)
    c4_x1 = sum(cluster2_x1)/len(cluster2_x1)
    c4_x2 = sum(cluster2_x2)/len(cluster2_x2)
	
    cluster2 (c3_x1,c3_x2,c4_x1,c4_x2)
    
def cluster2(c3_x1,c3_x2,c4_x1,c4_x2):    
    cluster3_x1 = []
    cluster3_x2 = []
    cluster4_x1 = []
    cluster4_x2 = []
    
    for i,j in zip(x1,x2):
    	c = math.sqrt(((i-c3_x1)**2 + (j-c3_x2)**2))
    	d = math.sqrt(((i-c4_x1)**2 + (j-c4_x2)**2))
    	if c < d:
    		cluster3_x1.append(i)
    		cluster3_x2.append(j)
    	else: 
    		cluster4_x1.append(i)
    		cluster4_x2.append(j)
    
    plots(cluster3_x1,cluster3_x2,cluster4_x1,cluster4_x2)

x1 = [15, 19, 15, 5, 13, 17, 15, 12, 8, 6, 9, 13]
x2 = [13, 16, 17, 6, 17, 14, 15, 13, 7, 6, 4, 12]

plots(x1,x2,[],[])
cluster1(x1[4],x2[4],x1[10],x2[10])

