cost = 3 
rate = 0.01 
precision = 0.000001 
diff = 1  
iters = 0 
df = lambda x: 2*(x+5) 
while diff > precision:
    prev_x = cost 
    cost = cost - rate * df(prev_x) 
    diff = abs(cost - prev_x) 
    iters = iters+1 
    print("Iteration",iters,"\nX value is",cost) 
    
print("The local minimum occurs at", cost)