import matplotlib.pyplot as plt

x = [[6], [8], [10], [14], [18], [21]]
y = [[7], [9], [13], [17.5], [18], [21]]

plt.figure()
plt.title('Pizza Price statistics')
plt.xlabel('Diameter')
plt.ylabel('dollar price')
plt.plot(x,y,'.')
plt.axis([0,25,0,25])
plt.grid(True)
plt.show()
