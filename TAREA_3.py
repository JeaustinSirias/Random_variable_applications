import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import scipy.stats as stats
import math

with open('xy.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        lote = list(reader)

with open('xyp.csv', 'r') as file:
	reader1 = csv.reader(file)
	for row in reader1:
		listado = list(reader1)



lst=[]
for i in range(0,11):
	mylist = [(lote[i][j]) for j in range(0,21)]
	lista = [float(j) for j in mylist]
	lst.append(lista)

Xsum = []
for i in range(0,11):
	P=sum(lst[i])
	Xsum.append(P)
print(len(Xsum))

#print(Xsum)

#****************************************************************

Ylst=[]
for j in range(0,21):
	mylistY = [(lote[i][j]) for i in range(0,11)]
	listaY = [float(j) for j in mylistY]
	Ylst.append(listaY)

Ysum = []
for i in range(0,21):
	Q=sum(Ylst[i])
	Ysum.append(Q)
print(len(Ysum))

X=range(5, 16)
Y=range(5,26)

#plt.plot(y, Ysum)
#plt.plot(x,Xsum)
#plt.show()


def gaussian(x, mu, sigma):
    return (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(x-mu)**2/(2*sigma**2))


def rayleigh(x, beta):
    return beta*np.exp(-beta*x)


param1, _ = curve_fit(gaussian, X, Xsum)
param2, _ = curve_fit(gaussian, Y, Ysum)
#param, _ = curve_fit(ray, X, Xsum)

print('Los parametros mu y sigma para X son:' + str(param1))
print('Los parametros mu y sigma para Y son:' + str(param2))


#plt.plot(Y, Ysum)
#plt.plot(X, Xsum)
#mu=9.90484381
#sigma=3.29944286
#mu=15.0794609
#sigma=6.02693775
#plt.plot(X, stats.norm.pdf(X, mu, sigma))
#plt.plot(Y, stats.norm.pdf(Y, mu, sigma))


Plst = []
for i in range(0, 231):
	array = listado[i]
	listZ = [float(j) for j in array]
	Plst.append(listZ)

#print(Plst)


product = []
for i in range(0, 231):
	result1 = np.prod(Plst[i])
	product.append(result1)



correlacion = sum(product)

print('La correlacion es:' + str(correlacion))


for i in range(0,231):

	Plst[i][0] = Plst[i][0] - 9.90484381
	Plst[i][1] = Plst[i][1] - 15.0794609

#print(Plst)

#print(Plst)

covarianza_prod=[]
for i in range (0,231):
  #np.prod(Plst[i])
  covarianza_prod.append(np.prod(Plst[i]))

covarianza = sum(covarianza_prod)
print('La covarianza es:' + str(covarianza))

pearson = covarianza/(3.29944286*6.02639775)
print('El coeficiente de Pearson es:' +  str(pearson))


#PARTE 4: GRAFICAS PARA LAS FUNCIONES DE DENSIDAD MARGINALES
'''
plt.plot(X, Xsum, label = 'Xcolumns')
mu=9.90484381
sigma=3.29944286
plt.plot(X, stats.norm.pdf(X, mu, sigma), label = 'Gaussian')
plt.xlabel('Datos')
plt.ylabel('Probabilidad')
plt.legend(framealpha=1, frameon=True);

plt.show()
#plt.savefig('new.png')

plt.plot(Y, Ysum, label = 'YrowSum', color = 'r')
mu=15.0794609
sigma=6.02693775
plt.plot(Y, stats.norm.pdf(Y, mu, sigma), label = 'Gaussian', color = 'g')
plt.xlabel('Datos')
plt.ylabel('Probabilidad')
plt.legend(framealpha=1, frameon=True);

plt.show()
'''

#3D PLOT

def densidad_conjunta(x, y):
  return 0.008*np.exp((-(x-9.905)**2)/21.773)*np.exp((-(y-15.079)**2)/72.648)


x=range(5, 16)
y=range(5,26)

X, Y = np.meshgrid(x, y)
Z = densidad_conjunta(X, Y)

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_wireframe(X, Y, Z, color='green')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()






#print(np.prod(Plst[0]))