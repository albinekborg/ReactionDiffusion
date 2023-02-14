import numpy as np
from scipy import signal as sig
import tkinter as tk
import time
from PIL import Image, ImageTk


## Contribution from reacting species
def reaction(u,v,feed,decay):
    newU = -u*v**2 + feed*(1 - u)
    newV = u*v**2 -(decay + feed)*v
    return newU, newV

## Laplacian of the neighbourhood - Finite difference method using convolution. 
def laplacian(lattice, kernel):
    newLattice = sig.convolve2d(lattice,kernel,mode="same",boundary="wrap")
    return newLattice

def dynamics(dt, laplacianKernel, diffusionConstants, u, v, feed, decay):
    reactionContribution = reaction(u,v,feed,decay)
    tempU = (diffusionConstants[0]*laplacian(u,laplacianKernel) + reactionContribution[0])*dt
    tempV = (diffusionConstants[1]*laplacian(v,laplacianKernel) + reactionContribution[1])*dt
    
    u += tempU
    v += tempV
    
    # Cannot have negative or larger than 1 concentrations:
    np.clip(u, 0, 1, u)
    np.clip(v, 0, 1, v)

    return u,v

## Parameters
N = 700
diffusionU = 0.8
diffusionV = 0.4
diffusionConstants = [diffusionU,diffusionV]

feed = 0.055
decay = 0.062
dt = 1
laplacianKernel = np.array([[0.05,     0.2,     0.05],
                            [0.2,      -1,      0.2],
                            [0.05,     0.2,     0.05]]) 


## Lattices containing concentration for U & V
u = np.zeros((N,N))
u[:,:] = 0.6 + 0.2*np.random.rand(N,N)

v = np.zeros((N,N))
v[:,:] = 0.2*np.random.rand(N,N)

## Add a region in the center were the concentration of U & V is different
rangeFrom = int(N/2) - int(N/10)
rangeTo = int(N/2) + int(N/10)

v[rangeFrom:rangeTo,rangeFrom:rangeTo] = 0.5
u[rangeFrom:rangeTo,rangeFrom:rangeTo] = 0.25

## Draw dynamics using Tk.
root = tk.Tk()

## Initialise
u,v = dynamics(dt, laplacianKernel, diffusionConstants, u, v, feed, decay)
content = ImageTk.PhotoImage(Image.fromarray(np.uint8(np.floor(u*255)),'L'))
canvas = tk.Canvas(root, width=N,height=N)
canvas.pack()
canvas.create_image(0, 0, anchor=tk.NW, image=content)

## Run dynamics
running = True
while running:
    u,v = dynamics(dt, laplacianKernel, diffusionConstants, u, v, feed, decay)

    content = ImageTk.PhotoImage(Image.fromarray(np.uint8(np.floor(u*255)),'L'))
    canvas.create_image(0, 0, anchor=tk.NW, image=content)
    root.update()

root.mainloop()


