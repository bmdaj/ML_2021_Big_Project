# -*- coding: utf-8 -*-
"""
Created on Mon May 31 16:10:20 2021

@author: Jacob-PC
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle

plt.close('all')

def super_ellipse(n,e,r,t,a):
    """
    Function for computing points on a superellipse
    @ r: normalized hole radius
    @ e: hole eccentricity
    @ t: rotation of hole
    @ n: square-ness of hole
    @ a: angle along the curve
    """
    x = np.abs(np.cos(a))**(2/n)*r*np.sign(np.cos(a))
    y = np.abs(np.sin(a))**(2/n)*r*e*np.sign(np.sin(a))
    
    xR = x*np.cos(t) - y*np.sin(t)
    yR = x*np.sin(t) + y*np.cos(t)
    return .5*xR,.5*yR
    

def get_structure(params, ax, num_cells = 5):
    """
    @ params: list of ellipse parameters [n,e,r,t]
    @ ax: axes object to plot the structure in
    @ num_cells: number of repeated unit cells
    """
    angles = np.linspace(0, 2*np.pi)
    points = super_ellipse(params[0],params[1],params[2]/290,params[3]*np.pi/180,angles)
    
    pos = []
    for i in range(num_cells):
        for j in range(num_cells):
            pos.append([i,j])
            
    for p in pos:
        ax.add_patch(Rectangle(p, 1, 1,color = 'grey'))
        ax.fill(points[0]+0.5+p[0],points[1]+0.5+p[1],'w' )

    ax.set(aspect='equal')
    return



def plot_connections(layer1, layer2,ax):
    for l1 in layer1:
        for l2 in layer2:
        
            ax.plot((l1.center[0],l2.center[0]), (l1.center[1],l2.center[1]), 'k-',lw=0.1)
            ax.add_patch(l1)
            ax.add_patch(l2)

def plot_model(N_nodes, ax):
    
    layers = []
    n_off = np.max(N_nodes)
    center = n_off/2
    for j,ns in enumerate(N_nodes):
        layers.append([Circle((n_off/3*j, center-ns/2 + i,),0.4,zorder=10) for i in range(ns)])
    
   # print(len(layers))
    for l1,l2 in zip(layers,layers[1:]):
        plot_connections(l1,l2,ax)
    return

            
if __name__ == "__main__":
    fig, ax = plt.subplots()
    params = [3,1,245,2,10]

    get_structure(params,ax,1)
    ax.set(aspect='equal')
