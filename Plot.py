import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


fig, axis = plt.subplots()
axis.set_xlim(0, 2*np.pi)
axis.set_ylim(-1.2, 1.2)

# Sine function
t = np.linspace(0, 2*np.pi, 100)
y = np.sin(t)


animated_plot = axis.plot([], [])
print("animated_plot", animated_plot)


# "func", updates the data input, at frame=1, produce a plot, at frame=10, produce another plot. 

def update_data(frame):

    # set_data updates the previous plot with our current data, the array from 0 up to frame
    animated_plot[0].set_data(t[:frame], y[:frame])

    return animated_plot


# Animation function
animation = FuncAnimation(
                    fig = fig, 
                    func = update_data,
                    frames = len(t) ,  
                    interval = 25
                          )

plt.show()
