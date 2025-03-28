import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from skimage import measure

def load_data(filename):
    """Load 3D data from solver output"""
    data = np.loadtxt(filename)
    grid_size = int(round(len(data)**(1/3)))
    x = data[:, 0].reshape(grid_size, grid_size, grid_size)
    y = data[:, 1].reshape(grid_size, grid_size, grid_size)
    z = data[:, 2].reshape(grid_size, grid_size, grid_size)
    u = data[:, 3].reshape(grid_size, grid_size, grid_size)
    return x, y, z, u

def plot_heat_slices(u):
    """Plot 2D slices through the 3D data"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    slices = {
        'XY': u[:, :, u.shape[2]//2],
        'XZ': u[:, u.shape[1]//2, :],
        'YZ': u[u.shape[0]//2, :, :]
    }
    
    for ax, (title, slice_data) in zip(axes, slices.items()):
        im = ax.imshow(slice_data, cmap='hot', origin='lower')
        ax.set_title(f'{title} Plane Slice')
        ax.set_xlabel(title[1])
        ax.set_ylabel(title[0])
        fig.colorbar(im, ax=ax, label='Temperature')
    
    plt.tight_layout()
    plt.savefig('data/heat_slices.png', dpi=150)
    plt.show()

def plot_3d_isosurface(x, y, z, u):
    """3D visualization of isothermal surfaces"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot multiple isosurfaces by thresholding
    levels = np.linspace(u.min(), u.max(), 6)[1:-1]
    for level in levels:
        verts, faces, _, _ = measure.marching_cubes(u, level=level)
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                       cmap='hot', alpha=0.3, antialiased=True)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Isothermal Surfaces')
    plt.savefig('data/heat_3d.png', dpi=150)
    plt.show()

def create_heat_animation():
    """Create animation of heat diffusion over time"""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    # Load all time steps
    frames = []
    for t in range(0, 1000, 100):
        _, _, _, u = load_data(f'data/heat_step_{t}.txt')
        frames.append(u[:, :, u.shape[2]//2])  # Middle Z-slice
    
    im = ax.imshow(frames[0], cmap='hot', vmin=100, vmax=200)
    fig.colorbar(im, label='Temperature')
    ax.set_title('Heat Diffusion Over Time')
    
    def update(frame):
        im.set_array(frames[frame])
        ax.set_xlabel(f'Time Step: {frame*100}')
        return im
    
    ani = FuncAnimation(fig, update, frames=len(frames), interval=200)
    ani.save('data/heat_diffusion.gif', writer='pillow', fps=2)
    plt.close()
    print("Saved animation to data/heat_diffusion.gif")

if __name__ == "__main__":
    x, y, z, u = load_data('data/heat_final.txt')
    plot_heat_slices(u)
    plot_3d_isosurface(x, y, z, u)
    create_heat_animation()