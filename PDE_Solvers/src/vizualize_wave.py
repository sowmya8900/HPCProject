import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from skimage import measure

def load_wave_data(filename):
    """Load 3D wave data from solver output"""
    data = np.loadtxt(filename)
    grid_size = int(round(len(data)**(1/3)))
    u = data[:, 3].reshape(grid_size, grid_size, grid_size)
    return u

def plot_wave_slices(u):
    """Plot 2D slices of wave amplitude"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    slices = {
        'XY': u[:, :, u.shape[2]//2],
        'XZ': u[:, u.shape[1]//2, :],
        'YZ': u[u.shape[0]//2, :, :]
    }
    
    vmax = np.abs(u).max()
    
    for ax, (title, slice_data) in zip(axes, slices.items()):
        im = ax.imshow(slice_data, cmap='seismic', vmin=-vmax, vmax=vmax, origin='lower')
        ax.set_title(f'{title} Plane Slice')
        ax.set_xlabel(title[1])
        ax.set_ylabel(title[0])
        fig.colorbar(im, ax=ax, label='Amplitude')
    
    plt.tight_layout()
    plt.savefig('data/wave_slices.png', dpi=150)
    plt.show()

def plot_wave_3d(u):
    """3D visualization of wavefronts"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot positive and negative wavefronts
    print(f"u.min(): {u.min()}, u.max(): {u.max()}")  # Debugging line
    for level in [0.3, -0.3]:
        if(level < u.min() or level > u.max()):
            print(f"Warning: level {level} is out of bounds for the data range.")
            continue
        verts, faces, _, _ = measure.marching_cubes(u, level=level)
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                       cmap='RdBu' if level > 0 else 'RdBu_r',
                       alpha=0.5, antialiased=True)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Wavefront Visualization')
    plt.savefig('data/wave_3d.png', dpi=150)
    plt.show()

def create_wave_animation():
    """Create animation of wave propagation"""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    # Load all time steps
    frames = []
    for t in range(0, 1000, 50):
        u = load_wave_data(f'data/wave_step_{t}.txt')
        frames.append(u[:, :, u.shape[2]//2])  # Middle Z-slice
    
    vmax = max(np.abs(frame).max() for frame in frames)
    
    im = ax.imshow(frames[0], cmap='seismic', vmin=-vmax, vmax=vmax)
    fig.colorbar(im, label='Amplitude')
    ax.set_title('Wave Propagation Over Time')
    
    def update(frame):
        im.set_array(frames[frame])
        ax.set_xlabel(f'Time Step: {frame*50}')
        return im
    
    ani = FuncAnimation(fig, update, frames=len(frames), interval=200)
    ani.save('data/wave_propagation.gif', writer='pillow', fps=5)
    plt.close()
    print("Saved animation to data/wave_propagation.gif")

if __name__ == "__main__":
    u = load_wave_data('data/wave_final.txt')
    plot_wave_slices(u)
    plot_wave_3d(u)
    create_wave_animation()