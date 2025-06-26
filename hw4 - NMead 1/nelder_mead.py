# %% [markdown]
import numpy as np

np.random.seed(42) 
    
class Simplex:
    """ For generality """
    def __init__(self, vertices):        
        self.vertices = np.array(vertices)
        self.n_dim = self.vertices[0].shape[0]  # Number of dimensions
        
        assert self.vertices.shape[0] == self.n_dim + 1, "Simplex must have n+1 vertices in n dimensions"
        
    def __repr__(self):
        return f"Simplex(vertices={self.vertices})"
    
    def sort_simplex(self, func):
        vals = [func(vertex) for vertex in self.vertices]
        sorted_vertices = np.array([self.vertices[i] for i in np.argsort(vals)])
        self.vertices = sorted_vertices
            
    def centroid(self):
        centroid = np.mean(self.vertices[:-1], axis=0)  # Exclude the worst point      
        return centroid
    
    def converged(self, tol=1e-6):
        """ Check if the simplex has converged based on the tolerance. """
        return np.max(np.abs(self.vertices - np.mean(self.vertices, axis=0))) < tol

    def reflect(self, c, alpha):
        """Reflect worst point across the centriod - also used for extension and contraction"""
        x_w = self.vertices[-1]
        return c + alpha * (c - x_w)
    
    def contract(self, c, x_r, beta, mode="outside"):
        if mode == 'outside':
            return c + beta * (x_r - c)
        else:  # inside contraction
            return c - beta * (c - self.vertices[-1])
    
    def diameter(self):
        """Compute the maximum distance between any two vertices."""
        return np.max(np.linalg.norm(self.vertices - self.vertices[:, np.newaxis], axis=-1))
    
    def shrink(self, delta):
        self.vertices[1:] = self.vertices[0] + delta * (self.vertices[1:] - self.vertices[0])

 
    
alpha = 1.0  # Reflection
beta = 0.5 # Contraction
gamma = 2.0  # Expansion
delta = 0.5 # Shrinkage    
    
def nelder_mead(func, simplex: Simplex, max_iter=1000, tol=1e-6, verbose=False):
    """
    Perform the Nelder-Mead optimization algorithm in 2D.

    Parameters:
    - func: The objective function to minimize.
    - simplex: A list of points defining the initial simplex.
    - max_iter: Maximum number of iterations.
    - tol: Tolerance for convergence.

    Returns:
    - The point that minimizes the function.
    """

    for k in range(max_iter):
        simplex.sort_simplex(func)

        if simplex.diameter() < tol:
            print(f"Converged at iteration {k}.")
            break
        
        if verbose:
            best, worst = simplex.vertices[0], simplex.vertices[-1]
            print(f"iter {k:3d}, best={best}, f={func(best):.6f}, worst={worst}, f={func(worst):.6f}, diameter={simplex.diameter():.6f} ")

        centroid = simplex.centroid()
        
        # Reflection & Extension
        reflected = simplex.reflect(centroid, alpha)
        reflected_value = func(reflected)   # Dont recompute (can precompute for all vertices each iteration)
        
        if reflected_value < func(simplex.vertices[-2]):
            
            if reflected_value < func(simplex.vertices[0]): # Expand reflection if it was better than the second worst
                extended = simplex.reflect(centroid, gamma)  

                if func(extended) < reflected_value:
                    simplex.vertices[-1] = extended
                    print("Extended point:", extended)
                    
                else:
                    simplex.vertices[-1] = reflected
                    print("Reflected point:", reflected)
            else:
                simplex.vertices[-1] = reflected
            
            continue

        # Contraction
        if reflected_value >= func(simplex.vertices[-2]):
            # Outside contraction
            contracted = simplex.contract(centroid, reflected, beta, mode='outside')
            contracted_value = func(contracted)
            
            if contracted_value < reflected_value:
                simplex.vertices[-1] = contracted
                print("Contracted point:", contracted)
            else:
                # Shrink
                simplex.shrink(delta)
                print("Shrinking simplex")
        
        else:
            # Inside contraction
            contracted = simplex.contract(centroid, reflected, beta, mode='inside')
            contracted_value = func(contracted)
            
            if contracted_value < func(simplex.vertices[-1]):
                simplex.vertices[-1] = contracted
                print("Contracted point (inside):", contracted)
            else:
                # Shrink
                simplex.shrink(delta)
                print("Shrinking simplex (inside contraction)")
            
    return simplex.vertices[0]

import subprocess

mode = 1

def bbox(x ,i=mode):
    x,y,z = x
    proc = subprocess.run(["./hw4_1_linux", "63200306", str(i), str(x), str(y), str(z)], check=True, stdout=subprocess.PIPE)
    
    value = proc.stdout.decode('utf-8').strip()
    value = float(value)
    
    return value

if __name__ == "__main__":
    
    
    start_f1 = np.array([0,0,0], dtype=float)
    simplex_f1 = Simplex([start_f1,
                          start_f1 + np.array([10, 0, 0], dtype=float),
                          start_f1 + np.array([0, 10, 0], dtype=float),    
                          start_f1 + np.array([0, 0, 10], dtype=float)])
    
    nelder_mead_f1 = nelder_mead(bbox, simplex_f1, max_iter=1000, tol=1e-7, verbose=True)
    print(nelder_mead_f1)
    print(bbox(nelder_mead_f1))
    print(simplex_f1)
    
    
    
    
    