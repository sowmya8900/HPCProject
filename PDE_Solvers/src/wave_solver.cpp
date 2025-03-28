#include "wave_solver.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>

void initializeGrid(std::vector<std::vector<std::vector<double>>>& u,
                  std::vector<std::vector<std::vector<double>>>& u_prev,
                  int grid_size) {
    u.resize(grid_size, std::vector<std::vector<double>>(
        grid_size, std::vector<double>(grid_size, 0.0)));
    u_prev = u;
    
    // Gaussian wave packet initial condition
    const int center = grid_size/2;
    const double sigma = 5.0;
    
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            for (int k = 0; k < grid_size; ++k) {
                double r = sqrt(pow(i-center,2) + pow(j-center,2) + pow(k-center,2));
                u[i][j][k] = exp(-r*r/(2*sigma*sigma));
            }
        }
    }
    u_prev = u;
}

void updateGrid(std::vector<std::vector<std::vector<double>>>& u,
              std::vector<std::vector<std::vector<double>>>& u_prev,
              int grid_size, double c, double dt, double dx) {
    const double coeff = (c * dt / dx) * (c * dt / dx);
    auto u_next = u; // Temporary grid
    
    #pragma omp parallel for collapse(3)
    for (int i = 1; i < grid_size-1; ++i) {
        for (int j = 1; j < grid_size-1; ++j) {
            for (int k = 1; k < grid_size-1; ++k) {
                // Standard wave equation finite difference
                u_next[i][j][k] = 2.0*u[i][j][k] - u_prev[i][j][k] + coeff * (
                    u[i+1][j][k] + u[i-1][j][k] +
                    u[i][j+1][k] + u[i][j-1][k] +
                    u[i][j][k+1] + u[i][j][k-1] -
                    6.0*u[i][j][k]);
            }
        }
    }
    
    // Apply absorbing boundary conditions
    const double damping = 0.1;
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            // Damp boundaries to reduce reflections
            u_next[i][j][0] *= (1.0 - damping);
            u_next[i][j][grid_size-1] *= (1.0 - damping);
            u_next[i][0][j] *= (1.0 - damping);
            u_next[i][grid_size-1][j] *= (1.0 - damping);
            u_next[0][i][j] *= (1.0 - damping);
            u_next[grid_size-1][i][j] *= (1.0 - damping);
        }
    }
    
    u_prev = u;
    u = u_next;
}

void saveResults(const std::vector<std::vector<std::vector<double>>>& u,
               const std::string& filename) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error opening file: " << filename << "\n";
        return;
    }
    
    int grid_size = u.size();
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            for (int k = 0; k < grid_size; ++k) {
                file << i << " " << j << " " << k << " " << u[i][j][k] << "\n";
            }
        }
    }
    file.close();
}

int main() {
    const int grid_size = 50;
    const double c = 1.0;  // Wave speed
    const double dx = 1.0;
    
    // Calculate stable time step (CFL condition)
    const double dt = 0.9 * dx / (c * sqrt(3.0)); // 3D stability factor
    
    std::vector<std::vector<std::vector<double>>> u, u_prev;
    initializeGrid(u, u_prev, grid_size);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < 1000; ++t) {
        updateGrid(u, u_prev, grid_size, c, dt, dx);
        
        // Optional: Save intermediate results for animation
        if (t % 50 == 0) {
            saveResults(u, "../data/wave_step_" + std::to_string(t) + ".txt");
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Simulation time: " << elapsed.count() << " seconds\n";
    
    saveResults(u, "../data/wave_final.txt");
    return 0;
}