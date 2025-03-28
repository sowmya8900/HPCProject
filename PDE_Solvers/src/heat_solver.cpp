#include "heat_solver.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>

void initializeGrid(std::vector<std::vector<std::vector<double>>>& u, 
                   int grid_size, double initial_value) {
    u.resize(grid_size, std::vector<std::vector<double>>(
        grid_size, std::vector<double>(grid_size, initial_value)));
    
    // Add a heat source in the center
    int center = grid_size/2;
    int radius = 5;
    for (int i = center-radius; i <= center+radius; ++i) {
        for (int j = center-radius; j <= center+radius; ++j) {
            for (int k = center-radius; k <= center+radius; ++k) {
                double r = sqrt(pow(i-center,2) + pow(j-center,2) + pow(k-center,2));
                if (r <= radius) {
                    u[i][j][k] = initial_value * 2.0; // Hotter center
                }
            }
        }
    }
}

void updateGrid(std::vector<std::vector<std::vector<double>>>& u, 
               std::vector<std::vector<std::vector<double>>>& u_new,
               int grid_size, double alpha, double dt, double dx) {
    double coeff = alpha * dt / (dx * dx);
    
    #pragma omp parallel for collapse(3)
    for (int i = 1; i < grid_size-1; ++i) {
        for (int j = 1; j < grid_size-1; ++j) {
            for (int k = 1; k < grid_size-1; ++k) {
                // Finite difference scheme
                u_new[i][j][k] = u[i][j][k] + coeff * (
                    u[i+1][j][k] + u[i-1][j][k] +
                    u[i][j+1][k] + u[i][j-1][k] +
                    u[i][j][k+1] + u[i][j][k-1] - 
                    6.0 * u[i][j][k]);
            }
        }
    }
    
    // Apply boundary conditions (fixed temperature)
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            u_new[i][j][0] = u_new[i][j][grid_size-1] = 100.0; // Fixed boundaries
            u_new[i][0][j] = u_new[i][grid_size-1][j] = 100.0;
            u_new[0][i][j] = u_new[grid_size-1][i][j] = 100.0;
        }
    }
    
    std::swap(u, u_new); // More efficient than copying
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
    const double alpha = 0.01;  // Thermal diffusivity
    const double dx = 1.0;
    
    // Calculate stable time step
    const double dt = 0.9 * (dx*dx)/(6.0 * alpha); // Stability condition
    
    std::vector<std::vector<std::vector<double>>> u(grid_size, 
        std::vector<std::vector<double>>(grid_size, 
        std::vector<double>(grid_size)));
    std::vector<std::vector<std::vector<double>>> u_new = u;
    
    initializeGrid(u, grid_size, 100.0);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < 1000; ++t) {
        updateGrid(u, u_new, grid_size, alpha, dt, dx);
        
        // Optional: Save intermediate results for animation
        if (t % 100 == 0) {
            saveResults(u, "../data/heat_step_" + std::to_string(t) + ".txt");
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Simulation time: " << elapsed.count() << " seconds\n";
    
    saveResults(u, "../data/heat_final.txt");
    return 0;
}