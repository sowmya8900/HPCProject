#ifndef WAVE_SOLVER_H
#define WAVE_SOLVER_H

#include <vector>
#include <string>

// Function to initialize the grid
void initializeGrid(std::vector<std::vector<std::vector<double>>>& u, std::vector<std::vector<std::vector<double>>>& u_prev, int grid_size);

// Function to update the grid using explicit FDM (OpenMP)
void updateGrid(std::vector<std::vector<std::vector<double>>>& u, 
                      std::vector<std::vector<std::vector<double>>>& u_prev, 
                      int grid_size, double c, double dt, double dx);

// Function to write results to a file
void saveResults(const std::vector<std::vector<std::vector<double>>>& u, const std::string& filename);

#endif // WAVE_SOLVER_H
