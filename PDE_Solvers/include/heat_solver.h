#ifndef HEAT_SOLVER_H
#define HEAT_SOLVER_H

#include <vector>
#include <string>

// Function to initialize the grid
void initializeGrid(std::vector<std::vector<std::vector<double>>>& u, int grid_size, double initial_value);

// Function to update the grid using explicit FDM (OpenMP)
void updateGrid(std::vector<std::vector<std::vector<double>>>& u, 
                      std::vector<std::vector<std::vector<double>>>& u_new, 
                      int grid_size, double alpha, double dt, double dx);

// Function to save results to a file
void saveResults(const std::vector<std::vector<std::vector<double>>>& u, const std::string& filename);

#endif // HEAT_SOLVER_H
