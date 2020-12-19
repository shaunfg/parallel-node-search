# Parallel Node Search on Optimal Classification Trees
(Shaun Fendi Gan, Arkira Tanglertsumpun, MIT MBAn Dec 2020)

The local-search heuristic was introduced as an attempt to remedy the limitations on the scalability of Bertsimas and Dunn's Optimal Classification Tree mixed-integer optimization formulation. This repo examined the effectiveness of this heuristic as well as introducing four parallelized versions that could improve model efficiency. 

## Key Results
* The Half-Split method was found to be the most scalable and time efficient, reducing run time by 30%. 
* Standard Multi-threading on random restarts did not have a significant improvement (<5%), whereas class assignments and 
* Deep Subtree Search were found to only benefit on datasets with many features (m > 15).
  
## How to Use
Open `benchmark.jl` to find all functions used to test and benchmark different local search. Original local search can be found as the function `LocalSearch` in `local_search.jl` The MIO formulation implement in JuMP/ Gurobi can also be found in `oct.jl`

## Files 
- `local_search.jl`- Main file containing serially optimized functions to run the local search heuristic
- `tree.jl` - Serially optimized functions to create tree structures
- `local_search_z.jl` - Contains function for Local Search with parallel Observation assignments
- `local_search_half.jl` - Contains function for Local Search with parallel half splits
- `local_search_deep.jl` - Contains function for Local Search for a deep subtree search
- `oct.jl` - Contains the JuMP MIO Formulation for an Optimal Classification Tree

#### Testing Files
- `model_evaluation.jl` - Contains function for calculating accuracy of a tree's predictions
- `benchmark.jl` - Contains code used to benchmark performance of different Local Search Functions 
- `unit_test.jl` - Contains function `test_tree` to verify Tree struct output from Local Search
- `testing OCT.ipynb` - Contains IntepretableAI Optimal Classification Tree structure, used as reference
- `onethread_oct.ipynb` - Contains MIO OCT function used to test on one thread only (Gurobi Default != 1) 
