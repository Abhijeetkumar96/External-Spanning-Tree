#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <cuda_runtime.h>

#include "graph.cuh"
#include "cuda_utility.cuh"
#include "spanning_tree.cuh"

int main(int argc, char const *argv[]) {
    // Check if a filename is provided
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    try {
        undirected_graph G(filename);
        
        int numVert = G.getNumVertices();
        long numEdges = G.getNumEdges();
        uint64_t* h_edge_list = G.getEdgeList();

        #ifdef DEBUG
            // Print the edge list
            std::cout << "Printing from main function:\n";
            print_edge_list(h_edge_list, numEdges);
        #endif

        std::cout << "Starting Spanning Tree Construction" << std::endl;
        construct_spanning_tree(h_edge_list, numVert, numEdges);

    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
