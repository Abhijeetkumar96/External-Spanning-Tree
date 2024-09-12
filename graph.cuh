#ifndef GRAPH_H
#define GRAPH_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>
#include <cuda_runtime.h>

#include "cuda_utility.cuh"

class undirected_graph {
public:
    undirected_graph(const std::string& filename) 
        : numVert(0), numEdges(0), h_edge_list(nullptr) {
        readGraphFile(filename);
    }

    ~undirected_graph() {
        if (h_edge_list != nullptr) {
            CUDA_CHECK(cudaFreeHost(h_edge_list), "Failed to free h_edge_list");
        }
    }

    // Accessors
    int getNumVertices() const { return numVert; }
    long getNumEdges() const { return numEdges; }
    uint64_t* getEdgeList() const { return h_edge_list; }


private:
    int numVert;
    long numEdges;
    uint64_t* h_edge_list;

    std::vector<long> vertices;
    std::vector<int>  edges;

    void readGraphFile(const std::string& filename);
    void readEdgeList(const std::string& filename);
    void readMTXgraph(const std::string& filename);
    void readMETISgraph(const std::string& filename);
    void readECLgraph(const std::string& filename);
    void csr_to_coo();
};


#endif // GRAPH_H