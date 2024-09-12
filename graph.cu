#include <vector>
#include <fstream>
#include <cassert>
#include <filesystem>
#include <cuda_runtime.h>

#include "graph.cuh"
#include "cuda_utility.cuh"

void undirected_graph::readGraphFile(const std::string& filename) {

    std::filesystem::path filepath = filename;  // Correct declaration

    if (!std::filesystem::exists(filepath)) {
        throw std::runtime_error("File does not exist: " + filepath.string());
    }

    std::string ext = filepath.extension().string();  // Correct way to get file extension

    if (ext == ".edges" || ext == ".eg2" || ext == ".txt") {
        readEdgeList(filename);
    }
    else if (ext == ".mtx") {
        readMTXgraph(filename);
    }
    else if (ext == ".gr") {
        readMETISgraph(filename);
    }
    else if (ext == ".egr" || ext == ".bin" || ".csr") {
        readECLgraph(filename);
    }
    else {
        throw std::runtime_error("Unsupported graph format: " + ext);
    }
}

void undirected_graph::readEdgeList(const std::string& filename) {
    std::ifstream inFile(filename);
    if (!inFile) {
        std::cerr << "Error opening file: " << filename << std::endl;
        throw std::runtime_error("Failed to open input file");
    }

    inFile >> numVert >> numEdges;
    if (inFile.fail()) {
        std::cerr << "Error reading vertices and edges from file." << std::endl;
        throw std::runtime_error("Failed to read vertices and edges");
    }

    // Allocate host pinned memory
    size_t bytes = (numEdges / 2) * sizeof(uint64_t);
    CUDA_CHECK(cudaMallocHost((void**)&h_edge_list, bytes), "Failed to allocate edgelist");

    long ctr = 0;
    int u, v;
    for (long i = 0; i < numEdges; ++i) {
        inFile >> u >> v;
        if (inFile.fail()) {
            std::cerr << "Error reading edges from file." << std::endl;
            CUDA_CHECK(cudaFreeHost(h_edge_list), "Failed to free h_edge_list");
            throw std::runtime_error("Failed to read edges");
        }
        if (u < v) {
            h_edge_list[ctr] = (static_cast<uint64_t>(u) << 32) | (v);
            ctr++;
        }
    }

    assert(ctr == numEdges / 2);
    numEdges /= 2;  // Adjust numEdges to match the reduced count
}

void undirected_graph::readMTXgraph(const std::string& filename) {
    // std::cout << "Reading mtx file: " << getFilename() << std::endl;
    std::ifstream inFile(filename);
    if (!inFile) {
        throw std::runtime_error("Error opening file: ");
    }
}

void undirected_graph::readMETISgraph(const std::string& filename) {
    // std::cout << "Reading metis file: " << getFilename() << std::endl;
    std::ifstream inFile(filename);
    if (!inFile) {
        throw std::runtime_error("Error opening file: ");
    }
}   

void undirected_graph::readECLgraph(const std::string& filename) {
    // std::cout << "Reading ECL file: " << getFilename() << std::endl;

    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        throw std::runtime_error("Error opening file: ");
    }

    // Reading sizes
    size_t size;
    inFile.read(reinterpret_cast<char*>(&size), sizeof(size));
    vertices.resize(size);
    inFile.read(reinterpret_cast<char*>(&size), sizeof(size));
    edges.resize(size);

    // Reading data
    inFile.read(reinterpret_cast<char*>(vertices.data()), vertices.size() * sizeof(long));
    inFile.read(reinterpret_cast<char*>(edges.data()), edges.size() * sizeof(int));

    numVert = vertices.size() - 1;
    numEdges = edges.size();

    csr_to_coo();

    numEdges /= 2;  // Adjust numEdges to match the reduced count
}

void undirected_graph::csr_to_coo() {

    // Allocate host pinned memories
    size_t bytes = (numEdges/2) * sizeof(int);

    // Host pinned memory ds
    CUDA_CHECK(cudaMallocHost((void**)&h_edge_list,  bytes),  "Failed to allocate pinned memory for src");

    long ctr = 0;

    for (int i = 0; i < numVert; ++i) {
        for (long j = vertices[i]; j < vertices[i + 1]; ++j) {
            if(i < edges[j]) {
                int u  = i;
                int v = edges[j];
                h_edge_list[ctr] = (static_cast<uint64_t>(u) << 32) | (v);
                ctr++;
            }
        }
    }    

    assert(ctr == numEdges/2);
}