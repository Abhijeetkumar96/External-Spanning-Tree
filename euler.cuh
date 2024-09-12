#ifndef EULER_TOUR_CUH
#define EULER_TOUR_CUH

void cuda_euler_tour(
    int N, 
    int root, 
    uint64_t* d_edges_input);

#endif // EULER_TOUR_CUH