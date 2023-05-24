#include <iostream>
#include <vector>
#include <unordered_map>
#include <dlib/clustering.h>

using namespace dlib;

std::vector<kcentroid<kernel_type>> get_subClusterings(
    const std::vector<sample_type>& points,
    const std::vector<int>& y_pred,
    int n_clusters,
    const std::unordered_map<int, int>& dict_distribute
) {
    std::vector<std::vector<sample_type>> pointsClusters(n_clusters);
    for (int i = 0; i < y_pred.size(); i++) {
        pointsClusters[y_pred[i]].push_back(points[i]);
    }

    std::vector<kcentroid<kernel_type>> subclusterings;
    for (int cluster = 0; cluster < n_clusters; cluster++) {
        subclusterings.push_back(kcentroid<kernel_type>(radial_basis_kernel<sample_type>(0.1), 0.01, dict_distribute.at(cluster)));
        subclusterings[cluster].train(pointsClusters[cluster]);
    }

    return subclusterings;
}

/*

This C++ function uses the [dlib](http://dlib.net/) library to create and train KMeans models. You will need to install this library and link it to your program in order to use this code.

Keep in mind that there may be some differences in behavior between the Python and C++ versions of the code due to differences in the behavior of certain functions and data structures between the two languages. Also note that the `kernel_type` and `sample_type` types used in this function are not defined in the code provided and will need to be defined based on your specific use case.
*/