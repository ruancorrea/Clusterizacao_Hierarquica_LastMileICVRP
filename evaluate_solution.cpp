#include <iostream>
#include <cassert>
#include <set>
#include <algorithm>

double evaluate_solution(
    const CVRPInstance& instance,
    const CVRPSolution& solution,
    const OSRMConfig& config = OSRMConfig()
) {
    std::set<Delivery> solution_demands;
    for (const auto& vehicle : solution.vehicles) {
        for (const auto& delivery : vehicle.deliveries) {
            solution_demands.insert(delivery);
        }
    }

    std::set<Delivery> instance_demands(instance.deliveries.begin(), instance.deliveries.end());
    assert(solution_demands == instance_demands);

    int max_capacity = 0;
    for (const auto& vehicle : solution.vehicles) {
        int sum = 0;
        for (const auto& delivery : vehicle.deliveries) {
            sum += delivery.size;
        }
        max_capacity = std::max(max_capacity, sum);
    }
    assert(max_capacity <= instance.vehicle_capacity);

    std::set<Point> origins;
    for (const auto& vehicle : solution.vehicles) {
        origins.insert(vehicle.origin);
    }
    assert(origins.size() <= 1);

    std::vector<double> route_distances_m;
    for (const auto& vehicle : solution.vehicles) {
        route_distances_m.push_back(calculate_route_distance_m(vehicle.circuit(), config));
    }

    double sum = 0;
    for (double distance : route_distances_m) {
        sum += distance;
    }

    return round(sum / 1000, 4);
}

/*

This C++ function assumes that the `calculate_route_distance_m` function has been defined and takes a `std::vector<Point>` and an `OSRMConfig` object as arguments.

Keep in mind that there may be some differences in behavior between the Python and C++ versions of the code due to differences in the behavior of certain functions and data structures between the two languages.
*/