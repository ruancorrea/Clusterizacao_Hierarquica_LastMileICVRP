#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class Point {
public:
    double lng;
    double lat;

    Point(double lng, double lat) : lng(lng), lat(lat) {}
};

class Delivery {
public:
    std::string id;
    Point point;
    int size;

    Delivery(std::string id, Point point, int size) : id(id), point(point), size(size) {}
};

class DeliveryProblemInstance {
public:
    std::string name;
    std::string region;
    int max_hubs;
    int vehicle_capacity;
    std::vector<Delivery> deliveries;

    DeliveryProblemInstance(std::string name, std::string region, int max_hubs, int vehicle_capacity, std::vector<Delivery> deliveries)
        : name(name), region(region), max_hubs(max_hubs), vehicle_capacity(vehicle_capacity), deliveries(deliveries) {}

    static DeliveryProblemInstance from_file(const std::string& path) {
        std::ifstream file(path);
        json data;
        file >> data;

        std::vector<Delivery> deliveries;
        for (const auto& delivery : data["deliveries"]) {
            deliveries.push_back(Delivery(
                delivery["id"],
                Point(delivery["point"]["lng"], delivery["point"]["lat"]),
                delivery["size"]
            ));
        }

        return DeliveryProblemInstance(
            data["name"],
            data["region"],
            data["max_hubs"],
            data["vehicle_capacity"],
            deliveries
        );
    }

    void to_file(const std::string& path) const {
        json data;
        data["name"] = name;
        data["region"] = region;
        data["max_hubs"] = max_hubs;
        data["vehicle_capacity"] = vehicle_capacity;

        for (const auto& delivery : deliveries) {
            json delivery_json;
            delivery_json["id"] = delivery.id;
            delivery_json["point"]["lng"] = delivery.point.lng;
            delivery_json["point"]["lat"] = delivery.point.lat;
            delivery_json["size"] = delivery.size;

            data["deliveries"].push_back(delivery_json);
        }

        std::ofstream file(path);
        file << data.dump(4);
    }
};

class CVRPInstance {
public:
    std::string name;
    std::string region;
    Point origin;
    int vehicle_capacity;
    std::vector<Delivery> deliveries;

    CVRPInstance(std::string name, std::string region, Point origin, int vehicle_capacity, std::vector<Delivery> deliveries)
        : name(name), region(region), origin(origin), vehicle_capacity(vehicle_capacity), deliveries(deliveries) {}

    static CVRPInstance from_file(const std::string& path) {
        std::ifstream file(path);
        json data;
        file >> data;

        std::vector<Delivery> deliveries;
        for (const auto& delivery : data["deliveries"]) {
            deliveries.push_back(Delivery(
                delivery["id"],
                Point(delivery["point"]["lng"], delivery["point"]["lat"]),
                delivery["size"]
            ));
        }

        return CVRPInstance(
            data["name"],
            data["region"],
            Point(data["origin"]["lng"], data["origin"]["lat"]),
            data["vehicle_capacity"],
            deliveries
        );
    }

    void to_file(const std::string& path) const {
        json data;
        data["name"] = name;
        data["region"] = region;
        data["origin"]["lng"] = origin.lng;
        data["origin"]["lat"] = origin.lat;
        data["vehicle_capacity"] = vehicle_capacity;

        for (const auto& delivery : deliveries) {
            json delivery_json;
            delivery_json["id"] = delivery.id;
            delivery_json["point"]["lng"] = delivery.point.lng;
            delivery_json["point"]["lat"] = delivery.point.lat;
            delivery_json["size"] = delivery.size;

            data["deliveries"].push_back(delivery_json);
        }

        std::ofstream file(path);
        file << data.dump(4);
    }
};

class CVRPSolutionVehicle {
public:
    Point origin;
    std::vector<Delivery> deliveries;

    CVRPSolutionVehicle(Point origin, std::vector<Delivery> deliveries)
        : origin(origin), deliveries(deliveries) {}

    std::vector<Point> circuit() const {
        std::vector<Point> points{origin};
        
        for (const auto& delivery : deliveries) {
            points.push_back(delivery.point);
        }
        
        points.push_back(origin);

        return points; 
    }

    int occupation() const {
      int sum = 0;

      for (const auto& delivery : deliveries) {
          sum += delivery.size; 
      }

      return sum; 
  }
};

class CVRPSolution {
public:
  std::string name; 
  std::vector<CVRPSolutionVehicle> vehicles; 

  CVRPSolution(std::string name, std::vector<CVRPSolutionVehicle> vehicles)
      : name(name), vehicles(vehicles) {}

  static CVRPSolution from_file(const std::string& path) {
      std::ifstream file(path);
      json data; 
      file >> data; 

      std::vector<CVRPSolutionVehicle> vehicles; 
      for (const auto& vehicle : data["vehicles"]) {
          Point origin(vehicle["origin"]["lng"], vehicle["origin"]["lat"]);

          std::vector<Delivery> deliveries; 
          for (const auto& delivery : vehicle["deliveries"]) {
              deliveries.push_back(Delivery(
                  delivery["id"],
                  Point(delivery["point"]["lng"],delivery ["point"]["lat"]),
                  delivery ["size"]
              ));
          }

          vehicles.push_back(CVRPSolutionVehicle(origin,deliveries));
      }

      return CVRPSolution(data ["name"], vehicles); 
  }

  void to_file(const std::string& path) const{
      json data; 
      data ["name"]=name; 

      for (const auto& vehicle: vehicles){
          json vehicle_json; 
          vehicle_json ["origin"]["lng"]=vehicle.origin.lng; 
          vehicle_json ["origin"]["lat"]=vehicle.origin.lat; 

          for (const auto&delivery:vehicle.deliveries){
              json delivery_json; 
              delivery_json ["id"]=delivery.id; 
              delivery_json ["point"]["lng"]=delivery.point.lng; 
              delivery_json ["point"]["lat"]=delivery.point.lat; 
              delivery_json ["size"]=delivery.size; 

              vehicle_json ["deliveries"].push_back(delivery_json); 
          }
          data ["vehicles"].push_back(vehicle_json); 
      }
      
      std::ofstream file(path); 
      file <<data.dump(4); 
  }
};


Here is a C++ version of the Python classes you provided:

```cpp
#include <iostream>
#include <vector>
#include <unordered_map>
#include <dlib/clustering.h>
#include <nlohmann/json.hpp>

using namespace dlib;
using json = nlohmann::json;

class UCModel {
public:
    int C;
    std::vector<Delivery> deliveries;

    UCModel(int C, std::vector<Delivery> deliveries) : C(C), deliveries(deliveries) {}

    static UCModel get_baseline() {
        return UCModel(0, {});
    }
};

class ORToolsParams {
public:
    int max_vehicles;
    int time_limit_ms;

    ORToolsParams(int max_vehicles, int time_limit_ms) : max_vehicles(max_vehicles), time_limit_ms(time_limit_ms) {}
};

class Params {
public:
    std::optional<int> num_clusters;
    ORToolsParams ortools_tsp_params;
    std::optional<int> NUM_UCS;
    int seed;

    Params(std::optional<int> num_clusters, ORToolsParams ortools_tsp_params, std::optional<int> NUM_UCS, int seed)
        : num_clusters(num_clusters), ortools_tsp_params(ortools_tsp_params), NUM_UCS(NUM_UCS), seed(seed) {}

    static Params from_file(const std::string& path) {
        std::ifstream file(path);
        json data;
        file >> data;

        return Params(
            data["num_clusters"],
            ORToolsParams(data["ortools_tsp_params"]["max_vehicles"], data["ortools_tsp_params"]["time_limit_ms"]),
            data["NUM_UCS"],
            data["seed"]
        );
    }

    void to_file(const std::string& path) const {
        json data;
        data["num_clusters"] = num_clusters;
        data["ortools_tsp_params"]["max_vehicles"] = ortools_tsp_params.max_vehicles;
        data["ortools_tsp_params"]["time_limit_ms"] = ortools_tsp_params.time_limit_ms;
        data["NUM_UCS"] = NUM_UCS;
        data["seed"] = seed;

        std::ofstream file(path);
        file << data.dump(4);
    }

    static Params get_baseline() {
        return Params(
            {},
            ORToolsParams(1, 1000),
            {},
            0
        );
    }
};

class ParamsModel {
public:
    Params params;
    kcentroid<kernel_type> clustering;
    std::optional<std::vector<kcentroid<kernel_type>>> subclustering;
    std::optional<CVRPInstance> subinstance;
    std::optional<std::vector<int>> list_distribute;
    std::optional<std::unordered_map<int, int>> dict_distribute;

    ParamsModel(
        Params params,
        kcentroid<kernel_type> clustering,
        std::optional<std::vector<kcentroid<kernel_type>>> subclustering,
        std::optional<CVRPInstance> subinstance,
        std::optional<std::vector<int>> list_distribute,
        std::optional<std::unordered_map<int, int>> dict_distribute
    ) : params(params), clustering(clustering), subclustering(subclustering), subinstance(subinstance),
        list_distribute(list_distribute), dict_distribute(dict_distribute) {}
};
/*

This C++ code uses the [dlib](http://dlib.net/) library to define the `kcentroid` type used in the `ParamsModel` class and the [nlohmann/json](https://github.com/nlohmann/json) library to parse and generate JSON files. You will need to install these libraries and link them to your program in order to use this code.

Keep in mind that there may be some differences in behavior between the Python and C++ versions of the code due to differences in the behavior of certain functions and data structures between the two languages. Also note that the `kernel_type` type used in this code is not defined and will need to be defined based on your specific use case.

*/


/*

This C++ code uses the [nlohmann/json](https://github.com/nlohmann/json) library to parse and generate JSON files. You will need to install this library and link it to your program in order to use this code.

Keep in mind that there may be some differences in behavior between the Python and C++ versions of the code due to differences in the behavior of certain functions and data structures between the two languages.
*/