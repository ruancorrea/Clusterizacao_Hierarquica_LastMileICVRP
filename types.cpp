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

/*

This C++ code uses the [nlohmann/json](https://github.com/nlohmann/json) library to parse and generate JSON files. You will need to install this library and link it to your program in order to use this code.

Keep in mind that there may be some differences in behavior between the Python and C++ versions of the code due to differences in the behavior of certain functions and data structures between the two languages.
*/