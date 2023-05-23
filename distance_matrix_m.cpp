#include <iostream>
#include <vector>
#include <string>
#include <curl/curl.h> // instalar
#include <nlohmann/json.hpp> // instalar

using json = nlohmann::json;

struct Point {
    double lat;
    double lng;
};

struct OSRMConfig {
    std::string host = "http://localhost:5000";
    int timeout_s = 30;
};

size_t writeFunction(void *ptr, size_t size, size_t nmemb, std::string* data) {
    data->append((char*) ptr, size * nmemb);
    return size * nmemb;
}

std::vector<std::vector<double>> calculate_distance_matrix_m(
    const std::vector<Point>& points,
    const OSRMConfig& config = OSRMConfig()
) {
    if (points.size() < 2) {
        return {};
    }

    std::string coords_uri;
    for (const auto& point : points) {
        coords_uri += std::to_string(point.lng) + "," + std::to_string(point.lat) + ";";
    }
    coords_uri.pop_back();

    CURL *curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("Failed to initialize curl");
    }

    std::string url = config.host + "/table/v1/driving/" + coords_uri + "?annotations=distance";
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, config.timeout_s);

    std::string response_string;
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeFunction);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        throw std::runtime_error("Failed to get response from OSRM server");
    }

    curl_easy_cleanup(curl);

    json response_json = json::parse(response_string);
    std::vector<std::vector<double>> distances = response_json["distances"];

    return distances;
}

/*
This C++ code uses the [libcurl](https://curl.se/libcurl/) library to send an HTTP GET request to the OSRM server and the [nlohmann/json](https://github.com/nlohmann/json) library to parse the JSON response. You will need to install these libraries and link them to your program in order to use this code.

Keep in mind that there may be some differences in behavior between the Python and C++ versions of the code due to differences in the behavior of certain functions and data structures between the two languages.
*/