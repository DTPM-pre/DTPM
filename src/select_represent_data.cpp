#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <random>
#include <thread>
#include <future>
#include <limits>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/dataframe.h"  //DataFrame

namespace py = pybind11;

//  (Z-score normalization)
std::vector<std::vector<double>> standardize_data(std::vector<std::vector<double>>& sorce_data) {
    //std::cout << "Starting data standardization..." << std::endl;

    std::vector<std::vector<double>> data = sorce_data;

    size_t num_samples = data.size();
    size_t num_features = data[0].size();

    std::vector<double> means(num_features, 0.0);
    std::vector<double> std_devs(num_features, 0.0);

    
    //std::cout << "Calculating means for each feature..." << std::endl;
    for (const auto& sample : data) {
        for (size_t j = 0; j < num_features; ++j) {
            means[j] += sample[j];
        }
    }

    for (double& mean : means) {
        mean /= num_samples;
    }

    //std::cout << "Calculating standard deviations for each feature..." << std::endl;
    for (const auto& sample : data) {
        for (size_t j = 0; j < num_features; ++j) {
            std_devs[j] += (sample[j] - means[j]) * (sample[j] - means[j]);
        }
    }

    for (double& std_dev : std_devs) {
        std_dev = std::sqrt(std_dev / num_samples);
    }

    
    //std::cout << "Normalizing data..." << std::endl;
    for (auto& sample : data) {
        for (size_t j = 0; j < num_features; ++j) {
            if (std_devs[j] != 0) {
                sample[j] = (sample[j] - means[j]) / std_devs[j];
            }
        }
    }

    //std::cout << "Data standardization completed." << std::endl;
    return data;
}

// 
double euclidean_distance(const std::vector<double>& point1, const std::vector<double>& point2) {
    //std::cout << "Calculating Euclidean distance..." << std::endl;

    double sum = 0.0;
    for (size_t i = 0; i < point1.size(); ++i) {
        sum += (point1[i] - point2[i]) * (point1[i] - point2[i]);
    }

    //std::cout << "Euclidean distance calculated." << std::endl;
    return std::sqrt(sum);
}

// 
std::vector<size_t> max_min_sampling(const std::vector<std::vector<double>>& data, size_t num_samples) {
    //std::cout << "Starting max-min sampling..." << std::endl;

    std::vector<size_t> sampled_indices;

    sampled_indices.push_back(std::rand() % data.size());

    //std::cout << "random: " << sampled_indices.back() << std::endl;

    //std::cout << "Initial random sample index selected: " << sampled_indices.back() << std::endl;

    for (size_t i = 1; i < num_samples; ++i) {
        double max_min_dist = -1.0;
        size_t best_index = 0;

        std::vector<size_t> remaining_indices;
        for (size_t idx = 0; idx < data.size(); ++idx) {
            if (std::find(sampled_indices.begin(), sampled_indices.end(), idx) == sampled_indices.end()) {
                remaining_indices.push_back(idx);
            }
        }

        //std::cout << "Remaining indices size: " << remaining_indices.size() << std::endl;

        auto process_chunk = [&data, &sampled_indices](size_t start, size_t end, const std::vector<size_t>& remaining_indices) 
            -> std::pair<double, size_t> {
            //std::cout << "Processing chunk from " << start << " to " << end << std::endl;

            double local_max_min_dist = -1.0;
            size_t local_best_index = 0;

            for (size_t idx = start; idx < end; ++idx) {
                double min_dist = std::numeric_limits<double>::max();
                for (const auto& sampled_idx : sampled_indices) {
                    double dist = euclidean_distance(data[sampled_idx], data[remaining_indices[idx]]);
                    min_dist = std::min(min_dist, dist);
                }
                if (min_dist > local_max_min_dist) {
                    local_max_min_dist = min_dist;
                    local_best_index = remaining_indices[idx];
                }
            }

            //std::cout << "Chunk processed, local best index: " << local_best_index << std::endl;
            return {local_max_min_dist, local_best_index};
        };

        size_t num_threads = std::thread::hardware_concurrency();
        //std::cout << "Using " << num_threads << " threads for sampling." << std::endl;
        
        std::vector<std::future<std::pair<double, size_t>>> futures;

        size_t chunk_size = remaining_indices.size() / num_threads;
        for (size_t t = 0; t < num_threads; ++t) {
            size_t start = t * chunk_size;
            size_t end = (t == num_threads - 1) ? remaining_indices.size() : (t + 1) * chunk_size;
            futures.push_back(std::async(std::launch::async, process_chunk, start, end, std::ref(remaining_indices)));
        }

        for (auto& future : futures) {
            auto [local_max_min_dist, local_best_index] = future.get();
            if (local_max_min_dist > max_min_dist) {
                max_min_dist = local_max_min_dist;
                best_index = local_best_index;
            }
        }

        sampled_indices.push_back(best_index);
        //std::cout << "Sampled index " << best_index << " selected." << std::endl;
    }

    //std::cout << "Max-min sampling completed." << std::endl;
    return sampled_indices;
}

// 
std::tuple<std::vector<std::vector<double>>, std::vector<double>, std::vector<std::string>> 
process_and_select_samples(const std::string& file_path, size_t num_samples, const std::string& method) {
    //std::cout << "Reading CSV file: " << file_path << std::endl;
    std::srand(42);

    DataFrame df;
    df.read_csv(file_path);
    //std::cout << "CSV file read successfully." << std::endl;

    std::vector<std::string> column_names = df.get_column_names();
    std::string target_column = column_names.back();
    std::vector<double> y = df.get_column(target_column);
    //std::cout << "Extracted target column: " << target_column << std::endl;

    std::vector<std::vector<double>> X(df.get_column(column_names[0]).size(),
                                       std::vector<double>(column_names.size() - 1));

    for (size_t i = 0; i < column_names.size() - 1; ++i) {
        std::vector<double> feature_column = df.get_column(column_names[i]);
        for (size_t j = 0; j < feature_column.size(); ++j) {
            X[j][i] = feature_column[j];
        }
    }

    //std::cout << "Features extracted. Starting data standardization." << std::endl;
    std::vector<std::vector<double>> X_scaled = standardize_data(X);
    //std::cout << "Data standardization complete." << std::endl;

    std::vector<size_t> representative_indices;
    if (method == "max_min") {
        std::cout << "Starting max-min sampling..." << std::endl;
        representative_indices = max_min_sampling(X_scaled, num_samples);
    } else {
        throw std::invalid_argument("Unsupported sampling method");
    }

    std::vector<std::vector<double>> X_representative;
    std::vector<double> y_representative;
    for (const auto& idx : representative_indices) {
        X_representative.push_back(X[idx]);
        y_representative.push_back(y[idx]);
    }

    //std::cout << "Representative samples selected." << std::endl;
    column_names.pop_back();
    column_names.push_back("target");
    return {X_representative, y_representative, column_names};
}

PYBIND11_MODULE(represent_data, m) {
    m.def("standardize_data", &standardize_data, "A function to standardize data");
    m.def("max_min_sampling", &max_min_sampling, "A function for max-min sampling");
    m.def("process_and_select_samples", &process_and_select_samples, 
        "A function to process and select representative samples from a CSV file.");
}
