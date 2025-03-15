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
#include <pybind11/numpy.h>
#include "../include/new_dataframe.h"

namespace py = pybind11;

nc::NdArray<double> standardize_data(const nc::NdArray<double>& source_data) {
    nc::NdArray<double> data = source_data;
    size_t num_samples = data.numRows();
    size_t num_features = data.numCols();

    nc::NdArray<double> means = nc::mean(data, nc::Axis::ROW);
    nc::NdArray<double> std_devs(1, num_features);
    for (size_t j = 0; j < num_features; ++j) {
        nc::NdArray<double> feature_col = data({0, static_cast<int>(num_samples)}, j);
        nc::NdArray<double> diff = feature_col - means[j];
        std_devs[j] = std::sqrt(nc::sum(nc::square(diff))[0] / num_samples);
    }
    for (size_t i = 0; i < num_samples; ++i) {
        for (size_t j = 0; j < num_features; ++j) {
            if (std_devs[j] != 0) {
                data(i, j) = (data(i, j) - means[j]) / std_devs[j];
            }
        }
    }
    return data;
}

double euclidean_distance(const nc::NdArray<double>& point1, const nc::NdArray<double>& point2) {
    nc::NdArray<double> diff = point1 - point2;
    double sum = nc::sum(nc::square(diff))[0];
    return std::sqrt(sum);
}

std::vector<size_t> max_min_sampling(const nc::NdArray<double>& data, size_t num_samples) {
    std::vector<size_t> sampled_indices;
    sampled_indices.push_back(std::rand() % data.numRows());

    for (size_t i = 1; i < num_samples; ++i) {
        double max_min_dist = -1.0;
        size_t best_index = 0;

        std::vector<size_t> remaining_indices;
        for (size_t idx = 0; idx < data.numRows(); ++idx) {
            if (std::find(sampled_indices.begin(), sampled_indices.end(), idx) == sampled_indices.end()) {
                remaining_indices.push_back(idx);
            }
        }

        auto process_chunk = [&data, &sampled_indices](size_t start, size_t end, const std::vector<size_t>& remaining_indices) 
            -> std::pair<double, size_t> {
            double local_max_min_dist = -1.0;
            size_t local_best_index = 0;

            for (size_t idx = start; idx < end; ++idx) {
                double min_dist = std::numeric_limits<double>::max();
                for (const auto& sampled_idx : sampled_indices) {
                    nc::NdArray<double> point1 = data(sampled_idx, {0, static_cast<int>(data.numCols())});
                    nc::NdArray<double> point2 = data(remaining_indices[idx], {0, static_cast<int>(data.numCols())});
                    double dist = euclidean_distance(point1, point2);
                    min_dist = std::min(min_dist, dist);
                }
                if (min_dist > local_max_min_dist) {
                    local_max_min_dist = min_dist;
                    local_best_index = remaining_indices[idx];
                }
            }
            return {local_max_min_dist, local_best_index};
        };

        size_t num_threads = std::thread::hardware_concurrency();
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
    }
    return sampled_indices;
}

void process_and_select_samples(const std::string& file_path, size_t num_samples, const std::string& output_path) {
    std::srand(42);

    NewDataFrame df;
    df.read_csv(file_path);

    std::vector<std::string> column_names = df.get_column_names();
    std::string target_column = column_names.back();
    
    std::vector<std::string> feature_columns(column_names.begin(), column_names.end() - 1);
    nc::NdArray<double> X = df.get_data()({0, static_cast<int>(df.get_num_rows())}, 
                                         {0, static_cast<int>(df.get_num_columns() - 1)});
    
    nc::NdArray<double> y = df.get_data()({0, static_cast<int>(df.get_num_rows())}, 
                                         df.get_num_columns() - 1);

    nc::NdArray<double> X_scaled = standardize_data(X);

    std::vector<size_t> representative_indices;
    if (num_samples >= X_scaled.numRows()) {
        representative_indices.resize(X_scaled.numRows());
        std::iota(representative_indices.begin(), representative_indices.end(), 0);
    } else {
        representative_indices = max_min_sampling(X_scaled, num_samples);
    }

    nc::NdArray<double> X_representative(representative_indices.size(), X.numCols());
    nc::NdArray<double> y_representative(representative_indices.size(), 1);
    
    for (size_t i = 0; i < representative_indices.size(); ++i) {
        for (size_t j = 0; j < X.numCols(); ++j) {
            X_representative(i, j) = X(representative_indices[i], j);
        }
        y_representative(i, 0) = y[representative_indices[i]];
    }

    nc::NdArray<double> combined_data = nc::hstack({X_representative, y_representative});
    
    NewDataFrame result_df(combined_data, column_names);
    result_df.write_csv(output_path);
    
    std::cout << "Representative samples saved to " << output_path << std::endl;
}

PYBIND11_MODULE(represent_data, m) {
    m.def("standardize_data", [](py::array_t<double> data) {
        py::buffer_info buf = data.request();
        size_t rows = buf.shape[0];
        size_t cols = buf.shape[1];
        
        double* ptr = static_cast<double*>(buf.ptr);
        nc::NdArray<double> nc_data(ptr, rows, cols);
        
        nc::NdArray<double> result = standardize_data(nc_data);
        
        return py::array_t<double>(
            {result.numRows(), result.numCols()},
            {result.numCols() * sizeof(double), sizeof(double)},
            result.data()
        );
    }, "A function to standardize data");
    
    m.def("max_min_sampling", [](py::array_t<double> data, size_t num_samples) {
        py::buffer_info buf = data.request();
        size_t rows = buf.shape[0];
        size_t cols = buf.shape[1];
        
        double* ptr = static_cast<double*>(buf.ptr);
        nc::NdArray<double> nc_data(ptr, rows, cols);
        
        return max_min_sampling(nc_data, num_samples);
    }, "A function for max-min sampling");
    
    m.def("process_and_select_samples", &process_and_select_samples, 
          "A function to process and select representative samples from a CSV file.");
}