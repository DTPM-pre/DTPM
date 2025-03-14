#include <iostream>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <NumCpp.hpp> // 

namespace py = pybind11;

class NewDataFrame
{
public:
    nc::NdArray<double> data;                                 // 
    std::vector<std::string> column_names;                    // 
    std::unordered_map<std::string, size_t> column_index_map; // 

    // 
    NewDataFrame() = default;

    // 
    NewDataFrame(const nc::NdArray<double> &data_array, const std::vector<std::string> &columns)
    {
        set_data(data_array);
        set_column_names(columns);
    }

    // 
    void set_data(const nc::NdArray<double> &new_data)
    {
        data = new_data;
    }

    // 
    void set_column_names(const std::vector<std::string> &new_column_names)
    {
        column_names = new_column_names;
        column_index_map.clear();
        if (data.numCols() != column_names.size())
        {
            throw std::invalid_argument("Number of columns does not match the number of columns in the data.");
        }
        for (size_t i = 0; i < column_names.size(); ++i)
        {
            column_index_map[column_names[i]] = i;
        }
    }

    // 
    nc::NdArray<double> get_data() const
    {
        return data;
    }

    // 
    std::vector<std::string> get_column_names() const
    {
        return column_names;
    }

    // 
    nc::NdArray<double> get_column(const std::string &column_name) const
    {
        if (column_index_map.find(column_name) == column_index_map.end())
        {
            throw std::invalid_argument("Column does not exist.");
        }
        size_t col_index = column_index_map.at(column_name);
        int num_rows = static_cast<int>(data.numRows());
        return data({0, num_rows}, col_index); // 
    }

    // 
    void add_column(std::string column_name, nc::NdArray<double> column_data)
    {
        if (column_data.numRows() != data.numRows())
        {
            throw std::invalid_argument("Column size does not match the number of rows.");
        }

        // 
        column_names.push_back(column_name);
        column_index_map[column_name] = column_names.size() - 1;

        // 
        data = nc::hstack({data, column_data.reshape(column_data.numRows(), 1)});
    }

    // 
    void remove_column(const std::string &column_name)
    {
        if (column_index_map.find(column_name) == column_index_map.end())
        {
            throw std::invalid_argument("Column does not exist.");
        }
        size_t col_index = column_index_map.at(column_name);
        column_names.erase(column_names.begin() + col_index);
        column_index_map.erase(column_name);

        // 
        data = nc::deleteIndices(data, col_index, nc::Axis::COL);

        // 
        for (size_t i = col_index; i < column_names.size(); ++i)
        {
            column_index_map[column_names[i]] = i;
        }
    }

    // 
    void print() const
    {
        for (const auto &col_name : column_names)
        {
            std::cout << col_name << " ";
        }
        std::cout << std::endl;

        std::cout << data << std::endl;
    }

    //
    void read_csv(const std::string &file_path)
    {
        std::ifstream file(file_path);
        if (!file.is_open())
        {
            throw std::runtime_error("Could not open file.");
        }

        std::string line, cell;
        std::getline(file, line);
        std::stringstream ss(line);

        while (std::getline(ss, cell, ','))
        {
            column_names.push_back(cell);
        }

        std::vector<std::vector<double>> temp_data;
        while (std::getline(file, line))
        {
            std::stringstream line_stream(line);
            std::vector<double> row;
            while (std::getline(line_stream, cell, ','))
            {
                row.push_back(std::stod(cell));
            }
            temp_data.push_back(row);
        }

        std::vector<double> flat_data;
        for (const auto &row : temp_data)
        {
            flat_data.insert(flat_data.end(), row.begin(), row.end());
        }

        data = nc::NdArray<double>(flat_data.data(), temp_data.size(), temp_data[0].size());

        for (size_t i = 0; i < column_names.size(); ++i)
        {
            column_index_map[column_names[i]] = i;
        }
        file.close();
    }

    // 
    void write_csv(const std::string &file_path) const
    {
        std::ofstream file(file_path);
        if (!file.is_open())
        {
            throw std::runtime_error("Could not open file.");
        }

        // 
        for (size_t i = 0; i < column_names.size(); ++i)
        {
            file << column_names[i];
            if (i < column_names.size() - 1)
            {
                file << ",";
            }
        }
        file << "\n";

        // 
        for (size_t i = 0; i < data.numRows(); ++i)
        {
            for (size_t j = 0; j < data.numCols(); ++j)
            {
                file << data(i, j);
                if (j < data.numCols() - 1)
                {
                    file << ",";
                }
            }
            file << "\n";
        }
        file.close();
    }

    // 
    size_t get_num_rows() const
    {
        return data.numRows();
    }

    // 
    size_t get_num_columns() const
    {
        return data.numCols();
    }

    //
    py::array_t<double> get_columns_as_numpy(const std::vector<std::string> &col_names) const
    {
        size_t num_rows = data.numRows();
        size_t num_cols = col_names.size();

        // 
        py::array_t<double> result({num_rows, num_cols});
        auto result_mutable = result.mutable_unchecked<2>();

        // 
        std::vector<size_t> col_indices;
        col_indices.reserve(num_cols);
        for (const auto &col_name : col_names)
        {
            if (column_index_map.find(col_name) == column_index_map.end())
            {
                throw std::invalid_argument("Column " + col_name + " does not exist.");
            }
            col_indices.push_back(column_index_map.at(col_name));
        }

        // 
        int i; // 
        #pragma omp parallel for
        for (i = 0; i < static_cast<int>(num_rows); ++i)
        {
            for (size_t j = 0; j < num_cols; ++j)
            {
                size_t col_idx = col_indices[j];
                result_mutable(i, j) = data(i, col_idx);
            }
        }

        return result;
    }

    // 
    void generate_and_save_new_features(const std::string &save_path)
    {

        // 
        std::vector<std::string> feature_names(column_names.begin(), column_names.end() - 1);
        std::string label_name = column_names.back(); // 

        // 
        size_t num_features = feature_names.size();

        // 
        size_t num_new_features = (num_features * (num_features - 1)) / 2 * 3;
        nc::NdArray<double> new_features(data.numRows(), num_new_features);

        size_t new_feature_idx = 0;
        int num_rows = static_cast<int>(data.numRows()); // 

        // 
        std::vector<std::string> new_feature_names;

        // 
        for (size_t i = 0; i < num_features; ++i)
        {
            for (size_t j = i + 1; j < num_features; ++j)
            {
                // 
                size_t col_i = column_index_map.at(feature_names[i]);
                size_t col_j = column_index_map.at(feature_names[j]);

                // 
                new_features.put({0, num_rows}, new_feature_idx, data({0, num_rows}, col_i) + data({0, num_rows}, col_j));
                new_feature_names.push_back(feature_names[i] + "+" + feature_names[j]); // 
                ++new_feature_idx;

                // 
                new_features.put({0, num_rows}, new_feature_idx, data({0, num_rows}, col_i) - data({0, num_rows}, col_j));
                new_feature_names.push_back(feature_names[i] + "-" + feature_names[j]); // 
                ++new_feature_idx;

                // 
                new_features.put({0, num_rows}, new_feature_idx, data({0, num_rows}, col_i) * data({0, num_rows}, col_j));
                new_feature_names.push_back(feature_names[i] + "*" + feature_names[j]); // 
                ++new_feature_idx;
            }
        }

        // 
        // nc::NdArray<double> all_features = nc::hstack({data({0, num_rows}, {0,num_features}), new_features});
        nc::NdArray<double> all_features = nc::hstack({data({0, num_rows}, {0, static_cast<int>(num_features)}), new_features, data({0, num_rows}, num_features)});

        // 
        // all_features = nc::hstack({all_features, data({0, num_rows}, num_features)});

        // 
        
        std::vector<std::string> all_column_names = feature_names;
        all_column_names.insert(all_column_names.end(), new_feature_names.begin(), new_feature_names.end());
        all_column_names.push_back(label_name);

        // 
        std::ofstream file(save_path);
        if (!file.is_open())
        {
            throw std::runtime_error("Could not open file.");
        }

        // 
        std::ostringstream header_stream;
        // 
        for (size_t i = 0; i < all_column_names.size(); ++i)
        {
            header_stream << all_column_names[i];
            if (i < all_column_names.size() - 1)
            {
                header_stream << ",";
            }
        }
        header_stream << "\n";
        // 
        file << header_stream.str();
        // 
        std::ostringstream data_stream;
        // 
        for (size_t i = 0; i < all_features.numRows(); ++i)
        {
            for (size_t j = 0; j < all_features.numCols(); ++j)
            {
                data_stream << all_features(i, j);
                if (j < all_features.numCols() - 1)
                {
                    data_stream << ",";
                }
            }
            data_stream << "\n";
        }
        // 
        file << data_stream.str();
        file.close();

        //NewDataFrame new_df = NewDataFrame(all_features, all_column_names);
        //new_df.print();
    }
};

