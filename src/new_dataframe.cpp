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
#include <NumCpp.hpp> // NumCpp
#include <omp.h>      // OpenMP
#include <../include/new_dataframe.h>

namespace py = pybind11;

// Pandas DataFrame to NewDataFrame
NewDataFrame from_pandas(py::object pandas_df)
{

    py::list columns = pandas_df.attr("columns");
    std::vector<std::string> col_names;
    for (py::handle col : columns)
    {
        col_names.push_back(col.cast<std::string>());
    }

    //  NumCpp NdArray
    py::object values = pandas_df.attr("values");
    py::array_t<double> numpy_values = values.cast<py::array_t<double>>();

    std::vector<double> data_flat;
    ssize_t num_rows = numpy_values.shape(0);
    ssize_t num_cols = numpy_values.shape(1);

    for (ssize_t i = 0; i < num_rows; ++i)
    {
        for (ssize_t j = 0; j < num_cols; ++j)
        {
            data_flat.push_back(*numpy_values.data(i, j));
        }
    }

    // creat NumCpp NdArray
    nc::NdArray<double> data_array(data_flat.data(), num_rows, num_cols);

    return NewDataFrame(data_array, col_names);
}

//  NewDataFrame to Pandas DataFrame
py::object to_pandas(const NewDataFrame &df)
{
    // 
    const nc::NdArray<double> &data = df.get_data();
    const std::vector<std::string> &column_names = df.get_column_names();

    // 
    py::array_t<double> numpy_data({data.numRows(), data.numCols()});
    auto numpy_data_mutable = numpy_data.mutable_unchecked<2>();

    // 
    for (size_t i = 0; i < data.numRows(); ++i)
    {
        for (size_t j = 0; j < data.numCols(); ++j)
        {
            numpy_data_mutable(i, j) = data(i, j);
        }
    }

    // 
    py::list pandas_columns;
    for (const auto &col_name : column_names)
    {
        pandas_columns.append(col_name);
    }

    // 
    py::module_ pd = py::module_::import("pandas");
    py::object pandas_df = pd.attr("DataFrame")(numpy_data, py::arg("columns") = pandas_columns);

    return pandas_df;
}



PYBIND11_MODULE(new_dataframe_module, m)
{
    py::class_<NewDataFrame>(m, "NewDataFrame")
        .def(py::init<>())
        .def(py::init<const nc::NdArray<double> &, const std::vector<std::string> &>())
        .def("set_data", &NewDataFrame::set_data)
        .def("set_column_names", &NewDataFrame::set_column_names)
        .def("get_data", &NewDataFrame::get_data)
        .def("get_column_names", &NewDataFrame::get_column_names)
        .def("get_column", &NewDataFrame::get_column)
        .def("add_column", &NewDataFrame::add_column)
        .def("remove_column", &NewDataFrame::remove_column)
        .def("print", &NewDataFrame::print)
        .def("read_csv", &NewDataFrame::read_csv)
        .def("write_csv", &NewDataFrame::write_csv)
        .def("get_num_rows", &NewDataFrame::get_num_rows)
        .def("get_num_columns", &NewDataFrame::get_num_columns)
        .def("get_columns_as_numpy", &NewDataFrame::get_columns_as_numpy)
        .def("generate_and_save_new_features", &NewDataFrame::generate_and_save_new_features);
    // 
    m.def("from_pandas", &from_pandas, "Convert Pandas DataFrame to C++ NewDataFrame", py::arg("pandas_df"));
    m.def("to_pandas", &to_pandas, "Convert C++ NewDataFrame to Pandas DataFrame", py::arg("new_dataframe"));
}
