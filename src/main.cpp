#include <CLI/CLI.hpp>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <magic_enum/magic_enum.hpp>
#include <safetensors.hpp>
#include <string>
#include <tensor.hpp>
#include <unordered_map>

void print_safetensors_info(const safetensors::SafeTensors& st) {
    std::cout << "\nSafeTensors Information:\n";
    std::cout << "========================\n\n";

    std::cout << "Total tensors: " << st.size() << "\n";
    if (st.empty()) {
        std::cout << "No tensors found.\n";
        return;
    }

    // Calculate total file statistics
    std::size_t total_parameters = 0;
    std::size_t total_data_size = 0;
    std::unordered_map<std::string, std::size_t> dtype_counts;
    std::unordered_map<std::string, std::size_t> dtype_sizes;

    auto tensor_names = st.names();
    // Sort tensor names with proper numeric ordering by splitting on '.'
    std::ranges::sort(tensor_names, [](const std::string_view& a, const std::string_view& b) {
        auto split = [](std::string_view str) {
            std::vector<std::string> parts;
            size_t start = 0;
            size_t pos = 0;
            while ((pos = str.find('.', start)) != std::string_view::npos) {
                parts.emplace_back(str.substr(start, pos - start));
                start = pos + 1;
            }
            parts.emplace_back(str.substr(start));
            return parts;
        };

        auto parts_a = split(a);
        auto parts_b = split(b);

        for (size_t i = 0; i < std::min(parts_a.size(), parts_b.size()); ++i) {
            // Try to parse as numbers first
            bool is_num_a = !parts_a[i].empty() && std::all_of(parts_a[i].begin(), parts_a[i].end(), ::isdigit);
            bool is_num_b = !parts_b[i].empty() && std::all_of(parts_b[i].begin(), parts_b[i].end(), ::isdigit);

            if (is_num_a && is_num_b) {
                int num_a = std::stoi(parts_a[i]);
                int num_b = std::stoi(parts_b[i]);
                if (num_a != num_b) return num_a < num_b;
            } else {
                if (parts_a[i] != parts_b[i]) return parts_a[i] < parts_b[i];
            }
        }
        return parts_a.size() < parts_b.size();
    });

    for (size_t i = 0; i < tensor_names.size(); ++i) {
        const auto& name = tensor_names[i];
        auto tensor_view = st.tensor(name);

        std::cout << "Tensor #" << i + 1 << ":\n";
        std::cout << "  Name: " << name << "\n";

        auto dtype_name = std::string(magic_enum::enum_name(tensor_view.dtype()));
        std::cout << "  Dtype: " << dtype_name << "\n";

        std::cout << "  Shape: [";
        const auto& shape = tensor_view.shape();
        for (size_t j = 0; j < shape.size(); ++j) {
            std::cout << shape[j];
            if (j < shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";

        std::cout << "  Dimensions: " << shape.size() << "D\n";

        std::size_t total_elements = 1;
        for (auto dim : shape) {
            total_elements *= dim;
        }
        std::cout << "  Elements: " << total_elements << "\n";

        auto data_size = tensor_view.data_len();
        std::cout << "  Data size: " << data_size << " bytes";
        if (data_size >= 1024 * 1024) {
            std::cout << " (" << std::fixed << std::setprecision(2) << data_size / (1024.0 * 1024.0) << " MB)";
        } else if (data_size >= 1024) {
            std::cout << " (" << std::fixed << std::setprecision(2) << data_size / 1024.0 << " KB)";
        }
        std::cout << "\n";

        auto element_size = safetensors::dtype_bitsize(tensor_view.dtype()) / 8;
        std::cout << "  Element size: " << element_size << " bytes\n";

        // Accumulate statistics
        total_parameters += total_elements;
        total_data_size += data_size;
        dtype_counts[dtype_name]++;
        dtype_sizes[dtype_name] += data_size;

        std::cout << "\n";
    }

    // Print summary statistics
    std::cout << "File Summary:\n";
    std::cout << "=============\n";
    std::cout << "Total parameters: " << total_parameters << "\n";
    std::cout << "Total data size: " << total_data_size << " bytes";
    if (total_data_size >= 1024 * 1024 * 1024) {
        std::cout << " (" << std::fixed << std::setprecision(2) << total_data_size / (1024.0 * 1024.0 * 1024.0)
                  << " GB)";
    } else if (total_data_size >= 1024 * 1024) {
        std::cout << " (" << std::fixed << std::setprecision(2) << total_data_size / (1024.0 * 1024.0) << " MB)";
    }
    std::cout << "\n";

    std::cout << "\nData type breakdown:\n";
    for (const auto& [dtype, count] : dtype_counts) {
        std::cout << "  " << dtype << ": " << count << " tensor(s), ";
        auto size = dtype_sizes[dtype];
        std::cout << size << " bytes";
        if (size >= 1024 * 1024) {
            std::cout << " (" << std::fixed << std::setprecision(2) << size / (1024.0 * 1024.0) << " MB)";
        }
        std::cout << "\n";
    }
}

int main(int argc, char** argv) {
    CLI::App app{"SafeTensors Loader"};

    std::string filename;
    bool show_tensors = false;

    app.add_option("file", filename, "SafeTensors file to load")->required();
    app.add_flag("-t,--tensors", show_tensors, "Show tensor information");

    CLI11_PARSE(app, argc, argv);

    try {
        std::cout << "Loading SafeTensors file: " << filename << "\n";
        safetensors::SafeTensors st(filename);

        std::cout << "File loaded successfully\n";
        std::cout << "Total tensors: " << st.size() << "\n";

        if (!st.empty()) {
            // Show basic summary even without -t flag
            std::size_t total_parameters = 0;
            std::size_t total_data_size = 0;

            auto tensor_names = st.names();
            for (const auto& name : tensor_names) {
                auto tensor_view = st.tensor(name);
                std::size_t elements = 1;
                for (auto dim : tensor_view.shape()) {
                    elements *= dim;
                }
                total_parameters += elements;
                total_data_size += tensor_view.data_len();
            }

            std::cout << "Total parameters: " << total_parameters << "\n";
            std::cout << "Total data size: " << total_data_size << " bytes";
            if (total_data_size >= 1024 * 1024 * 1024) {
                std::cout << " (" << std::fixed << std::setprecision(2) << total_data_size / (1024.0 * 1024.0 * 1024.0)
                          << " GB)";
            } else if (total_data_size >= 1024 * 1024) {
                std::cout << " (" << std::fixed << std::setprecision(2) << total_data_size / (1024.0 * 1024.0)
                          << " MB)";
            }
            std::cout << "\n";
        }

        if (show_tensors) {
            print_safetensors_info(st);
        } else if (!st.empty()) {
            std::cout << "\nUse -t flag to see detailed tensor information.\n";
        }

        // Demonstrate tensor usage with first float tensor
        auto names = st.names();
        for (const auto& name : names) {
            auto view = st.tensor(name);
            if (view.dtype() == safetensors::Dtype::F32) {
                std::cout << "\nTensor demonstration with: " << name << "\n";
                infer::Tensor<float> tensor(view);
                std::cout << "Created tensor view with " << tensor.size() << " elements\n";
                std::cout << "First few values: ";
                for (size_t i = 0; i < std::min(size_t(5), tensor.size()); ++i) {
                    std::cout << tensor[i] << " ";
                }
                std::cout << "\n";
                break;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
