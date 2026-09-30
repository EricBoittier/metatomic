// A metatomic-core plugin that runs TorchScript models exported with metatomic-torch
// (`AtomisticModel`, `*.pt` files), so engines using the metatomic C API can run
// existing models such as PET-MAD and PET-OMol.
//
// The plugin loads the model with `metatomic_torch::load_atomistic_model`, declares
// its capabilities, pair lists and inputs in metatomic-core terms, and in
// `execute_inner` converts the systems (DLPack tensors), their pair lists and
// custom data to metatomic-torch, runs the model, and returns the outputs as
// metatensor-core tensor maps. When an energy is requested with gradients, the
// plugin computes them with autograd: the positions and cell are multiplied by a
// per-system strain, and `positions` and `strain` gradients of the total energy
// are returned.
//
// Units and dtype are the model's own: metatomic-core converts the systems to the
// model's length unit before calling the plugin, and the outputs to the requested
// units afterwards. The model runs on `device` (load option, or the
// METATOMIC_TORCH_DEVICE environment variable; default: CUDA when available), while
// inputs and outputs are CPU arrays.

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <ATen/DLConvertor.h>
#include <torch/cuda.h>
#include <torch/script.h>

#include <metatensor.hpp>
#include <metatensor/torch.hpp>
#include <metatomic.hpp>
#include <metatomic/plugin.hpp>
#include <metatomic/torch.hpp>

namespace {

namespace mta  = metatomic;
namespace mtt  = metatomic_torch;
namespace mtst = metatensor_torch;

bool ends_with(const std::string& s, const std::string& suffix)
{
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

//! Whether metatomic-core accepts this output name: a standard quantity (optionally with a
//! "/variant") or a namespaced one. Legacy models also list deprecated names such as
//! "features" next to "feature"; those are not declared.
bool declarable(const std::string& name)
{
    static const std::vector<std::string> standard = {
        "charge", "energy_ensemble", "energy_uncertainty", "energy", "feature", "heat_flux", "mass",
        "momentum", "non_conservative_force", "non_conservative_stress", "position",
        "spin_multiplicity", "velocity"
    };
    if (name.find("::") != std::string::npos)
    {
        return true;
    }
    const auto base = name.substr(0, name.find('/'));
    return std::find(standard.begin(), standard.end(), base) != standard.end();
}

mta::SampleKind sample_kind(const std::string& kind)
{
    if (kind == "atom")
    {
        return mta::SampleKind::Atom;
    }
    if (kind == "atom_pair")
    {
        return mta::SampleKind::AtomPair;
    }
    return mta::SampleKind::System;
}

std::string sample_kind_name(mta::SampleKind kind)
{
    switch (kind)
    {
        case mta::SampleKind::Atom: return "atom";
        case mta::SampleKind::AtomPair: return "atom_pair";
        default: return "system";
    }
}

//! A metatensor-core Labels as metatensor-torch Labels
mtst::Labels to_torch(const metatensor::Labels& labels, torch::Device device)
{
    const auto count = static_cast<int64_t>(labels.count());
    const auto size  = static_cast<int64_t>(labels.size());
    auto       values = torch::zeros({ count, size }, torch::kInt32);
    if (count > 0 && size > 0)
    {
        const auto cpu = labels.values_cpu();
        std::memcpy(values.data_ptr<int32_t>(), cpu.data(), sizeof(int32_t) * count * size);
    }
    std::vector<std::string> names;
    for (const auto& name : labels.names())
    {
        names.emplace_back(name);  // core names are C strings, torch wants std::string
    }
    return torch::make_intrusive<mtst::LabelsHolder>(names, values.to(device));
}

//! The float64 values of a metatensor-core block as a torch tensor
torch::Tensor block_values(metatensor::TensorBlock& block)
{
    auto                 array = block.values<double>();
    std::vector<int64_t> shape(array.shape().begin(), array.shape().end());
    return torch::from_blob(const_cast<double*>(array.data()), shape, torch::kFloat64).clone();
}

//! A metatensor-core tensor map (custom data, e.g. charges) as a metatensor-torch one
mtst::TensorMap to_torch(metatensor::TensorMap map, torch::Device device, torch::Dtype dtype)
{
    std::vector<mtst::TensorBlock> blocks;
    for (size_t b = 0; b < map.keys().count(); ++b)
    {
        auto                      block = map.block_by_id(b);
        std::vector<mtst::Labels> components;
        for (const auto& c : block.components())
        {
            components.push_back(to_torch(c, device));
        }
        blocks.push_back(torch::make_intrusive<mtst::TensorBlockHolder>(
                block_values(block).to(device, dtype), to_torch(block.samples(), device), components,
                to_torch(block.properties(), device)));
    }
    return torch::make_intrusive<mtst::TensorMapHolder>(to_torch(map.keys(), device), blocks);
}

//! A DLPack tensor owned by the system as a torch tensor on `device`
torch::Tensor from_dlpack(mta::DLPackTensor tensor, torch::Device device)
{
    return at::fromDLPackVersioned(tensor.release()).to(device);
}

//! A copy of a torch tensor (any device) as a metatensor-core array of dtype T
template<typename T>
std::unique_ptr<metatensor::SimpleDataArray<T>> to_array(const torch::Tensor& tensor)
{
    const auto cpu = tensor.detach().to(torch::kCPU, c10::CppTypeToScalarType<T>::value).contiguous();
    std::vector<uintptr_t> shape(cpu.sizes().begin(), cpu.sizes().end());
    std::vector<T>         data(cpu.template data_ptr<T>(), cpu.template data_ptr<T>() + cpu.numel());
    return std::make_unique<metatensor::SimpleDataArray<T>>(shape, std::move(data));
}

metatensor::Labels to_core_labels(const mtst::Labels& labels)
{
    const auto values = labels->values().to(torch::kCPU, torch::kInt32).contiguous();
    const auto count  = static_cast<size_t>(values.size(0));
    return metatensor::Labels(labels->names(), count > 0 ? values.data_ptr<int32_t>() : nullptr, count);
}

class TorchModel final : public mta::BaseModel
{
public:
    TorchModel(const std::string& path, const std::map<std::string, std::string>& options)
    {
        std::optional<std::string> extensions;
        if (auto it = options.find("extensions_directory"); it != options.end())
        {
            extensions = it->second;
        }
        model_ = mtt::load_atomistic_model(path, extensions);
        capabilities_ = model_.run_method("capabilities").toCustomClass<mtt::ModelCapabilitiesHolder>();
        metadata_     = model_.run_method("metadata").toCustomClass<mtt::ModelMetadataHolder>();
        for (const auto& value : model_.run_method("requested_neighbor_lists").toList())
        {
            neighbors_.push_back(value.get().toCustomClass<mtt::NeighborListOptionsHolder>());
        }
        // current input names (e.g. "charge", "spin_multiplicity"); older models only have the old ones
        c10::IValue inputs;
        try
        {
            inputs = model_.run_method("requested_inputs", true);
        }
        catch (const std::exception&)
        {
            inputs = model_.run_method("requested_inputs");
        }
        for (const auto& entry : inputs.toGenericDict())
        {
            inputs_.emplace_back(entry.key().toStringRef(),
                                 entry.value().toCustomClass<mtt::ModelOutputHolder>());
        }

        dtype_ = capabilities_->dtype() == "float64" ? torch::kFloat64 : torch::kFloat32;
        std::string device = torch::cuda::is_available() ? "cuda" : "cpu";
        if (const char* env = std::getenv("METATOMIC_TORCH_DEVICE"))
        {
            device = env;
        }
        if (auto it = options.find("device"); it != options.end())
        {
            device = it->second;
        }
        device_ = torch::Device(device);
        model_.to(device_);

        if (device_.is_cuda())
        {
            // PyTorch loads its CUDA linear algebra lazily and not thread-safely; load it once
            // here, so that engines evaluating from several threads don't race on first use.
            static std::once_flag loaded;
            std::call_once(loaded,
                           [device = device_]()
                           {
                               torch::NoGradGuard noGrad;
                               const auto a = torch::eye(2, torch::TensorOptions().dtype(torch::kFloat64).device(device));
                               (void)torch::linalg_solve_triangular(a, a, /*upper=*/false);
                           });
        }
    }

    mta::ModelCapabilities capabilities() const override
    {
        auto builder = mta::ModelCapabilities::builder();
        for (const auto& entry : capabilities_->outputs())
        {
            if (!declarable(entry.key()))
            {
                continue;
            }
            builder.add_output(mta::Quantity::builder()
                                       .name(entry.key())
                                       .unit(entry.value()->unit())
                                       .sample_kind(sample_kind(entry.value()->sample_kind()))
                                       .build());
        }
        return builder.atomic_types(capabilities_->atomic_types)
                .interaction_range(capabilities_->interaction_range)
                .length_unit(capabilities_->length_unit())
                // the engine hands over CPU arrays; the plugin moves them to its device
                .supported_devices({ mta::ModelCapabilities::Device::CPU })
                .dtype(dtype_ == torch::kFloat64 ? mta::ModelCapabilities::DType::Float64
                                                 : mta::ModelCapabilities::DType::Float32)
                .build();
    }

    mta::ModelMetadata metadata() const override
    {
        return mta::ModelMetadata::builder().name(metadata_->name).build();
    }

    std::vector<mta::PairListOptions> requested_pair_lists() const override
    {
        std::vector<mta::PairListOptions> pairs;
        for (const auto& options : neighbors_)
        {
            pairs.push_back(core_pairs(options));
        }
        return pairs;
    }

    std::vector<mta::Quantity> requested_inputs() const override
    {
        std::vector<mta::Quantity> inputs;
        for (const auto& [name, option] : inputs_)
        {
            inputs.push_back(mta::Quantity::builder()
                                     .name(name)
                                     .unit(option->unit())
                                     .sample_kind(sample_kind(option->sample_kind()))
                                     .build());
        }
        return inputs;
    }

    std::vector<metatensor::TensorMap> execute_inner(const std::vector<mta::System>& systems,
                                                     const metatensor::Labels*        selected,
                                                     const std::vector<mta::Quantity>& requested) override
    {
        bool gradients = false;
        for (const auto& output : requested)
        {
            gradients = gradients || !output.gradients().empty();
        }

        // Torch systems: positions and cell scaled by a per-system strain, for the virial
        std::vector<mtt::System>   torchSystems;
        std::vector<torch::Tensor> positions, strains;
        for (const auto& system : systems)
        {
            auto types = from_dlpack(system.types(), device_).to(torch::kInt32);
            auto pos   = from_dlpack(system.positions(), device_).to(dtype_).detach().requires_grad_(gradients);
            auto cell  = from_dlpack(system.cell(), device_).to(dtype_);
            auto pbc   = from_dlpack(system.pbc(), device_).to(torch::kBool);
            auto strain = torch::eye(3, torch::TensorOptions().dtype(dtype_).device(device_)).requires_grad_(gradients);
            positions.push_back(pos);
            strains.push_back(strain);
            auto torchSystem = torch::make_intrusive<mtt::SystemHolder>(
                    types, torch::matmul(pos, strain), torch::matmul(cell, strain), pbc);

            for (const auto& options : neighbors_)
            {
                auto       block   = system.pairs(core_pairs(options));
                auto       vectors = block_values_as(block).to(device_, dtype_);
                auto       nl      = torch::make_intrusive<mtst::TensorBlockHolder>(
                        vectors, to_torch(block.samples(), device_),
                        std::vector<mtst::Labels>{ to_torch(block.components()[0], device_) },
                        to_torch(block.properties(), device_));
                mtt::register_autograd_neighbors(torchSystem, nl, /*check_consistency=*/false);
                torchSystem->add_neighbor_list(options, nl);
            }
            for (const auto& [name, option] : inputs_)
            {
                torchSystem->add_data(name, to_torch(system.custom_data(name), device_, dtype_));
            }
            torchSystems.push_back(torchSystem);
        }

        // Run the model for the requested outputs, in the model's own units
        auto evaluation = torch::make_intrusive<mtt::ModelEvaluationOptionsHolder>();
        evaluation->set_length_unit(capabilities_->length_unit());
        torch::Dict<std::string, mtt::ModelOutput> outputs;
        for (const auto& output : requested)
        {
            auto option = torch::make_intrusive<mtt::ModelOutputHolder>();
            option->set_unit(capabilities_->outputs().at(output.name())->unit());
            option->set_sample_kind(sample_kind_name(output.sample_kind()));
            outputs.insert(output.name(), option);
        }
        evaluation->outputs = outputs;
        if (selected != nullptr)
        {
            evaluation->set_selected_atoms(to_torch(*selected, device_));
        }
        auto result = model_.forward({ torchSystems, evaluation, /*check_consistency=*/false }).toGenericDict();

        std::vector<metatensor::TensorMap> tensors;
        for (const auto& output : requested)
        {
            auto map = result.at(output.name()).toCustomClass<mtst::TensorMapHolder>();
            tensors.push_back(dtype_ == torch::kFloat64 ? to_core<double>(map, output, positions, strains)
                                                         : to_core<float>(map, output, positions, strains));
        }
        return tensors;
    }

private:
    static mta::PairListOptions core_pairs(const mtt::NeighborListOptions& options)
    {
        return mta::PairListOptions::builder()
                .cutoff(options->cutoff())
                .full_list(options->full_list())
                .strict(options->strict())
                .build();
    }

    torch::Tensor block_values_as(metatensor::TensorBlock& block) const
    {
        // the engine sends pair vectors in the model dtype
        if (dtype_ == torch::kFloat64)
        {
            return block_values(block);
        }
        auto                 array = block.values<float>();
        std::vector<int64_t> shape(array.shape().begin(), array.shape().end());
        return torch::from_blob(const_cast<float*>(array.data()), shape, torch::kFloat32).clone();
    }

    //! One output as a metatensor-core tensor map, with the gradients the engine asked for
    template<typename T>
    metatensor::TensorMap to_core(const mtst::TensorMap&            map,
                                  const mta::Quantity&              output,
                                  const std::vector<torch::Tensor>& positions,
                                  const std::vector<torch::Tensor>& strains) const
    {
        bool wantPositions = false, wantStrain = false;
        for (const auto g : output.gradients())
        {
            wantPositions = wantPositions || g == mta::Gradients::Positions;
            wantStrain    = wantStrain || g == mta::Gradients::Strain;
        }

        std::vector<metatensor::TensorBlock> blocks;
        const auto                           keys = map->keys();
        for (int64_t b = 0; b < keys->count(); ++b)
        {
            auto       block   = mtst::TensorMapHolder::block_by_id(map, b);
            const auto values  = block->values();
            const auto samples = block->samples();
            std::vector<metatensor::Labels> components;
            for (const auto& c : block->components())
            {
                components.push_back(to_core_labels(c));
            }
            metatensor::TensorBlock core(to_array<T>(values), to_core_labels(samples), components,
                                         to_core_labels(block->properties()));

            if (wantPositions || wantStrain)
            {
                // gradients of the total, attached to the first sample of each system
                std::vector<torch::Tensor> inputs;
                for (size_t s = 0; s < positions.size(); ++s)
                {
                    if (wantPositions)
                    {
                        inputs.push_back(positions[s]);
                    }
                    if (wantStrain)
                    {
                        inputs.push_back(strains[s]);
                    }
                }
                const auto grads = torch::autograd::grad(
                        { values.sum() }, inputs, {}, /*retain_graph=*/true, /*create_graph=*/false,
                        /*allow_unused=*/true);

                const auto sampleSystems = samples->column("system").to(torch::kCPU);
                std::vector<int32_t> firstRow(positions.size(), -1);
                for (int64_t row = 0; row < sampleSystems.size(0); ++row)
                {
                    auto& first = firstRow[sampleSystems[row].item<int32_t>()];
                    if (first < 0)
                    {
                        first = static_cast<int32_t>(row);
                    }
                }

                const int32_t xyz[3] = { 0, 1, 2 };
                size_t        k      = 0;
                std::vector<int32_t> posSamples, strainSamples;
                std::vector<torch::Tensor> posValues, strainValues;
                for (size_t s = 0; s < positions.size(); ++s)
                {
                    torch::Tensor gp, gs;
                    if (wantPositions)
                    {
                        gp = grads[k++];
                    }
                    if (wantStrain)
                    {
                        gs = grads[k++];
                    }
                    if (firstRow[s] < 0)
                    {
                        continue;  // no output sample for this system
                    }
                    const int64_t n = positions[s].size(0);
                    if (wantPositions)
                    {
                        for (int64_t atom = 0; atom < n; ++atom)
                        {
                            posSamples.insert(posSamples.end(), { firstRow[s], static_cast<int32_t>(s), static_cast<int32_t>(atom) });
                        }
                        posValues.push_back(gp.defined() ? gp : torch::zeros_like(positions[s]));
                    }
                    if (wantStrain)
                    {
                        strainSamples.push_back(firstRow[s]);
                        strainValues.push_back((gs.defined() ? gs : torch::zeros_like(strains[s])).unsqueeze(0));
                    }
                }
                const auto nProps = values.size(-1);
                if (wantPositions)
                {
                    auto v = posValues.empty() ? torch::zeros({ 0, 3 }, values.options()) : torch::cat(posValues);
                    core.add_gradient("positions",
                                      metatensor::TensorBlock(to_array<T>(v.unsqueeze(-1).expand({ -1, 3, nProps }).contiguous()),
                                                              metatensor::Labels({ "sample", "system", "atom" },
                                                                                 posSamples.empty() ? nullptr : posSamples.data(),
                                                                                 posSamples.size() / 3),
                                                              { metatensor::Labels({ "xyz" }, xyz, 3) },
                                                              to_core_labels(block->properties())));
                }
                if (wantStrain)
                {
                    auto v = strainValues.empty() ? torch::zeros({ 0, 3, 3 }, values.options()) : torch::cat(strainValues);
                    core.add_gradient("strain",
                                      metatensor::TensorBlock(to_array<T>(v.unsqueeze(-1).expand({ -1, 3, 3, nProps }).contiguous()),
                                                              metatensor::Labels({ "sample" },
                                                                                 strainSamples.empty() ? nullptr : strainSamples.data(),
                                                                                 strainSamples.size()),
                                                              { metatensor::Labels({ "xyz_1" }, xyz, 3),
                                                                metatensor::Labels({ "xyz_2" }, xyz, 3) },
                                                              to_core_labels(block->properties())));
                }
            }
            blocks.push_back(std::move(core));
        }
        return metatensor::TensorMap(to_core_labels(keys), std::move(blocks));
    }

    mutable torch::jit::Module                             model_;
    mtt::ModelCapabilities                                 capabilities_;
    mtt::ModelMetadata                                     metadata_;
    std::vector<mtt::NeighborListOptions>                  neighbors_;
    std::vector<std::pair<std::string, mtt::ModelOutput>>  inputs_;
    torch::Dtype                                           dtype_  = torch::kFloat32;
    torch::Device                                          device_ = torch::kCPU;
};

std::unique_ptr<mta::BaseModel> load_torch_model(const std::string& load_from, const std::map<std::string, std::string>& options)
{
    if (!ends_with(load_from, ".pt"))
    {
        return nullptr;  // not a TorchScript model: let another plugin try
    }
    return std::make_unique<TorchModel>(load_from, options);
}

} // namespace

MTA_REGISTER_CXX_PLUGIN("metatomic-torch", load_torch_model);
