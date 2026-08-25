#include "be/mlp.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace be {

namespace {

constexpr char kMagic[4] = {'B', 'E', 'P', 'P'};
constexpr uint32_t kSupportedVersion = 1;

// Reads exactly n raw bytes from in into out, throwing rather than leaving
// out partially-filled/uninitialized if the stream runs out first - the
// truncated-file guard this whole loader exists for (see
// PolicyWeights::load's doc comment in mlp.hpp).
void read_exact(std::istream& in, void* out, size_t n, const std::string& what) {
  in.read(reinterpret_cast<char*>(out), static_cast<std::streamsize>(n));
  if (!in || static_cast<size_t>(in.gcount()) != n) {
    throw std::runtime_error("PolicyWeights::load: truncated weight file - short read on " + what);
  }
}

uint32_t read_u32(std::istream& in, const std::string& what) {
  // File is little-endian throughout (export_weights.py's struct.pack("<...")
  // convention) - this loader assumes a little-endian host, matching every
  // machine this project actually builds/runs on (Apple Silicon, x86_64).
  uint32_t value = 0;
  read_exact(in, &value, sizeof(value), what);
  return value;
}

MlpLayer read_layer(std::istream& in, const std::string& name) {
  MlpLayer layer;
  layer.out_dim = static_cast<int>(read_u32(in, name + ".out_dim"));
  layer.in_dim = static_cast<int>(read_u32(in, name + ".in_dim"));
  if (layer.out_dim <= 0 || layer.in_dim <= 0) {
    throw std::runtime_error("PolicyWeights::load: " + name + " declares a non-positive dimension (" +
                              std::to_string(layer.out_dim) + ", " + std::to_string(layer.in_dim) +
                              ") - malformed header");
  }
  layer.weight.resize(static_cast<size_t>(layer.out_dim) * static_cast<size_t>(layer.in_dim));
  read_exact(in, layer.weight.data(), layer.weight.size() * sizeof(float), name + ".weight");
  layer.bias.resize(static_cast<size_t>(layer.out_dim));
  read_exact(in, layer.bias.data(), layer.bias.size() * sizeof(float), name + ".bias");
  return layer;
}

void check_chain(const MlpLayer& upstream, const MlpLayer& downstream, const std::string& where) {
  if (upstream.out_dim != downstream.in_dim) {
    throw std::runtime_error("PolicyWeights::load: dimension-chain mismatch at " + where + ": " +
                              std::to_string(upstream.out_dim) + " != " + std::to_string(downstream.in_dim));
  }
}

void check_input_width(const MlpLayer& layer0, uint32_t vector_len, const std::string& branch) {
  if (layer0.in_dim != static_cast<int>(vector_len)) {
    throw std::runtime_error("PolicyWeights::load: " + branch + ".net[0].in_dim (" +
                              std::to_string(layer0.in_dim) + ") doesn't match the header's own vector_len (" +
                              std::to_string(vector_len) + ")");
  }
}

}  // namespace

std::vector<float> MlpWeights::forward(const std::vector<float>& input) const {
  // Double-precision accumulation per this header's own doc comment - see
  // mlp.hpp for why (reduces summation-order drift vs. PyTorch's own
  // accumulation, so the parity test can use a tolerance band).
  auto linear = [](const MlpLayer& layer, const std::vector<float>& x, bool apply_relu) {
    std::vector<float> out(static_cast<size_t>(layer.out_dim));
    for (int o = 0; o < layer.out_dim; ++o) {
      double acc = static_cast<double>(layer.bias[static_cast<size_t>(o)]);
      const float* row = &layer.weight[static_cast<size_t>(o) * static_cast<size_t>(layer.in_dim)];
      for (int i = 0; i < layer.in_dim; ++i) {
        acc += static_cast<double>(row[i]) * static_cast<double>(x[static_cast<size_t>(i)]);
      }
      if (apply_relu && acc < 0.0) {
        acc = 0.0;
      }
      out[static_cast<size_t>(o)] = static_cast<float>(acc);
    }
    return out;
  };

  std::vector<float> h0 = linear(layer0, input, /*apply_relu=*/true);
  std::vector<float> h1 = linear(layer1, h0, /*apply_relu=*/true);
  return linear(layer2, h1, /*apply_relu=*/false);
}

PolicyWeights PolicyWeights::load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("PolicyWeights::load: cannot open weight file: " + path);
  }

  char magic[4];
  read_exact(in, magic, sizeof(magic), "magic");
  if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
    throw std::runtime_error("PolicyWeights::load: bad magic in " + path + " - not a BEPP weight file");
  }

  uint32_t version = read_u32(in, "version");
  if (version != kSupportedVersion) {
    throw std::runtime_error("PolicyWeights::load: unsupported version " + std::to_string(version) +
                              " in " + path + " (this loader supports version " +
                              std::to_string(kSupportedVersion) + " only)");
  }

  uint32_t vector_len = read_u32(in, "vector_len");

  PolicyWeights weights;
  weights.actor.layer0 = read_layer(in, "actor.net[0]");
  weights.actor.layer1 = read_layer(in, "actor.net[2]");
  weights.actor.layer2 = read_layer(in, "actor.action_net");
  weights.critic.layer0 = read_layer(in, "critic.net[0]");
  weights.critic.layer1 = read_layer(in, "critic.net[2]");
  weights.critic.layer2 = read_layer(in, "critic.value_net");

  check_chain(weights.actor.layer0, weights.actor.layer1, "actor.net[0]->net[2]");
  check_chain(weights.actor.layer1, weights.actor.layer2, "actor.net[2]->action_net");
  check_chain(weights.critic.layer0, weights.critic.layer1, "critic.net[0]->net[2]");
  check_chain(weights.critic.layer1, weights.critic.layer2, "critic.net[2]->value_net");

  check_input_width(weights.actor.layer0, vector_len, "actor");
  check_input_width(weights.critic.layer0, vector_len, "critic");

  return weights;
}

}  // namespace be
