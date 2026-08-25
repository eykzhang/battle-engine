// M3: MlpWeights::forward + PolicyWeights::load. See mlp.hpp for the
// design rationale (fixed 3-layer struct, double-precision accumulation,
// exception-based load-time error handling).
#include <catch2/benchmark/catch_benchmark_all.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include "be/mlp.hpp"

using namespace be;
using Catch::Approx;

namespace {

// RAII scratch file under the system temp dir - every corrupt-file test
// below writes its own bytes here and PolicyWeights::load() reads them
// back, so the load-time validation is exercised against a REAL file on
// disk (matching how a real caller hits it), not an in-memory stream.
// Removed on destruction so a failed test run doesn't leave stray files
// behind in the temp dir across repeated `ctest` invocations.
class TempFile {
 public:
  explicit TempFile(const std::string& name)
      : path_((std::filesystem::temp_directory_path() / name).string()) {}
  ~TempFile() { std::filesystem::remove(path_); }

  const std::string& path() const { return path_; }

  void write(const std::vector<uint8_t>& bytes) const {
    std::ofstream out(path_, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  }

 private:
  std::string path_;
};

void push_u32(std::vector<uint8_t>& buf, uint32_t v) {
  for (int i = 0; i < 4; ++i) buf.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}

void push_f32(std::vector<uint8_t>& buf, float v) {
  uint32_t bits;
  static_assert(sizeof(bits) == sizeof(v));
  std::memcpy(&bits, &v, sizeof(v));
  push_u32(buf, bits);
}

void push_layer(std::vector<uint8_t>& buf, uint32_t out_dim, uint32_t in_dim,
                 const std::vector<float>& weight, const std::vector<float>& bias) {
  push_u32(buf, out_dim);
  push_u32(buf, in_dim);
  for (float w : weight) push_f32(buf, w);
  for (float b : bias) push_f32(buf, b);
}

// A minimal, internally-consistent 2->2->2->1 "ppo.bin"-shaped file
// (vector_len=2, identical tiny shape for both branches) - big enough to
// exercise every field the real 665->128->64->{13,1} file has, small
// enough to hand-construct and reason about byte-for-byte in a test.
std::vector<uint8_t> valid_minimal_file() {
  std::vector<uint8_t> buf;
  buf.insert(buf.end(), {'B', 'E', 'P', 'P'});
  push_u32(buf, /*version=*/1);
  push_u32(buf, /*vector_len=*/2);
  for (int branch = 0; branch < 2; ++branch) {
    push_layer(buf, 2, 2, {1, 0, 0, 1}, {0, 0});
    push_layer(buf, 2, 2, {1, 0, 0, 1}, {0, 0});
    push_layer(buf, 1, 2, {1, 1}, {0});
  }
  return buf;
}

}  // namespace

// ---------------------------------------------------------------------------
// MlpWeights::forward - known input, hand-computed output
// ---------------------------------------------------------------------------

TEST_CASE("MlpWeights::forward: matches a hand-computed 3-layer output", "[mlp]") {
  MlpWeights w;
  // layer0 (2->2): row0=[1,2], row1=[-1,1], bias=[0,1]
  w.layer0 = {2, 2, {1, 2, -1, 1}, {0, 1}};
  // layer1 (2->2): row0=[2,-1], row1=[0,1], bias=[-1,0]
  w.layer1 = {2, 2, {2, -1, 0, 1}, {-1, 0}};
  // layer2 (2->1): row0=[1,1], bias=[0.5]
  w.layer2 = {2, 1, {1, 1}, {0.5f}};

  // input=[1,1]:
  //   layer0 pre = [1*1+2*1+0, -1*1+1*1+1] = [3, 1] -> relu -> [3, 1]
  //   layer1 pre = [2*3-1*1-1, 0*3+1*1+0] = [4, 1] -> relu -> [4, 1]
  //   layer2 pre = [1*4+1*1+0.5] = [5.5] -> no relu -> [5.5]
  auto out = w.forward({1.0f, 1.0f});
  REQUIRE(out.size() == 1);
  REQUIRE(out[0] == Approx(5.5));
}

TEST_CASE("MlpWeights::forward: ReLU clamps a negative pre-activation to zero, not through to the output",
          "[mlp]") {
  MlpWeights w;
  w.layer0 = {2, 2, {1, 2, -1, 1}, {0, 1}};
  w.layer1 = {2, 2, {2, -1, 0, 1}, {-1, 0}};
  w.layer2 = {2, 1, {1, 1}, {0.5f}};

  // input=[-1,-1]:
  //   layer0 pre = [1*-1+2*-1+0, -1*-1+1*-1+1] = [-3, 1] -> relu -> [0, 1]
  //   layer1 pre = [2*0-1*1-1, 0*0+1*1+0] = [-2, 1] -> relu -> [0, 1]
  //   layer2 pre = [1*0+1*1+0.5] = [1.5] -> no relu -> [1.5]
  auto out = w.forward({-1.0f, -1.0f});
  REQUIRE(out.size() == 1);
  REQUIRE(out[0] == Approx(1.5));
}

// ---------------------------------------------------------------------------
// PolicyWeights::load - positive path + every named malformed-file case
// ---------------------------------------------------------------------------

TEST_CASE("PolicyWeights::load: a well-formed minimal file loads with the declared dimensions", "[mlp]") {
  TempFile file("be_test_mlp_valid.bin");
  file.write(valid_minimal_file());

  PolicyWeights pw = PolicyWeights::load(file.path());
  REQUIRE(pw.actor.layer0.in_dim == 2);
  REQUIRE(pw.actor.layer2.out_dim == 1);
  REQUIRE(pw.critic.layer0.in_dim == 2);
  REQUIRE(pw.critic.layer2.out_dim == 1);
}

TEST_CASE("PolicyWeights::load: throws on a truncated weight file rather than reading garbage", "[mlp]") {
  TempFile file("be_test_mlp_truncated.bin");
  std::vector<uint8_t> bytes = valid_minimal_file();
  bytes.resize(bytes.size() - 5);  // cut off mid-way through the last layer's bias
  file.write(bytes);

  REQUIRE_THROWS_AS(PolicyWeights::load(file.path()), std::runtime_error);
}

TEST_CASE("PolicyWeights::load: throws on a truncated header (fewer than 12 bytes total)", "[mlp]") {
  TempFile file("be_test_mlp_truncated_header.bin");
  file.write({'B', 'E', 'P'});  // not even a full magic

  REQUIRE_THROWS_AS(PolicyWeights::load(file.path()), std::runtime_error);
}

TEST_CASE("PolicyWeights::load: throws on a bad magic header", "[mlp]") {
  TempFile file("be_test_mlp_bad_magic.bin");
  std::vector<uint8_t> bytes = valid_minimal_file();
  bytes[0] = 'X';
  file.write(bytes);

  REQUIRE_THROWS_AS(PolicyWeights::load(file.path()), std::runtime_error);
}

TEST_CASE("PolicyWeights::load: throws on an unsupported version", "[mlp]") {
  TempFile file("be_test_mlp_bad_version.bin");
  std::vector<uint8_t> bytes = valid_minimal_file();
  bytes[4] = 2;  // version field's low byte, was 1
  file.write(bytes);

  REQUIRE_THROWS_AS(PolicyWeights::load(file.path()), std::runtime_error);
}

TEST_CASE("PolicyWeights::load: throws when the header's vector_len disagrees with layer0.in_dim",
          "[mlp]") {
  TempFile file("be_test_mlp_bad_vector_len.bin");
  std::vector<uint8_t> bytes = valid_minimal_file();
  bytes[8] = 3;  // vector_len field's low byte, was 2 - now disagrees with every layer0.in_dim
  file.write(bytes);

  REQUIRE_THROWS_AS(PolicyWeights::load(file.path()), std::runtime_error);
}

TEST_CASE("PolicyWeights::load: throws on a dimension-chain mismatch between consecutive layers",
          "[mlp]") {
  std::vector<uint8_t> buf;
  buf.insert(buf.end(), {'B', 'E', 'P', 'P'});
  push_u32(buf, /*version=*/1);
  push_u32(buf, /*vector_len=*/2);
  // actor.net[0]: 2->2 (fine)
  push_layer(buf, 2, 2, {1, 0, 0, 1}, {0, 0});
  // actor.net[2]: declares in_dim=3, but net[0] only produced 2 outputs -
  // the chain mismatch this test targets.
  push_layer(buf, 2, 3, {1, 0, 0, 1, 0, 0}, {0, 0});
  push_layer(buf, 1, 2, {1, 1}, {0});
  // critic branch, internally consistent, so the failure is unambiguously
  // attributable to the actor chain above.
  push_layer(buf, 2, 2, {1, 0, 0, 1}, {0, 0});
  push_layer(buf, 2, 2, {1, 0, 0, 1}, {0, 0});
  push_layer(buf, 1, 2, {1, 1}, {0});

  TempFile file("be_test_mlp_chain_mismatch.bin");
  file.write(buf);

  REQUIRE_THROWS_AS(PolicyWeights::load(file.path()), std::runtime_error);
}

TEST_CASE("PolicyWeights::load: throws when the weight file doesn't exist", "[mlp]") {
  REQUIRE_THROWS_AS(PolicyWeights::load("/nonexistent/path/does_not_exist.bin"), std::runtime_error);
}

// ---------------------------------------------------------------------------
// DW-3.3: microbenchmark against the REAL ppo.bin - hidden by default
// (Catch2's leading-"!" tag convention) so it doesn't add noise/flakiness
// to a normal `ctest` run; run explicitly with `--benchmark-samples` etc.
// via `./be_tests "[mlp][!benchmark]"` to get the real µs number recorded
// in this phase's discovery file / report.
// ---------------------------------------------------------------------------

TEST_CASE("MlpWeights::forward: microbenchmark one actor+critic pass on the real ppo.bin",
          "[mlp][!benchmark]") {
  PolicyWeights pw = PolicyWeights::load(BE_TEST_PPO_WEIGHTS_PATH);
  std::vector<float> input(static_cast<size_t>(pw.actor.layer0.in_dim), 0.1f);

  BENCHMARK("actor forward") { return pw.actor.forward(input); };
  BENCHMARK("critic forward") { return pw.critic.forward(input); };
  BENCHMARK("actor+critic forward (one PUCT node expansion)") {
    auto a = pw.actor.forward(input);
    auto c = pw.critic.forward(input);
    return a.size() + c.size();
  };
}
