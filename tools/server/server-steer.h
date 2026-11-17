#pragma once

// Per-request activation steering ("steer" field of a completion request).
//
// A request carries a list of pushes.  Each push adds one constant vector to the feed-forward output of one layer,
// at every token, for that request only:
//
//   {"layer": 32, "expert": 480, "neuron": 617, "scale": -340.0}   routed-expert neuron (MoE models)
//   {"layer": 44, "neuron": 136, "scale": 20.0}                    shared-expert neuron, or a neuron of a dense FFN
//   {"layer": 14, "vector": [ ...n_embd floats... ], "scale": 1.0}  raw direction
//
// For a neuron push the direction is that neuron's column of the layer's FFN down-projection, normalised to unit
// length, so "scale" is the (signed) norm of the vector that gets added.  A raw vector is multiplied by "scale" as is.
// Pushes on the same layer add up.  A request without "steer" (or with an empty list) runs unsteered.
//
// "steer_from": N (optional) leaves the first N prompt tokens unsteered, e.g. the system prompt and tool definitions:
// they are read by the plain model (and stay cached across requests with different steering); everything after,
// including the reply, is steered.
//
// The vector is installed with llama_set_adapter_cvec(), so it only has an effect on architectures whose graph calls
// build_cvec().  It belongs to the whole context: with more than one slot, concurrent requests share it.

#include "ggml.h"
#include "gguf.h"

#include "server-common.h"

#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <tuple>
#include <vector>

struct server_steer_push {
    int32_t layer  = -1;
    int32_t expert = -1;        // -1: shared expert / dense FFN
    int32_t neuron = -1;        // -1: raw vector
    float   scale  = 1.0f;
    std::vector<float> vector;  // raw direction (when neuron < 0)
};

struct server_steer {
    std::vector<server_steer_push> pushes;
    std::string key;            // canonical text of the request field; equal keys mean equal steering
    int32_t from = 0;           // "steer_from": prompt tokens before this index are processed unsteered
};

// throws std::runtime_error on a malformed field
server_steer parse_steer_request(const json & data);

// Reads neuron output directions straight from the model file(s), so it does not depend on where or in what packed
// form the backend holds the weights.
struct server_steer_source {
    ~server_steer_source();

    // builds the buffer llama_set_adapter_cvec() expects: n_embd floats for each of layers 1 .. n_layer-1
    bool build(const server_steer & steer, const std::string & model_path, int32_t n_layer, int32_t n_embd,
               std::vector<float> & buf, std::string & err);

private:
    struct file {
        std::string    path;
        FILE *         fp   = nullptr;
        gguf_context * gguf = nullptr;
        ggml_context * meta = nullptr;
    };

    bool open(const std::string & model_path, std::string & err);
    void close();
    bool neuron_dir(int32_t layer, int32_t expert, int32_t neuron, int32_t n_embd, const std::vector<float> ** out, std::string & err);

    bool opened = false;
    std::vector<file> files;
    std::map<std::tuple<int32_t, int32_t, int32_t>, std::vector<float>> cache; // unit directions
};
