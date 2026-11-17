#include "server-steer.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

server_steer parse_steer_request(const json & data) {
    server_steer res;
    if (data.is_null()) {
        return res;
    }
    if (!data.is_array()) {
        throw std::runtime_error("Error: 'steer' must be an array of objects with 'layer', 'scale' and either 'neuron' (optionally 'expert') or 'vector'");
    }
    for (const auto & entry : data) {
        if (!entry.is_object() || !entry.contains("layer")) {
            throw std::runtime_error("Error: each 'steer' entry must be an object with a 'layer'");
        }
        server_steer_push push;
        push.layer = entry.at("layer").get<int32_t>();
        push.scale = entry.contains("scale") ? entry.at("scale").get<float>() : 1.0f;
        if (!std::isfinite(push.scale)) {
            throw std::runtime_error("Error: 'steer' scale must be finite");
        }
        if (entry.contains("vector")) {
            push.vector = entry.at("vector").get<std::vector<float>>();
            for (float v : push.vector) {
                if (!std::isfinite(v)) {
                    throw std::runtime_error("Error: 'steer' vector must be finite");
                }
            }
        } else if (entry.contains("neuron")) {
            push.neuron = entry.at("neuron").get<int32_t>();
            if (entry.contains("expert") && !entry.at("expert").is_null()) {
                push.expert = entry.at("expert").get<int32_t>();
            }
            if (push.neuron < 0 || push.expert < -1) {
                throw std::runtime_error("Error: 'steer' neuron and expert must not be negative");
            }
        } else {
            throw std::runtime_error("Error: each 'steer' entry needs a 'neuron' or a 'vector'");
        }
        if (push.scale != 0.0f) {
            res.pushes.push_back(std::move(push));
        }
    }
    if (!res.pushes.empty()) {
        res.key = data.dump();
    }
    return res;
}

server_steer_source::~server_steer_source() {
    close();
}

void server_steer_source::close() {
    for (auto & f : files) {
        if (f.fp)   { fclose(f.fp); }
        if (f.gguf) { gguf_free(f.gguf); }
        if (f.meta) { ggml_free(f.meta); }
    }
    files.clear();
    opened = false;
}

bool server_steer_source::open(const std::string & model_path, std::string & err) {
    if (opened) {
        return true;
    }

    auto open_one = [&](const std::string & path) {
        file f;
        f.path = path;
        gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ &f.meta };
        f.gguf = gguf_init_from_file(path.c_str(), params);
        if (!f.gguf) {
            err = "steer: cannot read model file " + path;
            return false;
        }
        f.fp = fopen(path.c_str(), "rb");
        files.push_back(f); // close() releases it
        if (!f.fp) {
            err = "steer: cannot open model file " + path;
            return false;
        }
        return true;
    };

    if (!open_one(model_path)) {
        close();
        return false;
    }

    // the other parts of a split model: <prefix>-00001-of-00002.gguf
    const int64_t kid = gguf_find_key(files[0].gguf, "split.count");
    const int n_split = kid >= 0 ? gguf_get_val_u16(files[0].gguf, kid) : 1;
    if (n_split > 1) {
        char suffix[32];
        std::string prefix;
        int first = -1;
        for (int i = 1; i <= n_split && first < 0; i++) {
            snprintf(suffix, sizeof(suffix), "-%05d-of-%05d.gguf", i, n_split);
            const size_t n = strlen(suffix);
            if (model_path.size() > n && model_path.compare(model_path.size() - n, n, suffix) == 0) {
                prefix = model_path.substr(0, model_path.size() - n);
                first  = i;
            }
        }
        if (first < 0) {
            err = "steer: cannot work out the split file names of " + model_path;
            close();
            return false;
        }
        for (int i = 1; i <= n_split; i++) {
            if (i == first) {
                continue;
            }
            snprintf(suffix, sizeof(suffix), "-%05d-of-%05d.gguf", i, n_split);
            if (!open_one(prefix + suffix)) {
                close();
                return false;
            }
        }
    }
    opened = true;
    return true;
}

bool server_steer_source::neuron_dir(int32_t layer, int32_t expert, int32_t neuron, int32_t n_embd,
                                     const std::vector<float> ** out, std::string & err) {
    const auto id = std::make_tuple(layer, expert, neuron);
    auto it = cache.find(id);
    if (it != cache.end()) {
        *out = &it->second;
        return true;
    }

    const std::string blk = "blk." + std::to_string(layer) + ".";
    const std::vector<std::string> names = expert >= 0
        ? std::vector<std::string>{ blk + "ffn_down_exps.weight" }
        : std::vector<std::string>{ blk + "ffn_down_shexp.weight", blk + "ffn_down.weight" };

    for (auto & f : files) {
        for (const auto & name : names) {
            const int64_t tid = gguf_find_tensor(f.gguf, name.c_str());
            if (tid < 0) {
                continue;
            }
            const ggml_tensor * t = ggml_get_tensor(f.meta, name.c_str());
            // [n_ff, n_embd] or [n_ff, n_embd, n_expert]: each row holds one output dimension's weight for every neuron
            const int64_t n_ff = t->ne[0];
            const int64_t n_ex = t->ne[2];
            if (t->ne[1] != n_embd) {
                err = "steer: " + name + " has an unexpected shape";
                return false;
            }
            if (neuron >= n_ff) {
                err = "steer: layer " + std::to_string(layer) + " has only " + std::to_string(n_ff) + " neurons per FFN";
                return false;
            }
            if (expert >= n_ex || (expert < 0 && n_ex != 1)) {
                err = "steer: layer " + std::to_string(layer) + " has " + std::to_string(n_ex) + " experts";
                return false;
            }
            const enum ggml_type type = gguf_get_tensor_type(f.gguf, tid);
            const auto * traits = ggml_get_type_traits(type);
            if (type != GGML_TYPE_F32 && !traits->to_float) {
                err = "steer: cannot decode tensor type " + std::string(ggml_type_name(type));
                return false;
            }
            const size_t row_size = ggml_row_size(type, n_ff);
            const size_t offset   = gguf_get_data_offset(f.gguf) + gguf_get_tensor_offset(f.gguf, tid)
                                  + (size_t) (expert < 0 ? 0 : expert) * n_embd * row_size;

            std::vector<uint8_t> raw(row_size * n_embd);
#ifdef _WIN32
            if (_fseeki64(f.fp, (long long) offset, SEEK_SET) != 0 || fread(raw.data(), 1, raw.size(), f.fp) != raw.size()) {
#else
            if (fseeko(f.fp, (off_t) offset, SEEK_SET) != 0 || fread(raw.data(), 1, raw.size(), f.fp) != raw.size()) {
#endif
                err = "steer: read error in " + f.path;
                return false;
            }

            std::vector<float> row(n_ff);
            std::vector<float> dir(n_embd);
            double sq = 0.0;
            for (int32_t j = 0; j < n_embd; j++) {
                const uint8_t * src = raw.data() + (size_t) j * row_size;
                if (type == GGML_TYPE_F32) {
                    memcpy(row.data(), src, n_ff * sizeof(float));
                } else {
                    traits->to_float(src, row.data(), n_ff);
                }
                dir[j] = row[neuron];
                sq += (double) dir[j] * dir[j];
            }
            if (!(sq > 0.0) || !std::isfinite(sq)) {
                err = "steer: that neuron has no output direction";
                return false;
            }
            const float inv = (float) (1.0 / std::sqrt(sq));
            for (auto & v : dir) {
                v *= inv;
            }
            if (cache.size() > 4096) {
                cache.clear();
            }
            *out = &(cache[id] = std::move(dir));
            return true;
        }
    }

    err = "steer: the model has no " + names[0] + (expert >= 0 ? "" : " or " + names[1]);
    return false;
}

bool server_steer_source::build(const server_steer & steer, const std::string & model_path, int32_t n_layer, int32_t n_embd,
                                std::vector<float> & buf, std::string & err) {
    buf.assign((size_t) (n_layer - 1) * n_embd, 0.0f);

    for (const auto & push : steer.pushes) {
        // the engine keeps no vector for layer 0
        if (push.layer < 1 || push.layer >= n_layer) {
            err = "steer: layer must be between 1 and " + std::to_string(n_layer - 1);
            return false;
        }
        float * dst = buf.data() + (size_t) (push.layer - 1) * n_embd;
        if (push.neuron < 0) {
            if ((int32_t) push.vector.size() != n_embd) {
                err = "steer: a vector must have " + std::to_string(n_embd) + " values";
                return false;
            }
            for (int32_t j = 0; j < n_embd; j++) {
                dst[j] += push.scale * push.vector[j];
            }
        } else {
            if (!open(model_path, err)) {
                if (err.empty()) {
                    err = "steer: model file not available";
                }
                return false;
            }
            const std::vector<float> * dir = nullptr;
            if (!neuron_dir(push.layer, push.expert, push.neuron, n_embd, &dir, err)) {
                return false;
            }
            for (int32_t j = 0; j < n_embd; j++) {
                dst[j] += push.scale * (*dir)[j];
            }
        }
    }
    return true;
}
