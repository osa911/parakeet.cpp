// Warm transcriber daemon over a Unix domain socket.
//
// Usage:
//   example-server <socket-path> <model.safetensors> <vocab.txt> [options]
//
// Startup options:
//   --model TYPE   tdt-ctc-110m (default) or tdt-600m
//   --gpu          Move the loaded model to Metal GPU
//   --fp16         Cast model to fp16 before --gpu
//   --vad PATH     Load Silero VAD weights once at startup

#include <parakeet/parakeet.hpp>

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace {

volatile std::sig_atomic_t g_should_stop = 0;

void handle_signal(int) { g_should_stop = 1; }

std::string json_escape(std::string_view input) {
    std::string out;
    out.reserve(input.size() + 16);
    for (char ch : input) {
        switch (ch) {
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\b':
            out += "\\b";
            break;
        case '\f':
            out += "\\f";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            out += ch;
            break;
        }
    }
    return out;
}

std::optional<size_t> find_key(const std::string &json,
                               const std::string &key) {
    const auto token = "\"" + key + "\"";
    const auto pos = json.find(token);
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    const auto colon = json.find(':', pos + token.size());
    if (colon == std::string::npos) {
        return std::nullopt;
    }
    return colon + 1;
}

void skip_ws(const std::string &json, size_t &pos) {
    while (pos < json.size() &&
           std::isspace(static_cast<unsigned char>(json[pos]))) {
        ++pos;
    }
}

std::optional<std::string> parse_json_string_at(const std::string &json,
                                                size_t pos) {
    skip_ws(json, pos);
    if (pos >= json.size() || json[pos] != '"') {
        return std::nullopt;
    }
    ++pos;
    std::string out;
    while (pos < json.size()) {
        char ch = json[pos++];
        if (ch == '"') {
            return out;
        }
        if (ch == '\\' && pos < json.size()) {
            char esc = json[pos++];
            switch (esc) {
            case '"':
            case '\\':
            case '/':
                out += esc;
                break;
            case 'b':
                out += '\b';
                break;
            case 'f':
                out += '\f';
                break;
            case 'n':
                out += '\n';
                break;
            case 'r':
                out += '\r';
                break;
            case 't':
                out += '\t';
                break;
            default:
                return std::nullopt;
            }
        } else {
            out += ch;
        }
    }
    return std::nullopt;
}

std::optional<std::string> get_string_field(const std::string &json,
                                            const std::string &key) {
    auto pos = find_key(json, key);
    if (!pos) {
        return std::nullopt;
    }
    return parse_json_string_at(json, *pos);
}

std::optional<bool> get_bool_field(const std::string &json,
                                   const std::string &key) {
    auto pos = find_key(json, key);
    if (!pos) {
        return std::nullopt;
    }
    skip_ws(json, *pos);
    if (json.compare(*pos, 4, "true") == 0) {
        return true;
    }
    if (json.compare(*pos, 5, "false") == 0) {
        return false;
    }
    return std::nullopt;
}

std::optional<double> get_number_field(const std::string &json,
                                       const std::string &key) {
    auto pos = find_key(json, key);
    if (!pos) {
        return std::nullopt;
    }
    skip_ws(json, *pos);
    size_t end = *pos;
    while (end < json.size() &&
           std::string_view("+-0123456789.eE").find(json[end]) !=
               std::string::npos) {
        ++end;
    }
    if (end == *pos) {
        return std::nullopt;
    }
    return std::strtod(json.c_str() + *pos, nullptr);
}

std::optional<std::vector<std::string>>
get_string_array_field(const std::string &json, const std::string &key) {
    auto pos = find_key(json, key);
    if (!pos) {
        return std::nullopt;
    }
    skip_ws(json, *pos);
    if (*pos >= json.size() || json[*pos] != '[') {
        return std::nullopt;
    }
    ++(*pos);
    std::vector<std::string> out;
    while (*pos < json.size()) {
        skip_ws(json, *pos);
        if (*pos < json.size() && json[*pos] == ']') {
            ++(*pos);
            return out;
        }
        auto value = parse_json_string_at(json, *pos);
        if (!value) {
            return std::nullopt;
        }
        out.push_back(*value);
        skip_ws(json, *pos);
        if (*pos < json.size() && json[*pos] == ',') {
            ++(*pos);
            continue;
        }
        if (*pos < json.size() && json[*pos] == ']') {
            ++(*pos);
            return out;
        }
        return std::nullopt;
    }
    return std::nullopt;
}

struct Request {
    std::string request_id;
    std::string audio_path;
    std::string decoder = "tdt";
    bool timestamps = false;
    bool use_vad = false;
    int beam_width = 8;
    std::string lm_path;
    double lm_weight = 0.5;
    double boost_score = 5.0;
    std::vector<std::string> boost_phrases;
};

std::optional<Request> parse_request(const std::string &line,
                                     std::string &error) {
    Request req;
    if (auto value = get_string_field(line, "request_id")) {
        req.request_id = *value;
    }
    if (auto value = get_string_field(line, "audio_path")) {
        req.audio_path = *value;
    } else {
        error = "audio_path is required";
        return std::nullopt;
    }
    if (auto value = get_string_field(line, "decoder")) {
        req.decoder = *value;
    }
    if (auto value = get_bool_field(line, "timestamps")) {
        req.timestamps = *value;
    }
    if (auto value = get_bool_field(line, "use_vad")) {
        req.use_vad = *value;
    }
    if (auto value = get_number_field(line, "beam_width")) {
        req.beam_width = static_cast<int>(*value);
    }
    if (auto value = get_string_field(line, "lm_path")) {
        req.lm_path = *value;
    }
    if (auto value = get_number_field(line, "lm_weight")) {
        req.lm_weight = *value;
    }
    if (auto value = get_number_field(line, "boost_score")) {
        req.boost_score = *value;
    }
    if (auto value = get_string_array_field(line, "boost_phrases")) {
        req.boost_phrases = *value;
    }
    return req;
}

std::string make_error_response(const std::string &request_id,
                                const std::string &message) {
    std::ostringstream oss;
    oss << "{\"ok\":false";
    if (!request_id.empty()) {
        oss << ",\"request_id\":\"" << json_escape(request_id) << "\"";
    }
    oss << ",\"error\":\"" << json_escape(message) << "\"}\n";
    return oss.str();
}

parakeet::Decoder parse_decoder(const std::string &decoder_name) {
    if (decoder_name == "ctc") {
        return parakeet::Decoder::CTC;
    }
    if (decoder_name == "ctc-beam") {
        return parakeet::Decoder::CTC_BEAM;
    }
    if (decoder_name == "tdt-beam") {
        return parakeet::Decoder::TDT_BEAM;
    }
    return parakeet::Decoder::TDT;
}

struct ServerConfig {
    std::string socket_path;
    std::string weights_path;
    std::string vocab_path;
    std::string model_type = "tdt-ctc-110m";
    bool use_gpu = false;
    bool use_fp16 = false;
    std::string vad_path;
};

void print_usage(const char *prog) {
    std::cerr << "Usage: " << prog
              << " <socket-path> <model.safetensors> <vocab.txt> [options]\n"
              << "\nOptions:\n"
              << "  --model TYPE   tdt-ctc-110m (default) or tdt-600m\n"
              << "  --gpu          Move the loaded model to Metal GPU\n"
              << "  --fp16         Cast model to fp16 before --gpu\n"
              << "  --vad PATH     Load Silero VAD weights once at startup\n";
}

std::optional<ServerConfig> parse_server_args(int argc, char *argv[]) {
    if (argc < 4) {
        print_usage(argv[0]);
        return std::nullopt;
    }

    ServerConfig config;
    config.socket_path = argv[1];
    config.weights_path = argv[2];
    config.vocab_path = argv[3];

    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            config.model_type = argv[++i];
        } else if (arg == "--gpu") {
            config.use_gpu = true;
        } else if (arg == "--fp16") {
            config.use_fp16 = true;
        } else if (arg == "--vad" && i + 1 < argc) {
            config.vad_path = argv[++i];
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return std::nullopt;
        }
    }

    if (config.model_type != "tdt-ctc-110m" &&
        config.model_type != "tdt-600m") {
        std::cerr << "Unsupported model type: " << config.model_type << "\n";
        return std::nullopt;
    }

    return config;
}

template <typename T>
void configure_transcriber(T &transcriber, const ServerConfig &config) {
    if (config.use_fp16) {
        transcriber.to_half();
    }
    if (config.use_gpu) {
        transcriber.to_gpu();
    }
    if (!config.vad_path.empty()) {
        transcriber.enable_vad(config.vad_path);
    }
}

parakeet::TranscribeOptions make_options(const Request &request) {
    parakeet::TranscribeOptions options;
    options.decoder = parse_decoder(request.decoder);
    options.timestamps = request.timestamps;
    options.use_vad = request.use_vad;
    options.beam_width = request.beam_width;
    options.lm_path = request.lm_path;
    options.lm_weight = static_cast<float>(request.lm_weight);
    options.boost_phrases = request.boost_phrases;
    options.boost_score = static_cast<float>(request.boost_score);
    return options;
}

std::string make_success_response(const Request &request,
                                  const parakeet::TranscribeResult &result,
                                  long long elapsed_ms) {
    std::ostringstream oss;
    oss << "{\"ok\":true";
    if (!request.request_id.empty()) {
        oss << ",\"request_id\":\"" << json_escape(request.request_id) << "\"";
    }
    oss << ",\"text\":\"" << json_escape(result.text) << "\"";
    oss << ",\"elapsed_ms\":" << elapsed_ms;
    if (request.timestamps) {
        oss << ",\"word_timestamps\":[";
        for (size_t i = 0; i < result.word_timestamps.size(); ++i) {
            const auto &word = result.word_timestamps[i];
            if (i > 0) {
                oss << ",";
            }
            oss << "{\"word\":\"" << json_escape(word.word) << "\""
                << ",\"start\":" << std::fixed << std::setprecision(2)
                << word.start << ",\"end\":" << std::fixed
                << std::setprecision(2) << word.end
                << ",\"confidence\":" << std::fixed << std::setprecision(3)
                << word.confidence << "}";
        }
        oss << "]";
    }
    oss << "}\n";
    return oss.str();
}

struct ServerTranscriber {
    virtual ~ServerTranscriber() = default;
    virtual parakeet::TranscribeResult transcribe(const Request &request) = 0;
};

struct WarmTranscriber final : ServerTranscriber {
    explicit WarmTranscriber(const ServerConfig &config)
        : transcriber(config.weights_path, config.vocab_path) {
        configure_transcriber(transcriber, config);
    }

    parakeet::TranscribeResult transcribe(const Request &request) override {
        return transcriber.transcribe(request.audio_path,
                                      make_options(request));
    }

    parakeet::Transcriber transcriber;
};

struct WarmTDT600Transcriber final : ServerTranscriber {
    explicit WarmTDT600Transcriber(const ServerConfig &config)
        : model_config(parakeet::make_tdt_600m_config()), model(model_config) {
        auto weights = axiom::io::safetensors::load(config.weights_path);
        model.load_state_dict(weights, "", false);
        tokenizer.load(config.vocab_path);

        if (config.use_fp16) {
            model.to(axiom::DType::Float16);
            use_fp16 = true;
        }
        if (config.use_gpu) {
            model.to(axiom::Device::GPU);
            use_gpu = true;
        }
        if (!config.vad_path.empty()) {
            vad = std::make_unique<parakeet::audio::SileroVAD>(config.vad_path);
            if (use_fp16) {
                vad->to_half();
            }
            if (use_gpu) {
                vad->to_gpu();
            }
        }
    }

    parakeet::TranscribeResult transcribe(const Request &request) override {
        if (request.decoder == "ctc" || request.decoder == "ctc-beam") {
            throw std::runtime_error("tdt-600m server only supports "
                                     "decoder=\"tdt\" or \"tdt-beam\"");
        }

        auto audio_data = parakeet::read_audio(request.audio_path);
        auto options = make_options(request);

        axiom::Tensor audio_for_asr = audio_data.samples;
        std::unique_ptr<parakeet::audio::TimestampRemapper> remapper;
        if (options.use_vad && vad) {
            auto segments = vad->detect(audio_data.samples);
            if (!segments.empty()) {
                audio_for_asr = parakeet::audio::collect_speech(
                    audio_data.samples, segments);
                if (options.timestamps) {
                    remapper =
                        std::make_unique<parakeet::audio::TimestampRemapper>(
                            segments);
                }
            }
        }

        parakeet::AudioConfig audio_cfg;
        audio_cfg.n_mels = model_config.encoder.mel_bins;
        auto features = parakeet::preprocess_audio(audio_for_asr, audio_cfg);
        if (use_fp16) {
            features = features.half();
        }
        if (use_gpu) {
            features = features.gpu();
        }

        auto encoder_out = model.encoder()(features);

        parakeet::ContextTrie trie;
        const bool use_boost =
            !options.boost_phrases.empty() && tokenizer.loaded();
        if (use_boost) {
            trie.build(options.boost_phrases, tokenizer);
        }

        parakeet::ArpaLM lm;
        if (options.decoder == parakeet::Decoder::TDT_BEAM &&
            !options.lm_path.empty()) {
            lm.load(options.lm_path);
        }

        const int blank_id = model_config.prediction.vocab_size - 1;
        parakeet::TranscribeResult result;

        if (options.timestamps) {
            std::vector<std::vector<parakeet::TimestampedToken>> all_tokens;
            if (options.decoder == parakeet::Decoder::TDT_BEAM) {
                parakeet::TDTBeamSearchOptions beam_options;
                beam_options.beam_width = options.beam_width;
                beam_options.blank_id = blank_id;
                beam_options.lm = lm.loaded() ? &lm : nullptr;
                beam_options.lm_weight = options.lm_weight;
                beam_options.pieces =
                    tokenizer.loaded() ? &tokenizer.pieces() : nullptr;
                all_tokens = parakeet::tdt_beam_decode_with_timestamps(
                    model, encoder_out, model_config.durations, beam_options);
            } else {
                all_tokens =
                    use_boost
                        ? parakeet::tdt_greedy_decode_with_timestamps_boosted(
                              model, encoder_out, model_config.durations, trie,
                              options.boost_score, blank_id)
                        : parakeet::tdt_greedy_decode_with_timestamps(
                              model, encoder_out, model_config.durations,
                              blank_id);
            }

            if (!all_tokens.empty()) {
                result.timestamped_tokens = all_tokens[0];
                if (remapper) {
                    result.timestamped_tokens =
                        remapper->remap_tokens(result.timestamped_tokens);
                }
                for (const auto &token : result.timestamped_tokens) {
                    result.token_ids.push_back(token.token_id);
                }
                result.text = tokenizer.decode(result.token_ids);
                result.word_timestamps = parakeet::group_timestamps(
                    result.timestamped_tokens, tokenizer.pieces());
            }
            return result;
        }

        std::vector<std::vector<int>> all_tokens;
        if (options.decoder == parakeet::Decoder::TDT_BEAM) {
            parakeet::TDTBeamSearchOptions beam_options;
            beam_options.beam_width = options.beam_width;
            beam_options.blank_id = blank_id;
            beam_options.lm = lm.loaded() ? &lm : nullptr;
            beam_options.lm_weight = options.lm_weight;
            beam_options.pieces =
                tokenizer.loaded() ? &tokenizer.pieces() : nullptr;
            all_tokens = parakeet::tdt_beam_decode(
                model, encoder_out, model_config.durations, beam_options);
        } else {
            all_tokens =
                use_boost ? parakeet::tdt_greedy_decode_boosted(
                                model, encoder_out, model_config.durations,
                                trie, options.boost_score, blank_id)
                          : parakeet::tdt_greedy_decode(model, encoder_out,
                                                        model_config.durations,
                                                        blank_id);
        }

        if (!all_tokens.empty()) {
            result.token_ids = all_tokens[0];
            result.text = tokenizer.decode(result.token_ids);
        }
        return result;
    }

    parakeet::TDTConfig model_config;
    parakeet::ParakeetTDT model;
    parakeet::Tokenizer tokenizer;
    bool use_gpu = false;
    bool use_fp16 = false;
    std::unique_ptr<parakeet::audio::SileroVAD> vad;
};

std::unique_ptr<ServerTranscriber>
make_transcriber(const ServerConfig &config) {
    if (config.model_type == "tdt-600m") {
        return std::make_unique<WarmTDT600Transcriber>(config);
    }
    return std::make_unique<WarmTranscriber>(config);
}

parakeet::TranscribeResult run_request(ServerTranscriber &transcriber,
                                       const Request &request) {
    return transcriber.transcribe(request);
}

int bind_socket(const std::string &socket_path) {
    if (socket_path.size() >= sizeof(sockaddr_un::sun_path)) {
        throw std::runtime_error("socket path is too long for sockaddr_un");
    }
    std::filesystem::remove(socket_path);

    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        throw std::runtime_error(std::string("socket() failed: ") +
                                 std::strerror(errno));
    }

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

    if (::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
        const auto message =
            std::string("bind() failed: ") + std::strerror(errno);
        ::close(fd);
        throw std::runtime_error(message);
    }

    if (::listen(fd, 16) < 0) {
        const auto message =
            std::string("listen() failed: ") + std::strerror(errno);
        ::close(fd);
        throw std::runtime_error(message);
    }

    return fd;
}

void write_all(int fd, const std::string &data) {
    size_t written = 0;
    while (written < data.size()) {
        const auto rv =
            ::write(fd, data.data() + written, data.size() - written);
        if (rv <= 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error(std::string("write() failed: ") +
                                     std::strerror(errno));
        }
        written += static_cast<size_t>(rv);
    }
}

void handle_client(int client_fd, ServerTranscriber &transcriber) {
    std::string buffer;
    char chunk[4096];
    while (!g_should_stop) {
        const auto rv = ::read(client_fd, chunk, sizeof(chunk));
        if (rv == 0) {
            return;
        }
        if (rv < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error(std::string("read() failed: ") +
                                     std::strerror(errno));
        }

        buffer.append(chunk, static_cast<size_t>(rv));
        size_t newline = std::string::npos;
        while ((newline = buffer.find('\n')) != std::string::npos) {
            std::string line = buffer.substr(0, newline);
            buffer.erase(0, newline + 1);
            if (line.empty()) {
                continue;
            }

            std::string error;
            auto request = parse_request(line, error);
            if (!request) {
                write_all(client_fd, make_error_response("", error));
                continue;
            }

            try {
                const auto started = std::chrono::steady_clock::now();
                const auto result = run_request(transcriber, *request);
                const auto elapsed_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - started)
                        .count();
                std::cerr << "request"
                          << (request->request_id.empty()
                                  ? ""
                                  : " id=" + request->request_id)
                          << " audio=" << request->audio_path
                          << " elapsed_ms=" << elapsed_ms << "\n";
                write_all(client_fd,
                          make_success_response(*request, result, elapsed_ms));
            } catch (const std::exception &ex) {
                std::cerr << "request failed"
                          << (request->request_id.empty()
                                  ? ""
                                  : " id=" + request->request_id)
                          << ": " << ex.what() << "\n";
                write_all(client_fd,
                          make_error_response(request->request_id, ex.what()));
            }
        }
    }
}

} // namespace

int main(int argc, char *argv[]) {
    auto config = parse_server_args(argc, argv);
    if (!config) {
        return 1;
    }

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);
    std::signal(SIGPIPE, SIG_IGN);

    try {
        auto transcriber = make_transcriber(*config);
        int server_fd = bind_socket(config->socket_path);

        std::cerr << "example-server listening on " << config->socket_path
                  << " model=" << config->model_type
                  << " gpu=" << (config->use_gpu ? "true" : "false")
                  << " fp16=" << (config->use_fp16 ? "true" : "false")
                  << " vad=" << (config->vad_path.empty() ? "false" : "true")
                  << "\n";

        while (!g_should_stop) {
            int client_fd = ::accept(server_fd, nullptr, nullptr);
            if (client_fd < 0) {
                if (errno == EINTR && g_should_stop) {
                    break;
                }
                if (errno == EINTR) {
                    continue;
                }
                throw std::runtime_error(std::string("accept() failed: ") +
                                         std::strerror(errno));
            }

            try {
                handle_client(client_fd, *transcriber);
            } catch (const std::exception &ex) {
                std::cerr << "client error: " << ex.what() << "\n";
            }
            ::close(client_fd);
        }

        ::close(server_fd);
        std::filesystem::remove(config->socket_path);
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "fatal: " << ex.what() << "\n";
        return 1;
    }
}
