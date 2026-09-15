// One-off recorder for the Seam A vision fixtures (GitHub #176).
//
// Links the reference frontend (ninfer `build-ninja` static libraries) and
// writes what its processor prepares for each case: token ids, token types,
// axis-major positions, rope_delta, the prompt identity frontiers, and per
// media item the grid, token spans, content digest and the SHA-256 of the
// packed BF16 patch rows (little-endian u16). It also records each image's
// decoded RGB8 digest, so a mismatch on the ignis side can be attributed to
// decode or to resize/pack. A case the reference refuses records its error.
//
// Usage: record <frontend-resource-dir> <cases.json> <images-dir> <out-dir>
// Build: tools/vision-fixtures/build.ps1

#include "media/decode/decode.h"
#include "targets/qwen3_6/impl/frontend/digest.h"
#include "targets/qwen3_6/impl/frontend/test_access.h"

#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace q  = ninfer::targets::qwen3_6;
namespace fi = ninfer::targets::qwen3_6::frontend_internal;
using Json   = nlohmann::ordered_json;

namespace {

std::string read_text(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot read " + path.string()); }
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::vector<std::uint8_t> read_bytes(const fs::path& path) {
    const std::string text = read_text(path);
    return {text.begin(), text.end()};
}

std::string hex(std::span<const std::uint8_t> bytes) { return fi::sha256_hex(fi::sha256(bytes)); }

ninfer::ChatRole role_of(const std::string& role) {
    if (role == "system") { return ninfer::ChatRole::System; }
    if (role == "user") { return ninfer::ChatRole::User; }
    if (role == "assistant") { return ninfer::ChatRole::Assistant; }
    if (role == "tool") { return ninfer::ChatRole::Tool; }
    throw std::invalid_argument("unknown role " + role);
}

ninfer::MessagePart text_part(std::string text) {
    return ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}};
}

Json record_case(const q::Frontend& frontend, const Json& spec, const fs::path& images) {
    ninfer::PromptInput input;
    input.options.enable_thinking   = spec.value("enable_thinking", false);
    input.options.preserve_thinking = spec.value("preserve_thinking", false);
    Json decoded                    = Json::array();
    for (const Json& message : spec.at("messages")) {
        ninfer::ChatMessage chat;
        chat.role              = role_of(message.at("role").get<std::string>());
        chat.reasoning_content = message.value("reasoning_content", "");
        chat.tool_call_id      = message.value("tool_call_id", "");
        if (message.contains("tool_calls")) {
            for (const Json& call : message.at("tool_calls")) {
                chat.tool_calls.push_back(ninfer::ToolCall{
                    .id             = call.value("id", ""),
                    .name           = call.at("name").get<std::string>(),
                    .arguments_json = call.at("arguments").dump(),
                });
            }
        }
        const Json& content = message.at("content");
        if (content.is_string()) {
            chat.parts.push_back(text_part(content.get<std::string>()));
        } else {
            for (const Json& part : content) {
                const std::string type = part.at("type").get<std::string>();
                if (type == "text") {
                    chat.parts.push_back(text_part(part.at("text").get<std::string>()));
                    continue;
                }
                if (type != "image") { throw std::invalid_argument("unknown part type " + type); }
                const std::string file = part.at("file").get<std::string>();
                ninfer::MessagePart image;
                image.kind              = ninfer::MessagePartKind::Media;
                image.media.kind        = ninfer::MediaKind::Image;
                image.media.bytes       = read_bytes(images / file);
                image.media.source_name = file;
                Json entry{{"file", file}};
                try {
                    const auto rgb      = ninfer::media::decode::decode_image(image.media.bytes, {});
                    entry["width"]      = rgb.width;
                    entry["height"]     = rgb.height;
                    entry["rgb_sha256"] = hex(rgb.rgb);
                } catch (const std::exception& error) {
                    entry["error"] = error.what();
                }
                decoded.push_back(std::move(entry));
                chat.parts.push_back(std::move(image));
            }
        }
        input.messages.push_back(std::move(chat));
    }

    Json out{{"case", spec}, {"decoded", decoded}};
    try {
        const q::PreparedPrompt prompt    = frontend.prepare(std::move(input));
        const q::PreparedPromptData& data = q::FrontendTestAccess::inspect(prompt);
        Json expected;
        expected["token_ids"]   = data.token_ids;
        expected["token_types"] = data.token_types;
        expected["positions"]   = Json::array();
        for (int axis = 0; axis < 3; ++axis) {
            const auto values = data.position_axis(axis);
            expected["positions"].push_back(std::vector<std::int32_t>(values.begin(), values.end()));
        }
        expected["rope_delta"] = data.rope_delta;
        Json items             = Json::array();
        for (std::size_t i = 0; i < data.vision_items.size(); ++i) {
            const q::VisionItem& item = data.vision_items[i];
            const auto patches        = data.media_payloads[i]->span();
            std::vector<std::uint8_t> le;
            le.reserve(patches.size() * 2);
            for (const std::uint16_t value : patches) {
                le.push_back(static_cast<std::uint8_t>(value & 0xff));
                le.push_back(static_cast<std::uint8_t>(value >> 8));
            }
            Json spans = Json::array();
            for (const q::TokenSpan span : item.token_spans) { spans.push_back({span.begin, span.count}); }
            items.push_back(Json{
                {"grid", {item.grid.temporal, item.grid.height, item.grid.width}},
                {"token_spans", spans},
                {"patch_count", item.patch_count},
                {"content_sha256", fi::sha256_hex(item.content_digest)},
                {"patch_sha256", hex(le)},
            });
        }
        expected["items"] = items;
        if (data.identity.rewrite_checkpoint) {
            expected["rewrite_checkpoint"] = {
                {"kind", data.identity.rewrite_checkpoint->kind == q::RewriteCheckpointKind::TurnClosure
                             ? "turn_closure"
                             : "response_replay"},
                {"frontier", data.identity.rewrite_checkpoint->frontier}};
        } else {
            expected["rewrite_checkpoint"] = nullptr;
        }
        expected["shared_prefix_frontier"] = data.identity.shared_prefix_frontier
                                                 ? Json(*data.identity.shared_prefix_frontier)
                                                 : Json(nullptr);
        out["expected"] = expected;
    } catch (const std::exception& error) {
        out["error"] = error.what();
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: record <frontend-resource-dir> <cases.json> <images-dir> <out-dir>\n";
        return 2;
    }
    const fs::path resources_dir = argv[1];
    const fs::path images        = argv[3];
    const fs::path out_dir       = argv[4];
    fs::create_directories(out_dir);

    q::FrontendResources resources{
        .tokenizer_json                 = read_text(resources_dir / "tokenizer.json"),
        .tokenizer_config_json          = read_text(resources_dir / "tokenizer_config.json"),
        .chat_template_jinja            = read_text(resources_dir / "chat_template.jinja"),
        .generation_config_json         = read_text(resources_dir / "generation_config.json"),
        .preprocessor_config_json       = read_text(resources_dir / "preprocessor_config.json"),
        .video_preprocessor_config_json = read_text(resources_dir / "video_preprocessor_config.json"),
    };
    const q::Frontend frontend = q::FrontendTestAccess::create_component(resources, true);

    for (const Json& spec : Json::parse(read_text(argv[2]))) {
        const std::string name = spec.at("name").get<std::string>();
        const Json out         = record_case(frontend, spec, images);
        const fs::path path    = out_dir / (name + ".json");
        std::ofstream(path, std::ios::binary) << out.dump() << '\n';
        std::cout << name << (out.contains("error") ? " error: " + out["error"].get<std::string>() : "")
                  << '\n';
    }
    return 0;
}
