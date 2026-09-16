// One-off recorder for the chat-render fixtures (GitHub #184).
//
// Links the reference's chat template (ninfer `build-ninja` static
// libraries) and writes the exact prompt text it renders for each case, so
// `crates/artifact/tests/chat_render.rs` can hold ignis's minijinja render
// of the same messages to it byte for byte.
//
// The template source is the artifact's own `frontend/chat_template.jinja`
// (dumped by `crates/artifact/examples/dump_frontend.rs`); the reference
// resolves it by SHA-256, so a source it accepts is the same file ignis
// renders.
//
// Usage: record <chat_template.jinja> <cases.json> <out-dir>
// Build: tools/chat-render-fixtures/build.ps1

#include "targets/qwen3_6/impl/frontend/chat_template.h"

#include <ninfer/types.h>
#include <nlohmann/json.hpp>

#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace fi = ninfer::targets::qwen3_6::frontend_internal;
// Ordered: a case's key order is the case file's own, and a tool call's
// `arguments` is carried as the wire string, so nothing here can reorder
// what the recorder feeds the template.
using Json = nlohmann::ordered_json;

namespace {

std::string read_text(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot read " + path.string()); }
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

ninfer::ChatRole role_of(const std::string& role) {
    if (role == "system") { return ninfer::ChatRole::System; }
    if (role == "user") { return ninfer::ChatRole::User; }
    if (role == "assistant") { return ninfer::ChatRole::Assistant; }
    if (role == "tool") { return ninfer::ChatRole::Tool; }
    throw std::invalid_argument("unknown role " + role);
}

ninfer::ReasoningEffort effort_of(const std::string& effort) {
    if (effort == "low") { return ninfer::ReasoningEffort::Low; }
    if (effort == "medium") { return ninfer::ReasoningEffort::Medium; }
    if (effort == "xhigh") { return ninfer::ReasoningEffort::XHigh; }
    throw std::invalid_argument("unknown reasoning effort " + effort);
}

std::vector<fi::ChatMessage> messages_of(const Json& casus) {
    std::vector<fi::ChatMessage> messages;
    for (const Json& item : casus.at("messages")) {
        fi::ChatMessage message;
        message.role = role_of(item.at("role").get<std::string>());
        message.parts.push_back(fi::ChatPart::text_part(item.at("content").get<std::string>()));
        if (item.contains("reasoning_content")) {
            message.reasoning_content = item.at("reasoning_content").get<std::string>();
        }
        if (item.contains("tool_call_id")) {
            message.tool_call_id = item.at("tool_call_id").get<std::string>();
        }
        if (item.contains("tool_calls")) {
            for (const Json& call : item.at("tool_calls")) {
                // `arguments` is the wire string, verbatim -- exactly what
                // `openai_schema.cpp` hands the template.
                message.tool_calls.push_back({.id             = call.at("id").get<std::string>(),
                                              .name           = call.at("name").get<std::string>(),
                                              .arguments_json = call.at("arguments").get<std::string>()});
            }
        }
        messages.push_back(std::move(message));
    }
    return messages;
}

fi::ChatRenderOptions options_of(const Json& casus) {
    fi::ChatRenderOptions options;
    options.add_generation_prompt = casus.value("add_generation_prompt", true);
    options.enable_thinking       = casus.value("enable_thinking", false);
    if (casus.contains("reasoning_effort")) {
        options.reasoning_effort = effort_of(casus.at("reasoning_effort").get<std::string>());
    }
    // Always engaged, the way a request leaves it (GitHub #185). The serving
    // path fills `PromptOptions::preserve_thinking`, a plain bool defaulting
    // to false, and `frontend.cpp`'s `render_options()` engages the optional
    // from it — so nullopt is a state no request can produce, and here it
    // would mean the opposite of the default: `chat_template.cpp:359` reads
    // `value_or(effort_template)`, and this artifact's template is the
    // reasoning-effort one, so an unset option renders as *preserve*.
    options.preserve_thinking = casus.value("preserve_thinking", false);
    if (casus.contains("tools")) {
        for (const Json& tool : casus.at("tools")) {
            // A tool definition reaches the template through the request's
            // own `nlohmann::json`, which sorts keys; re-parsing the
            // (ordered) case entry as one reproduces that exactly.
            options.tool_jsons.push_back(nlohmann::json::parse(tool.dump()).dump());
        }
    }
    return options;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: record <chat_template.jinja> <cases.json> <out-dir>\n";
        return 2;
    }
    try {
        const fi::CompiledChatTemplate template_ = fi::CompiledChatTemplate::resolve(read_text(argv[1]));
        const Json cases                         = Json::parse(read_text(argv[2]));
        const fs::path out(argv[3]);
        fs::create_directories(out);
        for (const Json& casus : cases) {
            const std::string name    = casus.at("name").get<std::string>();
            const fi::RenderedChat rendered = template_.render(messages_of(casus), options_of(casus));
            Json fixture;
            fixture["case"]     = casus;
            fixture["expected"] = Json{{"text", rendered.text}};
            if (rendered.shared_prefix_offset) {
                fixture["expected"]["shared_prefix_offset"] = *rendered.shared_prefix_offset;
            }
            std::ofstream file(out / (name + ".json"), std::ios::binary);
            file << fixture.dump(2) << "\n";
            std::cout << name << ": " << rendered.text.size() << " bytes\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "record failed: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
