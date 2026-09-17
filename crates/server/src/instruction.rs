//! Instruction-message policies (GitHub #209, spec
//! `.scratch/vram-budget/specs/01-vram-budget.md` §Slice 2).
//!
//! An **instruction message** is a `system` or `developer` message. The Qwen
//! 3.8 template accepts exactly one `system`, at index 0, and no `developer`
//! at all, while real clients send more: qwen-code a leading run of `system`
//! messages (the agent prompt, then a hook line), OpenAI SDKs `developer`
//! messages wherever they please. The engine normalizes the message list here,
//! before any template runs, into a list whose roles are only `system`,
//! `user`, `assistant` and `tool`, where
//!
//! - a `system` at index 0 is the **system prompt**, and
//! - a `system` anywhere else is an instruction rendered as a system block of
//!   its own, in place (the provider's contract, as ninfer renders one).
//!
//! No policy reorders messages across roles. Joining and gathering re-render
//! earlier history when a new message arrives mid-conversation, which is why
//! [`DeveloperMessagePolicy::rerenders_history`] exists: `main` warns about it.
//!
//! A block joined from several messages is carried as **text parts, one per
//! message**, every part but the last ending in `"\n"`. The reference's part
//! join (`template_text_parts`) adds one more `"\n"` between adjacent text
//! parts, so the block renders as the texts joined by `"\n\n"` — and the
//! provider can still see where the first message ends. That is where a
//! retained prefix is cut (`artifact_template.rs`): a qwen-code hook line that
//! changes between requests then leaves every page before the one it starts in
//! shared.

use crate::template::{ChatMessage, ContentPart, MessageContent, TemplateRejection};

/// What the engine does with a `system` message that is not the first
/// message (`--system-message-policy`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SystemMessagePolicy {
    /// The leading run joins the system prompt; a later one is its own block
    /// in place.
    #[default]
    Merge,
    /// Any `system` message that is not first is refused.
    Strict,
}

impl SystemMessagePolicy {
    /// Every value, in the order `--help` and errors list them.
    pub const ALL: [Self; 2] = [Self::Merge, Self::Strict];

    /// The value's name on the command line.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Merge => "merge",
            Self::Strict => "strict",
        }
    }
}

/// What the engine does with a `developer` message
/// (`--developer-message-policy`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeveloperMessagePolicy {
    /// Each is its own system block where it stands.
    #[default]
    Inplace,
    /// All are joined into the system prompt.
    IntoSystem,
    /// All are gathered into one system block right after the system prompt.
    AfterSystem,
    /// Exactly one, immediately after the system prompt; any other is refused.
    OneAfterSystem,
    /// Any is refused.
    Reject,
}

impl DeveloperMessagePolicy {
    /// Every value, in the order `--help` and errors list them.
    pub const ALL: [Self; 5] = [
        Self::Inplace,
        Self::IntoSystem,
        Self::AfterSystem,
        Self::OneAfterSystem,
        Self::Reject,
    ];

    /// The value's name on the command line.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Inplace => "inplace",
            Self::IntoSystem => "into-system",
            Self::AfterSystem => "after-system",
            Self::OneAfterSystem => "one-after-system",
            Self::Reject => "reject",
        }
    }

    /// Whether a `developer` message arriving mid-conversation changes the
    /// render of history before it, so prefix reuse is lost from there.
    pub fn rerenders_history(self) -> bool {
        matches!(self, Self::IntoSystem | Self::AfterSystem)
    }
}

/// Both policies, as a request is normalized under them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct InstructionPolicy {
    pub system: SystemMessagePolicy,
    pub developer: DeveloperMessagePolicy,
}

impl InstructionPolicy {
    /// `messages` with every instruction message placed under this policy, or
    /// the 400 that refuses the request.
    ///
    /// Roles are expected to be checked already (`check_roles`); an unknown
    /// role passes through untouched for the provider to refuse. Instruction
    /// messages are expected to carry text only (`check_content_parts` refuses
    /// media in them), so their text is what is joined.
    ///
    /// - **Leading run:** the maximal run of `system` messages at index 0. A
    ///   `developer` interrupts it.
    /// - **Leading developer:** a `developer` at index 0 is the system prompt,
    ///   under every developer policy but `reject`.
    /// - **Joining** trims each text and puts `"\n\n"` between the non-empty
    ///   ones. A system prompt made of one message keeps that message as sent.
    ///
    /// Under `after-system` or `into-system` with no system prompt, the
    /// gathered or joined block opens the list and so becomes the system
    /// prompt: there is nothing for it to follow.
    pub fn normalize(&self, messages: &[ChatMessage]) -> Result<Vec<ChatMessage>, TemplateRejection> {
        let mut prompt: Vec<&ChatMessage> = Vec::new();
        let mut head = 0;
        match messages.first().map(|m| m.role.as_str()) {
            Some("system") => {
                prompt.push(&messages[0]);
                head = 1;
                while let Some(message) = messages.get(head).filter(|m| m.role == "system") {
                    if self.system == SystemMessagePolicy::Strict {
                        return Err(system_position(head));
                    }
                    prompt.push(message);
                    head += 1;
                }
            }
            Some("developer") if self.developer != DeveloperMessagePolicy::Reject => {
                prompt.push(&messages[0]);
                head = 1;
            }
            _ => {}
        }
        let has_prompt = !prompt.is_empty();
        // `one-after-system` accepts exactly one developer message: a leading
        // one that is the system prompt already is it.
        let developer_prompt = messages.first().is_some_and(|m| m.role == "developer");

        let mut gathered: Vec<&ChatMessage> = Vec::new();
        let mut body = Vec::with_capacity(messages.len() - head);
        for (index, message) in messages.iter().enumerate().skip(head) {
            match message.role.as_str() {
                "system" => match self.system {
                    SystemMessagePolicy::Strict => return Err(system_position(index)),
                    SystemMessagePolicy::Merge => body.push(instruction_block(&[message])),
                },
                "developer" => match self.developer {
                    DeveloperMessagePolicy::Inplace => body.push(instruction_block(&[message])),
                    DeveloperMessagePolicy::IntoSystem => prompt.push(message),
                    DeveloperMessagePolicy::AfterSystem => gathered.push(message),
                    DeveloperMessagePolicy::OneAfterSystem if has_prompt && !developer_prompt && index == head => {
                        body.push(instruction_block(&[message]))
                    }
                    DeveloperMessagePolicy::OneAfterSystem | DeveloperMessagePolicy::Reject => {
                        return Err(developer_position(index, self.developer))
                    }
                },
                _ => body.push(message.clone()),
            }
        }

        let mut out = Vec::with_capacity(body.len() + 2);
        match prompt.as_slice() {
            [] => {}
            [only] => out.push(ChatMessage { role: "system".to_owned(), ..(*only).clone() }),
            many => out.push(instruction_block(many)),
        }
        if !gathered.is_empty() {
            out.push(instruction_block(&gathered));
        }
        out.extend(body);
        Ok(out)
    }
}

/// One `system` message carrying `messages`' texts, trimmed and joined: a
/// plain string for one text, text parts (module docs) for more.
fn instruction_block(messages: &[&ChatMessage]) -> ChatMessage {
    let texts: Vec<String> = messages
        .iter()
        .map(|m| m.content.text().trim().to_owned())
        .filter(|text| !text.is_empty())
        .collect();
    let content = match texts.as_slice() {
        [] => MessageContent::Text(String::new()),
        [only] => MessageContent::Text(only.clone()),
        [.., last] => {
            let part = |text: String| ContentPart { kind: Some("text".to_owned()), text: Some(text), url: None };
            let mut parts: Vec<ContentPart> = texts[..texts.len() - 1].iter().map(|t| part(format!("{t}\n"))).collect();
            parts.push(part(last.clone()));
            MessageContent::Parts(parts)
        }
    };
    ChatMessage { content, ..ChatMessage::text("system", "") }
}

fn system_position(index: usize) -> TemplateRejection {
    TemplateRejection {
        code: "system_message_position",
        message: format!(
            "system message at index {index} is not the first message \
             (--system-message-policy strict accepts only one, at index 0)"
        ),
    }
}

fn developer_position(index: usize, policy: DeveloperMessagePolicy) -> TemplateRejection {
    let rule = match policy {
        DeveloperMessagePolicy::OneAfterSystem => {
            "accepts exactly one, immediately after the system prompt"
        }
        _ => "accepts none",
    };
    TemplateRejection {
        code: "developer_message_position",
        message: format!(
            "developer message at index {index} is not accepted \
             (--developer-message-policy {} {rule})",
            policy.as_str()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use DeveloperMessagePolicy as D;
    use SystemMessagePolicy as S;

    fn policy(system: S, developer: D) -> InstructionPolicy {
        InstructionPolicy { system, developer }
    }

    fn list(shape: &[(&str, &str)]) -> Vec<ChatMessage> {
        shape.iter().map(|(role, text)| ChatMessage::text(*role, *text)).collect()
    }

    /// `role:text` per message, the shape assertions read best in.
    fn shape(messages: &[ChatMessage]) -> Vec<String> {
        messages.iter().map(|m| format!("{}:{}", m.role, m.content.text())).collect()
    }

    fn normalized(p: InstructionPolicy, messages: &[(&str, &str)]) -> Vec<String> {
        shape(&p.normalize(&list(messages)).expect("accepted"))
    }

    fn refused(p: InstructionPolicy, messages: &[(&str, &str)]) -> TemplateRejection {
        p.normalize(&list(messages)).expect_err("refused")
    }

    const TWO_SYSTEMS: &[(&str, &str)] = &[("system", "A"), ("system", "B"), ("user", "q")];
    const LEADING_DEVELOPER: &[(&str, &str)] = &[("developer", "D"), ("user", "q")];
    const MID_DEVELOPER: &[(&str, &str)] =
        &[("system", "S"), ("user", "q"), ("developer", "D"), ("assistant", "a"), ("user", "r")];
    const INTERLEAVED: &[(&str, &str)] =
        &[("system", "S"), ("developer", "D"), ("system", "T"), ("user", "q")];

    #[test]
    fn the_defaults_are_merge_and_inplace() {
        assert_eq!(InstructionPolicy::default(), policy(S::Merge, D::Inplace));
    }

    #[test]
    fn merge_joins_the_leading_run_into_the_system_prompt() {
        for developer in D::ALL {
            assert_eq!(
                normalized(policy(S::Merge, developer), TWO_SYSTEMS),
                ["system:A\n\nB", "user:q"],
                "{developer:?}"
            );
        }
    }

    #[test]
    fn joining_trims_each_text_and_skips_empty_ones() {
        let p = InstructionPolicy::default();
        let messages = [("system", "  A.\n"), ("system", "   "), ("system", "\nB "), ("user", "q")];
        assert_eq!(normalized(p, &messages), ["system:A.\n\nB", "user:q"]);
    }

    #[test]
    fn a_joined_block_keeps_one_text_part_per_message() {
        let messages = [("system", "A"), ("system", "B"), ("system", "C"), ("user", "q")];
        let out = InstructionPolicy::default().normalize(&list(&messages)).unwrap();
        let MessageContent::Parts(parts) = &out[0].content else {
            panic!("a joined block is parts: {:?}", out[0]);
        };
        let texts: Vec<_> = parts.iter().map(|p| (p.kind.as_deref(), p.text.as_deref())).collect();
        assert_eq!(texts, [(Some("text"), Some("A\n")), (Some("text"), Some("B\n")), (Some("text"), Some("C"))]);
        assert_eq!(out[0].content.text(), "A\n\nB\n\nC", "the part join puts \\n\\n between them");
    }

    #[test]
    fn a_single_system_prompt_is_kept_as_sent() {
        let messages = [("system", "  A  "), ("user", "q")];
        assert_eq!(normalized(InstructionPolicy::default(), &messages), ["system:  A  ", "user:q"]);
    }

    #[test]
    fn strict_refuses_a_second_system_message_naming_its_index() {
        for developer in D::ALL {
            let rejection = refused(policy(S::Strict, developer), TWO_SYSTEMS);
            assert_eq!(rejection.code, "system_message_position");
            assert!(rejection.message.contains("index 1"), "{}", rejection.message);
        }
        let later = [("system", "S"), ("user", "q"), ("system", "T"), ("user", "r")];
        let rejection = refused(policy(S::Strict, D::Inplace), &later);
        assert_eq!(rejection.code, "system_message_position");
        assert!(rejection.message.contains("index 2"), "{}", rejection.message);
    }

    #[test]
    fn a_later_system_message_under_merge_is_a_block_in_place() {
        let later = [("system", "S"), ("user", "q"), ("system", " T "), ("user", "r")];
        assert_eq!(
            normalized(InstructionPolicy::default(), &later),
            ["system:S", "user:q", "system:T", "user:r"]
        );
    }

    #[test]
    fn a_leading_developer_is_the_system_prompt_in_every_mode_but_reject() {
        for system in S::ALL {
            for developer in [D::Inplace, D::IntoSystem, D::AfterSystem, D::OneAfterSystem] {
                assert_eq!(
                    normalized(policy(system, developer), LEADING_DEVELOPER),
                    ["system:D", "user:q"],
                    "{system:?} {developer:?}"
                );
            }
            let rejection = refused(policy(system, D::Reject), LEADING_DEVELOPER);
            assert_eq!(rejection.code, "developer_message_position");
            assert!(rejection.message.contains("index 0"), "{}", rejection.message);
        }
    }

    #[test]
    fn a_mid_conversation_developer_follows_its_policy() {
        let p = |developer| policy(S::Merge, developer);
        assert_eq!(
            normalized(p(D::Inplace), MID_DEVELOPER),
            ["system:S", "user:q", "system:D", "assistant:a", "user:r"]
        );
        assert_eq!(
            normalized(p(D::IntoSystem), MID_DEVELOPER),
            ["system:S\n\nD", "user:q", "assistant:a", "user:r"]
        );
        assert_eq!(
            normalized(p(D::AfterSystem), MID_DEVELOPER),
            ["system:S", "system:D", "user:q", "assistant:a", "user:r"]
        );
        for developer in [D::OneAfterSystem, D::Reject] {
            let rejection = refused(p(developer), MID_DEVELOPER);
            assert_eq!(rejection.code, "developer_message_position");
            assert!(rejection.message.contains("index 2"), "{}", rejection.message);
        }
    }

    #[test]
    fn an_interleaved_head_is_never_reordered() {
        let p = |developer| policy(S::Merge, developer);
        assert_eq!(
            normalized(p(D::Inplace), INTERLEAVED),
            ["system:S", "system:D", "system:T", "user:q"]
        );
        assert_eq!(normalized(p(D::IntoSystem), INTERLEAVED), ["system:S\n\nD", "system:T", "user:q"]);
        assert_eq!(normalized(p(D::AfterSystem), INTERLEAVED), ["system:S", "system:D", "system:T", "user:q"]);
        assert_eq!(
            normalized(p(D::OneAfterSystem), INTERLEAVED),
            ["system:S", "system:D", "system:T", "user:q"]
        );
        let rejection = refused(p(D::Reject), INTERLEAVED);
        assert_eq!(rejection.code, "developer_message_position");
        assert!(rejection.message.contains("index 1"), "{}", rejection.message);
        // Strict: the developer is fine where it is, the second system is not.
        let rejection = refused(policy(S::Strict, D::Inplace), INTERLEAVED);
        assert_eq!(rejection.code, "system_message_position");
        assert!(rejection.message.contains("index 2"), "{}", rejection.message);
    }

    #[test]
    fn after_system_gathers_every_developer_into_one_block() {
        let messages = [
            ("system", "S"),
            ("developer", "D1"),
            ("user", "q"),
            ("developer", "D2"),
            ("user", "r"),
        ];
        assert_eq!(
            normalized(policy(S::Merge, D::AfterSystem), &messages),
            ["system:S", "system:D1\n\nD2", "user:q", "user:r"]
        );
    }

    #[test]
    fn one_after_system_accepts_one_only_right_after_the_prompt() {
        let p = policy(S::Merge, D::OneAfterSystem);
        let after_run = [("system", "A"), ("system", "B"), ("developer", "D"), ("user", "q")];
        assert_eq!(normalized(p, &after_run), ["system:A\n\nB", "system:D", "user:q"]);
        let second = [("system", "S"), ("developer", "D1"), ("developer", "D2"), ("user", "q")];
        let rejection = refused(p, &second);
        assert_eq!(rejection.code, "developer_message_position");
        assert!(rejection.message.contains("index 2"), "{}", rejection.message);
        let no_prompt = [("user", "q"), ("developer", "D")];
        assert!(refused(p, &no_prompt).message.contains("index 1"));
        // A leading developer is the system prompt and the one developer
        // message this policy accepts; a second is refused wherever it stands.
        let two = [("developer", "D1"), ("developer", "D2"), ("user", "q")];
        let rejection = refused(p, &two);
        assert_eq!(rejection.code, "developer_message_position");
        assert!(rejection.message.contains("index 1"), "{}", rejection.message);
    }

    #[test]
    fn without_a_system_prompt_joined_developers_become_it() {
        let messages = [("user", "q"), ("developer", "D"), ("user", "r")];
        for developer in [D::IntoSystem, D::AfterSystem] {
            assert_eq!(
                normalized(policy(S::Merge, developer), &messages),
                ["system:D", "user:q", "user:r"],
                "{developer:?}"
            );
        }
    }

    #[test]
    fn other_messages_pass_through_whole() {
        let mut messages = list(&[("system", "S"), ("assistant", "a"), ("tool", "t"), ("user", "q")]);
        messages[1].reasoning_content = Some("why".to_owned());
        messages[2].tool_call_id = Some("call_1".to_owned());
        let out = InstructionPolicy::default().normalize(&messages).unwrap();
        assert_eq!(out, messages);
    }

    #[test]
    fn only_rerendering_developer_policies_warn() {
        let warned: Vec<_> = D::ALL.into_iter().filter(|d| d.rerenders_history()).collect();
        assert_eq!(warned, [D::IntoSystem, D::AfterSystem]);
    }
}
