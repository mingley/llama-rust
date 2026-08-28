//! Chat-message plumbing on top of the template engine.
//!
//! Variable names match what HuggingFace `transformers` passes to
//! `apply_chat_template`: `messages`, `add_generation_prompt`, and the
//! `special_tokens_map` entries (`bos_token`, `eos_token`, ...). Anything a
//! template asks for that is not bound stays undefined, which is what makes
//! `{% if tools %}` and `{% if strftime_now is defined %}` take their
//! no-tools, fixed-date branches.

use super::{Template, TemplateError, Value};

/// One turn of a conversation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatMessage {
    /// `system`, `user`, `assistant`, or whatever role the template expects.
    pub role: String,
    /// Message text.
    pub content: String,
}

impl ChatMessage {
    /// A message with the given role and content.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    /// A `system` message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    /// A `user` message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    /// An `assistant` message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }

    fn to_value(&self) -> Value {
        Value::Map(vec![
            ("role".into(), Value::str(&self.role)),
            ("content".into(), Value::str(&self.content)),
        ])
    }
}

/// The non-message variables a chat template renders against.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ChatOptions {
    /// Append the "assistant speaks next" prefix. Defaults to `false`, like
    /// `apply_chat_template`.
    pub add_generation_prompt: bool,
    /// `bos_token`, when the tokenizer has one.
    pub bos_token: Option<String>,
    /// `eos_token`, when the tokenizer has one.
    pub eos_token: Option<String>,
    /// `unk_token`, when the tokenizer has one.
    pub unk_token: Option<String>,
    /// `pad_token`, when the tokenizer has one.
    pub pad_token: Option<String>,
}

impl ChatOptions {
    /// Options that only ask for the generation prompt.
    pub fn generation_prompt(add: bool) -> Self {
        Self {
            add_generation_prompt: add,
            ..Self::default()
        }
    }

    fn vars(&self, messages: &[ChatMessage]) -> Vec<(String, Value)> {
        let mut vars = vec![
            (
                "messages".to_string(),
                Value::List(messages.iter().map(ChatMessage::to_value).collect()),
            ),
            (
                "add_generation_prompt".to_string(),
                Value::Bool(self.add_generation_prompt),
            ),
        ];
        for (name, tok) in [
            ("bos_token", &self.bos_token),
            ("eos_token", &self.eos_token),
            ("unk_token", &self.unk_token),
            ("pad_token", &self.pad_token),
        ] {
            if let Some(t) = tok {
                vars.push((name.to_string(), Value::str(t)));
            }
        }
        vars
    }
}

/// Render a `tokenizer.chat_template` for `messages`.
///
/// Errors rather than guessing: an unsupported construct, a missing variable a
/// template walks into, or a `raise_exception(...)` the template itself uses to
/// reject a conversation shape all come back as [`TemplateError`].
pub fn render_chat_template(
    template: &str,
    messages: &[ChatMessage],
    opts: &ChatOptions,
) -> Result<String, TemplateError> {
    Template::parse(template)?.render(opts.vars(messages))
}

#[cfg(test)]
mod tests {
    use super::{render_chat_template, ChatMessage, ChatOptions};
    use crate::template::TemplateError;

    // Templates below are the verbatim `chat_template` strings shipped in each
    // model's `tokenizer_config.json` on HuggingFace. Expected renders were
    // captured from `transformers.apply_chat_template` (transformers 5.16.1)
    // for the same messages; see the PR description for the capture script.

    /// `Qwen/Qwen2.5-0.5B-Instruct` (ChatML). Also Qwen3 and most Qwen forks.
    const QWEN25: &str = include_str!("testdata/qwen2_5.jinja");
    /// `NousResearch/Meta-Llama-3.1-8B-Instruct`, the plain Llama-3 template.
    const LLAMA31: &str = include_str!("testdata/llama3_1.jinja");
    /// `unsloth/Llama-3.2-1B-Instruct`, the full Meta template with tool slots.
    const LLAMA32: &str = include_str!("testdata/llama3_2.jinja");
    /// `mistralai/Mistral-7B-Instruct-v0.2`. `mistralai/Mixtral-8x7B-Instruct-v0.1`
    /// ships the same body.
    const MISTRAL: &str = include_str!("testdata/mistral.jinja");
    /// `unsloth/gemma-2-2b-it`.
    const GEMMA2: &str = include_str!("testdata/gemma2.jinja");
    /// `microsoft/Phi-3-mini-4k-instruct`.
    const PHI3: &str = include_str!("testdata/phi3.jinja");

    fn two_turn() -> Vec<ChatMessage> {
        vec![
            ChatMessage::user("What is 2+2?"),
            ChatMessage::assistant("4"),
            ChatMessage::user("And 3+3?"),
        ]
    }

    fn with_system() -> Vec<ChatMessage> {
        vec![
            ChatMessage::system("You are terse."),
            ChatMessage::user("Hi"),
        ]
    }

    fn opts(add: bool, bos: Option<&str>, eos: Option<&str>) -> ChatOptions {
        ChatOptions {
            add_generation_prompt: add,
            bos_token: bos.map(str::to_string),
            eos_token: eos.map(str::to_string),
            ..ChatOptions::default()
        }
    }

    fn render(t: &str, m: &[ChatMessage], o: &ChatOptions) -> String {
        render_chat_template(t, m, o).expect("render")
    }

    #[test]
    fn qwen25_chatml_injects_the_default_system_prompt() {
        let got = render(QWEN25, &two_turn(), &opts(true, None, Some("<|im_end|>")));
        assert_eq!(
            got,
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n\
             <|im_start|>user\nWhat is 2+2?<|im_end|>\n\
             <|im_start|>assistant\n4<|im_end|>\n\
             <|im_start|>user\nAnd 3+3?<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn qwen25_chatml_uses_a_supplied_system_message_and_honours_the_flag() {
        let o = opts(true, None, Some("<|im_end|>"));
        assert_eq!(
            render(QWEN25, &with_system(), &o),
            "<|im_start|>system\nYou are terse.<|im_end|>\n\
             <|im_start|>user\nHi<|im_end|>\n\
             <|im_start|>assistant\n"
        );
        let o = opts(false, None, Some("<|im_end|>"));
        assert_eq!(
            render(QWEN25, &with_system(), &o),
            "<|im_start|>system\nYou are terse.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n"
        );
    }

    #[test]
    fn llama31_prefixes_bos_and_always_opens_an_assistant_header() {
        let o = opts(true, Some("<|begin_of_text|>"), Some("<|eot_id|>"));
        assert_eq!(
            render(LLAMA31, &two_turn(), &o),
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is 2+2?<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n4<|eot_id|>\
             <|start_header_id|>user<|end_header_id|>\n\nAnd 3+3?<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn llama32_takes_the_no_tools_fixed_date_branch() {
        // This template probes `strftime_now is defined`. We leave it
        // undefined, so it uses its own hard-coded fallback date and the render
        // is reproducible. `apply_chat_template` substitutes today's date here;
        // with `strftime_now` unregistered its output is byte-identical to this.
        let o = opts(true, Some("<|begin_of_text|>"), Some("<|eot_id|>"));
        assert_eq!(
            render(LLAMA32, &with_system(), &o),
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\
             Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n\
             You are terse.<|eot_id|>\
             <|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        );
        // An empty system slot when no system message is supplied, and no
        // assistant header at all without the generation prompt.
        let o = opts(false, Some("<|begin_of_text|>"), Some("<|eot_id|>"));
        assert_eq!(
            render(LLAMA32, &[ChatMessage::user("Hello")], &o),
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\
             Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|>\
             <|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>"
        );
    }

    #[test]
    fn mistral_folds_the_system_message_into_the_first_inst_block() {
        let o = opts(true, Some("<s>"), Some("</s>"));
        assert_eq!(
            render(MISTRAL, &with_system(), &o),
            "<s> [INST] You are terse.\n\nHi [/INST]"
        );
        assert_eq!(
            render(MISTRAL, &two_turn(), &o),
            "<s> [INST] What is 2+2? [/INST] 4</s> [INST] And 3+3? [/INST]"
        );
    }

    #[test]
    fn gemma2_renames_assistant_to_model() {
        let o = opts(true, Some("<bos>"), Some("<eos>"));
        assert_eq!(
            render(GEMMA2, &two_turn(), &o),
            "<bos><start_of_turn>user\nWhat is 2+2?<end_of_turn>\n\
             <start_of_turn>model\n4<end_of_turn>\n\
             <start_of_turn>user\nAnd 3+3?<end_of_turn>\n\
             <start_of_turn>model\n"
        );
    }

    #[test]
    fn phi3_emits_eos_when_no_generation_prompt_is_asked_for() {
        let o = opts(true, Some("<s>"), Some("<|endoftext|>"));
        assert_eq!(
            render(PHI3, &with_system(), &o),
            "<|system|>\nYou are terse.<|end|>\n<|user|>\nHi<|end|>\n<|assistant|>\n"
        );
        let o = opts(false, Some("<s>"), Some("<|endoftext|>"));
        assert_eq!(
            render(PHI3, &with_system(), &o),
            "<|system|>\nYou are terse.<|end|>\n<|user|>\nHi<|end|>\n<|endoftext|>"
        );
    }

    #[test]
    fn templates_that_refuse_a_conversation_return_the_raised_message() {
        // Gemma has no system role.
        let err = render_chat_template(
            GEMMA2,
            &with_system(),
            &opts(true, Some("<bos>"), Some("<eos>")),
        )
        .expect_err("gemma system");
        assert_eq!(
            err,
            TemplateError::Raised("System role not supported".into())
        );
        assert!(err.to_string().contains("System role not supported"));

        // Mistral requires strictly alternating user/assistant turns.
        let bad = vec![ChatMessage::user("a"), ChatMessage::user("b")];
        let err = render_chat_template(MISTRAL, &bad, &opts(true, Some("<s>"), Some("</s>")))
            .expect_err("mistral alternation");
        assert!(err.to_string().contains("must alternate"), "{err}");

        // Gemma too, with the same rule.
        let err = render_chat_template(GEMMA2, &bad, &opts(true, Some("<bos>"), Some("<eos>")))
            .expect_err("gemma alternation");
        assert!(err.to_string().contains("must alternate"), "{err}");
    }

    #[test]
    fn every_shipped_template_parses() {
        for (name, src) in [
            ("qwen2.5", QWEN25),
            ("llama3.1", LLAMA31),
            ("llama3.2", LLAMA32),
            ("mistral", MISTRAL),
            ("gemma2", GEMMA2),
            ("phi3", PHI3),
        ] {
            let _parsed = crate::template::Template::parse(src)
                .unwrap_or_else(|e| panic!("{name} failed to parse: {e}"));
        }
    }

    #[test]
    fn single_user_turn_round_trips_for_every_family() {
        let m = [ChatMessage::user("Hello")];
        let o = opts(true, Some("<s>"), Some("</s>"));
        for (name, src) in [
            ("qwen2.5", QWEN25),
            ("llama3.1", LLAMA31),
            ("llama3.2", LLAMA32),
            ("mistral", MISTRAL),
            ("gemma2", GEMMA2),
            ("phi3", PHI3),
        ] {
            let got = render_chat_template(src, &m, &o)
                .unwrap_or_else(|e| panic!("{name} failed to render: {e}"));
            assert!(got.contains("Hello"), "{name}: {got:?}");
            assert!(!got.is_empty(), "{name}");
        }
    }
}
