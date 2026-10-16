"""serve/test_oneshot_rules.py - the one-shot marker rule set (see oneshot_rules.py).

Two kinds of test live here:

* **the real DSH samples** - the session-title call and an ordinary multi-turn turn, both taken from
  this machine's spill sidecars (`strata-conv-3/7.meta`, decoded with the pack's vocab.json) and the
  engine log.  They pin the rule that has to hit, and - more importantly - the requests that must
  NOT hit it.
* **the config contract** - a typo must fail at startup, never silently change caching.

Runs without an engine, a tokenizer or a GPU:

    python -m unittest serve.test_oneshot_rules -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))          # the repo root: `serve` next to this file is the package under test

from serve import oneshot_rules as R  # noqa: E402

# The title system prompt, verbatim from the two real spill entries.  The two recorded prompts are
# byte-identical in this part and differ only in the user turn (`seq:10` vs `seq:8`).
TITLE_SYSTEM = (
    "Create a concise title for an AI coding-assistant session from the supplied human messages.\n"
    "Return only the title on one line, **in plain text of natural language**, with no quotes, "
    "prefix, explanation, Markdown, XML, or terminal control codes. No code is allowed.\n"
    "Use the language of the messages.\n"
    "Aim for about 5 words in non-CJK languages or 10 CJK characters."
)
TITLE_PREFIX = "Create a concise title for an AI coding-assistant session"
TITLE_USER = ('Generate the session title from this JSON array of human messages:\n'
              '[{"seq":10,"text":"你好"}]')

# An ordinary turn of the session the title call sits beside: the deployment's system prompt is the
# workspace instructions, not the title text (this is the shape the engine logged: 150K+ token
# prompts, `systemPromptUpdate: "in-history"`).
MAIN_SYSTEM = ("You are DSH, a coding agent.\n\nInstructions from: ~/.dsh/AGENTS.md\n\n"
               "# 网络访问与代理回退策略（全局）\n\n本地代理：Clash Verge 混合端口 ...")

# The anchor these rules were validated against: the first 24 prompt tokens of the real title
# request (sidecar-derived), i.e. `<|im_start|>system ... <|im_end|>`.
TITLE_ANCHOR = (248045, 8678, 198, 3886, 264, 61446, 2192, 364, 449, 14791)

# DSH's context-compaction call.  The instruction is verbatim from the client's own source (`app.asar`:
# `const COMPACTION_INSTRUCTION = [...]`), and the call is assembled as
# `[...input.messages, deepFreeze({role: "user", content: [{type: "text", text: COMPACTION_INSTRUCTION}]})]`
# - the directive is the FINAL user turn of the very conversation being compacted, deliberately, "so the
# provider's KV cache is reused instead of invalidated".
COMPACTION_INSTRUCTION = (
    "You are now acting as a compaction engine for this AI coding assistant. Condense the conversation "
    "ABOVE into a structured checkpoint that lets another model resume the work with no loss of "
    "essential context.")
COMPACTION_PREFIX = "You are now acting as a compaction engine for this AI coding assistant"


def title_request(**extra):
    """DSH's session-title call as serve sees it: the title text is messages[0] (in-history)."""
    req = {"model": "qwen3.8-flash-next-iq3_xxs", "max_tokens": 64, "stream": False,
           "messages": [{"role": "system", "content": TITLE_SYSTEM},
                        {"role": "user", "content": TITLE_USER}]}
    req.update(extra)
    return req


def main_request(tokens=150000, **extra):
    """An ordinary turn of a long main conversation."""
    req = {"model": "qwen3.8-flash-next-iq3_xxs", "max_tokens": 64000,
           "messages": [{"role": "system", "content": MAIN_SYSTEM},
                        {"role": "user", "content": "继续修那个缓存问题"},
                        {"role": "assistant", "content": "好"},
                        {"role": "user", "content": "先看日志"}]}
    req.update(extra)
    req["_tokens"] = tokens
    return req


def compaction_request(**extra):
    """DSH's compaction call as serve sees it: the conversation replayed with the client's own system
    prompt (role `developer`), then the directive as the final user turn in a content block."""
    req = {"model": "qwen3.8-flash-next-iq3_xxs", "max_completion_tokens": 65536, "stream": True,
           "messages": [{"role": "developer", "content": MAIN_SYSTEM},
                        {"role": "user", "content": "继续修那个缓存问题"},
                        {"role": "assistant", "content": "好"},
                        {"role": "user", "content": [{"type": "text", "text": COMPACTION_INSTRUCTION}]}]}
    req.update(extra)
    return req


def markers(*rules, enabled=True, action=R.ACTION_CKPT0, log=True):
    return R.parse({"enabled": enabled, "action": action, "log": log, "rules": list(rules)})


TITLE_RULE = {"id": "ds-title", "type": "system_text", "prefix": TITLE_PREFIX,
              "description": "DSH session title"}
CODEX_RULE = {"id": "codex-compaction", "type": "request_kind", "equals": "compaction"}
FLAG_RULE = {"id": "client-flag", "type": "field", "path": "strata_checkpoint", "equals": False}
SUBAGENT_RULE = {"id": "ds-subagent", "type": "field",
                 "path": "dsh_session_log.session.origin", "equals": "subagent"}
COMPACTION_RULE = {"id": "ds-compaction", "type": "user_text", "prefix": COMPACTION_PREFIX,
                   "description": "DSH context compaction"}


class RealSamples(unittest.TestCase):
    """The DSH samples: the title call must be marked, everything else must not."""

    def setUp(self):
        self.rules = markers(TITLE_RULE)

    def test_the_title_call_is_marked(self):
        hits = R.match(self.rules, title_request())
        self.assertEqual(hits, ["ds-title"])

    def test_a_long_main_turn_is_not_marked(self):
        self.assertEqual(R.match(self.rules, main_request()), [])

    def test_a_small_main_turn_is_not_marked(self):
        """A short ordinary request is still not a title call: size is not a criterion."""
        req = main_request(tokens=200)
        self.assertEqual(R.match(self.rules, req), [])

    def test_a_request_that_only_looks_like_it_in_the_user_turn_is_not_marked(self):
        """The words of the title prompt inside a user message (an agent reading these docs)."""
        req = {"messages": [{"role": "system", "content": MAIN_SYSTEM},
                            {"role": "user", "content": TITLE_PREFIX + " ..."
                             "这是我的提示词，帮我改一改"}]}
        self.assertEqual(R.match(self.rules, req), [])

    def test_an_empty_system_prompt_is_not_marked(self):
        self.assertEqual(R.match(self.rules, {"messages": [{"role": "user", "content": "hi"}]}), [])

    def test_the_anchor_matches_the_real_prompt_tokens(self):
        """With the real prompt ids, the token anchor decides (the strict path)."""
        tokens_rule = dict(TITLE_RULE, match="tokens")
        prepared, notes = R.prepare_text_rules(markers(tokens_rule), encode=lambda text: TITLE_ANCHOR)
        self.assertEqual(notes, [])
        rule = prepared.text_rules()[0]
        self.assertEqual(rule.anchor, TITLE_ANCHOR)
        self.assertEqual(R.match(prepared, title_request(), prompt_ids=list(TITLE_ANCHOR) + [1, 2, 3]),
                         ["ds-title"])
        self.assertEqual(R.match(prepared, title_request(), prompt_ids=[999, 998]), [])

    def test_the_configured_prefix_matches_the_real_recorded_prompt(self):
        """The plan's A4 check, pinned: the config prefix is a prefix of the prompt the engine recorded.

        TITLE_SYSTEM above IS the recorded system text (decoded from `strata-conv-3.meta` with the
        pack's vocab), so a rule whose prefix is either the configured text or the whole system block
        must match it.  This is what makes the rule deployable rather than merely plausible.
        """
        prompt = {"messages": [{"role": "system", "content": TITLE_SYSTEM},
                               {"role": "user", "content": TITLE_USER}]}
        for prefix in (TITLE_PREFIX, TITLE_SYSTEM, TITLE_SYSTEM[:40]):
            with self.subTest(prefix=prefix[:24]):
                self.assertEqual(R.match(markers({"id": "ds-title", "type": "system_text",
                                                  "prefix": prefix}), prompt), ["ds-title"])
        # and one character more than the block is not a prefix of it any more
        self.assertEqual(R.match(markers({"id": "ds-title", "type": "system_text",
                                          "prefix": TITLE_SYSTEM + " "}), prompt), [])

    def test_the_main_conversation_system_prompt_is_not_a_title(self):
        for prefix in (TITLE_PREFIX, TITLE_SYSTEM):
            self.assertEqual(R.match(markers({"id": "ds-title", "type": "system_text",
                                              "prefix": prefix}),
                                     {"messages": [{"role": "system", "content": MAIN_SYSTEM}]}), [])

    def test_text_mode_is_independent_of_the_prompt_ids(self):
        """The default mode decides from the request's own system text.

        Measured reason for the default (2026-10-16): the prompt starts with the template's
        `<|im_start|>system\n`, and encoding the system text alone does not reproduce that block -
        rebuilding it agreed with the recorded prompt for 3 tokens out of 205.  Text matching has no
        such problem, so a rule must not silently depend on prompt ids.
        """
        rules = markers(TITLE_RULE)                      # no "match": the default
        self.assertEqual(R.match(rules, title_request()), ["ds-title"])
        self.assertEqual(R.match(rules, title_request(), prompt_ids=[999, 998, 997]), ["ds-title"])
        self.assertEqual(R.match(rules, title_request(), prompt_ids=[]), ["ds-title"])

    def test_the_guard_excludes_in_text_mode_too(self):
        rules = markers(dict(TITLE_RULE, max_prompt_tokens=512))
        self.assertEqual(R.match(rules, title_request(), prompt_ids=[0] * 100), ["ds-title"])
        self.assertEqual(R.match(rules, title_request(), prompt_ids=[0] * 5000), [])
        # without ids there is nothing to check the guard against: the text still decides
        self.assertEqual(R.match(rules, title_request()), ["ds-title"])

    def test_tokens_mode_needs_the_prompt_ids(self):
        prepared, _ = R.prepare_text_rules(markers(dict(TITLE_RULE, match="tokens")),
                                          encode=lambda text: TITLE_ANCHOR)
        self.assertEqual(R.match(prepared, title_request()), [],
                         "tokens mode never falls back to text")
        self.assertEqual(R.match(prepared, title_request(), prompt_ids=[*TITLE_ANCHOR, 1]), ["ds-title"])

    def test_the_optional_guard_excludes_a_long_prompt(self):
        """A guard can only exclude: it never selects a request by itself."""
        rule = dict(TITLE_RULE, match="tokens", max_prompt_tokens=512)
        prepared, _ = R.prepare_text_rules(markers(rule), encode=lambda text: TITLE_ANCHOR)
        short = list(TITLE_ANCHOR) + [7] * 100
        long = list(TITLE_ANCHOR) + [7] * 5000
        self.assertEqual(R.match(prepared, title_request(), prompt_ids=short), ["ds-title"])
        self.assertEqual(R.match(prepared, title_request(), prompt_ids=long), [],
                         "a 5K-token prompt is not a title call even if it opens the same way")
        # the same prompt with no guard is marked: the guard is what excluded it
        unguarded, _ = R.prepare_text_rules(markers(dict(TITLE_RULE, match="tokens")), encode=lambda text: TITLE_ANCHOR)
        self.assertEqual(R.match(unguarded, title_request(), prompt_ids=long), ["ds-title"])


class DeclaredIdentities(unittest.TestCase):
    """FIELD and REQUEST_KIND rules: the client's own declaration."""

    def test_a_nested_field_path(self):
        rules = markers(SUBAGENT_RULE)
        req = title_request(dsh_session_log={"session": {"origin": "subagent", "id": "session-1"}})
        self.assertEqual(R.match(rules, req), ["ds-subagent"])
        req["dsh_session_log"]["session"]["origin"] = "user"
        self.assertEqual(R.match(rules, req), [])
        del req["dsh_session_log"]["session"]["origin"]
        self.assertEqual(R.match(rules, req), [])

    def test_a_field_without_an_expected_value_only_has_to_exist(self):
        rules = markers({"id": "parent", "type": "field", "path": "report_usage"})
        self.assertEqual(R.match(rules, {"report_usage": False}), ["parent"])
        self.assertEqual(R.match(rules, {}), [])

    def test_json_types_stay_apart(self):
        rules = markers(FLAG_RULE)
        self.assertEqual(R.match(rules, {"strata_checkpoint": False}), ["client-flag"])
        self.assertEqual(R.match(rules, {"strata_checkpoint": 0}), [], "False is not 0")
        self.assertEqual(R.match(rules, {"strata_checkpoint": "false"}), [], "False is not 'false'")
        self.assertEqual(R.match(rules, {"strata_checkpoint": True}), [])

    def test_a_developer_turn_is_the_system_prompt(self):
        """Measured: DSH sends its system prompt as role "developer" over the OpenAI API.

        Strata's own adapter maps developer -> system (serve/frontend.py:526, issue #56); a rule that
        only looked at "system" missed every DSH request, including the session-title call.
        """
        req = {"messages": [{"role": "developer", "content": TITLE_SYSTEM},
                            {"role": "user", "content": TITLE_USER}]}
        self.assertEqual(R.system_text(req)[:40], TITLE_SYSTEM[:40])
        self.assertEqual(R.match(markers(TITLE_RULE), req), ["ds-title"])

    def test_developer_and_system_turns_are_joined(self):
        req = {"messages": [{"role": "system", "content": "first"},
                            {"role": "developer", "content": "second"}]}
        self.assertEqual(R.system_text(req), "first\nsecond")

    def test_a_developer_turn_is_not_confused_with_a_user_turn(self):
        """A user message that merely opens with the title words must still not match."""
        req = {"messages": [{"role": "developer", "content": MAIN_SYSTEM},
                            {"role": "user", "content": TITLE_PREFIX + " ... please shorten this"}]}
        self.assertEqual(R.match(markers(TITLE_RULE), req), [])

    def test_the_codex_request_kind(self):
        rules = markers(CODEX_RULE)
        import json
        self.assertEqual(R.match(rules, {"x-codex-turn-metadata": json.dumps(
            {"request_kind": "compaction", "session_id": "a", "thread_id": "a"})}),
            ["codex-compaction"])
        self.assertEqual(R.match(rules, {"x-codex-turn-metadata": json.dumps(
            {"request_kind": "turn", "session_id": "a", "thread_id": "a"})}), [])
        self.assertEqual(R.match(rules, {"request_kind": "compaction"}), ["codex-compaction"])
        self.assertEqual(R.match(rules, {"x-codex-turn-metadata": "{not json"}), [])
        self.assertEqual(R.match(rules, {"x-codex-turn-metadata": json.dumps(["compaction"])}), [])

    def test_rules_are_reported_in_config_order_and_all_of_them(self):
        rules = markers(TITLE_RULE, FLAG_RULE, SUBAGENT_RULE)
        req = title_request(strata_checkpoint=False,
                           dsh_session_log={"session": {"origin": "subagent"}})
        self.assertEqual(R.match(rules, req), ["ds-title", "client-flag", "ds-subagent"])

    def test_a_disabled_rule_set_never_marks(self):
        rules = markers(TITLE_RULE, FLAG_RULE, enabled=False)
        self.assertEqual(R.match(rules, title_request(strata_checkpoint=False)), [])


class DshCompaction(unittest.TestCase):
    """The client's auxiliary summarizer call: marked, and only when it really is that call.

    Why it is a one-shot: the call is a genuine prefix of the conversation (its whole point), but the
    turn it leaves is never continued - the client replaces the summarized range with the summary, so
    the next prompt diverges from it.  Measured: 24587 tokens -> a 436 MiB session file written at the
    next request's switch, never read back.
    """

    def setUp(self):
        self.rules = markers(COMPACTION_RULE)

    def test_the_compaction_call_is_marked(self):
        self.assertEqual(R.match(self.rules, compaction_request()), ["ds-compaction"])

    def test_a_plain_string_content_is_read_too(self):
        """A client that sends the directive as a bare string must match as well as a block list."""
        req = compaction_request(messages=[{"role": "user", "content": COMPACTION_INSTRUCTION}])
        self.assertEqual(R.match(self.rules, req), ["ds-compaction"])

    def test_an_ordinary_turn_of_the_same_conversation_is_not_marked(self):
        self.assertEqual(R.match(self.rules, main_request()), [])

    def test_the_other_auxiliary_call_beside_it_is_not_marked(self):
        self.assertEqual(R.match(self.rules, title_request()), [])

    def test_the_same_text_earlier_in_the_history_is_not_marked(self):
        """Only the request's FINAL user turn is a directive: a history that mentions it is not one."""
        req = compaction_request(messages=[
            {"role": "developer", "content": MAIN_SYSTEM},
            {"role": "user", "content": COMPACTION_INSTRUCTION},
            {"role": "assistant", "content": "（那段是压缩指令）"},
            {"role": "user", "content": "继续修那个缓存问题"}])
        self.assertEqual(R.match(self.rules, req), [])

    def test_a_final_turn_that_only_quotes_it_is_not_marked(self):
        req = compaction_request(messages=[
            {"role": "developer", "content": MAIN_SYSTEM},
            {"role": "user", "content": "日志里有句 " + '"' + COMPACTION_PREFIX +
                                        '..." ，它是什么意思？'}])
        self.assertEqual(R.match(self.rules, req), [])

    def test_the_system_prompt_is_not_a_user_turn(self):
        """The prefix in the system prompt is a different declaration (that is SYSTEM_TEXT's job)."""
        req = compaction_request(messages=[{"role": "developer", "content": COMPACTION_INSTRUCTION},
                                           {"role": "user", "content": "你好"}])
        self.assertEqual(R.match(self.rules, req), [])

    def test_a_request_without_a_user_turn_never_matches(self):
        for messages in ([], [{"role": "assistant", "content": COMPACTION_INSTRUCTION}],
                         [{"role": "user"}], [{"role": "user", "content": {"type": "image"}}],
                         "not a list", None):
            self.assertEqual(R.match(self.rules, {"messages": messages}), [], messages)

    def test_the_rule_carries_no_tokenizer_dependency(self):
        prepared, notes = R.prepare_text_rules(self.rules, encode=None)
        self.assertEqual(prepared.rule_ids(), ("ds-compaction",))
        self.assertEqual(notes, [])


class TheDecision(unittest.TestCase):
    def test_a_hit_asks_for_the_checkpoint_field(self):
        rules = markers(TITLE_RULE)
        hits, key = R.decision(rules, title_request())
        self.assertEqual(hits, ["ds-title"])
        self.assertEqual(key, R.CHECKPOINT_FIELD)

    def test_a_miss_asks_for_nothing(self):
        self.assertEqual(R.decision(markers(TITLE_RULE), main_request()), ([], None))

    def test_an_unimplemented_action_is_refused_at_startup(self):
        with self.assertRaises(R.MarkerConfigError):
            markers(TITLE_RULE, action="nosave")

    def test_no_config_at_all_is_the_release_behaviour(self):
        parsed = R.parse(None)
        self.assertFalse(parsed.enabled)
        self.assertEqual(parsed.rules, ())
        self.assertEqual(R.match(parsed, title_request(strata_checkpoint=False)), [])


class ConfigContract(unittest.TestCase):
    """A bad config must fail at startup, and it must say why."""

    def bad(self, block, needle=None):
        with self.assertRaises(R.MarkerConfigError) as caught:
            R.parse(block)
        if needle:
            self.assertIn(needle, str(caught.exception))

    def test_unknown_top_level_keys(self):
        self.bad({"enabled": True, "rules": [TITLE_RULE], "mode": "auto"}, "unknown key")

    def test_unknown_rule_keys(self):
        self.bad({"rules": [dict(TITLE_RULE, when="always")]}, "unknown key")

    def test_a_heuristic_key_is_refused_by_name(self):
        self.bad({"rules": [{"id": "x", "type": "field", "path": "a", "max_tokens": 64}]},
                 "guess which requests are one-shot")
        self.bad({"rules": [{"id": "x", "type": "system_text", "prefix": "p", "min_tokens": 1}]},
                 "guess")
        self.bad({"rules": [{"id": "x", "type": "field", "path": "a", "idle_s": 60}]}, "guess")

    def test_an_unknown_type(self):
        self.bad({"rules": [{"id": "x", "type": "regex", "prefix": "p"}]}, "not a rule type")

    def test_an_empty_text_prefix(self):
        self.bad({"rules": [{"id": "x", "type": "system_text", "prefix": "   "}]}, "needs the text")

    def test_a_request_kind_needs_its_value(self):
        self.bad({"rules": [{"id": "x", "type": "request_kind"}]}, "needs the kind")

    def test_an_empty_field_path(self):
        self.bad({"rules": [{"id": "x", "type": "field", "path": ""}]}, "needs a field path")
        self.bad({"rules": [{"id": "x", "type": "field", "path": "a..b"}]}, "empty segment")

    def test_keys_belonging_to_another_type(self):
        self.bad({"rules": [{"id": "x", "type": "field", "path": "a", "prefix": "p"}]}, "only a")
        self.bad({"rules": [{"id": "x", "type": "system_text", "prefix": "p", "path": "a"}]}, "only a")
        self.bad({"rules": [{"id": "x", "type": "system_text", "prefix": "p", "equals": "p"}]},
                 "prefix, not an exact value")
        self.bad({"rules": [{"id": "x", "type": "user_text", "prefix": "p", "path": "a"}]}, "only a")
        self.bad({"rules": [{"id": "x", "type": "user_text", "prefix": "p", "equals": "p"}]},
                 "prefix, not an exact value")

    def test_a_user_text_rule_takes_no_match_mode_and_no_guard(self):
        self.bad({"rules": [{"id": "x", "type": "user_text", "prefix": "p", "match": "tokens"}]},
                 "compares text only")
        self.bad({"rules": [{"id": "x", "type": "user_text", "prefix": "p", "max_prompt_tokens": 512}]},
                 "only a system_text rule")

    def test_an_empty_user_text_prefix(self):
        self.bad({"rules": [{"id": "x", "type": "user_text", "prefix": "   "}]}, "needs the text")

    def test_a_duplicate_rule_id(self):
        self.bad({"rules": [TITLE_RULE, dict(TITLE_RULE)]}, "used twice")

    def test_bad_ids(self):
        for rid in ("", "  ", "9 ok but space", "x" * 65, 7, None, "../etc/passwd"):
            self.bad({"rules": [dict(TITLE_RULE, id=rid)]}, "short identifier")

    def test_enabled_with_no_rules(self):
        self.bad({"enabled": True, "rules": []}, "no rules")

    def test_wrong_types(self):
        self.bad({"enabled": "yes", "rules": [TITLE_RULE]}, "true or false")
        self.bad({"log": 1, "rules": [TITLE_RULE]}, "true or false")
        self.bad({"rules": {}}, "expected a list")
        self.bad({"rules": ["ds-title"]}, "expected an object")
        self.bad({"rules": [dict(TITLE_RULE, description=7)]}, "expected a string")

    def test_an_unknown_match_mode(self):
        self.bad({"rules": [dict(TITLE_RULE, match="regex")]}, "not a way to compare")

    def test_a_guard_must_be_a_positive_integer(self):
        self.bad({"rules": [dict(TITLE_RULE, max_prompt_tokens=0)]}, "positive integer")
        self.bad({"rules": [dict(TITLE_RULE, max_prompt_tokens=True)]}, "positive integer")
        self.bad({"rules": [dict(TITLE_RULE, max_prompt_tokens="512")]}, "positive integer")

    def test_no_block_means_disabled(self):
        """The normal case: a config without the block is the release behaviour, unchanged."""
        self.assertFalse(R.parse({"args": ["--serve"], "port": 4321}.get("oneshot_markers")).enabled)
        self.assertFalse(R.parse(None).enabled)

    def test_a_valid_block_parses(self):
        parsed = R.parse({"enabled": True, "log": False, "action": "ckpt0",
                          "rules": [TITLE_RULE, CODEX_RULE, FLAG_RULE, COMPACTION_RULE]})
        self.assertTrue(parsed.enabled)
        self.assertFalse(parsed.log)
        self.assertEqual(parsed.rule_ids(),
                         ("ds-title", "codex-compaction", "client-flag", "ds-compaction"))


class TextAnchorPreparation(unittest.TestCase):
    """A rule that cannot be prepared is disabled, not fatal."""

    def test_without_a_tokenizer_the_text_rule_is_dropped(self):
        prepared, notes = R.prepare_text_rules(markers(dict(TITLE_RULE, match="tokens"), CODEX_RULE), encode=None)
        self.assertEqual(prepared.rule_ids(), ("codex-compaction",))
        self.assertTrue(any("no tokenizer" in n for n in notes))

    def test_an_encoder_that_raises_only_kills_its_own_rule(self):
        def encode(text):
            raise RuntimeError("no such token")

        prepared, notes = R.prepare_text_rules(markers(dict(TITLE_RULE, match="tokens"), CODEX_RULE), encode=encode)
        self.assertEqual(prepared.rule_ids(), ("codex-compaction",))
        self.assertIn("cannot tokenize", notes[0])

    def test_a_prefix_that_encodes_to_nothing(self):
        prepared, notes = R.prepare_text_rules(markers(dict(TITLE_RULE, match="tokens")), encode=lambda text: [])
        self.assertEqual(prepared.rule_ids(), ())
        self.assertIn("no token", notes[0])

    def test_a_dropped_anchor_makes_the_rule_match_nothing(self):
        """The strict path never falls back to text when it has ids - it just does not match."""
        prepared, notes = R.prepare_text_rules(markers(dict(TITLE_RULE, match="tokens")), encode=None)
        self.assertEqual(prepared.rule_ids(), ())
        self.assertEqual(R.match(prepared, title_request(), prompt_ids=list(TITLE_ANCHOR)), [])

    def test_the_anchor_is_kept_per_rule(self):
        prepared, _ = R.prepare_text_rules(
            markers(dict(TITLE_RULE, match="tokens"),
                    {"id": "other", "type": "system_text", "prefix": "Other system", "match": "tokens"}),
            encode=lambda text: (1, 2, 3) if text.startswith("Create") else (4, 5, 6))
        self.assertEqual([r.anchor for r in prepared.text_rules()], [(1, 2, 3), (4, 5, 6)])


class SystemTextExtraction(unittest.TestCase):
    def test_the_top_level_field(self):
        self.assertEqual(R.system_text({"system": "  hello  "}), "hello")

    def test_the_anthropic_block_form(self):
        self.assertEqual(R.system_text({"system": [{"type": "text", "text": "a"},
                                                   {"type": "text", "text": "b"}]}), "a\nb")

    def test_the_in_history_form(self):
        self.assertEqual(R.system_text({"messages": [{"role": "system", "content": "s"},
                                                     {"role": "user", "content": "u"}]}), "s")

    def test_the_top_level_field_wins(self):
        self.assertEqual(R.system_text({"system": "top",
                                        "messages": [{"role": "system", "content": "in-history"}]}),
                         "top")

    def test_a_single_block_and_a_mixed_list(self):
        """The bug the first run of this file caught: a lone block (a dict) produced no text at all."""
        self.assertEqual(R.system_text({"system": {"type": "text", "text": "lone"}}), "lone")
        self.assertEqual(R.system_text({"messages": [{"role": "system", "content":
                                                      {"type": "text", "text": "one"}}]}), "one")
        mixed = {"system": [{"type": "text", "text": "a"},
                            {"type": "image", "source": "x"},
                            {"type": "text", "text": "b"}]}
        self.assertEqual(R.system_text(mixed), "a\nb")
        self.assertEqual(R.system_text({"system": {"type": "image"}}), "")
        self.assertEqual(R.system_text({"system": [{"not": "a block"}]}), "")

    def test_nothing_to_read(self):
        for req in ({}, {"messages": []}, {"messages": [{"role": "user", "content": "u"}]},
                    {"system": 7}, {"system": [{"type": "image"}]}, "not a dict", None):
            self.assertEqual(R.system_text(req), "")


class StartupLog(unittest.TestCase):
    def test_the_lines_say_what_will_be_marked(self):
        prepared, _ = R.prepare_text_rules(markers(TITLE_RULE, SUBAGENT_RULE), encode=None)
        text = "\n".join(R.describe(prepared))
        self.assertIn("oneshot markers on", text)
        self.assertIn("ds-title", text)
        self.assertIn("ds-subagent", text)
        self.assertIn("system text starts with", text)
        self.assertIn("field dsh_session_log.session.origin", text)

    def test_disabled_reads_as_off(self):
        self.assertIn("oneshot markers off", "\n".join(R.describe(markers(enabled=False))))

    def test_the_compaction_rule_says_what_it_reads(self):
        text = "\n".join(R.describe(markers(COMPACTION_RULE)))
        self.assertIn("ds-compaction", text)
        self.assertIn("final user message starts with", text)


if __name__ == "__main__":
    unittest.main()
