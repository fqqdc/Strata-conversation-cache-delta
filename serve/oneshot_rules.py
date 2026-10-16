"""One-shot request markers: the rule set that says "the conversation this request leaves behind
will never be resumed with this suffix", read from the run config's `oneshot_markers` block.

Why this exists: with --conversation-cache-disk-only a
conversation is streamed to the spill directory when the next request switches it out.  Some
requests are one-shot by construction - a session-title call, a Codex compaction, any auxiliary
call whose prompt is never sent again - so writing their state to disk is pure waste (measured on
this machine: 130 tokens -> a 226 MiB file, twice per session).  The engine cannot tell: at save
time it only has tokens.  This module is the serve side's answer, and it is deliberately
**config-driven**: adding a marker is a config edit, not a code change.

The second class is the client's own context compaction.  DSH's auxiliary summarizer call replays the
conversation and appends ONE directive as the final user turn - deliberately, "so the provider's KV
cache is reused instead of invalidated" (its own comment on COMPACTION_INSTRUCTION) - and then the
branch it leaves is never continued: the client replaces the summarized range with the summary, so
the next prompt diverges from it.  Measured on this box (2026-10-16): prompt 22943 tokens = 22546
reused + 397 read, 1643 generated; at the NEXT request's switch the engine parked that branch as
`disk-saved 24587 tokens (436 MiB, switch)` and it was never read back (the following request
mounted 16384 tokens from the in-RAM checkpoint chain instead, with no disk read at all).  The
`user_text` rule type names that directive.

The only action implemented is "ckpt0" - the request is sent to the engine with `ckpt=0`, which
already means "this call's turn is not continued, not parked, and (in disk mode) not saved, while
a cached prefix is still reused" (src/program/generate.cpp).  Actions that are not implemented are
refused at startup rather than accepted and ignored.

Nothing here imports `serve.server`: the module is pure (config in, decision out) so it can be
unit-tested without an engine, a tokenizer or a GPU.

Design rules kept on purpose:

* **No heuristics.**  A rule matches a *declared identity* (a request-body field, a Codex request
  kind, a fixed system-prompt text), never a size, a message count, an idle time or a token
  length.  `SYSTEM_TEXT`'s optional `max_prompt_tokens` is a *guard* that can only exclude after
  the text already matched; it never selects a request.
* **Fail loud on a bad config.**  An unknown key, an unknown type, a heuristic key smuggled into a
  rule, or an unimplemented action raises `MarkerConfigError` at startup.  A typo in a config must
  not silently change caching behaviour.
* **Fail safe on a missing capability.**  A SYSTEM_TEXT rule whose text cannot be prepared is
  disabled with a warning; it never fails a request.
"""

from __future__ import annotations   # `str`/`text` are shadowed below: annotations must stay strings

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

# The rule types.  Each is a way a client *declares* what a request is for.
FIELD = "field"                 # a request-body field (dot path), optionally with an expected value
REQUEST_KIND = "request_kind"   # Codex's x-codex-turn-metadata request_kind
SYSTEM_TEXT = "system_text"     # the request's system prompt text starts with a fixed prefix
USER_TEXT = "user_text"         # the request's FINAL user turn starts with a fixed prefix

RULE_TYPES = (FIELD, REQUEST_KIND, SYSTEM_TEXT, USER_TEXT)

# The actions.  Only "ckpt0" exists today; anything else is refused (see the module docstring).
ACTION_CKPT0 = "ckpt0"
ACTIONS = (ACTION_CKPT0,)

# What the decision feeds into: the request dict key `sampling_keys()` already understands
# (serve/server.py: the `strata_checkpoint is False` branch).  Writing this is equivalent to the
# client having sent the field itself.
CHECKPOINT_FIELD = "strata_checkpoint"

# The engine key ckpt=0 is only about the *turn*, and a very long prompt was never a title call:
# a SYSTEM_TEXT rule may carry this guard (opt-in, absent = no guard).  It can only exclude.
DEFAULT_MAX_PROMPT_TOKENS = 512

ALLOWED_TOP = {"enabled", "log", "probe", "action", "rules"}
ALLOWED_RULE = {"id", "type", "description", "equals", "path", "prefix", "max_prompt_tokens", "match"}

# How a SYSTEM_TEXT rule compares.  "text" (the default) compares the declared text with the system
# prompt text the request carries.  "tokens" compares prompt token ids with the rule's anchor.
#
# Measured on this deployment (2026-10-16): "tokens" is NOT usable for the template-rendered shape.
# The engine's prompt starts with the chat template's `<|im_start|>system\n` ([248045, 8678, 198]),
# and encoding the system text on its own reproduces the rest of the block only in part: rebuilding
# `<|im_start|>system\n<text>` agreed with the recorded prompt for 3 tokens out of 205 (the
# tokenizer's merges differ across the template boundary).  "text" has no such problem: the body
# carries the system prompt verbatim, and the recorded prompt's system text begins exactly with the
# configured prefix.  Keep "tokens" for a deployment whose prompt starts at the system text itself.
MATCH_TEXT = "text"
MATCH_TOKENS = "tokens"
MATCHES = (MATCH_TEXT, MATCH_TOKENS)

# Keys that would turn this into a guess.  Refused by name so the error says what to do instead.
HEURISTIC_KEYS = (
    "max_tokens", "min_tokens", "tokens", "token_count", "prompt_tokens", "max_prompt_tokens_gte",
    "messages", "message_count", "n_messages", "idle_s", "idle_seconds", "age_s", "similarity",
    "length", "chars", "first_request", "is_first", "round", "turn", "step",
)

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")

# This module has `text` parameters and locals, which shadow the builtin `str` inside those
# functions; the alias keeps the isinstance checks readable and safe (see _text_of).
_STR = str


class MarkerConfigError(ValueError):
    """The `oneshot_markers` block is not usable: raised at startup, never per request."""


@dataclass(frozen=True)
class Rule:
    id: str
    type: str
    description: str = ""
    path: str = ""                  # FIELD
    equals: Any = None              # FIELD, REQUEST_KIND
    has_equals: bool = False
    prefix: str = ""                # SYSTEM_TEXT
    match: str = MATCH_TEXT         # SYSTEM_TEXT: "text" (default) or "tokens"
    max_prompt_tokens: int | None = None   # SYSTEM_TEXT guard (optional)
    anchor: tuple[int, ...] | None = None  # SYSTEM_TEXT "tokens" mode, filled in by prepare_text_rules()


@dataclass(frozen=True)
class Markers:
    """The parsed, ready-to-use rule set.  `enabled` False means "exactly as before this feature"."""
    enabled: bool = False
    log: bool = True
    probe: bool = False               # DIAG: print what each request's body looks like to the rules
    action: str = ACTION_CKPT0
    rules: tuple[Rule, ...] = field(default_factory=tuple)

    def text_rules(self) -> tuple[Rule, ...]:
        return tuple(r for r in self.rules if r.type == SYSTEM_TEXT)

    def rule_ids(self) -> tuple[str, ...]:
        return tuple(r.id for r in self.rules)


def parse(cfg: Any) -> Markers:
    """The `oneshot_markers` block -> Markers.  Raises MarkerConfigError on anything unusable.

    `None` (no such block) is the normal case for every config that has not opted in, and yields
    a disabled, empty rule set: the default path stays byte-identical to the release.
    """
    if cfg is None:
        return Markers(enabled=False)
    if not isinstance(cfg, dict):
        raise MarkerConfigError(f"oneshot_markers: expected an object, got {type(cfg).__name__}")

    unknown = sorted(set(cfg) - ALLOWED_TOP)
    if unknown:
        raise MarkerConfigError(f"oneshot_markers: unknown key(s) {', '.join(unknown)}; "
                                f"known: {', '.join(sorted(ALLOWED_TOP))}")

    enabled = cfg.get("enabled", False)
    if not isinstance(enabled, bool):
        raise MarkerConfigError(f"oneshot_markers.enabled: expected true or false, got {enabled!r}")
    log = cfg.get("log", True)
    if not isinstance(log, bool):
        raise MarkerConfigError(f"oneshot_markers.log: expected true or false, got {log!r}")

    probe = cfg.get("probe", False)
    if not isinstance(probe, bool):
        raise MarkerConfigError(f"oneshot_markers.probe: expected true or false, got {probe!r}")
    action = cfg.get("action", ACTION_CKPT0)
    if action not in ACTIONS:
        raise MarkerConfigError(f"oneshot_markers.action: {action!r} is not implemented; "
                                f"implemented: {', '.join(ACTIONS)}")

    raw_rules = cfg.get("rules", [])
    if not isinstance(raw_rules, list):
        raise MarkerConfigError("oneshot_markers.rules: expected a list of rules")
    rules: list[Rule] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_rules):
        rule = _parse_rule(raw, index)
        if rule.id in seen:
            raise MarkerConfigError(f"oneshot_markers.rules[{index}].id: {rule.id!r} is used twice")
        seen.add(rule.id)
        rules.append(rule)
    if enabled and not rules:
        raise MarkerConfigError("oneshot_markers: enabled with no rules; nothing would ever be marked")
    return Markers(enabled=enabled, log=log, probe=probe, action=action, rules=tuple(rules))


def _parse_rule(raw: Any, index: int) -> Rule:
    where = f"oneshot_markers.rules[{index}]"
    if not isinstance(raw, dict):
        raise MarkerConfigError(f"{where}: expected an object, got {type(raw).__name__}")

    for key in sorted(set(raw) & set(HEURISTIC_KEYS)):
        raise MarkerConfigError(
            f"{where}.{key}: this would guess which requests are one-shot; markers must name a "
            f"declared identity instead (a body field, a request kind, or a fixed text)")

    unknown = sorted(set(raw) - ALLOWED_RULE)
    if unknown:
        raise MarkerConfigError(f"{where}: unknown key(s) {', '.join(unknown)}; "
                                f"known: {', '.join(sorted(ALLOWED_RULE))}")

    rid = raw.get("id")
    if not isinstance(rid, str) or not _ID_RE.match(rid):
        raise MarkerConfigError(f"{where}.id: expected a short identifier "
                                f"(letters, digits, . _ : -), got {rid!r}")

    rtype = raw.get("type")
    if rtype not in RULE_TYPES:
        raise MarkerConfigError(f"{where}.type: {rtype!r} is not a rule type; "
                                f"known: {', '.join(RULE_TYPES)}")

    desc = raw.get("description", "")
    if not isinstance(desc, str):
        raise MarkerConfigError(f"{where}.description: expected a string, got {desc!r}")

    common = dict(id=rid, type=rtype, description=desc)

    if rtype == FIELD:
        path = raw.get("path")
        if not isinstance(path, str) or not path:
            raise MarkerConfigError(f"{where}.path: a field rule needs a field path, e.g. "
                                    f'"strata_checkpoint" or "dsh_session_log.session.origin"')
        if any(not part for part in path.split(".")):
            raise MarkerConfigError(f"{where}.path: {path!r} has an empty segment")
        if "prefix" in raw:
            raise MarkerConfigError(f"{where}.prefix: only a {SYSTEM_TEXT} rule has a prefix")
        has_equals = "equals" in raw
        if has_equals and not _scalar(raw["equals"]):
            raise MarkerConfigError(f"{where}.equals: expected a string, number or boolean, "
                                    f"got {type(raw['equals']).__name__}")
        return Rule(path=path, equals=raw.get("equals"), has_equals=has_equals, **common)

    if rtype == REQUEST_KIND:
        if "path" in raw or "prefix" in raw:
            raise MarkerConfigError(f"{where}: a {REQUEST_KIND} rule has no path or prefix")
        equals = raw.get("equals")
        if not isinstance(equals, str) or not equals:
            raise MarkerConfigError(f'{where}.equals: a request-kind rule needs the kind to match, '
                                    f'e.g. "compaction"')
        return Rule(equals=equals, has_equals=True, **common)

    if rtype == USER_TEXT:
        if "path" in raw:
            raise MarkerConfigError(f"{where}.path: only a {FIELD} rule has a path")
        prefix = raw.get("prefix")
        if not isinstance(prefix, str) or not prefix.strip():
            raise MarkerConfigError(f"{where}.prefix: a user-text rule needs the text to match")
        if "equals" in raw:
            raise MarkerConfigError(f"{where}.equals: a user-text rule matches a prefix, not an "
                                    f"exact value")
        if "match" in raw:
            raise MarkerConfigError(f"{where}.match: a user-text rule compares text only "
                                    f"(there is no token form of a message the client typed)")
        if "max_prompt_tokens" in raw:
            raise MarkerConfigError(f"{where}.max_prompt_tokens: only a {SYSTEM_TEXT} rule carries "
                                    f"the guard (a directive turn is not a short one)")
        return Rule(prefix=prefix, **common)

    # SYSTEM_TEXT
    if "path" in raw:
        raise MarkerConfigError(f"{where}.path: only a {FIELD} rule has a path")
    prefix = raw.get("prefix")
    if not isinstance(prefix, str) or not prefix.strip():
        raise MarkerConfigError(f"{where}.prefix: a system-text rule needs the text to match")
    if "equals" in raw:
        raise MarkerConfigError(f"{where}.equals: a system-text rule matches a prefix, not an "
                                f"exact value")
    guard = raw.get("max_prompt_tokens")
    if guard is not None:
        if not isinstance(guard, int) or isinstance(guard, bool) or guard <= 0:
            raise MarkerConfigError(f"{where}.max_prompt_tokens: expected a positive integer "
                                    f"(a guard, absent = no guard), got {guard!r}")
    how = raw.get("match", MATCH_TEXT)
    if how not in MATCHES:
        raise MarkerConfigError(f"{where}.match: {how!r} is not a way to compare; "
                                f"known: {', '.join(MATCHES)}")
    return Rule(prefix=prefix, match=how, max_prompt_tokens=guard, **common)


def _scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool))


def prepare_text_rules(markers: Markers, encode=None) -> tuple[Markers, list[str]]:
    """Give every token-matched SYSTEM_TEXT rule its anchor; -> (markers, warnings).

    Only rules with `match: "tokens"` need an encoder (the engine's tokenizer).  A rule whose anchor
    cannot be prepared is dropped with a note - a config problem must not fail requests.  The default
    `match: "text"` needs nothing and is passed through untouched.
    """
    notes: list[str] = []
    rules: list[Rule] = []
    for rule in markers.rules:
        if rule.type != SYSTEM_TEXT or rule.match != MATCH_TOKENS:
            rules.append(rule)
            continue
        if encode is None:
            notes.append(f"oneshot_markers rule {rule.id!r} disabled: no tokenizer to prepare it with")
            continue
        try:
            anchor = tuple(int(t) for t in encode(rule.prefix))
        except Exception as exc:                       # a tokenizer that cannot encode this text
            notes.append(f"oneshot_markers rule {rule.id!r} disabled: cannot tokenize its prefix "
                         f"({exc})")
            continue
        if not anchor:
            notes.append(f"oneshot_markers rule {rule.id!r} disabled: its prefix encodes to no token")
            continue
        rules.append(Rule(id=rule.id, type=rule.type, description=rule.description,
                          prefix=rule.prefix, match=rule.match,
                          max_prompt_tokens=rule.max_prompt_tokens, anchor=anchor))
    return Markers(enabled=markers.enabled, log=markers.log, probe=markers.probe,
                   action=markers.action, rules=tuple(rules)), notes


def system_text(req: Mapping[str, Any]) -> str:
    """The request's system prompt as text: the top-level `system` field (Anthropic, or an OpenAI
    request that sends it that way) or the system turns of `messages`.

    "System turns" means `role: "system"` **and `role: "developer"`**: Strata's own OpenAI adapter
    maps developer -> system (serve/frontend.py:526, issue #56), and a client that sends its system
    prompt that way - DSH does - would otherwise be invisible to every text rule.  Measured on this
    box: DSH's session-title call arrives as `["developer:<title text>", "user:<...>"]`, with no
    top-level `system` field at all.
    """
    if not isinstance(req, Mapping):
        return ""
    top = req.get("system")
    if isinstance(top, (_STR, Mapping, list)):         # text, one block, or the Anthropic block list
        text = _text_of(top)
        if text.strip():
            return text.strip()
    messages = req.get("messages")
    if isinstance(messages, list):
        parts = []
        for message in messages:
            if not isinstance(message, Mapping) or message.get("role") not in _SYSTEM_ROLES:
                continue
            text = _text_of(message.get("content"))
            if text:
                parts.append(text)
        joined = "\n".join(parts)
        if joined.strip():
            return joined.strip()
    return ""


_SYSTEM_ROLES = ("system", "developer")   # the adapter's own mapping (serve/frontend.py:526)


def user_text(req: Mapping[str, Any]) -> str:
    """The request's FINAL user turn as text: the last `messages` entry with `role: "user"`.

    "Final" is the point.  A client delivers a *directive* as the turn it wants answered and keeps the
    conversation in front of it: DSH's compaction call is built as
    `[...input.messages, {role: "user", content: [{type: "text", text: COMPACTION_INSTRUCTION}]}]`, so
    the instruction is the last message, while the instruction itself is private to that call and is
    never replayed into a later conversation ("only returned text is stored").  Reading the LAST user
    turn - not every user turn - keeps a conversation that merely mentions such a directive from
    matching: DSH sends system reminders as user turns too, so a text that appears earlier in a history
    is not this request's instruction.
    """
    if not isinstance(req, Mapping):
        return ""
    messages = req.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        return _text_of(message.get("content")).strip()
    return ""


def _text_of(content: Any) -> str:
    """The text of one message's content: a plain string, one content block ({"type": "text",
    "text": ...}), or a list of blocks joined.  Anything else (an image block, a number) is "".

    NB: this module has `text` locals and parameters that shadow builtins; only the `_STR` alias is
    safe for the isinstance checks here.
    """
    if isinstance(content, _STR):
        return content
    if isinstance(content, Mapping):                   # one block, e.g. Anthropic's system list
        block = content
        if block.get("type") == "text":
            value = block.get("text")
            if isinstance(value, _STR):
                return value
        return ""
    if isinstance(content, list):
        out = []
        for block in content:
            text = _text_of(block)
            if text:
                out.append(text)
        return "\n".join(out)
    return ""


def field_value(req: Any, path: str) -> tuple[bool, Any]:
    """`("a.b".split) -> (found, value)`; a missing path or a non-object step is (False, None)."""
    node = req
    for part in path.split("."):
        if isinstance(node, Mapping):
            if part not in node:
                return False, None
            node = node[part]
            continue
        if isinstance(node, (list, tuple)) and part.isdigit():
            index = int(part)
            if index >= len(node):
                return False, None
            node = node[index]
            continue
        return False, None
    return True, node


def request_kind(req: Mapping[str, Any]) -> str | None:
    """Codex's turn metadata.  Parsed here the same way serve/responses.py does (a JSON object in
    the x-codex-turn-metadata field, or the same keys flat on the request); kept local so this
    module stays free of serve imports."""
    import json

    raw = req.get("x-codex-turn-metadata") if isinstance(req, Mapping) else None
    kind = None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = None
        if isinstance(parsed, Mapping):
            kind = parsed.get("request_kind")
    elif isinstance(raw, Mapping):
        kind = raw.get("request_kind")
    if isinstance(kind, str) and kind:
        return kind
    flat = req.get("request_kind") if isinstance(req, Mapping) else None
    return flat if isinstance(flat, str) and flat else None


def match(markers: Markers, req: Any, prompt_ids: Sequence[int] | None = None) -> list[str]:
    """Every rule id this request matches, in config order.  Empty = not one-shot.

    `prompt_ids` is the request's tokenized prompt (for SYSTEM_TEXT anchors and their guard); it
    may be None, in which case SYSTEM_TEXT rules that carry a guard cannot be checked and are
    skipped, and the others are checked against the prompt's start only if ids are available.
    """
    if not markers.enabled or not markers.rules or not isinstance(req, Mapping):
        return []
    hits: list[str] = []
    text = None
    final_user = None
    for rule in markers.rules:
        if rule.type == FIELD:
            found, value = field_value(req, rule.path)
            if not found:
                continue
            if rule.has_equals and not _same(value, rule.equals):
                continue
            hits.append(rule.id)
        elif rule.type == REQUEST_KIND:
            if request_kind(req) == rule.equals:
                hits.append(rule.id)
        elif rule.type == SYSTEM_TEXT:
            if rule.max_prompt_tokens is not None and prompt_ids is not None \
                    and len(prompt_ids) > rule.max_prompt_tokens:
                continue                               # the guard excludes; it never selects
            if rule.match == MATCH_TOKENS:
                # compare the prompt's own tokens with the prepared anchor (a rule whose anchor
                # could not be prepared never matches here, and never falls back to text)
                if prompt_ids is None or not rule.anchor or len(prompt_ids) < len(rule.anchor):
                    continue
                if tuple(int(t) for t in prompt_ids[:len(rule.anchor)]) != rule.anchor:
                    continue
            else:
                if text is None:
                    text = system_text(req)
                if not text.startswith(rule.prefix):
                    continue
            hits.append(rule.id)
        elif rule.type == USER_TEXT:
            if final_user is None:
                final_user = user_text(req)
            if not final_user.startswith(rule.prefix):
                continue
            hits.append(rule.id)
    return hits


def _same(value: Any, expected: Any) -> bool:
    """Equality that keeps JSON's types apart: `True` is not 1 and not "true"."""
    if isinstance(expected, bool) or isinstance(value, bool):
        return isinstance(value, bool) and isinstance(expected, bool) and value is expected
    if isinstance(expected, (int, float)) and isinstance(value, (int, float)):
        return float(value) == float(expected)
    if isinstance(expected, str) and isinstance(value, str):
        return value == expected
    return False


def decision(markers: Markers, req: Any, prompt_ids: Sequence[int] | None = None) -> tuple[list[str], str | None]:
    """-> (matched rule ids, the request-body key to set) or ([], None) for an ordinary request.

    The caller sets the key on its own copy of the sampling dict; that is the whole integration.
    """
    hits = match(markers, req, prompt_ids)
    if not hits:
        return [], None
    if markers.action == ACTION_CKPT0:
        return hits, CHECKPOINT_FIELD
    raise MarkerConfigError(f"oneshot_markers.action: {markers.action!r} is not implemented")


def describe(markers: Markers) -> Iterable[str]:
    """Lines a startup log prints, so the server's log says what it will mark."""
    state = "on" if markers.enabled else "off"
    yield f"oneshot markers {state}: action={markers.action} log={bool(markers.log)} rules={len(markers.rules)}"
    for rule in markers.rules:
        if rule.type == FIELD:
            what = f"field {rule.path}" + (f" == {rule.equals!r}" if rule.has_equals else " exists")
        elif rule.type == REQUEST_KIND:
            what = f"request_kind == {rule.equals!r}"
        elif rule.type == USER_TEXT:
            what = f"final user message starts with {rule.prefix[:48]!r}"
        else:
            anchor = "" if not rule.anchor else f" [{len(rule.anchor)} tokens]"
            guard = "" if rule.max_prompt_tokens is None else f" (<= {rule.max_prompt_tokens} prompt tokens)"
            what = (f"system text starts with {rule.prefix[:48]!r} "
                    f"({rule.match}{anchor}){guard}")
        yield f"  {rule.id}: {what}" + (f" - {rule.description}" if rule.description else "")
