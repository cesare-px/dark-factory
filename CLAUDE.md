# dark-factory

## What this is

Central orchestrator (Repo A) for the Dark Software Factory: an autonomous,
multi-agent software-engineering pipeline driven by GitHub Issues. A ticket
(GitHub issue) runs through four real LLM agents in sequence -- Validator,
Planner, Developer, Reviewer -- and on approval, Developer's work gets
committed, pushed, and opened as a PR automatically.

**Project framing**: personal learning/experimentation project. Optimize for
interesting capability and understanding, not adoption polish, OSS onboarding,
or fitting a specific team's existing workflow. Provider-agnostic in
principle (Anthropic, OpenAI, or any OpenAI-compatible endpoint), but
agentic tool use is Anthropic-only for now -- a deliberate, named gap for the
others, not an oversight.

## North star: self-hosting

**Explicit goal, not just aspirational**: dark-factory should eventually be
able to meaningfully open real, mergeable PRs against *its own* repository.
This is a deliberately high bar -- multi-file, judgment-heavy,
security-sensitive work -- and it's the best available test of whether the
pipeline is actually capable yet, versus just able to handle toy fixtures.
See `ROADMAP.md` for the concrete maturity ladder and the ranked backlog of
work to get there.

**Corollary -- be careful about self-modifying safety code.** A ticket asking
dark-factory to modify its own guardrail/sandboxing code (`agents/tools.py`,
`vcs.py`'s auth checks, the budget tracker in `guardrails/`) is a real "who
watches the watchers" risk. Any PR touching those files deserves a human's
eyes specifically, always -- don't let self-hosting ambition erode that.

## Current maturity

Proven against real models (not just scripted mocks -- see "How we like to
work" below for why that distinction matters here):
- Create a new file with exact, specified content.
- Precisely edit **one existing function** in a multi-function file via
  `edit_file`'s unique-match search/replace, leaving sibling code
  byte-for-byte untouched -- validated with a real bug fix, not a toy.
- Ship a real PR on approval (commit, push, open PR via the GitHub REST
  API), gated on the Reviewer's own APPROVE/REJECT verdict -- not a
  string-matching heuristic; one was tried and explicitly rejected as
  overfit (see below).
- A per-ticket command-permission system: issue checkboxes can request
  extra shell-command prefixes beyond the safe defaults, honored only if
  GitHub confirms the checkbox-checker has write access -- never trusted
  from ticket text alone.

**Not yet capable of** (see `ROADMAP.md` for the plan to close these):
genuine multi-file architectural coordination; any follow-up on a human's
PR review comments (an issue can become a PR, but review comments on that
PR currently go nowhere -- intake only understands `issues` events); or a
ticket too large for the current tool-use/build-test-fix iteration caps.

## How we like to work on this repo

Collaboration patterns validated over the course of building this, worth
repeating rather than rediscovering:

- **Plan Mode for anything architecturally significant.** Multi-file
  changes, new capability, or a real "which approach" fork (e.g.
  hand-rolled-vs-SDK for tool use, where "shipping" belongs in the pipeline)
  go through an explicit plan with trade-offs surfaced, not straight to
  code.
- **Real-model validation beats a mocked test, every time.** A unit test
  with a scripted response proves the *mechanics* work; it does not prove
  the *capability* is real. Every major feature here (shipping, tool use,
  `edit_file`) was validated with an actual model call against a real
  scratch repo before being called done, in addition to (never instead of)
  unit tests.
- **Distrust a heuristic reverse-engineered from one failing example.**
  This happened once already: the Reviewer's acceptance-criteria matching
  was patched with a quoted-string/filename heuristic that happened to fix
  the one test case in front of us, then called out as overfit and replaced
  with the model's own semantic verdict instead. If a fix only works
  because it matches the shape of the example that motivated it, that's a
  signal to find the general mechanism, not tune the heuristic further.
- **A dedicated security-review pass on new attack surface is not
  optional.** Every time new tool-execution / path-handling /
  command-execution surface was added, an adversarial review pass (not
  self-review) found real, PoC-confirmable bugs: a `git grep` flag
  injection, `run_command`'s file-reading defaults with zero path
  containment, `list_files`' broken `..` containment, and a prefix-match
  word-boundary bypass. Budget time for this on anything touching
  `agents/tools.py`, `vcs.py`, or the permission system -- self-review alone
  has consistently missed things a dedicated pass caught.
- **Fix the root cause, even mid-session, over a narrower patch.** E.g.
  the whole JSON-patch mechanism was replaced outright once real tool use
  existed, rather than keeping both mechanisms alive.
- **`make check` (ruff + mypy strict + pytest) stays clean, no exceptions.**
  No `--no-verify`, no suppressed findings without a documented reason.
- **No local Anthropic API key in this environment.** Any validation that
  needs a real model call requires asking the user to run it and paste back
  the result -- never assume a key is available locally.

## Architecture, in one page

`README.md`'s own layout section predates this session's agentic tool-use
work and is now incomplete (no mention of `agents/`, `vcs.py`, etc.) --
treat this as the more current reference until that's refreshed.

```
src/dark_factory/
  intake/         webhook -> FactoryTicket, incl. HMAC verification,
                  prompt-injection sanitization, and per-ticket permission-
                  checkbox extraction (requested_permissions, sender_login)
  agents/
    pipeline.py     the 4-stage state machine (validator -> planner ->
                    developer -> reviewer), one shared budget tracker
    validator.py    structural pass/fail; its own LLM call is currently
                    decorative (ROADMAP FR-8)
    planner.py      explores via read-only tools, submits its own plan via
                    submit_plan -- no hardcoded plan_steps anymore
    developer.py    the build-test-fix loop; each attempt is a fresh
                    tool-use conversation ending on finish_implementation
    reviewer.py     ships (commit/push/PR) on its own APPROVE verdict,
                    gated by the security prefilter first
    tools.py        read_file/write_file/edit_file/list_files/grep/
                    run_command -- every path-taking tool shares one
                    containment check; this is the sandbox boundary
    tool_loop.py    the shared model-driven loop: call, dispatch tool
                    calls, feed results back, repeat until the model calls
                    its own designated finish tool
  llm/            provider-agnostic LLMClient + tool-call protocol
                  (ToolDefinition/ToolCall/ToolResultBlock); full tool
                  translation for Anthropic only; MeteredLLMClient is the
                  ONLY place spend is ever recorded, on every single call,
                  including every tool-use round trip
  vcs.py          the only module that shells out to git or calls GitHub's
                  REST API: git_diff, ship() (commit/push/PR), and
                  sender_has_write_access (the permission-system gate)
  config/         FactoryConfig dataclasses + loader; ToolUseSettings is
                  the tool-use budget + command-permission catalog
  guardrails/     TokenBudgetTracker ($ ceiling, checked before every LLM
                  call) and IterationLoopTracker (build-test-fix retry cap)
  harness.py      the configured test command (SubprocessHarness) plus a
                  ScriptedHarness for offline tests
  cli.py          `dark-factory doctor/run/init/config/prices` --
                  `init` is the interactive onboarding wizard (provider,
                  per-agent models, project type, permission catalog)
```

## Where the backlog lives

`ROADMAP.md` holds the current, ranked maturity ladder and feature-request
backlog -- each entry already written in the same `User Story` /
`Acceptance Criteria` shape as `.github/ISSUE_TEMPLATE/dark-factory-task.md`,
specifically so it can become a real GitHub issue with minimal editing.
