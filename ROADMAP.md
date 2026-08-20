# Roadmap: capability maturity and the self-hosting milestone

This is the working backlog for dark-factory. Framing (see `CLAUDE.md` for
the full context): personal capability-focused project, with **dark-factory
meaningfully opening real PRs against its own repository** as the explicit
north star, not just an aspiration.

Each feature request below is written in the same `User Story` /
`Acceptance Criteria` shape as `.github/ISSUE_TEMPLATE/dark-factory-task.md`,
specifically so it can become a real GitHub issue (`gh issue create`) with
minimal editing. `#N` labels are provisional ranking, not issue numbers.

## The self-hosting maturity ladder

A concrete, checkable progression rather than a vague goal. Each level should
be validated the way this repo already validates everything -- against a
real model, not just a scripted mock.

- [x] **L0 -- Greenfield file.** Create a new file with exact, specified
  content. (Proven: the `hello.txt` fixture.)
- [x] **L1 -- Precise single-function edit.** Fix a real bug in one function
  of an existing multi-function file via `edit_file`, leaving sibling code
  byte-for-byte untouched. (Proven: the `calc.py` `add()`/`multiply()`
  fixture.)
- [ ] **L2 -- Coordinated multi-file change.** A ticket that requires touching
  2-3 files coherently (e.g. add a small tool + wire it into an agent + add
  its test) and still converges within the existing iteration/budget caps.
- [ ] **L3 -- Self-verification via the project's own tooling.** A ticket
  where the harness is `make check` (lint + mypy strict + tests, not a
  synthetic one-liner) and the agent has to notice and fix its own lint/type
  error mid-loop, not just a test assertion failure.
- [ ] **L4 -- Self-hosting.** dark-factory opens a real PR against its own
  repository for a genuinely useful, non-trivial ticket, which a human
  reviews and merges as-is or with minor changes.
- [ ] **L5 -- Review round-trip.** A human leaves review comments on an
  L4-shaped PR, and a follow-up dark-factory run addresses them without
  starting from scratch. Blocked on FR-1 below (no `pull_request`/review
  event support today).

## Tier 1 -- blocking for L2 through L5

### FR-1: Support `pull_request` review events in intake

**Why this is ranked first**: right now a ticket can become a PR, but a
human's review comments on that PR go nowhere -- there is no path back into
the pipeline. This is the single biggest gap between "can open one PR" and
"can actually collaborate on one," and directly blocks L5.

```
## User Story
As the repo owner, I want to leave review comments on a dark-factory-opened
PR and have a follow-up run address them, so review feels like a normal
back-and-forth instead of a one-shot guess.

## Acceptance Criteria
- intake.parser supports event_type "pull_request" (currently
  SUPPORTED_EVENT_TYPES = {"issues"} only) for review_comment/synchronize
  actions on a dark-factory/* branch.
- The resulting FactoryTicket carries the review comment(s) as additional
  context, distinguishable from the original issue body.
- Developer's tool-use loop is seeded with "here's what changed since last
  time and here's the new feedback," not a from-scratch conversation.
- caller.yml's existing pull_request trigger + branch guard (already
  present, already unused) actually fires an end-to-end run.
```

### FR-2: Make `tool_use.max_iterations` overridable per-agent

**Why**: `loop.max_iterations` (the outer build-test-fix cap) is already
overridable per-agent via `agents.<name>.max_iterations`. The *inner*
tool-use budget (`tool_use.max_iterations`, currently a single global value,
default 8) is not -- so there's no way to give Developer more exploration
room for a genuinely large ticket without raising it for every agent,
including cheap gatekeeping ones that don't need it.

```
## User Story
As the repo owner, I want to give the Developer agent a bigger tool-use
budget than Planner/Validator/Reviewer without a global config change, so a
complex ticket has room to actually explore.

## Acceptance Criteria
- ToolUseSettings (or AgentSettings) supports a per-agent max_iterations
  override for the tool-use loop, following the same pattern
  max_iterations_for() already uses for the outer loop.
- developer.py and planner.py both resolve their own budget through this
  mechanism instead of reading tool_use.max_iterations directly.
- Documented in .dark-factory.yml's generated comments.
```

### FR-3: Planner-driven ticket decomposition for oversized tickets

**Why**: even with a bigger per-agent budget (FR-2), some tickets are just
too large for one Developer invocation. Right now there's no mechanism for
Planner to recognize that and split the work -- it always hands the whole
ticket to one Developer run.

```
## User Story
As the repo owner, I want a ticket that's too big for one Developer pass to
get split into an ordered sequence of smaller sub-tickets, each with its own
fresh budget, rather than failing or producing a shallow partial fix.

## Acceptance Criteria
- Planner's submit_plan tool (or a new one) can express "this needs N
  sequential sub-steps" instead of only a flat step list.
- pipeline.py can run Developer multiple times against the same ticket,
  once per sub-step, carrying forward the diff/state between them.
- A ticket that IS small enough behaves exactly as it does today (no
  regression for the common case).
```

### FR-4: Carry a ticket's own prior comments into a retry

**Why**: today, every run starts from the raw issue text alone. If a ticket
gets BLOCKED, a human comments, and the issue is re-triggered, the next run
has no idea what happened last time beyond the labels already on the issue.
For a self-hosting workflow where tickets realistically take a few rounds,
this makes iteration much less effective than it should be.

```
## User Story
As the repo owner, I want a retried ticket to know what happened on its
previous attempt(s), so a retry after I comment "try X instead" actually
incorporates that instead of repeating the same failed approach.

## Acceptance Criteria
- intake.parser (or a new step in cli.py) can fetch and include the
  issue's own prior comments as additional ticket context on a retry.
- Developer/Planner's first message includes a summary of the prior
  attempt's outcome when one exists.
- No change in behavior for a ticket's first-ever run (no prior comments).
```

## Tier 2 -- quality and trust, given what security review already found

### FR-5: Reviewer independently re-runs verification, doesn't just trust Developer's harness result

**Why**: today Reviewer's PASS/FAIL is gated entirely on the model's own
APPROVE/REJECT text plus a crude `eval(`/`exec(` substring check. It never
independently re-runs anything. For self-hosting specifically, Reviewer
re-running `make check` itself (rather than trusting Developer's last
harness result) would catch a broken build before shipping.

```
## User Story
As the repo owner, I want the Reviewer agent to independently verify the
final state (not just trust Developer's last harness run), so a subtle
regression introduced after the last passing harness run can't slip through.

## Acceptance Criteria
- ReviewerAgent re-runs the configured test harness itself before deciding,
  independent of Developer's own last recorded result.
- A harness failure at review time overrides an APPROVE verdict.
- The existing security-prefilter substring check is acknowledged as weak
  (trivially bypassable, e.g. getattr(builtins, "ev"+"al")) and either
  hardened or explicitly scoped down to "best-effort lint, not a security
  boundary" in its own docstring.
```

### FR-6: A standing self-hosting smoke test, not just a one-off manual session

**Why**: this whole tool-use feature was validated through a live,
manually-driven session (this one). Once L4 is reached, regressions in the
agent pipeline itself should be caught automatically, not only when a human
happens to run another live validation session.

```
## User Story
As the repo owner, I want a recurring (or push-triggered) job that runs a
small known-good ticket against dark-factory's own repo, so a regression in
the agent pipeline is caught by CI, not by accident.

## Acceptance Criteria
- A scheduled or push-triggered workflow provisions/reuses a disposable
  sandbox repo (the pattern this session already used manually) and runs a
  fixed, cheap (haiku, small budget) ticket end-to-end.
- The job fails loudly (not silently) if the ticket doesn't converge or the
  PR doesn't open.
- Documented as the project's own regression test for its hardest-to-unit-
  test property: does the real agentic loop still work.
```

## Tier 3 -- lower priority given the personal-project framing

### FR-7: OpenAI / openai_compatible tool-call support

Already a named, deliberate Phase 2 gap (see `CLAUDE.md`). Worth doing
eventually for provider-agnosticism's own sake, but Anthropic-only is fine
for a personal project and shouldn't compete with Tier 1/2 for attention.

### FR-8: Validator's LLM call is still decorative

`ValidatorAgent` calls the LLM and discards the response exactly like
Developer/Planner/Reviewer used to -- it was never fixed alongside them
since validation is pass/fail on structural checks anyway. Low effort, low
urgency; mostly a consistency cleanup.

```
## User Story
As the repo owner, I want Validator's own model call to actually inform its
verdict (or be removed if it's not needed), for consistency with how every
other agent's LLM call is now load-bearing.

## Acceptance Criteria
- Either Validator's VALID/INVALID verdict is parsed and used the way
  Reviewer's APPROVE/REJECT now is, or the LLM call is removed entirely if
  the structural checks are judged sufficient on their own.
- No regression in existing validator tests.
```

## Beyond this list

Everything above is deliberately scoped to "one step past what's currently
proven" -- that's a real bias, not a neutral ranking. It's short-term by
construction: each item closes a gap we can already name precisely because
we just hit it. It does not include a longer-horizon, more ambitious pass
(e.g. what would dark-factory look like with real multi-repo fleet
management, a genuinely different context/memory architecture instead of
per-attempt fresh conversations, or agent capability well beyond
"self-hosting on one repo"). That's a deliberate gap in this document, not
an oversight -- flagged as a real to-do to revisit with a more ambitious
lens, not just incrementally extend this list.
