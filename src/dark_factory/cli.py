"""`dark-factory` command-line entry point.

Exists so the GitHub Actions glue in orchestrate.yml stays thin (load event,
run, exit with a stable code) and so the whole pipeline is runnable locally
against a saved webhook payload with zero CI involved. `doctor` is the
single biggest lever on "extremely easy to configure": it tells an adopter
exactly what's missing before their first real run spends any money.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any, cast

from dark_factory.agents.pipeline import DarkFactoryPipeline
from dark_factory.config import ConfigError, FactoryConfig, find_config_file, load_config
from dark_factory.intake.parser import ParserError, parse_webhook_event
from dark_factory.llm.errors import LLMError
from dark_factory.llm.registry import resolve as resolve_provider
from dark_factory.pricing import ModelNotPricedError, PriceBook
from dark_factory.report import RunReport, exit_code_for

DEFAULT_CONFIG_TEMPLATE = """\
# Dark Factory configuration. Everything here is optional -- delete this
# file entirely and the built-in defaults (mock provider, $3 ceiling, 5
# build-test-fix retries) apply. Run `dark-factory doctor` to check.
llm:
  provider: {provider}
  model: {model}
  api_key_env: LLM_API_KEY

budget:
  max_usd: 3.00

loop:
  max_iterations: 5

test_harness:
  command: ./factory-test.sh
"""

CALLER_WORKFLOW_TEMPLATE = """\
name: Dark Factory
on:
  issues:
    types: [opened, edited, labeled]
  pull_request:
    types: [opened, synchronize, reopened]
  workflow_dispatch:
    inputs:
      issue_number:
        required: true
permissions:
  contents: read
  issues: read
  pull-requests: read
  id-token: write
concurrency:
  group: dark-factory-${{ github.repository }}
  cancel-in-progress: false
jobs:
  orchestrate:
    if: >
      github.event_name == 'workflow_dispatch' ||
      github.event_name == 'issues' ||
      startsWith(github.event.pull_request.head.ref, 'dark-factory/')
    uses: your-org/dark-factory/.github/workflows/orchestrate.yml@v1
    with:
      event_name: ${{ github.event_name }}
      repository: ${{ github.repository }}
      issue_number: >-
        ${{ github.event.issue.number || github.event.pull_request.number ||
        inputs.issue_number }}
    secrets:
      app_id: ${{ secrets.DARK_FACTORY_APP_ID }}
      app_private_key: ${{ secrets.DARK_FACTORY_APP_PRIVATE_KEY }}
      llm_api_key: ${{ secrets.LLM_API_KEY }}
"""

_AGENT_NAMES = ("validator", "planner", "developer", "reviewer")


def _parse_set_overrides(pairs: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--set expects KEY=VALUE, got {pair!r}")
        key, _, value = pair.partition("=")
        node = overrides
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return overrides


def _load_config_for_cli(args: argparse.Namespace, *, force_mock: bool = False) -> FactoryConfig:
    overrides = _parse_set_overrides(getattr(args, "set", None) or [])
    if force_mock:
        overrides.setdefault("llm", {})
        overrides["llm"]["provider"] = "mock"
        overrides["llm"]["model"] = "mock-default"
        overrides["agents"] = {}
    return load_config(
        root=Path(args.root),
        path=Path(args.config) if getattr(args, "config", None) else None,
        overrides=overrides,
    )


def cmd_run(args: argparse.Namespace) -> int:
    """Run the pipeline against a saved webhook event and print/write a report."""
    try:
        cfg = _load_config_for_cli(args, force_mock=args.dry_run)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    event_path = Path(args.event_path)
    try:
        payload = json.loads(event_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"could not read event payload {event_path}: {exc}", file=sys.stderr)
        return 2

    try:
        ticket = parse_webhook_event(
            payload,
            event_type=args.event_name,
            section_aliases=cfg.intake.section_aliases,
        )
    except ParserError as exc:
        print(f"could not parse webhook payload: {exc}", file=sys.stderr)
        return 2

    try:
        pipeline = DarkFactoryPipeline(cfg, root=Path(args.root))
        result = pipeline.run(ticket)
    except LLMError as exc:
        print(f"provider error: {exc}", file=sys.stderr)
        return 3

    report = RunReport.from_pipeline_result(result)
    print(report.to_markdown())
    if args.output:
        Path(args.output).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    return exit_code_for(result)


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check config, credentials, pricing, and the test harness before any spend."""
    root = Path(args.root)
    ok = True

    config_path = Path(args.config) if args.config else find_config_file(root)
    print(f"  config          {config_path if config_path else '(none found -- using defaults)'}")

    try:
        cfg = _load_config_for_cli(args)
    except ConfigError as exc:
        print(str(exc))
        return 1

    try:
        resolve_provider(cfg.llm.provider)
        print(f"  provider        {cfg.llm.provider} (resolvable)")
    except Exception as exc:
        print(f"  FAIL provider   {exc}")
        ok = False

    if cfg.llm.api_key_env:
        present = bool(os.environ.get(cfg.llm.api_key_env))
        status = "is set" if present else "is NOT set"
        marker = "" if present else "FAIL "
        print(f"  {marker}credentials     {cfg.llm.api_key_env} {status}")
        ok = ok and present

    price_book = PriceBook.builtin(on_unknown_model=cfg.pricing.on_unknown_model)
    for agent_name in _AGENT_NAMES:
        spec = cfg.resolve_llm(agent_name)
        try:
            price = price_book.price_for(spec.provider, spec.model)
            label = f"{agent_name}={spec.model}"
            print(f"  models          {label} [priced ${price.input_per_mtok_usd}/MTok]")
        except ModelNotPricedError as exc:
            print(f"  FAIL models     {exc}")
            ok = False

    print(f"  budget          ceiling ${cfg.budget.max_usd:.2f}")

    command = cfg.test_harness.command
    script = command.split()[0] if command else ""
    script_path = root / script
    if script_path.exists():
        executable = os.access(script_path, os.X_OK)
        marker = "" if executable else "FAIL "
        state = "executable" if executable else "not executable"
        print(f"  {marker}test harness    {command} ({state})")
        ok = ok and executable
    else:
        print(
            f"  test harness    {command} (not found at {script_path} -- may be a non-file command)"
        )

    print(
        f"  labels          spec-validated={cfg.labels.spec_validated} blocked={cfg.labels.blocked}"
    )

    return 0 if ok else 1


def cmd_config(args: argparse.Namespace) -> int:
    """Print the effective, merged configuration, optionally with provenance."""
    try:
        cfg = _load_config_for_cli(args)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(dataclasses.asdict(cfg), indent=2, default=str))
    if args.explain:
        print("\nprovenance:")
        for key, source in sorted(cfg.provenance.items()):
            print(f"  {key} <- {source}")
    return 0


def cmd_prices(args: argparse.Namespace) -> int:
    """List built-in per-model pricing, optionally filtered by provider."""
    from dark_factory.pricing import _load_builtin_raw

    raw = _load_builtin_raw()
    for key, price in raw.get("models", {}).items():
        provider, _, _model = key.partition(":")
        if args.provider and provider != args.provider:
            continue
        in_price, out_price = price["input_per_mtok_usd"], price["output_per_mtok_usd"]
        print(f"{key}: ${in_price}/MTok in, ${out_price}/MTok out")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    """Scaffold `.dark-factory.yml` and `caller.yml` into a downstream repo."""
    root = Path(args.root)
    config_path = root / ".dark-factory.yml"
    workflow_dir = root / ".github" / "workflows"
    workflow_path = workflow_dir / "caller.yml"

    if config_path.exists() and not args.force:
        print(f"{config_path} already exists (use --force to overwrite)", file=sys.stderr)
        return 1

    model_by_provider = {
        "anthropic": "claude-sonnet-5",
        "openai": "gpt-5",
        "openai_compatible": "qwen2.5-coder:32b",
        "mock": "mock-default",
    }
    config_path.write_text(
        DEFAULT_CONFIG_TEMPLATE.format(
            provider=args.provider, model=model_by_provider.get(args.provider, "mock")
        ),
        encoding="utf-8",
    )
    print(f"wrote {config_path}")

    workflow_dir.mkdir(parents=True, exist_ok=True)
    workflow_path.write_text(CALLER_WORKFLOW_TEMPLATE, encoding="utf-8")
    print(f"wrote {workflow_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the `dark-factory` argument parser and its subcommands."""
    parser = argparse.ArgumentParser(prog="dark-factory")
    parser.add_argument("--root", default=".", help="downstream repo checkout root (default: .)")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run the pipeline against a saved webhook event")
    run_p.add_argument("--event-path", required=True)
    run_p.add_argument("--event-name", default="issues")
    run_p.add_argument("--config", default=None)
    run_p.add_argument(
        "--set", action="append", default=[], help="dotted override, e.g. budget.max_usd=5"
    )
    run_p.add_argument(
        "--dry-run", action="store_true", help="force the mock provider; zero network calls"
    )
    run_p.add_argument("--output", default=None, help="write the JSON run report to this path")
    run_p.set_defaults(func=cmd_run)

    doctor_p = sub.add_parser("doctor", help="check that this repo is configured correctly")
    doctor_p.add_argument("--config", default=None)
    doctor_p.add_argument("--set", action="append", default=[])
    doctor_p.set_defaults(func=cmd_doctor)

    config_p = sub.add_parser("config", help="print the effective, merged configuration")
    config_p.add_argument("--config", default=None)
    config_p.add_argument("--set", action="append", default=[])
    config_p.add_argument(
        "--explain", action="store_true", help="also print which layer set each value"
    )
    config_p.set_defaults(func=cmd_config)

    prices_p = sub.add_parser("prices", help="list built-in per-model pricing")
    prices_p.add_argument("--provider", default=None)
    prices_p.set_defaults(func=cmd_prices)

    init_p = sub.add_parser("init", help="scaffold .dark-factory.yml and caller.yml into a repo")
    init_p.add_argument("--provider", default="anthropic")
    init_p.add_argument("--force", action="store_true")
    init_p.set_defaults(func=cmd_init)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the selected subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return cast(int, args.func(args))


if __name__ == "__main__":
    sys.exit(main())
