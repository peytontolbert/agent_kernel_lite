#!/usr/bin/env python3
"""Run the Agent Kernel Lite computer-use bridge.

This entry point intentionally delegates to the older Codex bridge filename so
existing installs keep working while the extension boundary moves to
`computer_use`.
"""

from run_agentkernel_lite_codex_bridge import main


if __name__ == "__main__":
    main()
