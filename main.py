import argparse

from orchestrator.orchestrator import OrchestratorLoop


def main():
    parser = argparse.ArgumentParser(description="Work buddy: watches your focus and nudges you.")
    parser.add_argument("--robot", action="store_true", help="use the Reachy robot instead of the Mac camera/speaker")
    parser.add_argument("--edge", action="store_true", help="run local (edge) models instead of the OpenAI cloud")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="shorter intervals: 15s nudge cooldown, summary every 3 minutes",
    )
    args = parser.parse_args()

    orchestrator = OrchestratorLoop(use_robot=args.robot, edge=args.edge, debug=args.debug)
    orchestrator.run()


if __name__ == "__main__":
    main()
