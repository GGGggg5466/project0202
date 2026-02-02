from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Task:
    title: str
    done: bool = False

    def toggle(self) -> None:
        self.done = not self.done


def main() -> None:
    print("=== PROJECT_0202 Python Test ===")
    print("Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    tasks = [
        Task("Initialize git repo", True),
        Task("First commit", True),
        Task("Push to GitHub", True),
        Task("Create a branch"),
        Task("Open a Pull Request"),
    ]

    # Demo: toggle the last task
    tasks[-1].toggle()

    print("\nTasks:")
    for i, t in enumerate(tasks, start=1):
        mark = "✅" if t.done else "⬜"
        print(f"{i:02d}. {mark} {t.title}")

    print("\nNext action suggestion: create a feature branch and open a PR.")


if __name__ == "__main__":
    main()
