"""학습 진행 상황 표시. 모방학습과 강화학습이 같이 쓴다.

터미널에 찍으면서 동시에 파일(train.log)에도 남긴다.
다른 터미널에서 watch.ps1 을 돌리면 실시간으로 볼 수 있다.
"""

import sys
import time

LOG_PATH = "train.log"
BAR_WIDTH = 24


def _clock(seconds):
    """초를 분:초 또는 시:분:초로."""
    seconds = int(max(seconds, 0))
    if seconds >= 3600:
        return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"
    return f"{seconds // 60}:{seconds % 60:02d}"


class Logger:
    """화면과 파일에 동시에 쓴다."""

    def __init__(self, path=LOG_PATH, reset=True):
        self.file = open(path, "w" if reset else "a", encoding="utf-8")

    def __call__(self, text=""):
        print(text, flush=True)
        self.file.write(text + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


class Progress:
    """남은 시간을 지금까지 걸린 시간으로 추정해서 보여준다."""

    def __init__(self, total, label="진행", log=print):
        self.total = total
        self.label = label
        self.log = log
        self.started = time.time()

    def bar(self, done):
        filled = int(BAR_WIDTH * done / self.total)
        return "█" * filled + "░" * (BAR_WIDTH - filled)

    def line(self, done, extra=""):
        elapsed = time.time() - self.started
        remain = elapsed / done * (self.total - done) if done else 0.0
        return (
            f"{self.label} {done:>3}/{self.total} [{self.bar(done)}] "
            f"{done / self.total:>4.0%}  {extra}"
            f"  경과 {_clock(elapsed)}  남음 {_clock(remain)}"
        )

    def step(self, done, extra=""):
        self.log(self.line(done, extra))

    def done(self):
        elapsed = time.time() - self.started
        self.log(f"{self.label} 완료  총 {_clock(elapsed)}")
        return elapsed


class Spinner:
    """총량을 모를 때 쓰는 한 줄짜리 표시. 화면에만 덮어쓴다."""

    def __init__(self, label):
        self.label = label
        self.started = time.time()

    def step(self, text):
        elapsed = _clock(time.time() - self.started)
        sys.stdout.write(f"\r{self.label} {text}  경과 {elapsed}   ")
        sys.stdout.flush()

    def clear(self):
        sys.stdout.write("\r" + " " * 78 + "\r")
        sys.stdout.flush()
