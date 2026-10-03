"""The queries the households send, built from conversations rather than a flat rate.

A household does not send queries at a steady rate through the day. Someone opens a
conversation, asks a few things a couple of minutes apart, and leaves. That clumping is what
the OLT's batching window sees, so the stream is built one conversation at a time, resampling
the shapes measured in BurstGPT (data/load_traces/burstgpt_sessions.json, written by
prepare_load_traces.py --sessions).

How much a household sends is not BurstGPT's to say: it has no user id and never states how
many people it serves. That rate comes from ChatGPT's own consumer figures instead
(config/traffic_sources.yaml).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SESSIONS = Path(__file__).resolve().parents[2] / "data" / "load_traces" / "burstgpt_sessions.json"
"""Conversation shapes measured in BurstGPT's conversation log."""


def load_sessions(path: Path = SESSIONS) -> dict:
    """Return the measured conversation shapes.

    Args:
        path: The prepared session statistics.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class Households:
    """The subscribers on one OLT, and the conversations they open.

    Attributes:
        count: Subscribers on the OLT (one household each).
        users: Weekly-active LLM users per household.
        per_user_day: Messages one active user sends a day.
        messages_per_day: What the whole population sends a day.
        sessions_per_day: How many conversations that is, at the measured length.
    """

    # ==========================================
    # Initialization
    # ==========================================
    def __init__(self, count: int, users: float, per_user_day: float, stats: dict | None = None):
        """Size the population.

        Args:
            count: Subscribers on the OLT.
            users: Weekly-active LLM users per household.
            per_user_day: Messages one active user sends a day.
            stats: The measured conversation shapes (load_sessions() by default).
        """
        self.stats = stats or load_sessions()
        self.count, self.users, self.per_user_day = count, users, per_user_day
        self.messages_per_day = count * users * per_user_day
        self.sessions_per_day = self.messages_per_day / self.stats["requests_per_session_mean"]

        self._len = np.array(self.stats["requests_per_session_icdf"])
        self._think = np.array(self.stats["think_time_s_icdf"])
        self.weekend = set(self.stats["weekend_days_mod7"])
        """Days of the week (index mod 7) the trace runs quiet."""

        per_day = self.stats["sessions_per_day"]
        self._starts = {
            name: np.array(v, dtype=float) / per_day
            for name, v in self.stats["starts_by_hour"].items()
        }
        """Bursts opening in each hour of a weekday and of a weekend day, per burst of an
        average day, so scaling by this population's own rate keeps both shapes."""

    # ==========================================
    # The query stream
    # ==========================================
    def stream(self, questions, days: int, first_day: int, seed: int) -> list:
        """Build the time-ordered queries the households send.

        Conversations open through the day following the measured hour-of-day shape. Each one
        holds a measured number of messages, spaced by measured pauses, and belongs to one
        household drawn at random: BurstGPT has no user id, so there is nothing to say which
        households talk more than others.

        Args:
            questions: The question indices that can be asked.
            days: Days to simulate.
            first_day: Day the simulation starts on, in days since the trace's start.
            seed: Random seed.

        Returns:
            [(question, time in hours, household)], sorted by time.
        """
        rng = np.random.default_rng([seed, 1709])
        qs = np.asarray(questions)

        # --- How many bursts open in each hour of the run, weekday or weekend ---
        expected = np.concatenate(
            [
                self.sessions_per_day
                * self._starts["weekend" if (first_day + d) % 7 in self.weekend else "weekday"]
                for d in range(days)
            ]
        )
        hours = days * 24
        opened = rng.poisson(expected)
        total = int(opened.sum())
        if total == 0:
            return []

        # --- When each one opens, and how many messages it holds ---
        hour_of = np.repeat(np.arange(hours), opened)
        opens_h = first_day * 24 + hour_of + rng.random(total)
        lengths = self._len[rng.integers(0, len(self._len), total)]

        # --- Message times: the opening one, then the pauses after it ---
        messages = int(lengths.sum())
        which = np.repeat(np.arange(total), lengths)
        gaps = self._think[rng.integers(0, len(self._think), messages)] / 3600
        gaps[np.cumsum(lengths) - lengths] = 0.0  # each conversation's first message opens it
        offsets = np.cumsum(gaps) - np.repeat(
            np.cumsum(gaps)[np.cumsum(lengths) - lengths], lengths
        )
        times = opens_h[which] + offsets

        # --- One household and one question per message ---
        home = rng.integers(0, self.count, total)[which]
        asked = qs[rng.integers(0, len(qs), messages)]

        order = np.argsort(times, kind="stable")
        return [
            (int(asked[i]), float(times[i]), int(home[i]))
            for i in order
            if times[i] < (first_day + days) * 24
        ]

    def weekday(self, day: int) -> bool:
        """Return whether a day of the run is a weekday.

        Args:
            day: Day index, in days since the trace's start.
        """
        return day % 7 not in self.weekend

    def describe(self) -> str:
        """Return the population, in one line of the printout."""
        return (
            f"{self.count:,} households x {self.users:g} active users x "
            f"{self.per_user_day:g} msg/day = {self.messages_per_day:,.0f} messages/day "
            f"in {self.sessions_per_day:,.0f} bursts, weekends quieter"
        )
