"""The OLT serving its queries the way a real LLM server does: continuous batching.

A query joins the running batch as soon as a slot is free and leaves when its answer is
done, as Orca's iteration-level scheduling does in vLLM and SGLang. Speed and energy follow
the batch moment by moment, read off the OLT's measured curve.

Every active sequence advances at the same tokens per second, so one virtual clock tracks
their shared progress and a second accumulates what one sequence's tokens have cost. A
query's decode energy is that accumulator when it leaves minus its value when it joined.
"""

from __future__ import annotations

import collections
import heapq

MAX_BATCH = 64
"""Largest batch the OLT's curve was measured at, and so its number of slots."""


class OltServer:
    """One OLT serving its PON's escalated queries with continuous batching.

    Attributes:
        clock: Wall clock, in seconds.
        admitted: Queries that reached the OLT.
        queued: Queries among them that found every slot busy and waited.
        last_broadcast: The rates the OLT last put on the PON, or None before its first send.
    """

    # ==========================================
    # Initialization
    # ==========================================
    def __init__(self, curve, prices, report_window_s: float = 300.0):
        """Set the server up against the OLT's measured curve.

        Args:
            curve: The OLT's measured batch curve (energy.three_tier.OltCurve).
            prices: True energy rates (olt_energy.Energy).
            report_window_s: Seconds of recent work the OLT averages over when it reports.
        """
        self.curve, self.prices, self.report_window_s = curve, prices, report_window_s
        self.clock = 0.0
        self.admitted = self.queued = 0
        self.last_broadcast = None

        self._progress = 0.0
        """Virtual clock: tokens every active sequence has generated since the run began."""

        self._token_joules = 0.0
        """Joules one sequence's generated tokens have cost along that virtual clock."""

        self._finishing: list[tuple[float, int]] = []
        """Heap of (virtual clock at which the query finishes, query id)."""

        self._active: dict[int, tuple[float, float]] = {}
        """Query id -> (accumulated token joules when it joined, joules of its prefill)."""

        self._waiting: collections.deque = collections.deque()
        """(query id, prompt tokens, generated tokens) waiting for a slot, in arrival order."""

        self._recent: collections.deque = collections.deque()
        """(clock, busy seconds, prompt tokens, prompt J, generated tokens, generated J) of recent work."""

        self._recent_totals = [0.0] * 5
        """Sums of the five quantities over the report window."""

        self._busy_s = self._batch_s = 0.0
        """Seconds spent generating, and those weighted by the batch held."""

    # ==========================================
    # The batch right now
    # ==========================================
    @property
    def batch(self) -> int:
        """Return how many sequences are generating."""
        return len(self._active)

    @property
    def mean_batch(self) -> float:
        """Return the mean batch the OLT held while generating, weighted by time."""
        return self._batch_s / self._busy_s if self._busy_s > 0 else 0.0

    def _rates(self, batch: int) -> tuple[float, float]:
        """Return (J per prompt token, J per generated token) at this batch size."""
        return self.prices.olt(min(max(batch, 1), MAX_BATCH))

    def _speed(self, batch: int) -> float:
        """Return the tokens per second one sequence gets at this batch size."""
        size = min(max(batch, 1), MAX_BATCH)
        return self.curve._at(self.curve.tps, size) / size  # pylint: disable=protected-access

    # ==========================================
    # Serving
    # ==========================================
    def admit(self, query: int, prompt_tokens: float, generated_tokens: float) -> None:
        """Take in a query reaching the OLT now; it waits for a slot if all are busy.

        The caller advances the server to the query's arrival first, so no completion
        before it is lost.

        Args:
            query: The query's id.
            prompt_tokens: Its prompt length.
            generated_tokens: How many tokens its answer runs to.
        """
        self.admitted += 1
        if self.batch < MAX_BATCH and not self._waiting:
            self._start(query, prompt_tokens, generated_tokens)
        else:
            self.queued += 1
            self._waiting.append((query, prompt_tokens, generated_tokens))

    def advance(self, until: float) -> list[tuple[int, float, float, float]]:
        """Run the server up to a wall-clock time, finishing whatever finishes on the way.

        Args:
            until: Wall clock to advance to, in seconds.

        Returns:
            [(query id, clock it finished, prefill joules, decode joules)], in finishing order.
        """
        finished = []
        while self._finishing and self.clock < until:
            batch = self.batch
            speed = self._speed(batch)
            joules_per_token = self._rates(batch)[1]

            # --- Generate up to the next completion, or up to `until` if that comes first ---
            to_next = self._finishing[0][0] - self._progress
            if self.clock + to_next / speed > until:
                self._generate((until - self.clock) * speed, until - self.clock, joules_per_token)
                self.clock = until
                break
            self._generate(to_next, to_next / speed, joules_per_token)
            self._progress = self._finishing[0][0]  # exact, so the heap comparison below holds

            # --- Everyone whose answer is done leaves, and the queue takes their slots ---
            while self._finishing and self._finishing[0][0] <= self._progress:
                _, query = heapq.heappop(self._finishing)
                joined_at, prefill_joules = self._active.pop(query)
                finished.append((query, self.clock, prefill_joules, self._token_joules - joined_at))
            while self._waiting and self.batch < MAX_BATCH:
                self._start(*self._waiting.popleft())

        self.clock = max(self.clock, until)
        self._forget_old()
        return finished

    def drain(self) -> list[tuple[int, float, float, float]]:
        """Finish everything still generating or waiting, and return what completed."""
        finished = []
        while self._finishing:
            finished += self.advance(self.clock + 3600)
        return finished

    def _start(self, query: int, prompt_tokens: float, generated_tokens: float) -> None:
        """Give a query a slot: its prefill runs at once, at the batch it makes."""
        prefill_joules = self._rates(self.batch + 1)[0] * prompt_tokens
        self._active[query] = (self._token_joules, prefill_joules)
        heapq.heappush(self._finishing, (self._progress + generated_tokens, query))
        self._record(0.0, prompt_tokens, prefill_joules, 0.0, 0.0)

    def _generate(self, tokens_each: float, seconds: float, joules_per_token: float) -> None:
        """Let every active sequence generate the same number of tokens."""
        batch = self.batch
        self._busy_s += seconds
        self._batch_s += seconds * batch
        self._progress += tokens_each
        self._token_joules += tokens_each * joules_per_token
        self.clock += seconds
        self._record(seconds, 0.0, 0.0, tokens_each * batch, tokens_each * batch * joules_per_token)

    # ==========================================
    # What the OLT can tell the PON
    # ==========================================
    def reported_rate(self) -> tuple[float, float]:
        """Return what the OLT tells the PON a query costs, from its last report window.

        Under marginal accounting it is what one more query would add: the slope while the
        OLT was busy, the net-of-idle rate while it was idle, weighted by how much of the
        window it spent busy. Under average accounting it is the mean cost per token of the
        work done, or a lone query's rate when there was none.

        Returns:
            (J per prompt token, J per generated token).
        """
        self._forget_old()
        busy_s, prompt_tokens, prompt_joules, generated_tokens, generated_joules = (
            self._recent_totals
        )
        if self.prices.accounting == "marginal":
            window = min(self.report_window_s, self.clock) or self.report_window_s
            busy = min(busy_s / window, 1.0)
            idle_rates, busy_rates = self.prices.added_rates(0), self.prices.added_rates(1)
            return tuple(
                busy * when_busy + (1 - busy) * when_idle
                for when_idle, when_busy in zip(idle_rates, busy_rates)
            )
        lone_prompt, lone_generated = self._rates(1)
        return (
            prompt_joules / prompt_tokens if prompt_tokens > 0 else lone_prompt,
            generated_joules / generated_tokens if generated_tokens > 0 else lone_generated,
        )

    def send_broadcast(self) -> None:
        """Put the current report on the PON, where households hear it until the next send."""
        self.last_broadcast = self.reported_rate()

    def _record(self, *work: float) -> None:
        """Add work done now to the report window."""
        self._recent.append((self.clock, *work))
        for index, amount in enumerate(work):
            self._recent_totals[index] += amount

    def _forget_old(self) -> None:
        """Drop work older than the report window."""
        cutoff = self.clock - self.report_window_s
        while self._recent and self._recent[0][0] < cutoff:
            _, *work = self._recent.popleft()
            for index, amount in enumerate(work):
                self._recent_totals[index] -= amount
        if not self._recent:  # clear rounding residue
            self._recent_totals = [0.0] * 5
