"""The OLT serving its queries the way a real LLM server does: continuous batching.

A query starts generating as soon as it arrives and leaves when it is done, so the batch is
whoever happens to be mid-answer. Nothing is imposed from outside: the batch is what the
households' own escalated queries make it. This is Orca's iteration-level scheduling, which
is what vLLM and SGLang run today; we drive it from the OLT's measured curve the way Vidur
drives a simulated cluster from a profile.

Energy follows the batch moment by moment. A query that starts alone pays the lonely rate
until company arrives, then pays less. Charging one batch size for a whole answer, as a
static model does, cannot show that.

Cost is kept off the hot path with a virtual token clock. Every active sequence advances at
the same tokens-per-second, so one counter tracks the progress they share and a second
accumulates the joules a token has cost since the run began. A sequence's decode energy is
then the difference between that accumulator when it left and when it joined, whatever
happened in between.
"""

from __future__ import annotations

import heapq

MAX_BATCH = 64
"""Largest batch the OLT curve was measured at; beyond it the rates hold at 64's."""


class OltServer:
    """One OLT serving its PON's escalated queries with continuous batching.

    Attributes:
        t: Wall clock, in seconds.
        served: Queries finished so far.
        admitted: Queries that have entered the server.
    """

    # ==========================================
    # Initialization
    # ==========================================
    def __init__(self, curve, prices, report_window_s: float = 300.0):
        """Set the server up against the OLT's measured curve.

        Args:
            curve: The OLT's measured batch curve (energy.three_tier.OltCurve).
            prices: True energy rates (olt_energy.Energy).
            report_window_s: Seconds the OLT averages over for what it reports.
        """
        self.curve, self.prices, self.window = curve, prices, report_window_s
        self.t = 0.0
        self.served = self.admitted = 0

        self._p = 0.0
        """Virtual clock: tokens every active sequence has produced since the run began."""

        self._e = 0.0
        """Joules one generated token has cost, accumulated over that virtual clock."""

        self._done: list[tuple[float, int]] = []
        """Heap of (virtual clock at which it finishes, query id)."""

        self._joined: dict[int, float] = {}
        """Virtual clock at which each active query joined."""

        self._recent: list[tuple[float, float, float]] = []
        """(wall clock, generated tokens, joules) of recent work, for what the OLT reports."""

    # ==========================================
    # The batch right now
    # ==========================================
    @property
    def batch(self) -> int:
        """Return how many sequences are generating."""
        return len(self._joined)

    def _rates(self, n: int) -> tuple[float, float]:
        """Return (J per prompt token, J per generated token) at a batch of n.

        Args:
            n: Sequences generating together.
        """
        return self.prices.olt(min(max(n, 1), MAX_BATCH))

    def _speed(self, n: int) -> float:
        """Return the tokens per second one sequence gets in a batch of n.

        Args:
            n: Sequences generating together.
        """
        b = min(max(n, 1), MAX_BATCH)
        return self.curve._at(self.curve.tps, b) / b  # pylint: disable=protected-access

    # ==========================================
    # Running time forward
    # ==========================================
    def advance(self, target_t: float) -> list[tuple[int, float, float]]:
        """Run the server up to a wall-clock time, finishing whatever finishes on the way.

        Args:
            target_t: Wall clock to advance to, in seconds.

        Returns:
            [(query id, wall clock it finished, joules it spent decoding)] for each query
            that completed, in the order they completed.
        """
        finished = []
        while self._done and self.t < target_t:
            n = self.batch
            speed = self._speed(n)
            j_token = self._rates(n)[1]

            # --- How far the next completion is, in the clock the batch shares ---
            p_next = self._done[0][0]
            dt = (p_next - self._p) / speed
            if self.t + dt > target_t:  # nobody finishes before the target: part-step there
                step = (target_t - self.t) * speed
                self._track(target_t - self.t, step * n, step * n * j_token)
                self._p += step
                self._e += step * j_token
                self.t = target_t
                return finished

            self._track(dt, (p_next - self._p) * n, (p_next - self._p) * n * j_token)
            self._e += (p_next - self._p) * j_token
            self._p = p_next
            self.t += dt

            # --- Everyone whose clock ran out leaves together ---
            while self._done and self._done[0][0] <= self._p + 1e-9:
                _, qid = heapq.heappop(self._done)
                finished.append((qid, self.t, self._e - self._joined.pop(qid)))
                self.served += 1

        self.t = max(self.t, target_t)
        return finished

    def admit(self, qid: int, prompt_tokens: float, gen_tokens: float, at_t: float) -> float:
        """Take one query in, and return the joules its prefill cost.

        The prefill is charged at the batch the query meets on arrival and treated as
        instant: the curve measures a batch's prefill as one pass, and chunking it across
        decode steps, as vLLM can, is below what the curve resolves.

        Args:
            qid: The query's id.
            prompt_tokens: Its prompt length.
            gen_tokens: How many tokens its answer runs to.
            at_t: Wall clock it reaches the OLT.

        Returns:
            Joules spent on its prefill.
        """
        self.advance(at_t)
        j_prompt = self._rates(self.batch + 1)[0]
        self._joined[qid] = self._e
        heapq.heappush(self._done, (self._p + gen_tokens, qid))
        self.admitted += 1
        return j_prompt * prompt_tokens

    def drain(self) -> list[tuple[int, float, float]]:
        """Finish everything still generating, and return what completed."""
        out = []
        while self._done:
            out += self.advance(self.t + 3600)
        return out

    # ==========================================
    # What the OLT can tell the PON
    # ==========================================
    def _track(self, dt: float, tokens: float, joules: float) -> None:
        """Record work done, and forget what has aged out of the report window."""
        if tokens > 0:
            self._recent.append((self.t + dt, tokens, joules))
        cut = self.t + dt - self.window
        while self._recent and self._recent[0][0] < cut:
            self._recent.pop(0)

    def reported_rate(self) -> tuple[float, float] | None:
        """Return the OLT's mean (J per prompt token, J per generated token) over its window.

        This is what the OLT broadcasts on the PON, and what rides back on an answer. It is
        backward-looking by construction: it averages work already done.

        Returns:
            The mean rates, or None when the window holds no work yet.
        """
        tokens = sum(r[1] for r in self._recent)
        if tokens <= 0:
            return None
        return self._rates(self.batch)[0], sum(r[2] for r in self._recent) / tokens
