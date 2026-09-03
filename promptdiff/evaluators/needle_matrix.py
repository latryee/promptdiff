"""Multi-Depth Needle-in-a-Haystack 2D Retrieval Matrix.

Benchmarks long-context prompt retrieval fidelity across context depths (1k, 4k, 8k, 16k, 32k)
and relative positions (0% to 100%), constructing a 2D performance heatmap matrix
to expose 'lost-in-the-middle' attention decay.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NeedleCellResult:
    """Outcome of needle test at a specific (depth, position) coordinate."""

    context_tokens: int
    needle_depth_pct: int  # 0%, 25%, 50%, 75%, 100%
    retrieved_successfully: bool
    retrieval_confidence: float


@dataclass
class NeedleMatrixReport:
    """Complete 2D context retrieval performance matrix."""

    context_lengths: list[int]
    depth_percentages: list[int]
    overall_retrieval_rate_pct: float
    lost_in_middle_observed: bool
    matrix: list[NeedleCellResult]

    def render_ascii_matrix(self) -> str:
        """Render a terminal 2D grid of retrieval accuracy."""
        lines = ["Needle Retrieval Matrix (Green = 100%, Red = Failed):"]
        header = "Depth %  | " + " | ".join(f"{c // 1000}k" for c in self.context_lengths)
        lines.append(header)
        lines.append("-" * len(header))

        for pos in self.depth_percentages:
            row_cells = []
            for c in self.context_lengths:
                cell = next((x for x in self.matrix if x.context_tokens == c and x.needle_depth_pct == pos), None)
                sym = "✓" if cell and cell.retrieved_successfully else "✗"
                row_cells.append(f" {sym} ")
            lines.append(f"{pos:3d}%     | " + " | ".join(row_cells))
        return "\n".join(lines)


class NeedleMatrixEvaluator:
    """Evaluates long-context prompt retrieval across 2D depth grids."""

    def __init__(self, context_lengths: list[int] | None = None, depth_percentages: list[int] | None = None):
        self.context_lengths = context_lengths or [1000, 4000, 8000, 16000]
        self.depth_percentages = depth_percentages or [0, 25, 50, 75, 100]

    def benchmark_mock(self) -> NeedleMatrixReport:
        """Deterministic simulation of needle matrix performance."""
        matrix: list[NeedleCellResult] = []
        success_count = 0

        for c in self.context_lengths:
            for pos in self.depth_percentages:
                # Middle positions (40%-60%) at 16k+ suffer lost-in-middle decay
                if c >= 16000 and 25 < pos < 75:
                    success = False
                    conf = 0.35
                else:
                    success = True
                    conf = 0.95

                if success:
                    success_count += 1

                matrix.append(
                    NeedleCellResult(
                        context_tokens=c,
                        needle_depth_pct=pos,
                        retrieved_successfully=success,
                        retrieval_confidence=conf,
                    )
                )

        rate = round((success_count / max(1, len(matrix))) * 100.0, 1)
        lost_in_middle = any(not cell.retrieved_successfully for cell in matrix)

        return NeedleMatrixReport(
            context_lengths=self.context_lengths,
            depth_percentages=self.depth_percentages,
            overall_retrieval_rate_pct=rate,
            lost_in_middle_observed=lost_in_middle,
            matrix=matrix,
        )
