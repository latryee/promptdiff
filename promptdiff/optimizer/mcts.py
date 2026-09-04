"""Monte Carlo Tree Search (MCTS) & Pareto-Optimal Active Prompt Optimizer.

Implements an MCTS exploration-exploitation tree search over prompt mutation spaces,
using Upper Confidence Bound for Trees (UCT / UCB1) and multi-objective Pareto Frontier
tracking across (Task Quality, Latency, Token Cost, Conciseness).
"""

from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

from promptdiff.core.models import PromptVersion, TestCase
from promptdiff.core.runner import PromptDiffRunner
from promptdiff.evaluators.registry import get_evaluators
from promptdiff.pricing import calculate_cost, estimate_tokens
from promptdiff.providers.registry import get_provider


@dataclass
class ParetoMetrics:
    """Multi-objective performance criteria for a prompt candidate."""

    quality_score: float  # 0.0 to 1.0 (higher is better)
    latency_ms: float  # ms (lower is better)
    cost_usd: float  # $ per request (lower is better)
    token_count: int  # token count (lower is better)

    def dominates(self, other: ParetoMetrics) -> bool:
        """Pareto Dominance check: True if self is >= other on all objectives and strictly > on at least one."""
        not_worse = (
            self.quality_score >= other.quality_score
            and self.latency_ms <= other.latency_ms
            and self.cost_usd <= other.cost_usd
            and self.token_count <= other.token_count
        )
        strictly_better = (
            self.quality_score > other.quality_score
            or self.latency_ms < other.latency_ms
            or self.cost_usd < other.cost_usd
            or self.token_count < other.token_count
        )
        return not_worse and strictly_better


@dataclass
class MCTSNode:
    """A single state in the MCTS Prompt Search Tree."""

    prompt_template: str
    parent: Optional[MCTSNode] = None
    children: list[MCTSNode] = field(default_factory=list)
    action_applied: str = "initial"
    depth: int = 0
    visits: int = 0
    total_reward: float = 0.0
    metrics: ParetoMetrics = field(default_factory=lambda: ParetoMetrics(0.0, 0.0, 0.0, 0))

    @property
    def average_reward(self) -> float:
        return (self.total_reward / self.visits) if self.visits > 0 else 0.0

    def ucb1(self, c: float = 1.414) -> float:
        """Compute Upper Confidence Bound 1 for node selection."""
        if self.visits == 0:
            return float("inf")
        if self.parent is None or self.parent.visits == 0:
            return self.average_reward
        exploitation = self.average_reward
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def is_leaf(self) -> bool:
        return len(self.children) == 0


@dataclass
class MCTSResult:
    """Comprehensive result of MCTS Prompt Optimization."""

    best_prompt: str
    best_quality: float
    initial_quality: float
    nodes_explored: int
    pareto_frontier: list[MCTSNode]
    tree_ascii: str


MAX_ALLOWED_ITERATIONS: int = 500


def validate_mcts_iterations(iterations: int) -> int:
    """Validate that MCTS iteration count is strictly positive and does not exceed upper safety limits."""
    if iterations <= 0:
        raise ValueError(f"MCTS iterations must be a positive integer, got {iterations}")
    if iterations > MAX_ALLOWED_ITERATIONS:
        raise ValueError(
            f"MCTS iterations ({iterations}) exceeds MAX_ALLOWED_ITERATIONS ({MAX_ALLOWED_ITERATIONS})"
        )
    return iterations


class MCTSPromptOptimizer:
    """Monte Carlo Tree Search active optimizer for prompt engineering."""

    def __init__(
        self,
        initial_prompt: str,
        test_cases: list[TestCase],
        model_name: str = "gpt-4o",
        max_iterations: int = 8,
        num_iterations: Optional[int] = None,
        exploration_constant: float = 1.414,
        force_mock: bool = True,
    ):
        raw_iterations = num_iterations if num_iterations is not None else max_iterations
        self.max_iterations = validate_mcts_iterations(raw_iterations)
        self.num_iterations = self.max_iterations
        self.initial_prompt = initial_prompt
        self.test_cases = test_cases
        self.model_name = model_name
        self.c = exploration_constant
        self.force_mock = force_mock
        self.provider = get_provider(model_name=self.model_name, force_mock=self.force_mock)
        self.all_nodes: list[MCTSNode] = []

    # Prompt Mutation Operators (Semantic Action Space)
    def _mutate_add_chain_of_thought(self, prompt: str) -> str:
        """Inject structured chain-of-thought instructions."""
        if "step-by-step" in prompt.lower() or "chain of thought" in prompt.lower():
            return prompt
        return f"{prompt.strip()}\n\nThink carefully step by step before providing your final concise answer."

    def _mutate_harden_constraints(self, prompt: str) -> str:
        """Inject strict negative constraints and format bounds."""
        directive = "Strict Constraints:\n- Never speculate or hallucinate.\n- Adhere strictly to the requested schema.\n- Be direct and factually grounded."
        if "Strict Constraints:" in prompt:
            return prompt
        return f"{prompt.strip()}\n\n{directive}"

    def _mutate_role_specialize(self, prompt: str) -> str:
        """Inject authoritative domain expert role specialization."""
        prefix = "You are a world-class domain expert. Your analysis must be authoritative, rigorous, and precise.\n"
        if "world-class domain expert" in prompt:
            return prompt
        return f"{prefix}{prompt.strip()}"

    def _mutate_prune_fluff(self, prompt: str) -> str:
        """Prune conversational pleasantries and filler tokens."""
        fluff_patterns = [
            r"please\s+kindly\s+",
            r"you\s+are\s+a\s+helpful\s+assistant\.\s*",
            r"feel\s+free\s+to\s+",
            r"as\s+an\s+ai\s+model,\s*",
        ]
        cleaned = prompt
        for pat in fluff_patterns:
            cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def _mutate_few_shot_structure(self, prompt: str) -> str:
        """Inject exemplar output format guidance."""
        if "Format Example:" in prompt:
            return prompt
        exemplar = "Format Example:\nInput: Example Query\nOutput: Crisp, structured response"
        return f"{prompt.strip()}\n\n{exemplar}"

    def _get_mutators(self) -> list[tuple[str, Callable[[str], str]]]:
        return [
            ("add_chain_of_thought", self._mutate_add_chain_of_thought),
            ("harden_constraints", self._mutate_harden_constraints),
            ("role_specialize", self._mutate_role_specialize),
            ("prune_fluff", self._mutate_prune_fluff),
            ("few_shot_structure", self._mutate_few_shot_structure),
        ]

    async def _evaluate_prompt(self, prompt: str) -> ParetoMetrics:
        """Evaluate a prompt candidate against the test case suite."""
        pv = PromptVersion(name="candidate", template=prompt, model=self.model_name)
        base_pv = PromptVersion(name="baseline", template=self.initial_prompt, model=self.model_name)

        runner = PromptDiffRunner(
            v1_prompt=base_pv,
            v2_prompt=pv,
            provider_v1=self.provider,
            provider_v2=self.provider,
            evaluators=get_evaluators(["similarity,latency,cost,json_validity"]),
            concurrency=2,
        )
        report = await runner.run(self.test_cases)

        # Compute aggregate quality score (0.0 to 1.0)
        passed_count = sum(1 for c in report.comparisons if all(s.passed for s in c.scores.values()))
        quality = passed_count / max(1, len(report.comparisons))

        # Reward prompt mutations that preserve variables
        vars_in_initial = set(re.findall(r"\{\{([a-zA-Z0-9_]+)\}\}", self.initial_prompt))
        vars_in_candidate = set(re.findall(r"\{\{([a-zA-Z0-9_]+)\}\}", prompt))
        if not vars_in_initial.issubset(vars_in_candidate):
            quality *= 0.5  # Penalize missing template variables

        avg_latency = report.verdict.avg_latency_v2
        tokens = estimate_tokens(prompt)
        cost = calculate_cost(self.model_name, tokens, max(10, tokens // 2))

        return ParetoMetrics(
            quality_score=round(quality, 3),
            latency_ms=round(avg_latency, 2),
            cost_usd=round(cost, 6),
            token_count=tokens,
        )

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select node to expand using UCB1."""
        curr = node
        while not curr.is_leaf():
            curr = max(curr.children, key=lambda child: child.ucb1(self.c))
        return curr

    def _expand(self, leaf: MCTSNode) -> list[MCTSNode]:
        """Expand leaf node by applying available prompt mutation operators."""
        children: list[MCTSNode] = []
        for name, mutator in self._get_mutators():
            new_prompt = mutator(leaf.prompt_template)
            if new_prompt != leaf.prompt_template:
                child = MCTSNode(
                    prompt_template=new_prompt,
                    parent=leaf,
                    action_applied=name,
                    depth=leaf.depth + 1,
                )
                children.append(child)
                self.all_nodes.append(child)
        leaf.children = children
        return children

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Propagate evaluation reward upward to the root."""
        curr: Optional[MCTSNode] = node
        while curr is not None:
            curr.visits += 1
            curr.total_reward += reward
            curr = curr.parent

    def _compute_pareto_frontier(self) -> list[MCTSNode]:
        """Extract non-dominated nodes across (Quality, Cost, Latency, Token Count)."""
        frontier: list[MCTSNode] = []
        for node in self.all_nodes:
            is_dominated = False
            for other in self.all_nodes:
                if other != node and other.metrics.dominates(node.metrics):
                    is_dominated = True
                    break
            if not is_dominated and node not in frontier:
                frontier.append(node)
        return frontier

    def _render_ascii_tree(self, root: MCTSNode) -> str:
        """Render a clean hierarchical ASCII visualization of the search tree."""
        lines = []

        def _traverse(node: MCTSNode, prefix: str = "", is_last: bool = True) -> None:
            connector = "└── " if is_last else "├── "
            lines.append(
                f"{prefix}{connector}[{node.action_applied}] "
                f"Visits={node.visits}, Quality={node.metrics.quality_score:.2f}, "
                f"Tokens={node.metrics.token_count}, Cost=${node.metrics.cost_usd:.6f}"
            )
            child_prefix = prefix + ("    " if is_last else "│   ")
            for idx, child in enumerate(node.children):
                _traverse(child, child_prefix, idx == len(node.children) - 1)

        lines.append(f"MCTS Tree Root (Depth=0): Visits={root.visits}, Quality={root.metrics.quality_score:.2f}")
        for idx, child in enumerate(root.children):
            _traverse(child, "", idx == len(root.children) - 1)
        return "\n".join(lines)

    async def optimize(self, num_iterations: Optional[int] = None) -> MCTSResult:
        """Run full Monte Carlo Tree Search optimization."""
        iterations = (
            validate_mcts_iterations(num_iterations)
            if num_iterations is not None
            else self.max_iterations
        )
        root_metrics = await self._evaluate_prompt(self.initial_prompt)
        root = MCTSNode(
            prompt_template=self.initial_prompt,
            action_applied="root_initial",
            depth=0,
            metrics=root_metrics,
        )
        self.all_nodes = [root]

        for _iteration in range(iterations):
            selected = self._select(root)
            if selected.visits > 0 or selected == root:
                expanded_children = self._expand(selected)
                target = expanded_children[0] if expanded_children else selected
            else:
                target = selected

            # Simulate evaluation
            metrics = await self._evaluate_prompt(target.prompt_template)
            target.metrics = metrics

            # Multi-objective composite reward (normalized 0 to 1)
            # Reward high quality, penalize high token cost/latency
            token_penalty = min(0.3, (metrics.token_count / 1000.0) * 0.1)
            reward = max(0.0, metrics.quality_score - token_penalty)

            # Backpropagate
            self._backpropagate(target, reward)

        # Select best overall node
        best_node = max(self.all_nodes, key=lambda n: (n.metrics.quality_score, -n.metrics.token_count))
        pareto_frontier = self._compute_pareto_frontier()
        tree_ascii = self._render_ascii_tree(root)

        return MCTSResult(
            best_prompt=best_node.prompt_template,
            best_quality=best_node.metrics.quality_score,
            initial_quality=root.metrics.quality_score,
            nodes_explored=len(self.all_nodes),
            pareto_frontier=pareto_frontier,
            tree_ascii=tree_ascii,
        )

    def optimize_sync(self, num_iterations: Optional[int] = None) -> MCTSResult:
        """Synchronous wrapper for MCTS optimization."""
        return asyncio.run(self.optimize(num_iterations=num_iterations))
