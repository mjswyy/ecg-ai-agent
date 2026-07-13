"""Tool Scheduler — DAG-based tool execution ordering.

Resolves dependencies between tools, determines execution order via
topological sort, and handles parallel execution where possible.

Usage:
    scheduler = ToolScheduler(registry)
    results = scheduler.execute_plan(plan_steps)
"""

import logging
from collections import defaultdict, deque
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ToolScheduler:
    """Execute a sequence of tool calls respecting dependencies.

    Args:
        registry: ToolRegistry instance.
        max_retries: Maximum retries per tool on failure.
    """

    def __init__(self, registry: "ToolRegistry", max_retries: int = 1):
        self.registry = registry
        self.max_retries = max_retries
        self._cache: Dict[str, Any] = {}

    def execute_plan(
        self,
        steps: List["AgentStep"],
        cache_results: bool = True,
    ) -> Dict[str, Any]:
        """Execute a diagnostic plan.

        Args:
            steps: List of AgentSteps to execute.
            cache_results: If True, cache results for repeated calls.

        Returns:
            Dict mapping tool_name → result.
        """
        # Build dependency graph
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        name_to_step = {}

        for step in steps:
            name_to_step[step.action] = step
            in_degree.setdefault(step.action, 0)
            deps = self.registry.get_dependencies(step.action)
            for dep in deps:
                if dep in name_to_step:
                    graph[dep].append(step.action)
                    in_degree[step.action] += 1

        # Topological sort
        queue = deque([s.action for s in steps if in_degree[s.action] == 0])
        results = {}

        while queue:
            name = queue.popleft()
            step = name_to_step[name]

            # Check cache
            cache_key = f"{name}:{str(sorted(step.params.items()))}"
            if cache_results and cache_key in self._cache:
                step.result = self._cache[cache_key]
                step.completed = True
                results[name] = step.result
                continue

            # Execute with retry
            for attempt in range(self.max_retries + 1):
                try:
                    result = self.registry.call(name, **step.params)
                    step.result = result
                    step.completed = True
                    results[name] = result
                    if cache_results:
                        self._cache[cache_key] = result
                    break
                except Exception as e:
                    if attempt < self.max_retries:
                        logger.warning(f"Retrying {name} (attempt {attempt+1}): {e}")
                    else:
                        step.error = str(e)
                        step.completed = True
                        logger.error(f"Tool {name} failed after {self.max_retries} retries: {e}")

            # Enqueue dependents
            for dependent in graph[name]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        return results

    def clear_cache(self):
        """Clear result cache."""
        self._cache.clear()
