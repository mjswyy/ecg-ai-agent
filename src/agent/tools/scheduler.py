# ⚠️ DEPRECATED（2026-08-24，检查报告第六步）: ToolScheduler 未接入主流程（agent 自行为 ReAct 循环）；环检测等缺陷已修复，保留备接入
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

        # 先注册全部步骤名，再连边（旧版在同一次遍历中连边，
        # 依赖指向尚未遍历到的步骤时会被误判为"不在计划中"）
        for step in steps:
            name_to_step[step.action] = step
            in_degree.setdefault(step.action, 0)
        for step in steps:
            deps = self.registry.get_dependencies(step.action)
            for dep in deps:
                if dep in name_to_step:
                    graph[dep].append(step.action)
                    in_degree[step.action] += 1
                else:
                    # 检查报告 1.9 修复：依赖工具不在计划中时显式告警
                    logger.warning(
                        f"步骤 {step.action} 依赖 {dep}，但 {dep} 不在计划中")

        # Topological sort
        queue = deque([s.action for s in steps if in_degree[s.action] == 0])
        results = {}
        executed = set()
        failed_set = set()  # 失败/被跳过节点（其依赖者递归跳过）

        while queue:
            name = queue.popleft()
            step = name_to_step[name]

            # 检查报告 1.9 修复：任一直接依赖失败/跳过 → 本步骤标记 skipped，
            # 而不是带着缺失前置继续执行
            failed_deps = [d for d in self.registry.get_dependencies(name)
                           if d in failed_set]
            if failed_deps:
                step.error = f"依赖步骤 {failed_deps} 失败/跳过"
                step.completed = True
                failed_set.add(name)
                executed.add(name)
                logger.warning(f"跳过 {name}: 依赖 {failed_deps} 失败/跳过")
                self._release_dependents(name, graph, in_degree, queue)
                continue

            # Check cache
            cache_key = f"{name}:{str(sorted(step.params.items()))}"
            if cache_results and cache_key in self._cache:
                step.result = self._cache[cache_key]
                step.completed = True
                results[name] = step.result
                executed.add(name)
                self._release_dependents(name, graph, in_degree, queue)
                continue

            # Execute with retry
            failed = False
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
                        failed = True
                        logger.error(f"Tool {name} failed after {self.max_retries} retries: {e}")

            executed.add(name)
            if failed:
                failed_set.add(name)
            self._release_dependents(name, graph, in_degree, queue)

        # 检查报告 1.9 修复：Kahn 结束仍有未执行步骤 → 存在依赖环，
        # 显式报错而不是静默返回不完整结果
        unexecuted = [s.action for s in steps if s.action not in executed]
        if unexecuted:
            raise ValueError(
                f"依赖环检测：以下步骤因循环依赖未执行: {unexecuted}")

        return results

    def _release_dependents(self, name, graph, in_degree, queue):
        """依赖满足后释放后继节点入队（失败/跳过经由 failed_set 传播）。"""
        for dependent in graph[name]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    def clear_cache(self):
        """Clear result cache."""
        self._cache.clear()
