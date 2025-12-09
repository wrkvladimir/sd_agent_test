from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

from .config import settings
from .schemas import ScenarioDefinition


logger = logging.getLogger("chat_app.scenario_registry")


class ScenarioRegistry:
    """
    In-memory registry of scenarios available to the agent.

    Scenarios can be loaded from disk on startup and updated via HTTP API.
    """

    def __init__(self) -> None:
        self._scenarios: Dict[str, ScenarioDefinition] = {}

    def add(self, scenario: ScenarioDefinition) -> None:
        self._scenarios[scenario.name] = scenario

    def get(self, name: str) -> ScenarioDefinition | None:
        return self._scenarios.get(name)

    def all(self) -> Dict[str, ScenarioDefinition]:
        return dict(self._scenarios)

    def load_default_from_disk(self) -> None:
        """
        Load default scenario definition from JSON file, if present.

        By convention, uses `test_scenario.json` in the configured
        scenario storage directory.
        """
        path = Path(settings.scenario_storage_path) / "test_scenario.json"
        if not path.exists():
            logger.warning("default scenario file not found at %s", path)
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            scenario = ScenarioDefinition.model_validate(data)
            self.add(scenario)
            logger.info("loaded default scenario %s from %s", scenario.name, path)
        except Exception as exc:  # noqa: BLE001
            logger.error("failed to load default scenario from %s: %s", path, exc)


registry = ScenarioRegistry()

