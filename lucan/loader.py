from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


class Lucan:
    def __init__(self, base_path: Path):
        self.personality_file = base_path / "personality.txt"
        self.modifiers_file = base_path / "modifiers.txt"

        self.personality: Dict[str, Any] = {}
        self.modifiers: Dict[str, int] = {}

        self.load()

    def load(self):
        with open(self.personality_file, "r") as f:
            self.personality = yaml.safe_load(f)

        with open(self.modifiers_file, "r") as f:
            self.modifiers = yaml.safe_load(f).get("modifiers", {})

    def save_modifiers(self) -> None:
        """Save current modifiers back to the modifiers.txt file."""
        data = {"modifiers": self.modifiers}
        with open(self.modifiers_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def set_modifier(self, key: str, value: int) -> Tuple[bool, str]:
        """
        Set a modifier to a specific value, respecting bounds.

        Args:
            key: The modifier to set (e.g., 'warmth', 'verbosity')
            value: The target value to set

        Returns:
            Tuple of (success: bool, message: str)
        """
        if key not in self.modifiers:
            return False, f"Unknown modifier: {key}"

        # Enforce bounds [-3, +3]
        if value < -3:
            value = -3
        elif value > 3:
            value = 3

        old_value = self.modifiers[key]
        self.modifiers[key] = value
        self.save_modifiers()

        return True, f"Set {key} from {old_value} to {value}"

    def adjust_modifier(self, key: str, adjustment: int) -> Tuple[bool, str]:
        """
        Adjust a modifier by the specified amount, respecting bounds.

        Args:
            key: The modifier to adjust (e.g., 'warmth', 'verbosity')
            adjustment: Amount to adjust by (positive or negative)

        Returns:
            Tuple of (success: bool, message: str)
        """
        if key not in self.modifiers:
            return False, f"Unknown modifier: {key}"

        current_value = self.modifiers[key]
        new_value = current_value + adjustment

        # Enforce bounds [-3, +3]
        if new_value < -3:
            new_value = -3
        elif new_value > 3:
            new_value = 3

        if new_value == current_value:
            return (
                False,
                f"Modifier '{key}' is already at the boundary (current: {current_value})",
            )

        old_value = self.modifiers[key]
        self.modifiers[key] = new_value
        self.save_modifiers()

        return True, f"Adjusted {key} from {old_value} to {new_value}"

    def build_prompt_profile(self) -> str:
        """
        Combine base personality and modifiers into a prompt-style instruction string.
        """
        profile = f"You are {self.personality.get('name', 'Lucan')}, {self.personality.get('description', '').strip()}\n\n"

        # Add modifier instructions if any modifiers are set
        modifiers = []
        for key, value in self.modifiers.items():
            if value != 0:
                modifiers.append(f"{key}: {value}")

        if modifiers:
            profile += "Personality modifiers (scale -3 to +3, where -3 is extreme negative, 0 is neutral, +3 is extreme positive):\n"
            for modifier in modifiers:
                profile += f"- {modifier}\n"
            profile += "\nAdjust your personality accordingly based on these modifier values.\n"

        return profile
