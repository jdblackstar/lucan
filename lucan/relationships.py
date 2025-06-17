"""Relationship tracking functionality for Lucan.

This module handles storing notes about the user's relationships when
Lucan decides something is worth remembering.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml


class RelationshipManager:
    """
    Manages relationship notes for the user.
    """

    def __init__(self, relationships_dir: Path) -> None:
        """
        Initialize the relationship manager.

        Args:
            relationships_dir: Path to the directory where relationship files are stored
        """
        self.relationships_dir = Path(relationships_dir)
        self.relationships_dir.mkdir(exist_ok=True)

    def _sanitize_filename(self, name: str) -> str:
        """
        Convert a name to a safe filename.

        Args:
            name: The name to sanitize

        Returns:
            A safe filename string
        """
        # Remove special characters and replace spaces with underscores
        sanitized = name.strip().lower()
        sanitized = "".join(c for c in sanitized if c.isalnum() or c in " -_")
        sanitized = sanitized.replace(" ", "_").replace("-", "_")
        return sanitized

    def add_note(self, name: str, relationship_type: str, note: str) -> bool:
        """
        Add a note about someone.

        Args:
            name: The name of the person/pet
            relationship_type: Type of relationship (e.g., "friend", "mother", "dog")
            note: The note to add (can be empty for initial record creation)

        Returns:
            True if successful, False otherwise
        """
        if not name.strip():
            return False

        filename = f"{self._sanitize_filename(name)}.txt"
        filepath = self.relationships_dir / filename

        current_time = datetime.now().isoformat()

        # Load existing data or create new
        if filepath.exists():
            content = filepath.read_text(encoding="utf-8")

            # Parse existing file
            if content.startswith("---\n"):
                parts = content.split("---\n", 2)
                if len(parts) >= 3:
                    frontmatter_text = parts[1]
                    notes_text = parts[2].strip()
                else:
                    frontmatter_text = ""
                    notes_text = content
            else:
                frontmatter_text = ""
                notes_text = content

            # Parse frontmatter
            try:
                frontmatter = (
                    yaml.safe_load(frontmatter_text) if frontmatter_text.strip() else {}
                )
                if frontmatter is None:
                    frontmatter = {}
            except yaml.YAMLError:
                frontmatter = {}

            frontmatter["last_updated"] = current_time

            # Parse existing notes
            notes = []
            if notes_text:
                for line in notes_text.split("\n"):
                    line = line.strip()
                    if line.startswith("-") or line.startswith("*"):
                        notes.append(line[1:].strip())
                    elif line:
                        notes.append(line)
        else:
            # New file
            frontmatter = {
                "name": name,
                "relationship": relationship_type,
                "first_mentioned": current_time,
                "last_updated": current_time,
            }
            notes = []

        # Update relationship type if provided
        if relationship_type:
            frontmatter["relationship"] = relationship_type

        # Add new note with date (only if note is not empty)
        if note.strip():
            notes.append(f"[{current_time[:10]}] {note.strip()}")

        # Write the file
        try:
            content = "---\n"
            content += yaml.dump(frontmatter, default_flow_style=False)
            content += "---\n\n"

            for note_item in notes:
                content += f"- {note_item}\n"

            filepath.write_text(content, encoding="utf-8")
            return True
        except Exception:
            return False

    def get_notes(self, name: str) -> Optional[Dict]:
        """
        Get notes about someone.

        Args:
            name: The name of the person/pet

        Returns:
            Dictionary with relationship information or None if not found
        """
        filename = f"{self._sanitize_filename(name)}.txt"
        filepath = self.relationships_dir / filename

        if not filepath.exists():
            return None

        content = filepath.read_text(encoding="utf-8")

        # Parse file
        if content.startswith("---\n"):
            parts = content.split("---\n", 2)
            if len(parts) >= 3:
                frontmatter_text = parts[1]
                notes_text = parts[2].strip()
            else:
                return None
        else:
            return None

        # Parse frontmatter
        try:
            frontmatter = (
                yaml.safe_load(frontmatter_text) if frontmatter_text.strip() else {}
            )
            if frontmatter is None:
                frontmatter = {}
        except yaml.YAMLError:
            return None

        # Parse notes
        notes = []
        if notes_text:
            for line in notes_text.split("\n"):
                line = line.strip()
                if line.startswith("-") or line.startswith("*"):
                    notes.append(line[1:].strip())
                elif line:
                    notes.append(line)

        return {
            "name": frontmatter.get("name", name),
            "relationship": frontmatter.get("relationship", ""),
            "notes": notes,
        }

    def find_by_relationship_type(self, relationship_type: str) -> List[Dict]:
        """
        Find people by their relationship type.

        Args:
            relationship_type: The type of relationship to search for

        Returns:
            List of dictionaries with relationship information
        """
        results = []
        relationship_type_lower = relationship_type.lower()

        for filepath in self.relationships_dir.glob("*.txt"):
            notes_data = self.get_notes(filepath.stem.replace("_", " ").title())
            if notes_data:
                stored_type = notes_data["relationship"].lower()

                # Check for exact match or common variations
                if (
                    relationship_type_lower == stored_type
                    or (
                        relationship_type_lower in ["mom", "mother"]
                        and stored_type in ["mom", "mother", "family"]
                    )
                    or (
                        relationship_type_lower in ["dad", "father"]
                        and stored_type in ["dad", "father", "family"]
                    )
                    or (
                        relationship_type_lower == "family"
                        and stored_type in ["mom", "mother", "dad", "father", "family"]
                    )
                    or (
                        relationship_type_lower in ["friend"]
                        and stored_type in ["friend"]
                    )
                    or (
                        relationship_type_lower in ["therapist", "counselor"]
                        and stored_type in ["therapist", "counselor"]
                    )
                    or (
                        relationship_type_lower in ["pet", "dog", "cat"]
                        and stored_type in ["pet", "dog", "cat"]
                    )
                ):
                    results.append(notes_data)

        return results
