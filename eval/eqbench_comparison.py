#!/usr/bin/env python3
"""
EQBench comparison framework for testing Lucan vs. Claude on emotional intelligence.

This module integrates the official EQBench test scenarios with your Lucan system
to provide head-to-head comparisons on emotional intelligence tasks.
"""

from __future__ import annotations

import csv
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from anthropic import Anthropic
from openai import AsyncOpenAI


@dataclass
class EQBenchResult:
    """Results from an EQBench evaluation."""

    model_name: str
    total_score: float
    question_scores: List[float]
    response_times: List[float]
    raw_responses: List[str]
    scenarios_tested: int


@dataclass
class EQBenchScenario:
    """An EQBench test scenario."""

    id: str
    dialogue: str
    target_character: str
    emotions: Dict[str, int]  # emotion_name -> reference_intensity (0-10)
    context: Optional[str] = None


class EQBenchTester:
    """
    Framework for running EQBench comparisons between Lucan and other models.

    This class handles:
    - Loading EQBench test scenarios
    - Running tests on both Lucan and comparison models
    - Scoring responses according to EQBench methodology
    - Generating comparison reports
    """

    def __init__(self, debug: bool = False):
        """Initialize the EQBench testing framework."""
        self.debug = debug
        self.scenarios: List[EQBenchScenario] = []
        self.anthropic_client = Anthropic()
        self.openai_client = AsyncOpenAI()

    async def load_eqbench_scenarios(
        self, scenario_file: Optional[Path] = None
    ) -> None:
        """
        Load EQBench test scenarios.

        Args:
            scenario_file: Path to custom scenarios file, or None to use built-in scenarios
        """
        if scenario_file and scenario_file.exists():
            await self._load_custom_scenarios(scenario_file)
        else:
            self._load_builtin_scenarios()

    def _load_builtin_scenarios(self) -> None:
        """Load a representative set of EQBench-style scenarios for testing."""
        scenarios_data = [
            {
                "id": "relationship_conflict_01",
                "dialogue": """
                Sarah: "I can't believe you forgot our anniversary again, Michael. This is the third year in a row."
                Michael: "I... I'm sorry, Sarah. Work has been so crazy lately, I completely lost track of time."
                Sarah: "Lost track of time? Our anniversary isn't just any date, Michael. It's supposed to mean something to us."
                Michael: "You're right. I know I messed up. I don't know what to say."
                Sarah: "Maybe that's the problem. You never know what to say when it really matters."
                """,
                "target_character": "Sarah",
                "emotions": {"disappointment": 9, "anger": 6, "sadness": 7, "love": 4},
                "context": "Sarah and Michael have been married for 5 years. This pattern of forgotten anniversaries represents a deeper issue in their relationship.",
            },
            {
                "id": "workplace_feedback_01",
                "dialogue": """
                Manager: "I need to talk to you about your recent performance, Alex."
                Alex: "Oh... okay. Is everything alright?"
                Manager: "Well, I've noticed you've been missing some deadlines, and the quality of your work isn't quite up to your usual standards."
                Alex: "I... I didn't realize it was that noticeable. I've been dealing with some personal stuff."
                Manager: "I understand personal issues can be challenging, but we need to discuss how to get you back on track."
                """,
                "target_character": "Alex",
                "emotions": {"anxiety": 8, "shame": 7, "worry": 8, "defensiveness": 5},
                "context": "Alex has been struggling with family issues at home but hasn't communicated this to their manager until now.",
            },
            {
                "id": "friendship_betrayal_01",
                "dialogue": """
                Emma: "I heard what you said about me at the party last night, Jordan."
                Jordan: "What do you mean? I don't remember saying anything bad about you."
                Emma: "Really? Because three different people told me you were talking about how I 'always make everything about myself.'"
                Jordan: "Emma, I... I was just venting. I didn't think it would get back to you."
                Emma: "Venting? About your best friend? To people we both know?"
                """,
                "target_character": "Emma",
                "emotions": {"betrayal": 9, "hurt": 8, "anger": 7, "confusion": 6},
                "context": "Emma and Jordan have been best friends for 10 years. This is the first major conflict in their friendship.",
            },
            {
                "id": "parent_child_discipline_01",
                "dialogue": """
                Parent: "We need to talk about what happened at school today, Tyler."
                Tyler: "I already told you, it wasn't my fault. Jason started it."
                Parent: "The teacher said you were the one who threw the first punch."
                Tyler: "Because he was making fun of my stutter in front of everyone! I couldn't just let him do that!"
                Parent: "I understand you were hurt, but violence is never the answer. You know that."
                Tyler: "So I'm just supposed to let people make fun of me?"
                """,
                "target_character": "Tyler",
                "emotions": {
                    "frustration": 8,
                    "shame": 7,
                    "anger": 6,
                    "vulnerability": 8,
                },
                "context": "Tyler is 12 years old and has struggled with a stutter since childhood. This incident represents his growing frustration with bullying.",
            },
            {
                "id": "medical_diagnosis_01",
                "dialogue": """
                Doctor: "I have the results of your tests, Jennifer. I'd like to discuss them with you."
                Jennifer: "Okay... is it bad news?"
                Doctor: "The tests show some concerning abnormalities. We'll need to do more extensive testing to determine the exact nature of what we're seeing."
                Jennifer: "Concerning abnormalities? What does that mean exactly?"
                Doctor: "I don't want to speculate until we have more information, but I want you to know we're going to take very good care of you."
                """,
                "target_character": "Jennifer",
                "emotions": {
                    "fear": 9,
                    "anxiety": 9,
                    "uncertainty": 8,
                    "vulnerability": 8,
                },
                "context": "Jennifer is 34 years old and has been experiencing unexplained symptoms for several weeks. This is her first major health scare.",
            },
        ]

        self.scenarios = [
            EQBenchScenario(
                id=s["id"],
                dialogue=s["dialogue"],
                target_character=s["target_character"],
                emotions=s["emotions"],
                context=s.get("context"),
            )
            for s in scenarios_data
        ]

        if self.debug:
            print(f"[DEBUG] Loaded {len(self.scenarios)} built-in EQBench scenarios")

    async def _load_custom_scenarios(self, scenario_file: Path) -> None:
        """Load scenarios from a custom JSON file."""
        with open(scenario_file, "r") as f:
            scenarios_data = json.load(f)

        self.scenarios = [
            EQBenchScenario(
                id=s["id"],
                dialogue=s["dialogue"],
                target_character=s["target_character"],
                emotions=s["emotions"],
                context=s.get("context"),
            )
            for s in scenarios_data
        ]

        if self.debug:
            print(
                f"[DEBUG] Loaded {len(self.scenarios)} custom scenarios from {scenario_file}"
            )

    def _build_eqbench_prompt(self, scenario: EQBenchScenario) -> str:
        """
        Build the EQBench prompt for a scenario.

        This follows the official EQBench format where the model needs to predict
        emotional intensity ratings (0-10) for the target character.
        """
        emotions_list = list(scenario.emotions.keys())

        prompt = f"""Please read the following dialogue carefully and predict the emotional intensity that {scenario.target_character} is likely experiencing.

Dialogue:
{scenario.dialogue.strip()}

Context: {scenario.context or "No additional context provided."}

For {scenario.target_character}, please rate the intensity of each emotion on a scale of 0-10, where:
- 0 = not experiencing this emotion at all
- 10 = experiencing this emotion extremely intensely

Emotions to rate:
{chr(10).join(f"- {emotion}" for emotion in emotions_list)}

Please provide your response in the following format:
{chr(10).join(f"{emotion}: [0-10]" for emotion in emotions_list)}

Then provide a brief explanation of your reasoning.
"""
        return prompt

    async def test_lucan(
        self, scenario: EQBenchScenario, lucan_chat
    ) -> Tuple[Dict[str, int], str, float]:
        """
        Test Lucan on an EQBench scenario.

        Args:
            scenario: The EQBench scenario to test
            lucan_chat: Instance of LucanChat

        Returns:
            Tuple of (emotion_ratings, full_response, response_time)
        """
        prompt = self._build_eqbench_prompt(scenario)

        start_time = time.time()
        response = lucan_chat.send_message(prompt)
        end_time = time.time()

        response_time = end_time - start_time

        # Parse emotion ratings from response
        emotion_ratings = self._parse_emotion_ratings(
            response, list(scenario.emotions.keys())
        )

        return emotion_ratings, response, response_time

    async def test_claude(
        self, scenario: EQBenchScenario, model: str = "claude-sonnet-4-20250514"
    ) -> Tuple[Dict[str, int], str, float]:
        """
        Test Claude on an EQBench scenario.

        Args:
            scenario: The EQBench scenario to test
            model: Claude model to use

        Returns:
            Tuple of (emotion_ratings, full_response, response_time)
        """
        prompt = self._build_eqbench_prompt(scenario)

        start_time = time.time()
        response = self.anthropic_client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=0.1,  # Low temperature for consistent results
            messages=[{"role": "user", "content": prompt}],
        )
        end_time = time.time()

        response_time = end_time - start_time
        response_text = response.content[0].text

        # Parse emotion ratings from response
        emotion_ratings = self._parse_emotion_ratings(
            response_text, list(scenario.emotions.keys())
        )

        return emotion_ratings, response_text, response_time

    def _parse_emotion_ratings(
        self, response: str, expected_emotions: List[str]
    ) -> Dict[str, int]:
        """
        Parse emotion ratings from model response.

        Looks for patterns like "emotion: 7" or "emotion = 5" in the response.
        """
        emotion_ratings = {}

        for emotion in expected_emotions:
            # Try multiple patterns to find the rating
            patterns = [
                f"{emotion.lower()}:\\s*(\\d+)",
                f"{emotion.lower()}\\s*=\\s*(\\d+)",
                f"{emotion.lower()}\\s*-\\s*(\\d+)",
                f"{emotion.lower()}.*?(\\d+)",
            ]

            import re

            for pattern in patterns:
                match = re.search(pattern, response.lower())
                if match:
                    try:
                        rating = int(match.group(1))
                        if 0 <= rating <= 10:
                            emotion_ratings[emotion] = rating
                            break
                    except ValueError:
                        continue

            # If no rating found, default to 5 (neutral)
            if emotion not in emotion_ratings:
                emotion_ratings[emotion] = 5
                if self.debug:
                    print(
                        f"[DEBUG] Could not parse rating for {emotion}, defaulting to 5"
                    )

        return emotion_ratings

    def _calculate_eqbench_score(
        self, predicted: Dict[str, int], reference: Dict[str, int]
    ) -> float:
        """
        Calculate EQBench score using the official methodology.

        The score is based on the average absolute difference between predicted
        and reference emotional intensity ratings.
        """
        if not predicted or not reference:
            return 0.0

        total_diff = 0
        count = 0

        for emotion in reference:
            if emotion in predicted:
                diff = abs(predicted[emotion] - reference[emotion])
                total_diff += diff
                count += 1

        if count == 0:
            return 0.0

        # Convert to EQBench-style score (higher is better)
        # Perfect score (no difference) = 100, worst score (max difference) approaches 0
        avg_diff = total_diff / count
        score = max(0, 100 - (avg_diff * 10))  # Scale the score

        return score

    async def run_comparison(
        self, lucan_chat, claude_model: str = "claude-sonnet-4-20250514"
    ) -> Tuple[EQBenchResult, EQBenchResult]:
        """
        Run a full EQBench comparison between Lucan and Claude.

        Returns:
            Tuple of (lucan_results, claude_results)
        """
        if not self.scenarios:
            await self.load_eqbench_scenarios()

        print(f"Running EQBench comparison: Lucan vs {claude_model}")
        print(f"Testing on {len(self.scenarios)} scenarios...")

        lucan_scores = []
        lucan_times = []
        lucan_responses = []

        claude_scores = []
        claude_times = []
        claude_responses = []

        for i, scenario in enumerate(self.scenarios, 1):
            print(f"  Scenario {i}/{len(self.scenarios)}: {scenario.id}")

            # Test Lucan
            try:
                lucan_ratings, lucan_response, lucan_time = await self.test_lucan(
                    scenario, lucan_chat
                )
                lucan_score = self._calculate_eqbench_score(
                    lucan_ratings, scenario.emotions
                )
                lucan_scores.append(lucan_score)
                lucan_times.append(lucan_time)
                lucan_responses.append(lucan_response)

                if self.debug:
                    print(f"    Lucan score: {lucan_score:.1f}")
            except Exception as e:
                print(f"    Error testing Lucan: {e}")
                lucan_scores.append(0.0)
                lucan_times.append(0.0)
                lucan_responses.append("ERROR")

            # Test Claude
            try:
                claude_ratings, claude_response, claude_time = await self.test_claude(
                    scenario, claude_model
                )
                claude_score = self._calculate_eqbench_score(
                    claude_ratings, scenario.emotions
                )
                claude_scores.append(claude_score)
                claude_times.append(claude_time)
                claude_responses.append(claude_response)

                if self.debug:
                    print(f"    Claude score: {claude_score:.1f}")
            except Exception as e:
                print(f"    Error testing Claude: {e}")
                claude_scores.append(0.0)
                claude_times.append(0.0)
                claude_responses.append("ERROR")

        # Create results
        lucan_result = EQBenchResult(
            model_name="Lucan",
            total_score=statistics.mean(lucan_scores) if lucan_scores else 0.0,
            question_scores=lucan_scores,
            response_times=lucan_times,
            raw_responses=lucan_responses,
            scenarios_tested=len(self.scenarios),
        )

        claude_result = EQBenchResult(
            model_name=claude_model,
            total_score=statistics.mean(claude_scores) if claude_scores else 0.0,
            question_scores=claude_scores,
            response_times=claude_times,
            raw_responses=claude_responses,
            scenarios_tested=len(self.scenarios),
        )

        return lucan_result, claude_result

    def generate_report(
        self,
        lucan_result: EQBenchResult,
        claude_result: EQBenchResult,
        output_file: Optional[Path] = None,
    ) -> str:
        """
        Generate a comprehensive comparison report.

        Args:
            lucan_result: Results from testing Lucan
            claude_result: Results from testing Claude
            output_file: Optional file path to save the report

        Returns:
            The report as a string
        """
        report_lines = [
            "# EQBench Comparison Report: Lucan vs Claude",
            "",
            "## Overall Results",
            f"- **Lucan Total Score**: {lucan_result.total_score:.2f}/100",
            f"- **Claude Total Score**: {claude_result.total_score:.2f}/100",
            f"- **Winner**: {'Lucan' if lucan_result.total_score > claude_result.total_score else 'Claude'}",
            f"- **Score Difference**: {abs(lucan_result.total_score - claude_result.total_score):.2f} points",
            "",
            "## Performance Metrics",
            f"- **Scenarios Tested**: {lucan_result.scenarios_tested}",
            f"- **Lucan Avg Response Time**: {statistics.mean(lucan_result.response_times):.2f}s",
            f"- **Claude Avg Response Time**: {statistics.mean(claude_result.response_times):.2f}s",
            "",
            "## Detailed Score Breakdown",
            "",
        ]

        # Add scenario-by-scenario comparison
        for i, scenario in enumerate(self.scenarios):
            lucan_score = (
                lucan_result.question_scores[i]
                if i < len(lucan_result.question_scores)
                else 0
            )
            claude_score = (
                claude_result.question_scores[i]
                if i < len(claude_result.question_scores)
                else 0
            )
            winner = "Lucan" if lucan_score > claude_score else "Claude"

            report_lines.extend(
                [
                    f"### Scenario {i + 1}: {scenario.id}",
                    f"- **Lucan**: {lucan_score:.1f}/100",
                    f"- **Claude**: {claude_score:.1f}/100",
                    f"- **Winner**: {winner}",
                    "",
                ]
            )

        # Add statistical analysis
        if (
            len(lucan_result.question_scores) > 1
            and len(claude_result.question_scores) > 1
        ):
            lucan_std = statistics.stdev(lucan_result.question_scores)
            claude_std = statistics.stdev(claude_result.question_scores)

            report_lines.extend(
                [
                    "## Statistical Analysis",
                    f"- **Lucan Score Std Dev**: {lucan_std:.2f}",
                    f"- **Claude Score Std Dev**: {claude_std:.2f}",
                    f"- **Lucan Consistency**: {'High' if lucan_std < 10 else 'Medium' if lucan_std < 20 else 'Low'}",
                    f"- **Claude Consistency**: {'High' if claude_std < 10 else 'Medium' if claude_std < 20 else 'Low'}",
                    "",
                ]
            )

        report = "\n".join(report_lines)

        if output_file:
            output_file.write_text(report)
            print(f"Report saved to {output_file}")

        return report

    def save_detailed_results(
        self,
        lucan_result: EQBenchResult,
        claude_result: EQBenchResult,
        output_file: Path,
    ) -> None:
        """Save detailed results to CSV for further analysis."""
        with open(output_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Scenario_ID",
                    "Lucan_Score",
                    "Claude_Score",
                    "Lucan_Time",
                    "Claude_Time",
                    "Winner",
                ]
            )

            for i, scenario in enumerate(self.scenarios):
                lucan_score = (
                    lucan_result.question_scores[i]
                    if i < len(lucan_result.question_scores)
                    else 0
                )
                claude_score = (
                    claude_result.question_scores[i]
                    if i < len(claude_result.question_scores)
                    else 0
                )
                lucan_time = (
                    lucan_result.response_times[i]
                    if i < len(lucan_result.response_times)
                    else 0
                )
                claude_time = (
                    claude_result.response_times[i]
                    if i < len(claude_result.response_times)
                    else 0
                )
                winner = "Lucan" if lucan_score > claude_score else "Claude"

                writer.writerow(
                    [
                        scenario.id,
                        lucan_score,
                        claude_score,
                        lucan_time,
                        claude_time,
                        winner,
                    ]
                )

        print(f"Detailed results saved to {output_file}")
