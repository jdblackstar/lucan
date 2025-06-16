#!/usr/bin/env python3
"""
Test script to run EQBench comparison between Lucan and Claude.

Usage:
    python run_eqbench_comparison.py [--debug] [--persona PERSONA_NAME]
"""

import asyncio
import sys
from pathlib import Path

# Add the current directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from lucan.core import LucanChat  
from eval.eqbench_comparison import EQBenchTester


async def main() -> None:
    """Run the EQBench comparison between Lucan and Claude."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run EQBench comparison between Lucan and Claude")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--persona", default="lucan", help="Persona to use for Lucan (default: lucan)")
    parser.add_argument("--claude-model", default="claude-sonnet-4-20250514", help="Claude model to use")
    parser.add_argument("--output-dir", type=Path, default=Path("eqbench_results"), help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(exist_ok=True)
    
    print("EQBench Comparison: Lucan vs Claude")
    print("=" * 50)
    
    try:
        # Initialize Lucan
        persona_path = Path("memory/personas") / args.persona
        if not persona_path.exists():
            print(f"Error: Persona path {persona_path} does not exist")
            print("Available personas:")
            personas_dir = Path("memory/personas")
            if personas_dir.exists():
                for persona_dir in personas_dir.iterdir():
                    if persona_dir.is_dir():
                        print(f"  - {persona_dir.name}")
            return
        
        print(f"Initializing Lucan with persona: {args.persona}")
        lucan_chat = LucanChat(persona_path, debug=args.debug)
        
        # Initialize EQBench tester
        print("Initializing EQBench tester...")
        tester = EQBenchTester(debug=args.debug)
        
        # Load scenarios
        print("Loading EQBench scenarios...")
        await tester.load_eqbench_scenarios()
        
        # Run comparison
        print("Running EQBench comparison...")
        lucan_result, claude_result = await tester.run_comparison(lucan_chat, args.claude_model)
        
        # Generate and save report
        print("Generating comparison report...")
        report_file = args.output_dir / "eqbench_comparison_report.md"
        
        # Save detailed results
        csv_file = args.output_dir / "eqbench_detailed_results.csv"
        tester.save_detailed_results(lucan_result, claude_result, csv_file)
        
        # Print summary
        print("\nRESULTS SUMMARY")
        print("=" * 30)
        print(f"Lucan Score:  {lucan_result.total_score:.2f}/100")
        print(f"Claude Score: {claude_result.total_score:.2f}/100")
        
        if lucan_result.total_score > claude_result.total_score:
            print(f"Winner: Lucan (+{lucan_result.total_score - claude_result.total_score:.2f} points)")
        else:
            print(f"Winner: Claude (+{claude_result.total_score - lucan_result.total_score:.2f} points)")
        
        print(f"\nFull report saved to: {report_file}")
        print(f"Detailed results saved to: {csv_file}")
        
        # Show some example responses
        print("\nEXAMPLE RESPONSES")
        print("=" * 30)
        if lucan_result.raw_responses and claude_result.raw_responses:
            scenario = tester.scenarios[0]
            print(f"Scenario: {scenario.id}")
            print(f"Expected emotions: {scenario.emotions}")
            print("\nLucan response (first 200 chars):")
            print(f"  {lucan_result.raw_responses[0][:200]}...")
            print("\nClaude response (first 200 chars):")
            print(f"  {claude_result.raw_responses[0][:200]}...")
        
        print("\nEQBench comparison completed successfully!")
        
    except Exception as e:
        print(f"Error during EQBench comparison: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 