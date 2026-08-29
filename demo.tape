# promptdiff v2.0 Terminal Demo Tape for Charmbracelet VHS
# Output target: assets/demo.gif
Output assets/demo.gif

# High-fidelity terminal aesthetics
Set FontSize 15
Set FontFamily "JetBrains Mono"
Set Width 1300
Set Height 850
Set LetterSpacing 0
Set LineHeight 1.25
Set Theme "TokyoNight"
Set Padding 24
Set Framerate 60
Set PlaybackSpeed 1.0
Set WindowBar Colorful

# 1. Warm-up and prompt initialization
Hide
Type "clear"
Enter
Show
Sleep 500ms

# 2. Run prompt regression test with side-by-side Rich diffs & LLM Judge
Type "promptdiff run prompts/system_v1.txt prompts/system_v2.txt --inputs testcases.jsonl --mock --eval json_validity,latency,cost,similarity,llm_judge"
Sleep 800ms
Enter

# Allow progress bar and table rendering
Sleep 6s

# 3. Demonstrate Multi-Model Arena (A/B/C/D)
Type "promptdiff arena --prompts prompts/system_v1.txt,prompts/system_v2.txt --models gpt-4o,claude-3-5-sonnet,gemini-2.0-flash --inputs testcases.jsonl --mock"
Sleep 800ms
Enter

# Allow arena leaderboard rendering
Sleep 6s

# Hold final frame
Sleep 3s
