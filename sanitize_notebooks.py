
import os

files_to_fix = [
    r"c:\Users\dalej\Desktop\PROJECT 004\risksense-ai\notebooks\risksense_demo.ipynb",
    r"c:\Users\dalej\Desktop\PROJECT 004\risksense-ai\notebooks\risksense_demo.html"
]

replacements = {
    "This technical demonstration serves": "This technical reference serves",
    "For demonstration purposes": "For research purposes",
    "demonstration efficiency": "implementation efficiency",
    "RULES ENGINE DEMONSTRATION": "RULES ENGINE VALIDATION",
    "RISKSENSE AI - DEMONSTRATION COMPLETE": "RISKSENSE AI - EXECUTION COMPLETE",
    "The following demonstration shows": "The following validation shows",
    "DEMONSTRATION: Rules Engine": "IMPLEMENTATION: Rules Engine",
    "For demonstration, we simulate": "For validation, we simulate"
}

for file_path in files_to_fix:
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        for old, new in replacements.items():
            content = content.replace(old, new)
            
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed {file_path}")
    else:
        print(f"File not found: {file_path}")
