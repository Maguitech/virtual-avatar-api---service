import os
import re

def fix_typeguard_import(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace the problematic imports
        changes_made = False
        
        # Fix check_argument_types import
        old_import1 = "from typeguard import check_argument_types"
        new_import1 = """# from typeguard import check_argument_types
def check_argument_types():
    return True"""
        
        if old_import1 in content:
            content = content.replace(old_import1, new_import1)
            changes_made = True
        
        # Fix check_return_type import
        old_import2 = "from typeguard import check_return_type"
        new_import2 = """# from typeguard import check_return_type
def check_return_type(value):
    return True"""
        
        if old_import2 in content:
            content = content.replace(old_import2, new_import2)
            changes_made = True
        
        # Fix mixed imports
        old_import3 = "from typeguard import check_argument_types, check_return_type"
        new_import3 = """# from typeguard import check_argument_types, check_return_type
def check_argument_types():
    return True
def check_return_type(value):
    return True"""
        
        if old_import3 in content:
            content = content.replace(old_import3, new_import3)
            changes_made = True
            
        if changes_made:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {file_path}")
            return True
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False
    return False

# List of files that need fixing
files_to_fix = [
    "extract_paraformer_feature.py",
    "funasr_local/tasks/asr.py"
]

# Add more files from the funasr_local directory
for root, dirs, files in os.walk("funasr_local"):
    for file in files:
        if file.endswith('.py'):
            file_path = os.path.join(root, file)
            files_to_fix.append(file_path)

# Remove duplicates
files_to_fix = list(set(files_to_fix))

# Fix all files
fixed_count = 0
for file_path in files_to_fix:
    if os.path.exists(file_path):
        if fix_typeguard_import(file_path):
            fixed_count += 1

print(f"Fixed {fixed_count} files")