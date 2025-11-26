"""
>> PYTHON PROJECT INITIALIZER <<
--------------------------------

This script creates lets user input a folder name, and then
creates a template project for data analysis. It will create
organized folders where you will store your files while ana-
lyzing data. 

The goal is pure organization and readability so others will
know what to look for when checking out your work.
--------------------------------

This version comes with `requirements.txt` containing basic
packages for data analysis and visualization and an empty
`README.md` file, no template.

"""

# -----
# About:
# -----

AUTHOR = 'D-Jlvc'
VERSION = 1.01


# -------
# Imports:
# -------

from pathlib import Path
import subprocess


# -------
# Program:
# -------

def info():
    """Returns and info about this script."""
    return f">>. Author: {AUTHOR} | Version: {VERSION}."


def create_project():
    """Creates a complete working directories for data analysis projects."""
    
    # -- User input:
    while True:
        
        project_name = input(">. Enter project name: ").strip()
    
        base_path = Path.cwd()/project_name  # -- Path: current working directory/project_name
    
        if base_path.exists():
            print(f"Folder '{project_name}' already exists!")
            continue
    
        base_path.mkdir()  # -- creating a folder.
    
    
        subfolders = [
            'datasets',
            'exports',
            'models',
            'model_source',
            'notebooks'
        ]
    
        for folder in subfolders:
            (base_path/folder).mkdir()  # -- creating subfolders.
        
    
    # -- Aditional files:
        (base_path/"README.md").write_text("")
        (base_path/"requirements.txt").write_text("pandas\nipykernel\njupyter\nmatplotlib\nseaborn\nscikitlearn")
    
        print(f">>. Project '{project_name}' successfuly created!")
        print(info())
        break


def git_init(base_path: Path):
    """Initializes git repository and, if not existent, creates a .gitignore file."""
    
    # -- Git init:
    subprocess.run(['git', 'init', str(base_path)])
    
    # -- .gitignore:
    gitignore_path = base_path/".gitignore"
    
    if not gitignore_path.exists():
        gitignore_content = "__pycache__/\n*.pyc\n#*.pkl\n.env\n.DS_Store"
        gitignore_path.write_text(gitignore_content)
    else:
        print("'.gitignore' file already exists. Try searching hidden files...")    


# -- Main:
def main() -> None:
    create_project()

if __name__ == "__main__":
    main()
    