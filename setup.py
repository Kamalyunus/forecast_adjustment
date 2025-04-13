"""
Setup script to prepare the environment for running the Forecast Adjustment RL system.
Creates necessary directories and empty __init__.py files.
"""

import os
import sys

def setup_environment():
    """Set up the environment for running the forecast adjustment RL system."""
    print("Setting up environment for Forecast Adjustment RL system...")
    
    # Create necessary directories
    dirs = [
        'data',
        'environment',
        'models',
        'saved_models',
        'plots'
    ]
    
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory: {d}")
    
    # Create empty __init__.py files
    modules = [
        'data',
        'environment',
        'models'
    ]
    
    for module in modules:
        init_file = os.path.join(module, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write(f"# {module.capitalize()} module\n")
            print(f"Created {init_file}")
    
    print("Setup complete. You can now run example.py")

if __name__ == "__main__":
    setup_environment()