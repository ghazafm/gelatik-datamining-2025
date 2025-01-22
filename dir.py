import os

def map_directory_structure(root_dir):
    for root, dirs, files in os.walk(root_dir):
        # Print the current directory
        print(f"Directory: {root}")
        
        # Print subdirectories
        if dirs:
            print(" Subdirectories:")
            for dir in dirs:
                print(f"  - {dir}")
        
        # Print files
        if files:
            print(" Files:")
            for file in files:
                print(f"  - {file}")
        
        print("-" * 40)

if __name__ == "__main__":
    project_dir = '.'  # Change this to your project directory path
    map_directory_structure(project_dir)
