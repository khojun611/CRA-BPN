import subprocess
import sys
import os

def run_script(script_path, *args):
    try:
        # Create a txt file with "a" content before running the script
        with open(os.path.join(os.path.dirname(script_path), "before_run.txt"), "a") as file:
            file.write("a\n")

        result = subprocess.run([sys.executable, script_path] + list(args), check=True, capture_output=True, text=True)
        print(f"Output of {script_path}:")
        print(result.stdout)
        print(result.stderr)  # Added to show error output as well
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_path}:")
        print(e.stderr)
        sys.exit(e.returncode)

current_dir = os.path.dirname(os.path.abspath(__file__))
image_folder_path = os.path.join(current_dir, "ICDAR13")  # 이미지 경로 설정
craft_model = os.path.join(current_dir, 'CRAFT-pytorch/weights/craft_mlt_25k.pth')
output_folder_path = os.path.join(current_dir, "inference")  # output 경로 설정

if __name__ == "__main__":
    # Project 1 - test.py
    project1_script = os.path.join(current_dir, "CRAFT-pytorch/test.py")
    project1_args = (
        "--test_folder", image_folder_path,
        "--trained_model", craft_model,
        "--refiner_model", os.path.join(current_dir, "CRAFT-pytorch/weights/craft_refiner_CTW1500.pth")
    )

    run_script(project1_script, *project1_args)
    print("CRAFT done")

    # Project 2 - inference.py
    project2_script = os.path.join(current_dir, "TextBPN/inference.py")
    project2_args = (
        "--image_dir", image_folder_path,
        "--output_dir", output_folder_path
    )

    run_script(project2_script, *project2_args)
    print("TextBPN done")

    # Project 3 - vis.py
    project3_script = os.path.join(current_dir, "vis.py")
    project3_args = (
        "--image_dir", image_folder_path
    )
    run_script(project3_script, *project3_args)
    print("Visualization done")
