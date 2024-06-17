# MLTasksCreation

This code automates the creation and submission of machine learning (ML) tasks for multi-ethnic cancer prognosis research. The provided Python script reads task specifications from an Excel file, generates job scripts and Python scripts for each task, and prepares them for submission to a job scheduler.

## Code Structure

- `main.py`: Template main Python script that should be executed for multiethnic machine learning schemes.
- `main.sh`: Template shell script for the job submission to execute main.py file
- `jobs/`: Create a Directory named "jobs" where generated job scripts and Python scripts will be stored.
- `SingleFeatureMLTasks.xlsx`: Excel file containing ML tasks specifications.

## Requirements

- Python 3.x
- pandas
- numpy
- dos2unix

## Installation

1. **Install required Python packages:**
    ```sh
    pip install pandas numpy
    ```

2. **Install `dos2unix` utility (if not already installed):**
    ```sh
    sudo apt-get install dos2unix
    ```

## Usage

1. **Prepare the Excel file:**
   Ensure `SingleFeatureMLTasks.xlsx` is in the same directory as `main.py`. This file should contain the following columns:
    - `Cancer Type`
    - `Feature Type`
    - `Clinical Outcome Endpoint`
    - `Event Time Threshold (Year)`
    - `Target Group`

2. **Run the Python script:**
    ```
    python SlurmJobs.py
    ```

3. **Script Execution:**
    - The script reads the Excel file and generates job scripts and Python scripts for each ML task.
    - It modifies the shell script template (`main.sh`) according to the task specifications.
    - It converts the scripts to Unix format using `dos2unix`.
    - It creates a batch file (`sbatch_commands_<feature_type>_<target_domain>.txt`) containing commands to submit each job script using `sbatch`.

4. **Submit Jobs:**
    After running the script, use the generated batch files to submit jobs:
    ```sh
    sh sbatch_commands_<feature_type>_<target_domain>.txt
    ```


**Acknowledgement**

This work has been supported by NIH R01 grant.

**Contact**

For any queries, please contact:

Prof. Yan Cui (ycui2@uthsc.edu)

Dr. Teena Sharma (tee.shar6@gmail.com)





