import pandas as pd
import numpy as np
import fileinput
import shutil
import subprocess

data = pd.read_excel('SingleFeatureMLTasks.xlsx',sheet_name='Protein-AA')
print(np.shape(data))

file_path_sh = 'main.sh'
file_path_py = 'main.py'
line_number = 0

def convert_dos_to_unix(input_file, output_file):
    subprocess.run(['dos2unix', input_file, output_file])

for i in range(np.shape(data)[0]):
    cancer_type = data['Cancer Type'].iloc[i]
    feature_type = data['Feature Type'].iloc[i]
    target = data['Clinical Outcome Endpoint'].iloc[i]
    years = data['Event Time Threshold (Year)'].iloc[i]
    target_domain = data['Target Group'].iloc[i]
    print(i, cancer_type, feature_type, target, years, target_domain)
    TaskName = cancer_type + '_'+ feature_type +'_' + target + '_' + str(years) + 'YR_' + target_domain + '_AE'

    new_sh_file = 'jobs/'+TaskName+'.sh'
    new_py_file = 'jobs/'+TaskName+'.py'
    shutil.copy(file_path_sh, new_sh_file)
    shutil.copy(file_path_py, new_py_file)
    
    original_line = '#SBATCH --job-name=main'
    original_line_8 = '#SBATCH --output=Protein_100Features_Test/o_e_files/main.o%j'
    original_line_9 = '#SBATCH --error=Protein_100Features_Test/o_e_files/main.e%j'
    original_line_20 = 'python Protein_100Features_Test/main.py'

    new_line = '#SBATCH --job-name=' + TaskName
    new_line_8 = '#SBATCH --output=Protein_100Features_Test/o_e_files/' + TaskName + '.o%j'
    new_line_9 = '#SBATCH --error=Protein_100Features_Test/o_e_files/' + TaskName + '.e%j'
    new_line_20 = 'python Protein_100Features_Test/' + TaskName + '.py ' + cancer_type + ' ' + \
                  feature_type + ' ' + target + ' ' + str(years) + ' ' + target_domain + ' 100 0 0 0 0 1'
    
    with fileinput.FileInput(new_sh_file, inplace=True) as file:
        for line_index, line in enumerate(file, start=1):
            if line_index == line_number+2:
                modified_line = line.replace(original_line, new_line)
                print(modified_line, end='')
            elif line_index == line_number+8:
                modified_line_8 = line.replace(original_line_8, new_line_8)
                print(modified_line_8, end='')
            elif line_index == line_number+9:
                modified_line_9 = line.replace(original_line_9, new_line_9)
                print(modified_line_9, end='')
            elif line_index == line_number+20: 
                modified_line_20 = line.replace(original_line_20, new_line_20)
                print(modified_line_20, end='')
            else:
                print(line, end='')
    convert_dos_to_unix(new_sh_file, new_sh_file)

    run_sh_command = 'sbatch Protein_100Features_Test/' + TaskName + '.sh'
    sbatch_file_path = 'sbatch_commands_'+target_domain+'.txt'
    # print(sbatch_file_path)
    if i==0:
        with open(sbatch_file_path, 'w') as file:
            # Write string lines to the file
            file.write(run_sh_command+"\n")
    else:
        with open(sbatch_file_path, 'a') as file:
            # Write string lines to the file
            file.write(run_sh_command+"\n")
    # Print a message to confirm the file creation
    # print(f"File '{sbatch_file_path}' created successfully!")






