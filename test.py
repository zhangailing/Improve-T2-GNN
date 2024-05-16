import subprocess
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_command(cmd):
    # Extract dataset name from cmd
    dataset = re.search(r'--dataset (\w+)', cmd).group(1)
    
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = process.communicate()[0].decode('utf-8')
    print(output)
    return output, dataset

def parse_output(output):
    results = {
        'teacher_fea': [],
        'teacher_str': [],
        'student': []
    }

    # Regular expressions to capture loss and accuracy
    pattern_fea = r"teacher_fea Test set results: loss= (\d+\.\d+) accuracy= (\d+\.\d+)"
    pattern_str = r"teacher_str Test set results: loss= (\d+\.\d+) accuracy= (\d+\.\d+)"
    pattern_stu = r"student Test set results: loss= (\d+\.\d+) accuracy= (\d+\.\d+)"

    # Iterate through each line in the log data
    for line in output.split('\n'):
        fea_match = re.search(pattern_fea, line)
        str_match = re.search(pattern_str, line)
        stu_match = re.search(pattern_stu, line)

        if fea_match:
            loss, accuracy = fea_match.groups()
            results['teacher_fea'].append({'loss': float(loss), 'accuracy': float(accuracy)})

        if str_match:
            loss, accuracy = str_match.groups()
            results['teacher_str'].append({'loss': float(loss), 'accuracy': float(accuracy)})

        if stu_match:
            loss, accuracy = stu_match.groups()
            results['student'].append({'loss': float(loss), 'accuracy': float(accuracy)})
    return results

def plot_training_results(results, dataset):
    # Prepare the data for plotting
    repeats = list(range(len(results['teacher_fea'])))
    
    # Extracting data for teacher_fea
    teacher_fea_loss = [item['loss'] for item in results['teacher_fea']]
    teacher_fea_accuracy = [item['accuracy'] for item in results['teacher_fea']]
    
    # Extracting data for teacher_str
    teacher_str_loss = [item['loss'] for item in results['teacher_str']]
    teacher_str_accuracy = [item['accuracy'] for item in results['teacher_str']]
    
    # Extracting data for student
    student_loss = [item['loss'] for item in results['student']]
    student_accuracy = [item['accuracy'] for item in results['student']]
    
    # Setting up the plot area
    plt.figure(figsize=(18, 5))

    # Plotting teacher_fea loss and accuracy
    plt.subplot(1, 3, 1)
    plt.plot(repeats, teacher_fea_loss, label='Loss', marker='o')
    plt.plot(repeats, teacher_fea_accuracy, label='Accuracy', marker='o')
    plt.title(f'{dataset} Teacher_fea Test Set Results')
    plt.xlabel('Repeat')
    plt.ylabel('Metrics')
    plt.legend()
    plt.grid(True)

    # Plotting teacher_str loss and accuracy
    plt.subplot(1, 3, 2)
    if len(repeats) != len(teacher_str_loss):
        print(f"Cannot plot because lengths of 'repeats' ({len(repeats)}) and 'teacher_str_loss' ({len(teacher_str_loss)}) do not match.")
    else:
        plt.plot(repeats, teacher_str_loss, label='Loss', marker='o', color='red')
    plt.plot(repeats, teacher_str_accuracy, label='Accuracy', marker='o', color='green')
    plt.title(f'{dataset} Teacher_str Test Set Results')
    plt.xlabel('Repeat')
    plt.ylabel('Metrics')
    plt.legend()
    plt.grid(True)

    # Plotting student loss and accuracy
    plt.subplot(1, 3, 3)
    plt.plot(repeats, student_loss, label='Loss', marker='o', color='purple')
    plt.plot(repeats, student_accuracy, label='Accuracy', marker='o', color='brown')
    plt.title(f'{dataset} Student Test Set Results')
    plt.xlabel('Repeat')
    plt.ylabel('Metrics')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'output/{dataset}.png')  # Save the figure in the output directory with the dataset name
  
cmd = "python run.py --dataset texas --Ts 4.0 --topk 10 --lambd 0.8 --epoch_fea 200 --epoch_str 1000 --epoch_stu 100 --repeat 5"
output, dataset = run_command(cmd)
results = parse_output(output)
plot_training_results(results, dataset)