import re

def calculate_execution_times(file_path):
    # Regular expression pattern to extract execution times
    pattern = r"Execution Time: (\d+\.\d+) seconds"

    # Read the log file
    with open(file_path, 'r') as file:
        log_data = file.read()

    # Find all matches and convert them to floats
    execution_times = [float(match) for match in re.findall(pattern, log_data)]

    # Calculate the average execution time
    average_time = sum(execution_times) / len(execution_times)
    print(f"Average Execution Time: {average_time} seconds")

    # Calculate the number of trials for 7 hours (7 hours * 3600 seconds/hour)
    total_seconds = 7 * 3600
    number_of_trials = total_seconds / average_time
    print(f"Number of Trials to Reach 7 Hours: {int(number_of_trials)}")

# Replace 'path_to_your_log_file.txt' with the path to your log file
calculate_execution_times('temp.txt')
