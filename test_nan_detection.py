import subprocess

# Variables to check for NaN errors in the main script
nan_vars = [
    'X_train_pca',
    'X_test_pca',
    'X_train',
    'X_test',
    'X',
]

nan_error_counts = {var: 0 for var in nan_vars}
no_error_count = 0
other_error_count = 0

for i in range(20):
    try:
        result = subprocess.run([
            'python', './ud120-projects/pca/eigenfaces.py'
        ], capture_output=True, text=True, check=True)
        no_error_count += 1
    except subprocess.CalledProcessError as e:
        output = e.stdout + e.stderr
        found = False
        for var in nan_vars:
            if f'NaN detected in {var}' in output:
                nan_error_counts[var] += 1
                print(f"Run {i+1}: ValueError in {var}")
                found = True
                break
        if not found:
            other_error_count += 1
            print(f"Run {i+1}: Other error encountered")
    else:
        print(f"Run {i+1}: No error")

print("\nSummary after 20 runs:")
for var in nan_vars:
    print(f"ValueError in {var}: {nan_error_counts[var]} times")
print(f"No error: {no_error_count} times")
print(f"Other errors: {other_error_count} times")
