import csv

file_path = "data/tweets.csv" # Replace "your_csv_file.csv" with the actual CSV file name
output_file_path = "data/modified_file_others.csv"  # Replace "modified_csv_file.csv" with the desired output file name

columns_to_delete_musk = [0, 2, 3, 4, 5, 6, 7,8, 11, 12, 13, 15]  
columns_to_delete_other=[2,4,5,6,7,9]
try:
    with open(file_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        
        # Get header and indices of columns to keep
        header = next(csv_reader)
        indices_to_keep = [i for i in range(len(header)) if i not in columns_to_delete_other]

        # Read rows, modify, and write to a new file
        with open(output_file_path, 'w', encoding='utf-8', newline='') as output_file:
            csv_writer = csv.writer(output_file)
            
            # Write header to the new file
            csv_writer.writerow([header[i] for i in indices_to_keep])

            for row in csv_reader:
                # Remove specified columns from each row
                modified_row = [row[i] for i in indices_to_keep]
                csv_writer.writerow(modified_row)

    print(f"CSV file modified and saved to: {output_file_path}")
except FileNotFoundError:
    print(f"CSV file not found at: {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
