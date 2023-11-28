import csv

file_path = "data/elon_musk_tweets.csv"  

try:
    with open(file_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        
        # Skip header
        header = next(csv_reader)
        print(f"CSV File Header: {header}")

        print("CSV File Content:")
        for row in csv_reader:
            print(row)
except FileNotFoundError:
    print(f"CSV file not found at: {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")