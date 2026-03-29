#!/usr/bin/env python3

import csv
import argparse


def divide_csv_by_4(input_file, output_file):
    with open(input_file, "r", newline="") as infile, open(output_file, "w", newline="") as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            new_row = []
            for value in row:
                try:
                    new_value = float(value) / 4
                    new_row.append(new_value)
                except ValueError:
                    continue
                    #raise ValueError(f"Non-numeric value found in CSV: {value!r}")
            writer.writerow(new_row)


def main():
    parser = argparse.ArgumentParser(description="Divide all numbers in a CSV file by 4.")
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument("output_csv", help="Path to output CSV file")
    args = parser.parse_args()

    divide_csv_by_4(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()
