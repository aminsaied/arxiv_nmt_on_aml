import argparse
import datetime

from arxiv_harvester import ArxivHarvester

# Define arguments
parser = argparse.ArgumentParser(description='arXiv scraping arg parser')
parser.add_argument('--output_dir', default='../data/raw/', type=str, help='Directory to store output raw data')
parser.add_argument('--start_date', type=str, help='Start date for harvest')
parser.add_argument('--end_date', type=str, help='End date for harvest')
args = parser.parse_args()

# Get arguments from parser
output_dir = args.output_dir
start_date = args.start_date
end_date = args.end_date

def validate(date_text):
    try:
        datetime.datetime.strptime(date_text, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Date should be in the form YYYY-MM-DD")

if __name__ == '__main__':

    # validate start and end date format
    validate(start_date)
    validate(end_date)

    # harvest data from arXiv
    harvester = ArxivHarvester(arxiv_data_path=output_dir)
    harvester.harvest(start_date=start_date, end_date=end_date)