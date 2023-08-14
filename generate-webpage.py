import json
import pandas as pd
from jinja2 import Environment, FileSystemLoader

# Input.
data_csv_file = 'benchmark-data-v1/house.txt'
results_json_file = 'update_base/specified-difficulties/base-benchmark-house-results.json'

data = json.load(open(results_json_file))

# Read the provided data using pandas
df = pd.read_csv(data_csv_file, delimiter = '\t') 
persons = df.to_dict('records')

# Set up the environment and load the template
env = Environment(loader=FileSystemLoader('.'))
template = env.get_template('template.html')

# Render the template with the data
html_output = template.render(data = data, persons = persons)

with open("website/output.html", "w") as f:
    f.write(html_output)

print("HTML generated as 'output.html'")
