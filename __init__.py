import os
from pkg_resources import get_distribution  # type: ignore
from sys import version
import csv

### input ###
working_server = "Precision3450"

### main ###
current_python_version = version.split()[0]
current_numpy_version = get_distribution("numpy").version
current_scipy_version = get_distribution("scipy").version
current_pandas_version = get_distribution("pandas").version
current_matplotlib_version = get_distribution("matplotlib").version

current_versions = [current_python_version, current_numpy_version, current_scipy_version, current_pandas_version, current_matplotlib_version]

nasu_dir = os.path.dirname(os.path.abspath(__file__))
version_table_path = os.path.join(nasu_dir, "version_table.csv")
with open(version_table_path, mode='r', newline='', encoding='utf-8') as file:
	reader = csv.DictReader(file)
	first_colnm = "name"

	i = 0
	for row in reader:
		current_version = current_versions[i]
		temp = {col: row[col] for col in [first_colnm, working_server]}
		module_name = temp[first_colnm]
		logged_version = temp[working_server]
		if current_version != logged_version:
			raise Exception(f"{module_name} version was changed from {logged_version} to {current_version}!! ")
		
		i += 1
