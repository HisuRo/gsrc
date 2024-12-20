import subprocess
import json
import os
from nasu import proc
import sys
from datetime import datetime
import pickle
import matplotlib.pyplot as plt  # type: ignore

def get_commit_id(repository):
	subprocess.run(["cd", repository], shell=True)
    # Gitコマンドを実行して現在のコミットIDを取得
	result = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
	if result.returncode == 0:
		# コミットIDを取得
		commit_id = result.stdout.strip()
		return commit_id
	else:
		raise Exception("Git コマンドが失敗しました")

def load_config(config_file):
	with open(config_file, 'r') as f:
		return json.load(f)
	
def check_working_directory(config_filename="config.json"):

	# check working directory
	config = load_config(config_filename)
	wd = os.path.normpath(config["wd"])
	cwd = os.getcwd()
	if cwd != wd:
		print(f"cwd: {cwd}")
		print(f"wd : {wd}")
		raise Exception("change working directory to analysis_scripts!!")
	
	return config, wd

def define_input_tmp_output_directories(script_path, config):

	# define input, tmp, and output directories
	input_filepath = os.path.join(os.path.dirname(script_path), config["inputs_dir"], f"{os.path.splitext(os.path.basename(script_path))[0]}.json")
	tmpdir = config["tmp_dir"]
	proc.ifNotMake(tmpdir)
	outdir_base = config["base_output_dir"]

	return input_filepath, tmpdir, outdir_base

def load_input(input_filepath, outdir_base):
	with open(input_filepath, "r") as file:
		inputs = json.load(file)
	outdir = os.path.join(outdir_base, inputs["outdirname"])
	proc.ifNotMake(outdir)
	
	return inputs, outdir

def get_logs(wd):
	now = datetime.now()
	logs = {
		'script': {sys.argv[0]}, 
		'analysis_scripts_gitid': {get_commit_id(wd)}, 
		'nasu_gitid': {get_commit_id("nasu")}, 
		'datetime': {now}
	}
	return now, logs

def output_pickle_file(outputs, inputs, logs, outdir):
	outputs.update(inputs)
	outputs.update(logs)
	output_filepath = os.path.join(outdir, f"{inputs['output_filename']}.pkl")

	with open(output_filepath, "wb") as f:
		pickle.dump(outputs, f)

	return output_filepath

def output_fig(fig, outdir, inputs, output_filepath, now):
	output_figureloc = os.path.join(outdir, f"{inputs['output_filename']}.png")
	metadata = {
		"Title": f"{inputs['output_filename']}.png", 
		"Author": "Tatsuhiro Nasu", 
		"Description": output_filepath, 
		"CreationTime": str(now)
	}
	fig.savefig(output_figureloc, format="png", metadata=metadata)
	plt.close(fig)

def load_pickle_data(inputs, key_name="input_datpath"):
	with open(inputs[key_name], "rb") as f:
		return pickle.load(f)
