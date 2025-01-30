import subprocess
import json
import os
from gsrc import proc
import sys
from datetime import datetime
import pickle
import matplotlib.pyplot as plt  # type: ignore
import numpy as np # type: ignore

def initial_setting(script_path, config_filename="config.json"):

	config, wd, tmpdir, outdir_base = initial_setting_via_config(config_filename)

	now, logs = get_logs(wd, script_path)
	input_filepath = define_input_directory(script_path, config)
	inputs, outdir = load_input(input_filepath, outdir_base)

	return inputs, tmpdir, outdir, logs, now

def initial_setting_in_gsrcmodule(script_path, class_name, func_name, outdir_name, config_filename="config.json"):

	_, wd, tmpdir, outdir_base = initial_setting_via_config(config_filename=config_filename)

	now, logs = get_logs_in_gsrcmodule(wd, script_path, class_name, func_name)
	outdir = os.path.join(outdir_base, outdir_name)

	return tmpdir, outdir, logs, now

def initial_setting_via_config(config_filename="config.json"):

	config, wd = check_working_directory(config_filename=config_filename)
	tmpdir, outdir_base = define_tmp_output_directories(config)

	return config, wd, tmpdir, outdir_base

def get_commit_id(repository):
    # Gitコマンドを実行して現在のコミットIDを取得
	result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repository, capture_output=True, text=True)
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

def define_input_directory(script_path, config):
	input_filepath = os.path.join(os.path.dirname(script_path), config["inputs_dir"], f"{os.path.splitext(os.path.basename(script_path))[0]}.json")
	return input_filepath

def define_tmp_output_directories(config):

	# define tmp and output directories
	tmpdir = config["tmp_dir"]
	proc.ifNotMake(tmpdir)
	outdir_base = config["base_output_dir"]

	return tmpdir, outdir_base

def load_input(input_filepath, outdir_base):
	with open(input_filepath, "r") as file:
		inputs = json.load(file)
	outdir = os.path.join(outdir_base, inputs["outdirname"])
	proc.ifNotMake(outdir)
	
	return inputs, outdir

def get_logs(wd, script_path):
	now = datetime.now()
	logs = {
		'script': {os.path.relpath(script_path, wd)}, 
		'anascrpts_gitid': {get_commit_id(wd)}, 
		'gsrc_gitid': {get_commit_id("gsrc")}, 
		'datetime': {now}
	}
	return now, logs

def get_logs_in_gsrcmodule(wd, script_path, class_name, func_name):
	now = datetime.now()
	logs = {
		'function': {func_name}, 
		'class': {class_name}, 
		'script': {os.path.relpath(script_path, wd)}, 
		'gsrc_gitid': {get_commit_id("gsrc")}, 
		'datetime': {now.strftime(r'%Y-%m-%d %H:%M:%S')}
	}
	return now, logs

def output_pickle_file(outputs, inputs, logs, outdir, suffix=""):
	outputs.update(inputs)
	outputs.update(logs)
	output_filepath = os.path.join(outdir, f"{inputs['output_filename']}{suffix}.pkl")

	with open(output_filepath, "wb") as f:
		pickle.dump(outputs, f)

	return output_filepath

def output_fig(fig, outdir, output_filepath, now, suffix=""):
	filename = f"{os.path.splitext(output_filepath)[0]}{suffix}_0.png"
	output_figure_path = os.path.join(outdir, filename)
	metadata = {
		"Title": filename, 
		"Author": "Tatsuhiro Nasu", 
		"Description": output_filepath, 
		"CreationTime": str(now)
	}
	fig.savefig(output_figure_path, format="png", metadata=metadata)
	plt.close(fig)

def output_dat(output_array, colnm_list, outdir, output_filepath, now, suffix=""):
	filename = f"{os.path.splitext(output_filepath)[0]}{suffix}_0.csv"
	colnm_str = ",".join(colnm_list)
	output_dat_path = os.path.join(outdir, filename)
	header_str = f"Title: {filename}\n" \
					f"Author: Tatsuhiro Nasu\n" \
					f"Description: {output_filepath}\n" \
					f"CreationTime: {str(now)}\n" \
					f"{colnm_str}" 
	np.savetxt(output_dat_path, output_array, delimiter=",", header=header_str)

def load_pickle_data(inputs, key_name="input_datpath"):
	with open(inputs[key_name], "rb") as f:
		return pickle.load(f)
	
def load_multiple_pickle_data(inputs, key_name="input_datpaths"):
	Ndat = len(inputs[key_name])
	data_list = [0]*Ndat
	for i in range(Ndat):
		input_datpath = inputs[key_name][i]
		with open(input_datpath, "rb") as f:
			data_list[i] = pickle.load(f)
	return data_list