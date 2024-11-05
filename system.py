import subprocess

def get_commit_id(repository):
	subprocess.run(["cd", repository])
    # Gitコマンドを実行して現在のコミットIDを取得
	result = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
	if result.returncode == 0:
		# コミットIDを取得
		commit_id = result.stdout.strip()
		return commit_id
	else:
		raise Exception("Git コマンドが失敗しました")