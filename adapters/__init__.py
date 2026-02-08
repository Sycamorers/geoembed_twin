from .repo_scan import scan_repos, register_repos_on_sys_path

# Register immediately so optional upstream repos are importable.
REPOS = register_repos_on_sys_path()
