# pyenv shell integration for login shells and for BASH_ENV (non-interactive bash).
export PYENV_ROOT=/home/agent/.pyenv
export PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"
export PYENV_SHELL=bash

pyenv() {
  local command="${1:-}"
  [ "$#" -gt 0 ] && shift
  case "$command" in
    rehash|shell)
      eval "$(/home/agent/.pyenv/bin/pyenv "sh-$command" "$@")"
      ;;
    *)
      command /home/agent/.pyenv/bin/pyenv "$command" "$@"
      ;;
  esac
}
