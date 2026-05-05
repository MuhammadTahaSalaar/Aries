pkill -u $(whoami) -9 python

pkill -9 -f "Aries"

# To check what processes are still running related to Aries, you can use:
ps aux | grep "Aries/.venv" | grep -v grep