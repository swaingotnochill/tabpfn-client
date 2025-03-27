install:
	uv pip install -e .

uninstall:
	uv pip uninstall tabpfn-client

reset: uninstall install
	
