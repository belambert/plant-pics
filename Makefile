.PHONY: format

format:
	isort src/
	black src/
