SRC = pci tests
DOCS = docs/source

lint: FORCE
	mypy $(SRC)
	ruff check $(SRC)
	ruff format --diff $(SRC)
	nbqa mypy $(DOCS)
	nbqa 'ruff check' $(DOCS)
	nbqa 'ruff format --diff' $(DOCS)

format: FORCE
	ruff check --fix-only $(SRC)
	ruff format $(SRC)
	nbqa 'ruff check --fix-only' $(DOCS)
	nbqa 'ruff format' $(DOCS)

FORCE:
