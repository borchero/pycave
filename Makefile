.PHONY: docs

docs: # requires `enchant` to be installed
	rm -rf build
	rm -rf docs/**/generated
	sphinx-build -W -b spelling docs build
	sphinx-build -W -b html docs build
