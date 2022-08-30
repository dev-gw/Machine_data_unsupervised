what:
	@echo "	drying    - Run drying_machine.py - main file"
	@echo "	ctest     - Run cluster_test.py - for testing cluster"
	@echo "	stest     - Run silouette_test.py - for testing silouette"
drying:
	python3 "drying_machine.py"

ctest:
	python3 "cluster_test.py"
 
stest:
	python3 "silouette_test.py"