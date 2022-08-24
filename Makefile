what:
	@echo "	drying    - Run drying.py - main file"
	@echo "	ctest     - Run cluster_test.py - for testing cluster"
	@echo "	stest     - Run silouette_test.py - for testing silouette"
drying:
	python3 "/UHome/kgw32395/kgw/drying.py"

ctest:
	python3 "/UHome/kgw32395/kgw/cluster_test.py"
 
stest:
	python3 "/UHome/kgw32395/kgw/silouette_test.py"
 
cnc:
	python3 "/UHome/kgw32395/kgw/CNC_machine.py" 