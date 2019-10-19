default:
	echo

deploy:
	ssh zgul.de mkdir -p web/zgul.de/stats
	scp index.md *.html zgul.de:web/zgul.de/stats/
