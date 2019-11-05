default: all

MP4_FILES = binomial-distribution-demo.mp4 normal-distribution-demo.mp4 regression-evaluation.mp4 dbscan.mp4

all: $(MP4_FILES) index.html

index.html: index.md
	pandoc --template github index.md > index.html

binomial-distribution-demo.mp4: binomial_distribution.py
	python $<
normal-distribution-demo.mp4: normal_distribution.py
	python $<
regression-evaluation.mp4: regression_evaluation.py
	python $<
dbscan.mp4: dbscan.py
	python $<

deploy: all
	ssh zgul.de mkdir -p web/zgul.de/stats
	scp index.md *.mp4 zgul.de:web/zgul.de/stats/

clean:
	rm -f *.html
	rm -f *.mp4
