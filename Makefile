default:
	@echo noop

HTML_FILES = binomial-distribution-demo.html normal-distribution-demo.html regression-evaluation.html

all: $(HTML_FILES)

binomial-distribution-demo.html: binomial_distribution.py
	python $<
normal-distribution-demo.html: normal_distribution.py
	python $<
regression-evaluation.html: regression_evaluation.py
	python $<

deploy:
	ssh zgul.de mkdir -p web/zgul.de/stats
	scp index.md *.html zgul.de:web/zgul.de/stats/

clean:
	rm -f *.html
