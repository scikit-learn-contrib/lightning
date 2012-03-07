PYTHON ?= python
CYTHON ?= cython
NOSETESTS ?= nosetests
DATADIR=$(HOME)/lightning_data

# Compilation...

CYTHONSRC= $(wildcard lightning/*.pyx)
CSRC= $(CYTHONSRC:.pyx=.c)

inplace: $(CSRC)
	$(PYTHON) setup.py build_ext -i

clean:
	rm -f lightning/*.c lightning/*.cpp lightning/*.so lightning/*.html lightning/*.pyc

%.c: %.pyx
	$(CYTHON) $<

# Tests...
#
test-code: in
	$(NOSETESTS) -s lightning

test-coverage:
	$(NOSETESTS) -s --with-coverage --cover-html --cover-html-dir=coverage \
	--cover-package=lightning lightning

test: test-code test-doc

# Datasets...
#
datadir:
	mkdir -p $(DATADIR)

# regression
download-abalone:
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/abalone_scale
	mv abalone_scale $(DATADIR)

download-cadata:
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/cadata
	mv cadata $(DATADIR)

download-cpusmall:
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/cpusmall_scale
	mv cpusmall_scale $(DATADIR)

download-space_ga:
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/space_ga_scale
	mv space_ga_scale $(DATADIR)

download-YearPredictionMSD:
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.bz2
	bunzip2 YearPredictionMSD.bz2
	mv YearPredictionMSD $(DATADIR)
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.t.bz2
	bunzip2 YearPredictionMSD.t.bz2
	mv YearPredictionMSD.t $(DATADIR)
	
# binary classification
download-adult: datadir
	./download.sh http://leon.bottou.org/_media/papers/lasvm-adult.tar.bz2
	tar xvfj lasvm-adult.tar.bz2
	mv adult $(DATADIR)
	rm lasvm-adult.tar.bz2

download-banana: datadir
	./download.sh http://leon.bottou.org/_media/papers/lasvm-banana.tar.bz2
	tar xvfj lasvm-banana.tar.bz2
	mv banana $(DATADIR)
	rm lasvm-banana.tar.bz2

download-covtype:
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2
	bunzip2 covtype.libsvm.binary.scale.bz2
	mv covtype.libsvm.binary.scale $(DATADIR)

download-reuters: datadir
	./download.sh http://leon.bottou.org/_media/papers/lasvm-reuters.tar.bz2
	tar xvfj lasvm-reuters.tar.bz2
	mv reuters $(DATADIR)
	rm lasvm-reuters.tar.bz2

download-waveform: datadir
	./download.sh http://leon.bottou.org/_media/papers/lasvm-waveform.tar.bz2
	tar xvfj lasvm-waveform.tar.bz2
	mv waveform $(DATADIR)
	rm lasvm-waveform.tar.bz2

# multi-class

download-dna: datadir
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/dna.scale.tr
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/dna.scale.t
	mv dna* $(DATADIR)

download-letter: datadir
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/letter.scale.tr
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/letter.scale.t
	mv letter* $(DATADIR)

download-mnist: datadir
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.bz2
	bunzip2 mnist.scale.bz2
	mv mnist.scale $(DATADIR)
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.t.bz2
	bunzip2 mnist.scale.t.bz2
	mv mnist.scale.t $(DATADIR)

download-news20: datadir
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/news20.scale.bz2
	bunzip2 news20.scale.bz2
	mv news20.scale $(DATADIR)
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/news20.t.scale.bz2
	bunzip2 news20.t.scale.bz2
	mv news20.t.scale $(DATADIR)

download-pendigits:
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits.t
	mv pendigits* $(DATADIR)

download-protein:
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/protein.tr.bz2
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/protein.t.bz2
	bunzip2 protein.tr.bz2
	bunzip2 protein.t.bz2
	mv protein* $(DATADIR)

download-satimage: datadir
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/satimage.scale.tr
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/satimage.scale.t
	mv satimage* $(DATADIR)

download-usps: datadir
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2
	bunzip2 usps.bz2
	mv usps $(DATADIR)
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2
	bunzip2 usps.t.bz2
	mv usps.t $(DATADIR)

