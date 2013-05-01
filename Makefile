PYTHON ?= python
CYTHON ?= cython
NOSETESTS ?= nosetests
DATADIR=$(HOME)/lightning_data

# Compilation...

CYTHONSRC= $(wildcard lightning/*.pyx lightning/random/*.pyx)
CSRC= $(CYTHONSRC:.pyx=.cpp)

inplace: cython
	$(PYTHON) setup.py build_ext -i

cython: $(CSRC)

clean:
	rm -f lightning/*.c lightning/*.so lightning/*.html lightning/*.pyc
	rm -f `find lightning -name "*.cpp"`

%.cpp: %.pyx
	$(CYTHON) --cplus $<

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
download-abalone: datadir
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/abalone_scale
	mv abalone_scale $(DATADIR)

download-cadata: datadir
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/cadata
	mv cadata $(DATADIR)

download-cpusmall: datadir
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/cpusmall_scale
	mv cpusmall_scale $(DATADIR)

download-space_ga: datadir
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/space_ga_scale
	mv space_ga_scale $(DATADIR)

download-YearPredictionMSD: datadir
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

download-covtype: datadir
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2
	bunzip2 covtype.libsvm.binary.scale.bz2
	mv covtype.libsvm.binary.scale $(DATADIR)

download-ijcnn: datadir
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.tr.bz2
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.t.bz2
	bunzip2 ijcnn1.tr.bz2
	bunzip2 ijcnn1.t.bz2
	mv ijcnn1* $(DATADIR)

download-real-sim: datadir
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/real-sim.bz2
	bunzip2 real-sim.bz2
	mv real-sim $(DATADIR)/realsim

download-reuters: datadir
	./download.sh http://leon.bottou.org/_media/papers/lasvm-reuters.tar.bz2
	tar xvfj lasvm-reuters.tar.bz2
	mv reuters $(DATADIR)
	rm lasvm-reuters.tar.bz2

download-url: datadir
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/url_combined.bz2
	bunzip2 url_combined.bz2
	mv url_combined $(DATADIR)

download-waveform: datadir
	./download.sh http://leon.bottou.org/_media/papers/lasvm-waveform.tar.bz2
	tar xvfj lasvm-waveform.tar.bz2
	mv waveform $(DATADIR)
	rm lasvm-waveform.tar.bz2

download-webspam: datadir
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/webspam_wc_normalized_unigram.svm.bz2
	bunzip2 webspam_wc_normalized_unigram.svm.bz2
	mv webspam_wc_normalized_unigram.svm $(DATADIR)/webspam

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

download-pendigits: datadir
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits.t
	mv pendigits* $(DATADIR)

download-protein: datadir
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/protein.tr.bz2
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/protein.t.bz2
	bunzip2 protein.tr.bz2
	bunzip2 protein.t.bz2
	mv protein* $(DATADIR)

download-rcv1: datadir
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/rcv1_train.multiclass.bz2
	bunzip2 rcv1_train.multiclass.bz2
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/rcv1_test.multiclass.bz2
	bunzip2 rcv1_test.multiclass.bz2
	mv rcv1* $(DATADIR)

download-satimage: datadir
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/satimage.scale.tr
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/satimage.scale.t
	mv satimage* $(DATADIR)

download-sector: datadir
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/sector/sector.scale.bz2
	bunzip2 sector.scale.bz2
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/sector/sector.t.scale.bz2
	bunzip2 sector.t.scale.bz2
	mv sector* $(DATADIR)

download-usps: datadir
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2
	bunzip2 usps.bz2
	mv usps $(DATADIR)
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2
	bunzip2 usps.t.bz2
	mv usps.t $(DATADIR)

