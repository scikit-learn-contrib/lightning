PYTHON ?= python
PYTEST ?= pytest
DATADIR=$(HOME)/lightning_data

# Compilation...

inplace:
	$(PYTHON) setup.py build_ext -i

all: inplace

clean:
	rm -f lightning/impl/*.cpp lightning/impl/*.html
	rm -f `find lightning -name "*.pyc"`
	rm -f `find lightning -name "*.so"`

# Tests...
#
test-code: inplace
	$(PYTEST) -s -v lightning

test: test-code

# Datasets...
#
datadir:
	mkdir -p $(DATADIR)

# regression
download-abalone: datadir
	wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/abalone_scale
	mv abalone_scale $(DATADIR)

download-cadata: datadir
	wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/cadata
	mv cadata $(DATADIR)

download-cpusmall: datadir
	wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/cpusmall_scale
	mv cpusmall_scale $(DATADIR)

download-space_ga: datadir
	wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/space_ga_scale
	mv space_ga_scale $(DATADIR)

download-YearPredictionMSD: datadir
	wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.bz2
	bunzip2 YearPredictionMSD.bz2
	mv YearPredictionMSD $(DATADIR)
	wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.t.bz2
	bunzip2 YearPredictionMSD.t.bz2
	mv YearPredictionMSD.t $(DATADIR)
	
# binary classification
download-adult: datadir
	wget http://leon.bottou.org/_media/papers/lasvm-adult.tar.bz2
	tar xvfj lasvm-adult.tar.bz2
	mv adult $(DATADIR)
	rm lasvm-adult.tar.bz2

download-banana: datadir
	wget http://leon.bottou.org/_media/papers/lasvm-banana.tar.bz2
	tar xvfj lasvm-banana.tar.bz2
	mv banana $(DATADIR)
	rm lasvm-banana.tar.bz2

download-covtype: datadir
	wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2
	bunzip2 covtype.libsvm.binary.scale.bz2
	mv covtype.libsvm.binary.scale $(DATADIR)

download-ijcnn: datadir
	wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.tr.bz2
	wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.t.bz2
	bunzip2 ijcnn1.tr.bz2
	bunzip2 ijcnn1.t.bz2
	mv ijcnn1* $(DATADIR)

download-real-sim: datadir
	wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/real-sim.bz2
	bunzip2 real-sim.bz2
	mv real-sim $(DATADIR)/realsim

download-reuters: datadir
	wget http://leon.bottou.org/_media/papers/lasvm-reuters.tar.bz2
	tar xvfj lasvm-reuters.tar.bz2
	mv reuters $(DATADIR)
	rm lasvm-reuters.tar.bz2

download-url: datadir
	wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/url_combined.bz2
	bunzip2 url_combined.bz2
	mv url_combined $(DATADIR)

download-waveform: datadir
	wget http://leon.bottou.org/_media/papers/lasvm-waveform.tar.bz2
	tar xvfj lasvm-waveform.tar.bz2
	mv waveform $(DATADIR)
	rm lasvm-waveform.tar.bz2

download-webspam: datadir
	wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/webspam_wc_normalized_unigram.svm.bz2
	bunzip2 webspam_wc_normalized_unigram.svm.bz2
	mv webspam_wc_normalized_unigram.svm $(DATADIR)/webspam

# multi-class

download-dna: datadir
	wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/dna.scale.tr
	wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/dna.scale.t
	mv dna* $(DATADIR)

download-letter: datadir
	wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/letter.scale.tr
	wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/letter.scale.t
	mv letter* $(DATADIR)

download-mnist: datadir
	wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.bz2
	bunzip2 mnist.scale.bz2
	mv mnist.scale $(DATADIR)
	wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.t.bz2
	bunzip2 mnist.scale.t.bz2
	mv mnist.scale.t $(DATADIR)

download-news20: datadir
	wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/news20.scale.bz2
	bunzip2 news20.scale.bz2
	mv news20.scale $(DATADIR)
	wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/news20.t.scale.bz2
	bunzip2 news20.t.scale.bz2
	mv news20.t.scale $(DATADIR)

download-pendigits: datadir
	wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits
	wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits.t
	mv pendigits* $(DATADIR)

download-protein: datadir
	wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/protein.tr.bz2
	wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/protein.t.bz2
	bunzip2 protein.tr.bz2
	bunzip2 protein.t.bz2
	mv protein* $(DATADIR)

download-rcv1: datadir
	wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/rcv1_train.multiclass.bz2
	bunzip2 rcv1_train.multiclass.bz2
	wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/rcv1_test.multiclass.bz2
	bunzip2 rcv1_test.multiclass.bz2
	mv rcv1* $(DATADIR)

download-satimage: datadir
	wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/satimage.scale.tr
	wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/satimage.scale.t
	mv satimage* $(DATADIR)

download-sector: datadir
	wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/sector/sector.scale.bz2
	bunzip2 sector.scale.bz2
	wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/sector/sector.t.scale.bz2
	bunzip2 sector.t.scale.bz2
	mv sector* $(DATADIR)

download-usps: datadir
	wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2
	bunzip2 usps.bz2
	mv usps $(DATADIR)
	wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2
	bunzip2 usps.t.bz2
	mv usps.t $(DATADIR)
