
DATADIR=$(HOME)/lightning_data

datadir:
	mkdir -p $(DATADIR)

download-news20: datadir
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/news20.scale.bz2
	bunzip2 news20.scale.bz2
	mv news20.scale $(DATADIR)
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/news20.t.scale.bz2
	bunzip2 news20.t.scale.bz2
	mv news20.t.scale $(DATADIR)

download-usps: datadir
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2
	bunzip2 usps.bz2
	mv usps $(DATADIR)
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2
	bunzip2 usps.t.bz2
	mv usps.t $(DATADIR)

download-mnist: datadir
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.bz2
	bunzip2 mnist.scale.bz2
	mv mnist.scale $(DATADIR)
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.t.bz2
	bunzip2 mnist.scale.t.bz2
	mv mnist.scale.t $(DATADIR)

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

download-covtype:
	./download.sh http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2
	bunzip2 covtype.libsvm.binary.scale.bz2
	mv covtype.libsvm.binary.scale $(DATADIR)
