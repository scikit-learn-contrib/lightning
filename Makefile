
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
