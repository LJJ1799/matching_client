# Deep Closest Point
conda create -n dcp -y python=3.7 numpy=1.20 numba
conda activate dcp
conda install cuda -c "nvidia/label/cuda-11.7.1" -y
pip install -y pytorch=1.13.1 torchvision cudatoolkit=11.7.0 -c pytorch -c nvidia
pip install -r requirements.txt

The Reisch data is saved in CustomData/val_data

## testing
If just match two specified slices,run:
python test.py --data src tgt (e.g python test.py --data Reisch_0 Reisch_1)
the matching result will be saved as image.

If matching all the slices in Customdata,run:
python test.py --testall
the error metrics will be save as json file in folder metrics

## Training

### DCP-v1

python main.py --exp_name=dcp_v1 --model=dcp --emb_nn=dgcnn --pointer=identity --head=svd

### DCP-v2

python main.py --exp_name=dcp_v2 --model=dcp --emb_nn=dgcnn --pointer=transformer --head=svd

## Testing

### DCP-v1

python main.py --exp_name=dcp_v1 --model=dcp --emb_nn=dgcnn --pointer=identity --head=svd --eval

or 

python main.py --exp_name=dcp_v1 --model=dcp --emb_nn=dgcnn --pointer=identity --head=svd --eval --model_path=xx/yy

### DCP-v2

python main.py --exp_name=dcp_v2 --model=dcp --emb_nn=dgcnn --pointer=transformer --head=svd --eval

or 

python main.py --exp_name=dcp_v2 --model=dcp --emb_nn=dgcnn --pointer=transformer --head=svd --eval --model_path=xx/yy

where xx/yy is the pretrained model or checkpoints

