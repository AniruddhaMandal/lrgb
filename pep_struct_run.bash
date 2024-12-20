python main.py --cfg configs/custom/pep_struct-KGCN-10.yaml wandb.use False | tee results/file1.txt
python main.py --cfg configs/custom/pep_struct-KGCN-10-full.yaml wandb.use False | tee results/file2.txt 
python main.py --cfg configs/custom/pep_struct-KGCN-full.yaml wandb.use False | tee results/file3.txt
python main.py --cfg configs/custom/pep_struct-KGCN.yaml wandb.use False | tee results/file4.txt