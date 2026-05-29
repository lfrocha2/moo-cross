#!/bin/bash

cd ~/python/ds

# app nvidia para exibir o status das placas
#nvidia-smi

# variaveis para alterar o comportamento do script python
export TF_ENABLE_ONEDNN_OPTS=0

#encontra onde o cuda está instalado
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/lib/cuda"
# Desabilita XLA
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"
# Reduz logs do TF (só erros)
export TF_CPP_MIN_LOG_LEVEL=2

#escolhe dispositivos para processamento
export CUDA_VISIBLE_DEVICES=1


# executa script python propriamente dito
# tee encaminha a saida para arquivo e para a tela
python experimentos_v12.py

exit
