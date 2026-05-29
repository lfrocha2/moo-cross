# -*- coding: utf-8 -*-
"""Experimentos v12

#Trabalho de Conclusão de Curso
#Especialização em Ciência de Dados
Prof. Eduardo Kugler Viegas<BR>
Alunos: Humberto Pradera e Leonardo Rocha

Experimentos para subsidiar a construção do TCC<BR>

Utilizamos a versão NetFlow v3 Datasets

3 Objetivos:
- F1 DS2
- F1 DS3
- F1 DS4

Variáveis
- Quantidade de features
- Quantidade de neuronios
- Quantidade de camadas

Este Script tem a gravacao para poder rodar em ambiente cuja disponibilidade varie.

"""

# Imports, variáveis e funções gerais"""


#Bibliotecas
import numpy as np
import pandas as pd

import gc

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

#shrink memory dataset
from fastai.tabular.core import df_shrink

#CNN/MLP
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense,  Dropout
from tensorflow.keras.models import Sequential

#from google.colab import drive

from time import time as time2
import datetime
#import time
import os




# Controle de alguns experimentos
QUANTIDADE_MINIMA_CLASSE = 1000
HIGIENIZAR_DATASETS = True
BALANCEAR_DATASETS = True
TEST_SIZE = 0.2
#BATCH_SIZE = 64
BATCH_SIZE = 512

# Define the expected path for the feature names file
LOG_DIR = os.path.expanduser('~/resultados/ds/')
LOG_EXECUCAO = 'exec.txt'
LOG_CHECKPOINT = 'checkpoint'
LOG_FINAL = 'final.pkl'

# Define o caminho do arquivo salvo
load_path = LOG_DIR+LOG_FINAL
#final_result_path = LOG_DIR+'nsga2_results_out.pkl'

#feature_names_path = LOG_DIR +'feature_names.txt'

#Desabilita logs verbosos do XLA
tf.get_logger().setLevel('ERROR')
print("Logs do TensorFlow desabilitados para nível ERROR ou superior.\n")

os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
print("✅ TF32 habilitado para operações de matriz em GPUs compatíveis (Ampere ou superior).\n")

'''
# Força GPU para modo de máximo desempenho
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'  # Aloca toda memória de uma vez

# Configurações de GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Para cada GPU
        for gpu in gpus:
            # Desabilita memory growth (força alocação total)
            tf.config.experimental.set_memory_growth(gpu, False)
            
            # Define opções de configuração
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=24000)]  # 24GB
            )
        
        # Força inicialização da GPU
        with tf.device('/GPU:1'):  # Sua GPU é a 1
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print("✅ GPU inicializada em modo de alto desempenho")
            
    except RuntimeError as e:
        print(e)
else:
    print("Nenhuma GPU encontrada. O treinamento será executado na CPU.")
print("----------------------------------------\n")
'''
#watch -n 0.5 'nvidia-smi --query-gpu=pstate,power.draw,clocks.gr,utilization.gpu --format=csv,noheader -i 1'

# Verificação de GPU
print("--- Verificando disponibilidade de GPU ---")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Configura o crescimento de memória dinâmico para evitar que o TensorFlow
    # aloque toda a memória da GPU de uma vez.
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(f"GPUs Físicas: {len(gpus)}, GPUs Lógicas: {len(logical_gpus)}")
    print("GPU disponível e configurada para uso.")
  except RuntimeError as e:
    # O crescimento de memória deve ser configurado antes da inicialização das GPUs
    print(e)
else:
    print("Nenhuma GPU encontrada. O treinamento será executado na CPU.")
print("----------------------------------------\n")

# --- Configurando a estratégia de distribuição para múltiplas GPUs ---
# print("--- Configurando a estratégia de distribuição ---")
# # Desativado para rodar em uma única GPU
# strategy = tf.distribute.MirroredStrategy()
# print(f"Estratégia de distribuição: MirroredStrategy com {strategy.num_replicas_in_sync} réplicas (GPUs).")
# print("----------------------------------------\n")

# Variavel que será redefinida posteriormente
attack_categories = ['Benigno', 'Ataque']

# Diretório para salvar os gráficos
#PLOT_DIR = os.path.expanduser('~/resultados/ds/plots')
#if not os.path.exists(PLOT_DIR):
#    os.makedirs(PLOT_DIR)

lista_campos_excluir = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L4_SRC_PORT', 'L4_DST_PORT', 'FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS',
                        'DNS_QUERY_ID', 'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT', 'DNS_QUERY_TYPE', 'DNS_TTL_ANSWER', 'MIN_TTL', 'MAX_TTL']
lista_campos_verif = ['SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES']

##########################################################################################
#Formata um número em bytes para Megabytes
def format_megabytes(size_in_bytes):
  if size_in_bytes is None:
    return "N/A"
  size_in_mb = size_in_bytes / (1024 * 1024)
  return f"{size_in_mb:.2f} MB"

##########################################################################################
def balacear_dataset(pdf):

  if not BALANCEAR_DATASETS:
    print("- Dataset não será balanceado")
    return pdf

  attack_counts = pdf['Attack'].value_counts()
  print("- attack_counts:",attack_counts,"\n")

  min_attack_count = attack_counts[attack_counts.index != 'Benign'].min()
  print("min_attack_count:",min_attack_count)

  benign_count = attack_counts['Benign']
  print("benign_count:", benign_count)

  # Calcula o valor alvo para as classes maliciosas
  target_non_benign_count = min(min_attack_count, benign_count // (len(attack_counts) - 1))
  print("- Quantidade inicial mínima para cada classe de ataque:",target_non_benign_count)

  if target_non_benign_count<QUANTIDADE_MINIMA_CLASSE:
    print(f"-- Então a quantidade inicial mínima é menor que {QUANTIDADE_MINIMA_CLASSE}. O Dataset terá problema no balanceamento.")
    # Se a quantidade inicial or menor que 1000, então teremos um Dataset balanceado com muito poucos casos.
    # Para evitar isto, busco a classe que possua menor representação que tenha quantidade de ocorrências maior que 1000
    # Pego a totalização por tipo de ataque
    vc = pdf['Attack'].value_counts()
    # Filtro de quantidade mínima, estipulada pelo professor como 1000
    vc1k = vc[vc>QUANTIDADE_MINIMA_CLASSE]
    # Pego o último valor, que deve ser o menor valor superior a 1000
    target_non_benign_count = vc1k.tail(1).values[0]
    print("-- Novo valor mínimo encontrado:",target_non_benign_count)

  # De fato faz o balanceamento
  balanced_dfs = []
  for attack_type in attack_counts.index:
    if attack_type == 'Benign':
      balanced_dfs.append(pdf[pdf['Attack'] == 'Benign'].sample(min(benign_count, target_non_benign_count * (len(attack_counts) - 1)), random_state=42))
    else:
      balanced_dfs.append(pdf[pdf['Attack'] == attack_type].sample(min(attack_counts[attack_type], target_non_benign_count), random_state=42))

  alldf = pd.concat(balanced_dfs)

  print("\nResultado do balanceamento:")
  attack_counts = alldf['Attack'].value_counts()
  print("- attack_counts:",attack_counts,"\n")

  return alldf

##########################################################################################
def higienizar_dataset(pdf):

  if not HIGIENIZAR_DATASETS:
    print("- Dataset não será higienizado")
    return

  qtd_ini = len(pdf)
  print("- Tamanho original do Dataset: {:,}".format(qtd_ini))

  # Drop rows where SRC_TO_DST_SECOND_BYTES or DST_TO_SRC_SECOND_BYTES have infinite values or values > np.finfo(np.float64).max
  print("- Removendo registros com campos com valor infinito ou superior a np.float64")
  for col in lista_campos_verif:
    pdf.drop(pdf[np.isinf(pdf[col])].index, inplace=True)
    pdf.drop(pdf[pdf[col] > np.finfo(np.float64).max].index, inplace=True)

  #retira registros com campos NA
  print("- Removendo registros com campos NA")
  pdf.dropna(subset=['SRC_TO_DST_SECOND_BYTES'], inplace=True)

  # verificando se há linhas duplicadas
  qtd_duplicadas = pdf.duplicated().sum()

  if qtd_duplicadas > 0:
    print("- Encontrados {:,} registros duplicados".format(qtd_duplicadas))
    # retirando as linhas duplicadas
    pdf.drop_duplicates(inplace=True)
    pdf.reset_index(inplace=True, drop=True)

  qtd_fim = len(pdf)
  print("- Ao total, foram eliminados {:,} registros".format(qtd_ini-qtd_fim))
  print("- Tamanho final {:,}".format(qtd_fim))


# Datasets ###############################################
#drive.mount('/content/gdrive/', force_remount=True)

df_url = []
df = []

#df_url.append('~/datasets/ds/NF-UNSW-NB15-v3.csv')
df_url.append('#')
df_url.append('~/datasets/ds/NF-ToN-IoT-v3.csv')
df_url.append('~/datasets/ds/NF-BoT-IoT-v3.csv')
df_url.append('~/datasets/ds/NF-CICIDS2018-v3.csv')

tamanho_antes = 0
tamanho_depois = 0

#limita a quantidade de registros a carregar ao mesmo tempo em memóris
#para permitir uma máquina com mesmo tempo processar os Datasets
chunk_size = 4000000
chunk_resultset = []

#loop carregando os datasets
for index, d in enumerate(df_url):
  #carrega
  print("\n--- Dataset",index+1,'---')
  if d == '#':
    print("Dataset desativado para processamento.")
    # Para manter compatibilidade com o restante do código, adiciona um DataFrame vazio
    df.append(pd.DataFrame())
    continue

  #cria um iterator para ler o arquivo em pedaços
  chunk_iter = pd.read_csv(d, chunksize=chunk_size)

  chunk_number = 0
  for chunk in chunk_iter:
    chunk_number += 1
    print(f"---- Processando chunk {chunk_number} ----")
    print("- Lidas {:,} linhas.".format(len(chunk)))

    tamanho_memoria_ds = chunk.memory_usage(deep=True).sum()
    print("Chunk carregado:",format_megabytes(tamanho_memoria_ds))
    tamanho_antes += tamanho_memoria_ds

    #elimina colunas desnecessárias
    chunk = chunk.drop(lista_campos_excluir, axis=1)
    print("Campos desnecessários removidos.")

    #higieniza dataset
    higienizar_dataset(chunk)
    print("Chunk higienizado.")

    chunk_resultset.append(chunk)
    gc.collect()
    print("Garbage collector invocado para o chunk")

  #unifica o que restou dos chunks
  mydf = pd.concat(chunk_resultset)
  chunk_resultset.clear()

  #Só para garantir que não teremos duplicados resultantes da união dos chunks
  higienizar_dataset(mydf)
  print("Dataset higienizado.")

  #balenceia o dataset
  mydf = balacear_dataset(mydf)
  print("Dataset balanceado.")

  #reduz o tamanho
  mydf = df_shrink(mydf)
  print("Dataset reduzido.")
  print("- Tamanho atual: {:,} linhas.".format(len(mydf)))

  gc.collect()
  print("Garbage Collector invocado para o final.")

  tamanho_memoria_ds = mydf.memory_usage(deep=True).sum()
  tamanho_depois += tamanho_memoria_ds

  #anexa na lista de datasets
  df.append(mydf)
  del mydf # Libera a memória do DataFrame unificado após ser adicionado à lista
  gc.collect()

print(" ")
print("Tamanho original dos Datasets:",format_megabytes(tamanho_antes))
print("Tamanho final dos Datasets:",format_megabytes(tamanho_depois))

print("Linhas lidas dos DS:")
for i, d in enumerate(df):
  print(f"DS{i+1}:",len(d))

print("Unifica os 3 Datasets para análise conjunta.")
df_all = pd.concat([df[1], df[2], df[3]])
print(f"DF Unificado:",len(df_all))


# Estrutura dos Datasets
# TODOS OS DATASETS possuem a mesma estrutura

# Definição dos conjuntos de dados
# Conjuntos de dados DS1 ###############################################
#DESATIVADO PARA NÂO SER POSSÌVEL RODAR - DATASET A SER DESCARTADO


### Conjunto de dados DS2 ###############################################
X2 = df[1].drop(['Attack', 'Label'], axis=1)
y2 = df[1].loc[:, 'Label']


### Conjunto de dados DS3 ###############################################
X3 = df[2].drop(['Attack', 'Label'], axis=1)
y3 = df[2].loc[:, 'Label']


### Conjunto de dados DS4 ###############################################
X4 = df[3].drop(['Attack', 'Label'], axis=1)
y4 = df[3].loc[:, 'Label']


### Conjunto de dados DS ALL ###############################################
XAll = df_all.drop(['Attack', 'Label'], axis=1)
yAll = df_all.loc[:, 'Label']


# FUNÇÂO DE CRIAÇÂO DE MODELO ###############################################

def cria_modelo(pShape, pOutput=True, pQtdNeurons=5, pQtdCamadas=1):
  # Validacao dos parâmetros
  if (pQtdCamadas < 1) or (pQtdCamadas > 9):
    raise ValueError("O número de camadas deve ser maior do que 0 e inferior a 10.")
  
  ## Modelo base
  modelo = Sequential([
      # Camada de Entrada
      Input(shape=(pShape,))
  ])

  # Adiciona camadas ocultas e Dropout baseadas em pQtdCamadas
  for _ in range(pQtdCamadas):
      modelo.add(Dense(2**pQtdNeurons, activation='relu'))
      # Dropout para prevenir overfitting
      modelo.add(Dropout(0.1))

  # Camada de saída
  modelo.add(Dense(1, activation='sigmoid'))

  # Arquitetura
  if pOutput:
    modelo.summary()
    print(f"Modelo criado com {pQtdCamadas} camadas ocultas e {2**pQtdNeurons} neurônios por camada.")
    print(f"Shape de entrada: {pShape}")

  ## Compilando o modelo
  if pOutput:
    print("\nCompilando modelo...")

  modelo.compile(
      optimizer='adam',
      #loss='sparse_categorical_crossentropy',
      loss='binary_crossentropy',
      metrics=['accuracy']
  )

  if pOutput:
    print(f"Learning rate: {modelo.optimizer.learning_rate.numpy()}")
    print("Compilação completa.\n")

  return modelo

##################################################################################
# MultiObjetivo Optimization com Pymoo
# !pip install -U pymoo

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.callback import Callback

import pickle

###############################################################################################
# Classe do smpling inicial da população
###############################################################################################

from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.ux import UniformCrossover
#from pymoo.operators.mutation.bitflip import BitflipMutation

class CustomFeatureSampling(Sampling):
    """
    Sampling customizado que inicia a população com diferentes quantidades de features.
    Garante diversidade na quantidade de features selecionadas.
    """
    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, problem.n_var), dtype=int)
        
        for i in range(n_samples):
            #sampling do pymoo
            # Para cada uma das 40 features independentemente:
            #feature[i] = np.random.randint(0, 2)  # 0 ou 1, com 50% de chance cada
            #Número de features | Probabilidade
            #0                  | 0.00000009%  (praticamente impossível)
            #5                  | 0.03%        (muito raro)
            #10                 | 3.5%
            #15                 | 12%
            #20                 | 18%          (PICO - mais provável)
            #25                 | 12%
            #30                 | 3.5%
            #35                 | 0.03%        (muito raro)
            #40                 | 0.00000009%  (praticamente impossível)

            #Huang & Wang (2006) - "Genetic Algorithm with Adaptive Elitist-population"
            #Recomendam: sampling uniforme em quantidades de features

            #Kabir et al. (2012) - "A New Hybrid AntTree for Feature Selection"
            #Usam: distribuição uniforme entre 10% e 90% das features

            #Xue et al. (2016) - PSO para feature selection
            #Inicialização: uniforme entre valores extremos
            
            # Nos testes anteriores, o limite inferior de features selecionadas nas soluções ótimas encontradas era 17 
            # Por outro lado, 90% de 40 é 36, o que é um valor alto para este problema.
            # Features (primeiras 40 variáveis): seleciona aleatoriamente entre 17 e 35 features
            # Teremos uma diversidade maior de soluções iniciais  distribuída igualmente entre as soluções 
            # (5,26% para cada quantidade de features entre 17 e 35)
            n_features_to_select = np.random.randint(17, 36)
            
            selected_indices = np.random.choice(40, n_features_to_select, replace=False)
            X[i, selected_indices] = 1
            
            # Neurônios (variável 40): entre 2 e 9
            X[i, 40] = np.random.randint(2, 10)
            
            # Camadas (variável 41): entre 1 e 5
            X[i, 41] = np.random.randint(1, 6)
        
        return X
    


###############################################################################################
# Custom Callback for saving checkpoints
###############################################################################################
class CheckpointCallback(Callback):

    def __init__(self, filename="checkpoint.pkl", verbose=False):
        super().__init__()
        self.filename = filename
        self.verbose = verbose
        self.n_eval = 0

    def notify(self, algorithm):
        # Save the algorithm state
        s = f"{self.filename}_{algorithm.n_gen}.pkl"
        with open(s, 'wb') as f:
            pickle.dump(algorithm, f)

        if self.verbose:
            print(f"Checkpoint saved at generation {algorithm.n_gen} to {s}")

        # reduz o consumo de memória a cada geração
        gc.collect()
        if self.verbose:
            print(f"Garbage collection completed at generation {algorithm.n_gen}")


def Minimize(problem):

    #inicialização padrão, com poucas features
    #amostragem_customizada = SamplingComFeaturesFixas(n_features_para_selecionar=5)

    # Define the path for saving and loading the checkpoint
    checkpoint_path = LOG_DIR + LOG_CHECKPOINT

    # Create the checkpoint callback
    checkpoint_callback = CheckpointCallback(filename=checkpoint_path, verbose=True)

    # Check if a checkpoint exists and load it
    if os.path.exists(f"{checkpoint_path}.pkl"):
        print(f"Loading checkpoint from: {checkpoint_path}.pkl")

        with open(f"{checkpoint_path}.pkl", 'rb') as f:
            algorithm = pickle.load(f)
        print(f"Resuming from generation {algorithm.n_gen}")

        # Continua otimização
        res = minimize(problem,
                       algorithm, # passa o algoritmo carregado
                       seed=42,   # mantém o mesmo seed original
                       save_history=True,
                       verbose=True,
                       termination=('n_gen', 100),  # continua até o número de gerações chega a 100
                       callback=checkpoint_callback # callback para gravação
                       )
    else:
        print(f"No checkpoint found with name {checkpoint_path}.pkl. Starting new optimization.")
       
        algorithm = NSGA2(pop_size=200
                          , sampling=CustomFeatureSampling()
                          , crossover=UniformCrossover(prob=0.9)
                          #, mutation=BitflipMutation(prob=0.05)  # 5% de chance de flipar cada bit RUIM PARA ESTE PROBLEMA
                          )
        #print(f"Algorithm Crossover: {algorithm.mating.crossover}")
        #print(f"Algorithm Mutação: {algorithm.mating.mutation}")

        res = minimize(problem,
                       algorithm,
                       seed=42,
                       save_history=True,
                       verbose=True,
                       termination=('n_gen', 100),
                       callback=checkpoint_callback # callback para gravação
                       )

    # Save the final result
    final_result_path = LOG_DIR + LOG_FINAL
    os.makedirs(os.path.dirname(final_result_path), exist_ok=True)

    with open(final_result_path, 'wb') as f:
        pickle.dump(res, f)
    print(f"Final result saved to: {final_result_path}")


    return res



###############################################################################################
# Classe do problema a ser resolvido
###############################################################################################
class NetFlowProblem(ElementwiseProblem): #ElementWise é orientado a objetos

    def is_to_delete_feature(self, prob):
        return prob == 0

    def adjust_features(self, x, X_features):

        delete_indices = [i for i in range(len(x)) if self.is_to_delete_feature(x[i])]

        cols_to_delete = X_features.columns[delete_indices]
        X_features = X_features.drop(columns=cols_to_delete)

        return X_features

    def create_and_train_model(self, pX, py, pQtdNeurons, pQtdCamadas):
        # Dados: separando da base original uma parte para teste
        x_train, x_test, y_train, y_test = train_test_split(pX, py, test_size=0.2, random_state=42, stratify=py)

        # Dados: normalização
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        # Cria modelo
        mymodel = cria_modelo(pShape=x_train_scaled.shape[1],
                              pOutput=False,
                              pQtdNeurons=pQtdNeurons,
                              pQtdCamadas=pQtdCamadas)

        # Treina o modelo        
        history = mymodel.fit(
          x_train_scaled,
          y_train,
          epochs=15,
          batch_size=BATCH_SIZE,
          validation_data=(x_test_scaled, y_test),
          #callbacks=[mc_accm], # desativei a callback para aumentar a velocidade. Resultado: não alterou tanto...
          verbose=0             # progress bar        
        )

        # Libera a memória dos dados de treino/teste que não são mais necessários
        del x_train, x_test, y_train, y_test, x_train_scaled, x_test_scaled, scaler, history
        gc.collect()

        return mymodel


    def get_predicted_class(self, pX, py, modelAll):

        # Separando da base original uma parte para validação. Aqui eu chamarei de teste
        x_train, x_test, y_train, y_test = train_test_split(pX, py, test_size=0.2, random_state=42, stratify=py)

        # normalização
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)       

        # Executa a predição no conjunto de dados de teste
        predictions_prob = modelAll.predict(x_test_scaled, verbose=0 )
        predictions = (predictions_prob > 0.5).astype("int32").flatten()

        #marca para o garbage collector remover da memória a fim de conter a expansão do consumo
        del x_train, x_test, y_train, x_train_scaled, x_test_scaled, predictions_prob #, mc_accm, fpm
        gc.collect()


        return predictions, y_test


    def evaluate_model_f1(self,pX
                              ,py
                              ,modelAll                              
                          ):
        # Executa a predição no conjunto de dados passado (algum dataset de teste)
        y_test_predict, y_test_true = self.get_predicted_class(pX, py, modelAll)
        # Calcula o F1 Score
        #f1_test = f1_score(y_true=y_test_true, y_pred=y_test_predict)

        # Calcula a matriz de confusão para obter TN, FP, FN, TP
        try:
            tn, fp, fn, tp = confusion_matrix(y_true=y_test_true, y_pred=y_test_predict).ravel()
            # Calcula TPR (True Positive Rate ou Recall) e TNR (True Negative Rate ou Specificity)
            # Adiciona uma pequena constante (epsilon) para evitar divisão por zero
            epsilon = 1e-7
            tpr = tp / (tp + fn + epsilon)
            tnr = tn / (tn + fp + epsilon)

            # Calcula o F1 Score com base em TP, FP e FN
            precision = tp / (tp + fp + epsilon)
            recall = tp / (tp + fn + epsilon)
            f1_test = 2 * (precision * recall) / (precision + recall + epsilon)
        except ValueError:
            # Caso a predição contenha apenas uma classe, o ravel() falhará.
            tpr, tnr, f1_test = 0.0, 0.0, 0.0

        del y_test_predict, y_test_true

        #retorno 1-, porque queremos minimizar o valor de comparação
        return (1 - f1_test), tpr, tnr



    def __init__(self, *args):
        
        # Define os limites inferiores (xl) e superiores (xu) para as 42 variáveis inteiras.
        # 40 variáveis para seleção de features (0 ou 1)
        # 1 variável para número de neurônios (2 a 9)
        # 1 variável para número de camadas (1 a 5)
        xl = np.array([0] * 40 + [2, 1])
        xu = np.array([1] * 40 + [9, 5])

        self.contagem = 0

        #Variaveis: 40 features + neuronios
        super().__init__(
            n_var=42,                               # 42 variáveis no total
            xl=xl,                                  # Limites inferiores
            xu=xu,                                  # Limites superiores
            vtype=int,                              # Define que todas as variáveis são inteiras
            n_obj=3,                                # Número de objetivos (menor 1-F1 DS1, DS2, DS3)
            n_ieq_constr=3                          # restrição de desigualdade
        )



    def _evaluate(self, x, out, *args, **kwargs):
      #variaveis
      f1_2 = 1.0               # Inicializa com 1 para caso a restrição não seja atendida
      f1_3 = 1.0               # Inicializa com 1 para caso a restrição não seja atendida
      f1_4 = 1.0               # Inicializa com 1 para caso a restrição não seja atendida
      tpr_2, tnr_2 = 0.0, 0.0
      tpr_3, tnr_3 = 0.0, 0.0
      tpr_4, tnr_4 = 0.0, 0.0
      qtd_features = 40        # O valor será determinado posteriormente
      num_neurons = int(x[-2]) # O penúltimo elemento de x é a quantidade de neurônios
      qtd_camadas = int(x[-1]) # O último elemento de x é a quantidade de camadas

      self.contagem += 1

      t0 = time2()
      #print("Current x:", x, end="") # esse x é o vetor que indica quais campos devem ser ativados ou não

      # Obtém o algoritmo do Pymoo e o número da avaliação atual a partir dos kwargs
      #algorithm = kwargs.get("algorithm")
      #n_eval = algorithm.evaluator.n_eval if algorithm else 0
      print(f"Avaliação {self.contagem:05d}", end='')

      # Extrai a parte de seleção de features do vetor x
      # Todas as posições exceto as duas últimas
      x_features_selection = x[:-2]

      # Dataset All #################
      # reinicializa os vetores, com os valores originais de definição do Dataset
      X_tmp = XAll.copy()
      y_tmp = yAll.copy()

      #acerta os campos do dataset
      X_tmp = self.adjust_features(x_features_selection, X_tmp)
      qtd_features = len(X_tmp.columns)

      parametros_validos = True

      # A restrição para o número de camadas é ser entre 1 e 5.
      if ((qtd_camadas<1) or (qtd_camadas>5)):
        print(", Número incorreto de camadas alcançado(<1 ou >5)", end="")        
        parametros_validos = False

      if ((qtd_features<2) or (num_neurons<2)):
        print(", Número insuficiente de features/neurônios alcançado(<2)", end="")        
        parametros_validos = False
      
      if ((num_neurons>9)):
        print(", Número de neurônios fora do limite superior a 9", end="")        
        parametros_validos = False

      if parametros_validos:
        # Treino o modelo no Dataset All e avalio nos outros datasets
        # Treino no All
        modelAll = self.create_and_train_model(X_tmp, y_tmp, num_neurons, qtd_camadas)

        # Dataset 2 ################
        X_tmp = X2.copy()
        y_tmp = y2.copy()
        X_tmp = self.adjust_features(x_features_selection, X_tmp)
        f1_2, tpr_2, tnr_2  = self.evaluate_model_f1(X_tmp, y_tmp, modelAll)

        # Dataset 3 ################
        X_tmp = X3.copy()
        y_tmp = y3.copy()
        X_tmp = self.adjust_features(x_features_selection, X_tmp)
        f1_3, tpr_3, tnr_3  = self.evaluate_model_f1(X_tmp, y_tmp, modelAll)

        # Dataset 4 ################
        X_tmp = X4.copy()
        y_tmp = y4.copy()
        X_tmp = self.adjust_features(x_features_selection, X_tmp)
        f1_4, tpr_4, tnr_4  = self.evaluate_model_f1(X_tmp, y_tmp, modelAll)

        # Limpa o modelo e a sessão do Keras imediatamente
        #del self.modelAll
        #self.modelAll = None
        if modelAll is not None:
          for layer in modelAll.layers:  # Deleta cada camada
            del layer
          del modelAll
          
        
        tf.keras.backend.clear_session()
        
        # Força limpeza de memória da GPU se disponível
        if tf.config.list_physical_devices('GPU'):
            try:
                tf.compat.v1.reset_default_graph()
            except:
                pass

      # Cálculo de tempo
      t1 = (time2() - t0)      
      print(f", duração: {t1:.2f}", end="")
             

      # --- CÁLCULO DA RESTRIÇÃO ---
      # A restrição é haver ao menos o valor 2 nos campos features e neuronios
      # Então a fórmula é `2 - qtd_features`.
      # Um valor > 0 significa que a restrição foi violada (qtd_features < 2).
      # Um valor <= 0 significa que a restrição foi satisfeita (qtd_features >= 2).
      g_features =  2 - qtd_features
      # O mesmo para os neurônios, que não podem ser inferior a 2
      g_neuronios = 2 - num_neurons
      # O mesmo para o mínimo de camadas, que não podem ser inferior a 1
      # e também nao podem ser maior que 5
      if qtd_camadas <= 1:
        g_camadas = 1 - qtd_camadas
      else:
        if qtd_camadas > 5:
          g_camadas = qtd_camadas
        else:
          g_camadas = qtd_camadas - 5

      out["G"] = [g_features, g_neuronios, g_camadas]

      # Os objetivos são: qtd_features, 1-F1 DS1, 1-F1 DS2, 1-F1 DS3, num_neurons
      out["F"] = [f1_2, f1_3, f1_4]

      print(f", qtd feat.:{qtd_features}, camadas:{qtd_camadas}, neurônios:{num_neurons} | 1-F1 DS2:{f1_2:.4f}, DS3:{f1_3:.4f}, DS4:{f1_4:.4f}")

      # gravacao de log para avaliação posterior
      str_log = f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, {self.contagem:05d}, {qtd_features}, {f1_2:.4f}, {f1_3:.4f}, {f1_4:.4f}, {num_neurons}, {t1:.2f}, {qtd_camadas}, {tpr_2:.4f}, {tnr_2:.4f}, {tpr_3:.4f}, {tnr_3:.4f}, {tpr_4:.4f}, {tnr_4:.4f}'

      # marca para o garbage collector remover da memória a fim de conter a expansão do consumo
      #self.modelAll = None
      
      with open(LOG_DIR+LOG_EXECUCAO, 'a') as f:
        f.write(str_log + '\n')
        f.flush()
      
      del X_tmp, y_tmp, f1_3, tpr_3, tnr_3, f1_2, tpr_2, tnr_2, f1_4, tpr_4, tnr_4, x_features_selection
      del parametros_validos, g_camadas, g_features, g_neuronios, num_neurons, qtd_camadas, qtd_features, t0, t1
      del str_log

      # Força garbage collection agressivo (múltiplas gerações)
      gc.collect()
      gc.collect()
      #gc.collect()

################################################ Fim da classe do problema ##########################################################

problem = NetFlowProblem()

### Execução do pymoo

res_dt = Minimize(problem)

print("Processamento concluído.")
