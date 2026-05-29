#!/bin/bash

# Define seu destino fixo
DESTINO="gdrive:Colab Notebooks/Mestrado/Orientação/Tese/Resultados"

# --- LÓGICA DA FLAG DE NÃO DELETAR ---
# Define o padrão como true (vai deletar)
DELETAR_ARQUIVO=true
SYNC_ALL=false

# Verifica flags (keep e all)
while true; do
    if [ "$1" == "--keep" ] || [ "$1" == "-k" ]; then
        DELETAR_ARQUIVO=false
        echo "Manter arquivos originais ativado."
        shift
    elif [ "$1" == "--all" ] || [ "$1" == "-a" ] || [ "$1" == "all" ]; then
        SYNC_ALL=true
        echo "Sincronização de todos os arquivos (incluindo .pth) ativada."
        shift
    else
        break
    fi
done

# Verifica se algum arquivo foi passado
if [ $# -eq 0 ]; then
    echo "Erro: Nenhum arquivo fornecido."
    echo "Uso: $0 <arquivo1> [arquivo2] ..."
    exit 1
fi

# Itera sobre CADA argumento (arquivo) passado para o script
for arquivo in "$@"
do
  # Verifica se o arquivo/diretório realmente existe antes de tentar copiar
  if [ -e "$arquivo" ]; then
  
    # Se for .pth e a flag SYNC_ALL não estiver ativa, pula
    if [[ "$arquivo" == *.pth ]] && [ "$SYNC_ALL" = false ]; then
        echo "Ignorando arquivo .pth: $arquivo (use -a ou all para incluir)"
        continue
    fi

    echo "----------------------------------------"
    echo "Copiando: $arquivo"
    echo "Para:     $DESTINO"
    echo "----------------------------------------"

    # Executa o comando rclone para CADA arquivo
    # Removi o --max-depth 1, pois estamos passando arquivos individuais.
    # Se você também passar diretórios e quiser limitar a profundidade, pode mantê-lo.
    ~/rclone copy -vP "$arquivo" "$DESTINO"
    
    if [ "$DELETAR_ARQUIVO" = true ]; then
    	rm "$arquivo"
	echo "Arquivo '$arquivo' removido localmente."
    else
        echo "Sucesso: Arquivo mantido localmente (--keep)."
    fi

  else
    echo "Aviso: Arquivo '$arquivo' não encontrado. Pulando."
  fi

done


echo "Processo de cópia concluído."
