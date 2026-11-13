# Desafio Final: Análise de Dados com OpenMP

Cálculo do **coeficiente de variação (CV)** de estatura e peso de 1000 pessoas, com dados agrupados em
classes (8 cm para estatura, 4 kg para peso), usando OpenMP. Inclui versão sequencial otimizada e versão
paralela para comparação de desempenho.

## Arquivos
| Arquivo | Descrição |
|---|---|
| `cv_openmp.c` | Código-fonte (versões sequencial e paralela) |
| `dados.txt` | 1000 linhas `estatura peso` (dados **sintéticos**, o enunciado não fornece base) |
| `gera_dados.py` | Gera o `dados.txt` (semente 42; estatura ~ N(170, 10), peso ~ N(70, 12)) |
| `relatorio.pdf` | Relatório: paralelização, resultados e análise de desempenho |
| `resultados/execucoes.txt` | Saída bruta das execuções com 1, 2, 4 e 8 threads |

## Como compilar e executar
```bash
gcc -O2 -fopenmp cv_openmp.c -o cv -lm
./cv dados.txt                     # usa 10000x de replicação no teste de desempenho
./cv dados.txt 10000 tabela        # também imprime a tabela de frequências
OMP_NUM_THREADS=4 ./cv dados.txt   # define o número de threads (Linux/Mac/Git Bash)
```
No PowerShell: `$env:OMP_NUM_THREADS=4; ./cv dados.txt`

Para usar outra base, troque o `dados.txt` mantendo o formato (uma pessoa por linha, estatura e peso
separados por espaço, até 1000 linhas).

## Resultados (dados sintéticos)
| Variável | Média | Desvio padrão | CV |
|---|---|---|---|
| Estatura (classe 8 cm) | 169,872 cm | 10,465 | **6,16%** |
| Peso (classe 4 kg) | 69,780 kg | 12,004 | **17,20%** |

## Desempenho (10 milhões de pessoas, dados replicados)
| Threads | Sequencial (s) | Paralelo (s) | Speedup |
|---|---|---|---|
| 1 | 0,0383 | 0,0389 | 0,98x |
| 2 | 0,0353 | 0,0257 | 1,37x |
| 4 | 0,0364 | 0,0160 | 2,27x |
| 8 | 0,0393 | 0,0155 | 2,54x |

Com apenas 1000 pessoas o paralelo é mais lento (o cálculo leva cerca de 3 us e o custo de criar threads
domina). Detalhes e análise no `relatorio.pdf`.
