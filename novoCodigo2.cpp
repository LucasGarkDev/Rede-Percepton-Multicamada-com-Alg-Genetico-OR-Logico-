/**
 * @file main.c
 * @brief Implementação de uma Rede Neural Artificial Evolutiva para aprendizado da função lógica 'A OU B'.
 * 
 * O código simula uma população de indivíduos que evolui para aprender a função lógica 'A OU B' utilizando
 * cruzamento genético, mutações e infecção viral. O processo de aprendizado é feito por gerações, e cada indivíduo
 * é avaliado com base na sua performance em relação a lições específicas.
 */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_Entradas 2   /**< Número máximo de entradas para a rede neural. */
#define MAX_Pesos 6      /**< Número de pesos genéticos para cada indivíduo. */

//===| Estrutura de Dados |==========================

/**
 * @typedef string
 * @brief Definição de tipo para representar strings de até 60 caracteres.
 */
typedef char string[60];

/**
 * @struct TLicao
 * @brief Estrutura que representa uma lição que a rede deve aprender (A OU B).
 */
typedef struct tipoLicao {
    int A;                 /**< Entrada A (0 ou 1). */
    int B;                 /**< Entrada B (0 ou 1). */
    int resultadoEsperado; /**< Resultado esperado da operação A OU B. */
    struct tipoLicao *prox; /**< Ponteiro para a próxima lição na lista encadeada. */
} TLicao;

/**
 * @struct TIndividuo
 * @brief Estrutura que representa um indivíduo da população.
 */
typedef struct tipoIndividuo {
    float genes[MAX_Pesos]; /**< Vetor de pesos (genes) do indivíduo. */
    int erros;              /**< Número de erros do indivíduo ao avaliar as lições. */
    int numero;             /**< Número identificador do indivíduo. */
    struct tipoIndividuo *prox; /**< Ponteiro para o próximo indivíduo na lista encadeada. */
} TIndividuo;

/**
 * @struct TLista
 * @brief Estrutura que representa a lista principal, contendo os indivíduos e as lições.
 */
typedef struct tipoLista {
    FILE *fp;                        /**< Arquivo de saída para o relatório de treinamento. */
    string objetivo;                 /**< Objetivo do treinamento. */
    TLicao *licoes;                  /**< Lista encadeada de lições a serem aprendidas. */
    TIndividuo *populacao;           /**< Lista encadeada de indivíduos da população. */
    TIndividuo *fimLista;            /**< Último indivíduo da lista de população. */
    int totalIndividuos;             /**< Número total de indivíduos na população. */
    int Qtd_Populacao;               /**< Quantidade de indivíduos por geração. */
    int Qtd_Mutacoes_por_vez;        /**< Quantidade de mutações por vez. */
    int Total_geracoes;              /**< Total de gerações a serem executadas. */
    int geracao_atual;               /**< Geração atual. */
    int Qtd_Geracoes_para_Mutacoes;  /**< Intervalo de gerações para ocorrência de mutações. */
    float sinapseThreshold;          /**< Limiar de sinapse para ativação dos neurônios. */
    float learningRate;              /**< Taxa de aprendizado para ajuste de pesos. */
    float probabilidadeInfeccao;     /**< Probabilidade de infecção viral entre indivíduos. */
} TLista;

TLista lista; /**< Instância principal da estrutura TLista. */

//====| Assinatura de Funções |=======================+

/**
 * @brief Inicializa a estrutura TLista com parâmetros fornecidos pelo usuário.
 * @param L Ponteiro para a estrutura TLista.
 */
void inicializa(TLista *L);

/**
 * @brief Gera os indivíduos iniciais da população.
 * @param L Ponteiro para a estrutura TLista.
 */
void geraIndividuos(TLista *L);

/**
 * @brief Gera as lições (entradas e saídas esperadas) para o treinamento.
 * @param L Ponteiro para a estrutura TLista.
 */
void geraLicoes(TLista *L);

/**
 * @brief Insere uma lição na lista encadeada de lições.
 * @param L Ponteiro para a estrutura TLista.
 * @param A Valor da entrada A.
 * @param B Valor da entrada B.
 * @param resultado Resultado esperado (A OU B).
 */
void insereLicao(TLista *L, int A, int B, int resultado);

/**
 * @brief Executa o processo de treinamento da rede neural.
 * @param L Ponteiro para a estrutura TLista.
 */
void treinamento(TLista *L);

/**
 * @brief Realiza o cruzamento genético entre indivíduos para gerar novos descendentes.
 * @param L Ponteiro para a estrutura TLista.
 */
void cruzamento(TLista *L);

/**
 * @brief Avalia os indivíduos da população em relação às lições.
 * @param L Ponteiro para a estrutura TLista.
 */
void avaliacaoIndividuos(TLista *L);

/**
 * @brief Ordena os indivíduos da população com base no número de erros.
 * @param L Ponteiro para a estrutura TLista.
 */
void ordenamentoIndividuos(TLista *L);

/**
 * @brief Realiza mutações genéticas nos indivíduos da população.
 * @param L Ponteiro para a estrutura TLista.
 * @param learningRate Taxa de aprendizado utilizada para alterar os genes.
 */
void promoveMutacoes(TLista *L, float learningRate);

/**
 * @brief Remove os indivíduos excedentes da população, mantendo apenas os mais aptos.
 * @param L Ponteiro para a estrutura TLista.
 */
void poda(TLista *L);

/**
 * @brief Imprime os indivíduos da população e seus respectivos erros.
 * @param L Ponteiro para a estrutura TLista.
 */
void printIndividuos(TLista *L); 

//===| Programa Principal |===========================

/**
 * @brief Função principal do programa.
 * @return Retorna 0 ao finalizar o programa.
 */
int main(){
    inicializa(&lista);
    treinamento(&lista);
    return 0;
}

//===| Funções |======================================

/**
 * @brief Abre um arquivo no modo especificado e retorna o ponteiro para ele.
 * 
 * A função verifica se o arquivo foi aberto corretamente. Caso contrário, exibe uma mensagem de erro.
 * 
 * @param nomeArq Nome do arquivo que será aberto.
 * @param modo Modo de abertura do arquivo (ex: "w" para escrita).
 * @return Retorna o ponteiro para o arquivo aberto ou NULL em caso de erro.
 */
FILE* abrirArquivo(const char* nomeArq, const char* modo) {
    FILE* arq = fopen(nomeArq, modo);
    if (arq == NULL) {
        printf("\n\n\tERRO ao abrir o arquivo.\n\n");
        return NULL;
    }
    return arq;
}

/**
 * @brief Inicializa os parâmetros da lista, popula a população e gera lições para a rede neural.
 * 
 * Esta função configura os parâmetros da lista TLista, como a quantidade de indivíduos e a taxa de aprendizado.
 * Também chama funções para gerar indivíduos e lições.
 * 
 * @param L Ponteiro para a estrutura TLista que será inicializada.
 */
void inicializa(TLista *L) {
    L->licoes = NULL;
    L->populacao = NULL;
    L->fimLista = NULL;  // Inicializa o ponteiro do último elemento da lista de indivíduos.
    L->totalIndividuos = 0;
    L->fp = abrirArquivo("relatorio_treinamento.txt", "w");

    printf("\t\t=====| REDE NEURAL ARTIFICIAL EVOLUTIVA |=====\n\n");

    // Solicita ao usuário a quantidade de indivíduos e gera a população inicial.
    printf("\tInforme o TAMANHO da POPULACAO (em termos de individuos): ");
    scanf("%d", &L->Qtd_Populacao);

    geraIndividuos(L);  // Chama a função para gerar a população de indivíduos.

    // Solicita e define parâmetros adicionais do sistema.
    printf("\n\tInforme a QUANTIDADE de GERACOES maxima: ");
    scanf("%d", &L->Total_geracoes);

    printf("\n\tInforme o INTERVALO de GERACOES para a ocorrencia de MUTACOES: ");
    scanf("%d", &L->Qtd_Geracoes_para_Mutacoes);

    printf("\n\tInforme a QUANTIDADE de MUTACOES que devem ocorrer POR VEZ: ");
    scanf("%d", &L->Qtd_Mutacoes_por_vez);

    printf("\n\tInforme a PROBABILIDADE de INFECCAO VIRAL (0 a 1): ");
    scanf("%f", &L->probabilidadeInfeccao);

    printf("\n\tInforme o SINAPSE THRESHOLD (Sinal limiar): ");
    scanf("%f", &L->sinapseThreshold);

    printf("\n\tInforme o LEARNING RATE (Taxa de aprendizado): ");
    scanf("%f", &L->learningRate);

    strcpy(L->objetivo, "Aprendizado da Funcao Logica A OU B");  // Define o objetivo do treinamento.

    geraLicoes(L);  // Chama a função para gerar as lições que a rede deve aprender.
}

/**
 * @brief Gera um relatório de progresso após cada geração.
 * 
 * O relatório inclui a geração atual, a população total, os parâmetros de aprendizado, e a lista de indivíduos
 * com seus respectivos erros.
 * 
 * @param fp Ponteiro para o arquivo onde o relatório será gravado.
 * @param geracao Número da geração atual.
 * @param L Ponteiro para a estrutura TLista contendo os dados da população.
 */
void gerarRelatorio(FILE* fp, int geracao, TLista *L) {
    fprintf(fp, "\n=== Geração %d ===\n", geracao);
    fprintf(fp, "População Total: %d\n", L->totalIndividuos);
    fprintf(fp, "Parâmetros:\n");
    fprintf(fp, "Learning Rate: %.2f\n", L->learningRate);
    fprintf(fp, "Sinapse Threshold: %.2f\n\n", L->sinapseThreshold);

    // Imprime os indivíduos e seus respectivos erros.
    TIndividuo *atual = L->populacao;
    fprintf(fp, "Indivíduos e Erros:\n");
    while (atual != NULL) {
        fprintf(fp, "Indivíduo %d: Erros = %d\n", atual->numero, atual->erros);
        atual = atual->prox;
    }
    fprintf(fp, "\n---------------------------------------\n");
}

/**
 * @brief Gera os indivíduos iniciais da população.
 * 
 * Esta função aloca memória para cada indivíduo da população, inicializa seus genes com valores aleatórios
 * e os insere na lista encadeada de indivíduos.
 * 
 * @param L Ponteiro para a estrutura TLista onde os indivíduos serão inseridos.
 */
void geraIndividuos(TLista *L) {
    TIndividuo *novo;
    srand((unsigned)time(NULL));  // Inicializa o gerador de números aleatórios.

    // Gera indivíduos com genes aleatórios.
    for (int i = 0; i < L->Qtd_Populacao; i++) {
        printf("Gerando individuo %d...\n", i+1);  // Imprime informações sobre a geração do indivíduo.

        novo = (TIndividuo *)malloc(sizeof(TIndividuo));  // Aloca memória para o novo indivíduo.
        if (novo == NULL) {
            printf("Erro ao alocar memória para novo indivíduo.\n");
            exit(1);  // Encerra o programa se houver erro de alocação.
        }

        novo->prox = NULL;
        novo->numero = i + 1;
        novo->erros = -1;

        // Inicializa os genes do indivíduo com valores aleatórios entre 0 e 1.
        for (int x = 0; x < MAX_Pesos; x++) {
            novo->genes[x] = (rand() % 101) / 100.0;  // Gera valores aleatórios para os genes.
        }

        // Insere o novo indivíduo na lista encadeada.
        if (L->populacao == NULL) {
            L->populacao = novo;
        } else {
            TIndividuo *atual = L->populacao;
            while (atual->prox != NULL) {
                atual = atual->prox;
            }
            atual->prox = novo;
        }

        L->totalIndividuos++;  // Atualiza o total de indivíduos.
        printf("Individuo %d gerado com sucesso.\n", novo->numero);  // Confirma a geração do indivíduo.
    }
    L->fimLista = novo;  // Define o último indivíduo na lista.
    printf("População inicial gerada com %d indivíduos.\n", L->totalIndividuos);  // Informa o total gerado.
}

/**
 * @brief Gera o conjunto de lições que a rede neural deverá aprender.
 * 
 * Esta função insere manualmente as lições da função lógica A OU B. Cada lição contém duas entradas (A e B)
 * e o resultado esperado.
 * 
 * @param L Ponteiro para a estrutura TLista onde as lições serão inseridas.
 */
void geraLicoes(TLista *L) {
    // Insere as lições correspondentes à operação lógica A OU B.
    insereLicao(L, 0, 0, 0);  // 0 OU 0 = 0
    insereLicao(L, 0, 1, 1);  // 0 OU 1 = 1
    insereLicao(L, 1, 0, 1);  // 1 OU 0 = 1
    insereLicao(L, 1, 1, 1);  // 1 OU 1 = 1
    printf("Lições geradas com sucesso.\n");  // Confirma a geração das lições.
}

/**
 * @brief Insere uma nova lição na lista de lições.
 * 
 * Cada lição representa uma combinação de entradas (A e B) e o resultado esperado da operação lógica A OU B.
 * 
 * @param L Ponteiro para a estrutura TLista onde a lição será inserida.
 * @param A Valor da entrada A (0 ou 1).
 * @param B Valor da entrada B (0 ou 1).
 * @param resultado Resultado esperado da operação A OU B.
 */
void insereLicao(TLista *L, int A, int B, int resultado) {
    TLicao *novo = (TLicao *)malloc(sizeof(TLicao));  // Aloca memória para a nova lição.
    if (novo == NULL) {
        printf("Erro ao alocar memória para nova lição.\n");
        exit(1);  // Encerra o programa em caso de erro de alocação.
    }

    novo->prox = NULL;
    novo->A = A;
    novo->B = B;
    novo->resultadoEsperado = resultado;

    // Insere a nova lição na lista encadeada.
    if (L->licoes == NULL) {
        L->licoes = novo;
    } else {
        TLicao *atual = L->licoes;
        while (atual->prox != NULL) {
            atual = atual->prox;
        }
        atual->prox = novo;
    }
}
/**
 * @brief Aplica a infecção viral nos indivíduos da população com base na probabilidade.
 * 
 * Esta função percorre a lista de indivíduos e, com base em uma probabilidade de infecção viral,
 * transfere parte dos genes do indivíduo mais apto (aquele com menos erros) para outros indivíduos.
 * 
 * @param L Ponteiro para a estrutura TLista que contém os parâmetros da população e probabilidade de infecção.
 */
void promoveInfeccaoViral(TLista *L) {
    if (L->populacao == NULL) {
        return;
    }

    TIndividuo *atual = L->populacao;
    TIndividuo *individuoMaisApto = atual;  // Inicialmente, o primeiro da lista é o mais apto.
    
    // Encontra o indivíduo mais apto (com menor número de erros).
    while (atual != NULL) {
        if (atual->erros < individuoMaisApto->erros) {
            individuoMaisApto = atual;
        }
        atual = atual->prox;
    }

    atual = L->populacao;  // Reseta o ponteiro para iterar sobre a lista novamente.

    while (atual != NULL) {
        // Aplica a infecção com base na probabilidade.
        if ((rand() % 100) / 100.0 < L->probabilidadeInfeccao) {
            // Transfere parte dos genes do indivíduo mais apto para o atual.
            printf("Indivíduo %d foi infectado viralmente por indivíduo %d.\n", atual->numero, individuoMaisApto->numero);
            
            // Infecta parte dos genes (50% por exemplo).
            for (int i = 0; i < MAX_Pesos; i++) {
                if (rand() % 2 == 0) {
                    atual->genes[i] = individuoMaisApto->genes[i];
                }
            }
        }
        atual = atual->prox;
    }
}

/**
 * @brief Executa o ciclo de treinamento da rede neural evolutiva.
 * 
 * O treinamento consiste em realizar cruzamentos genéticos, aplicar mutações, infecções virais,
 * avaliar os indivíduos, ordená-los, podar os indivíduos excedentes e gerar um relatório de cada geração.
 * 
 * @param L Ponteiro para a estrutura TLista que contém os parâmetros da população e o ciclo de treinamento.
 */
void treinamento(TLista *L) {
    printf("\n\n\t\t=====| INICIADO TREINAMENTO |=====\n\n");

    for (int i = 0; i < L->Total_geracoes; i++) {
        printf("Iniciando cruzamento na geração %d...\n", i+1);
        cruzamento(L);

        if (i % L->Qtd_Geracoes_para_Mutacoes == 0) {
            printf("Iniciando mutações...\n");
            promoveMutacoes(L, L->learningRate);
        }

        if ((rand() % 100) / 100.0 < L->probabilidadeInfeccao) {
            printf("Iniciando infecção viral na geração %d...\n", i + 1);
            promoveInfeccaoViral(L);
        }

        printf("Iniciando avaliação dos indivíduos...\n");
        avaliacaoIndividuos(L);
        
        printf("Ordenando indivíduos...\n");
        ordenamentoIndividuos(L);

        printf("Podando indivíduos excedentes...\n");
        poda(L);

        printf("Geração %d completa.\n", i + 1);
        printIndividuos(L);

        gerarRelatorio(L->fp, i + 1, L);
    }

    fclose(L->fp);
}

/**
 * @brief Realiza o cruzamento genético entre indivíduos.
 * 
 * A função combina pares de indivíduos da população para gerar novos descendentes.
 * Cada novo descendente recebe uma parte dos genes de cada progenitor, simulando um cruzamento genético.
 * 
 * @param L Ponteiro para a estrutura TLista que contém os parâmetros da população e os indivíduos.
 */
void cruzamento(TLista *L) {
    TIndividuo *pai1 = L->populacao;
    TIndividuo *pai2 = pai1->prox;
    int cont = L->totalIndividuos + 1;
    int cruzamentos = 0;

    // Limita o número de cruzamentos à metade da população inicial.
    while (pai2 != NULL && cruzamentos < L->Qtd_Populacao / 2) {
        printf("Cruzando individuos %d e %d...\n", pai1->numero, pai2->numero);

        TIndividuo *filho1 = (TIndividuo *)malloc(sizeof(TIndividuo));
        TIndividuo *filho2 = (TIndividuo *)malloc(sizeof(TIndividuo));
        if (filho1 == NULL || filho2 == NULL) {
            printf("Erro ao alocar memória para filhos.\n");
            exit(1);
        }

        filho1->prox = NULL;
        filho2->prox = NULL;

        int metade = MAX_Pesos / 2;
        for (int j = 0; j < metade; j++) {
            filho1->genes[j] = pai1->genes[j];
            filho2->genes[j] = pai2->genes[j];
        }
        for (int j = metade; j < MAX_Pesos; j++) {
            filho1->genes[j] = pai2->genes[j];
            filho2->genes[j] = pai1->genes[j];
        }

        filho1->erros = -1;
        filho2->erros = -1;
        filho1->numero = cont;
        filho2->numero = cont + 1;
        cont += 2;

        L->fimLista->prox = filho1;
        filho1->prox = filho2;
        L->fimLista = filho2;

        printf("Filhos %d e %d gerados com sucesso.\n", filho1->numero, filho2->numero);

        pai1 = pai2->prox;
        if (pai1 != NULL) {
            pai2 = pai1->prox;
        } else {
            pai2 = NULL;
        }

        cruzamentos++;
    }

    printf("Total de cruzamentos realizados: %d\n", cruzamentos);
}

/**
 * @brief Avalia o desempenho dos indivíduos em relação às lições fornecidas.
 * 
 * A função simula a execução dos indivíduos nas lições e ajusta o número de erros de cada indivíduo
 * com base no desempenho obtido.
 * 
 * @param L Ponteiro para a estrutura TLista contendo a população e as lições.
 */
void avaliacaoIndividuos(TLista *L) {
    TIndividuo *atual = L->populacao;

    while (atual != NULL) {
        if (atual->erros == -1) {
            atual->erros = 0;
            TLicao *licaoAtual = L->licoes;
            
            while (licaoAtual != NULL) {
                // Calcula as saídas dos neurônios da primeira camada.
                float n1 = (licaoAtual->A * atual->genes[0]) + (licaoAtual->B * atual->genes[1]);
                float n2 = (licaoAtual->A * atual->genes[2]) + (licaoAtual->B * atual->genes[3]);
                
                n1 = (n1 >= L->sinapseThreshold) ? 1 : 0;
                n2 = (n2 >= L->sinapseThreshold) ? 1 : 0;

                // Calcula a saída do neurônio da segunda camada.
                float n3 = (n1 * atual->genes[4]) + (n2 * atual->genes[5]);
                n3 = (n3 >= L->sinapseThreshold) ? 1 : 0;

                // Verifica se o resultado esperado corresponde ao calculado.
                if (licaoAtual->resultadoEsperado != n3) {
                    atual->erros++;
                }
                licaoAtual = licaoAtual->prox;
            }
        }
        atual = atual->prox;
    }
}

/**
 * @brief Ordena a população de indivíduos com base no número de erros.
 * 
 * Esta função utiliza um algoritmo de ordenação por troca para organizar os indivíduos
 * em ordem crescente de erros, de modo que os indivíduos mais aptos fiquem no topo da lista.
 * 
 * @param L Ponteiro para a estrutura TLista contendo a população de indivíduos.
 */
void ordenamentoIndividuos(TLista *L) {
    TIndividuo *atual, *anterior, *temp;
    int trocou;

    do {
        trocou = 0;
        atual = L->populacao;
        anterior = NULL;

        while (atual->prox != NULL) {
            if (atual->erros > atual->prox->erros) {
                temp = atual->prox;
                atual->prox = atual->prox->prox;
                temp->prox = atual;

                if (anterior != NULL) {
                    anterior->prox = temp;
                } else {
                    L->populacao = temp;
                }
                anterior = temp;
                trocou = 1;
            } else {
                anterior = atual;
                atual = atual->prox;
            }
        }
    } while (trocou);
}

/**
 * @brief Promove mutações nos genes dos indivíduos.
 * 
 * A função seleciona um indivíduo aleatoriamente e aplica uma mutação em um de seus genes,
 * ajustando-o para cima ou para baixo com base na taxa de aprendizado fornecida.
 * 
 * @param L Ponteiro para a estrutura TLista contendo os indivíduos.
 * @param learningRate Taxa de aprendizado que será utilizada para ajustar o gene durante a mutação.
 */
void promoveMutacoes(TLista *L, float learningRate) {
    if (L->populacao == NULL) {
        return;
    }

    // Seleciona um indivíduo aleatoriamente.
    int index = rand() % L->totalIndividuos;
    TIndividuo *individuo = L->populacao;

    for (int i = 0; i < index; i++) {
        individuo = individuo->prox;
    }

    // Escolhe um gene aleatório para mutação.
    int geneIndex = rand() % MAX_Pesos;
    int upOrDown = rand() % 2;

    if (upOrDown == 0) {
        individuo->genes[geneIndex] -= learningRate;
    } else {
        individuo->genes[geneIndex] += learningRate;
    }

    printf("Mutação promovida no gene %d do indivíduo %d.\n", geneIndex, individuo->numero);
}

/**
 * @brief Remove os indivíduos excedentes da população após o processo de ordenação.
 * 
 * A função remove os indivíduos menos aptos da lista, mantendo o número total de indivíduos
 * igual ao tamanho inicial da população.
 * 
 * @param L Ponteiro para a estrutura TLista contendo a população.
 */
void poda(TLista *L) {
    if (L->totalIndividuos > L->Qtd_Populacao) {
        int excedente = L->totalIndividuos - L->Qtd_Populacao;
        TIndividuo *atual = L->populacao;
        TIndividuo *anterior = NULL;

        while (atual != NULL && excedente > 0) {
            anterior = atual;
            atual = atual->prox;
            excedente--;
        }

        if (anterior != NULL) {
            L->fimLista = anterior;
            anterior->prox = NULL;
            L->totalIndividuos = L->Qtd_Populacao;
        } else {
            L->populacao = NULL;
            L->fimLista = NULL;
            L->totalIndividuos = 0;
        }

        printf("Podados %d indivíduos.\n", excedente);
    }
}

/**
 * @brief Imprime os indivíduos e seus respectivos números de erros.
 * 
 * A função percorre a lista de indivíduos e imprime os dados de cada um, incluindo
 * seu número de identificação e quantidade de erros.
 * 
 * @param L Ponteiro para a estrutura TLista contendo os indivíduos.
 */
void printIndividuos(TLista *L) {
    TIndividuo *atual = L->populacao;
    printf("\nLista de indivíduos:\n");
    while (atual != NULL) {
        printf("Indivíduo %d: Erros = %d\n", atual->numero, atual->erros);
        atual = atual->prox;
    }
}