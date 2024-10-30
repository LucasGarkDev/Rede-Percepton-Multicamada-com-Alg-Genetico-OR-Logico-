#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>  // Para a função exp() usada na sigmoid

#define MAX_Entradas 2
#define MAX_Pesos 6

//===| Estrutura de Dados |==========================
typedef char string[60];

// Estrutura para representar as lições de entrada
typedef struct tipoLicao {
    int p;                 // Proposição P
    int q;                 // Proposição Q
    int resultadoEsperado; // Resultado esperado da operação P "E" Q
    struct tipoLicao *prox;
} TLicao;

// Estrutura de indivíduos para o algoritmo genético
typedef struct tipoIndividuo {
    float genes[MAX_Pesos]; // Pesos (genes) que serão evoluídos
    int erros;
    int numero; // Identificador único
    struct tipoIndividuo *prox;
} TIndividuo;

// Sinapses entre neurônios
typedef struct tipoSinapse {
    int neuronio_origem;
    int neuronio_destino;
    float peso;
    struct tipoSinapse *prox;
} TSinapse;

// Estrutura de neurônios
typedef struct tipoNeuronio {
    int neuronio;
    float soma;
    float peso;
    struct tipoNeuronio *prox;
} TNeuronio;

// Estrutura para as camadas da rede
typedef struct tipoCamada {
    int numNeuronios;       // Número de neurônios na camada
    TNeuronio *neuronios;   // Lista de neurônios na camada
    struct tipoCamada *prox; // Próxima camada
} TCamada;

// Adicionar sinapses à estrutura da rede neural
typedef struct tipoRedeNeural {
    TCamada *camadas;      // Lista de camadas na rede
    int numCamadas;        // Número total de camadas
    TSinapse *pesos;       // Lista de sinapses (pesos) entre os neurônios
} TRedeNeural;

// Estrutura principal contendo a rede neural e informações gerais
typedef struct tipoLista {
    FILE *fp;
    string objetivo;
    TLicao *licoes;
    float entradas[MAX_Entradas];
    TRedeNeural *redeNeural; // A rede neural com múltiplas camadas
    TIndividuo *populacao;
    TIndividuo *fimLista;
    TIndividuo *individuoAtual;
    int totalIndividuos;
    int Qtd_Populacao;
    int Qtd_Mutacoes_por_vez;
    int Total_geracoes;
    int geracao_atual;
    int Qtd_Geracoes_para_Mutacoes;
    float sinapseThreshold;
    float learningRate;
} TLista;

TLista lista;

//====| Assinatura de Funções |=======================+
void inicializa(TLista *L);
void geraIndividuos(TLista *L);
void geraLicoes(TLista *L);
void insereLicao(TLista *L, int p, int q, int resultado);
void insereNeuronio(TCamada *camada, int neuronio);
void estabelecendoSinapse(TLista *L, int neuronioDe, int neuronioAte, int camada);
void treinamento(TLista *L);
void cruzamento(TLista *L);
void avaliacaoIndividuos(TLista *L);
void ordenamentoIndividuos(TLista *L);
void promoveMutacoes(TLista *L, float learningRate);
void poda(TLista *L);
void feedforward(TLista *L, float *entradas);
void conectaCamadas(TLista *L, int neuronioDe, int neuronioAte);
void inicializaRedeNeural(TLista *L, int numCamadasOcultas, int numNeuroniosPorCamada, int numSaidas);
void insereCamada(TLista *L, int numNeuronios);
void exibeIndividuos(TLista *L);

//====| Funções Auxiliares |==========================

void inicializa(TLista *L) {
    L->licoes = NULL;
    L->populacao = NULL;
    L->individuoAtual = NULL;
    L->totalIndividuos = 0;
    L->redeNeural = (TRedeNeural *)malloc(sizeof(TRedeNeural));
    L->redeNeural->camadas = NULL;
    L->redeNeural->numCamadas = 0;
    L->redeNeural->pesos = NULL;

    printf("\nSistema inicializado com sucesso.\n");
}


// Função sigmoid para ativação dos neurônios
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

// Inicializa a rede neural com as camadas especificadas
void inicializaRedeNeural(TLista *L, int numCamadasOcultas, int numNeuroniosPorCamada, int numSaidas) {
    // Adicionar camada de entrada
    insereCamada(L, MAX_Entradas); // Insere camada de entrada com o número de entradas

    // Adicionar camadas ocultas
    for (int i = 0; i < numCamadasOcultas; i++) {
        insereCamada(L, numNeuroniosPorCamada); // Cada camada oculta com 'numNeuroniosPorCamada' neurônios
    }

    // Adicionar camada de saída
    insereCamada(L, numSaidas); // Camada de saída com 'numSaidas' neurônios
}

float randomFloat(float min, float max) {
    float scale = rand() / (float) RAND_MAX; // Gera um número entre 0 e 1
    return min + scale * (max - min); // Retorna um número entre min e max
}


// Função para inserir uma nova camada
void insereCamada(TLista *L, int numNeuronios) {
    TCamada *novaCamada = (TCamada *)malloc(sizeof(TCamada));
    novaCamada->numNeuronios = numNeuronios;
    novaCamada->neuronios = NULL;
    novaCamada->prox = NULL;

    // Inserir os neurônios na nova camada, passando a camada correta
    for (int i = 0; i < numNeuronios; i++) {
        insereNeuronio(novaCamada, i);
    }

    // Inserir a nova camada na rede neural
    if (L->redeNeural->camadas == NULL) {
        L->redeNeural->camadas = novaCamada;
    } else {
        TCamada *atual = L->redeNeural->camadas;
        while (atual->prox != NULL) {
            atual = atual->prox;
        }
        atual->prox = novaCamada;
    }
}


// Função para inserir um neurônio em uma camada
void insereNeuronio(TCamada *camada, int neuronio) {
    TNeuronio *novoNeuronio = (TNeuronio *)malloc(sizeof(TNeuronio));
    novoNeuronio->neuronio = neuronio;
    novoNeuronio->peso = 0;
    novoNeuronio->soma = 0;
    novoNeuronio->prox = NULL;

    if (camada->neuronios == NULL) {
        camada->neuronios = novoNeuronio;
    } else {
        TNeuronio *atual = camada->neuronios;
        while (atual->prox != NULL) {
            atual = atual->prox;
        }
        atual->prox = novoNeuronio;
    }
}


// Função de feedforward para propagação de sinais entre as camadas
void feedforward(TLista *L, int *entradas) {
    TCamada *camadaAtual = L->redeNeural->camadas;

    // Definir a camada de entrada
    TNeuronio *neuronioEntrada = camadaAtual->neuronios;
    for (int i = 0; i < MAX_Entradas; i++) {
        neuronioEntrada->soma = entradas[i];
        neuronioEntrada = neuronioEntrada->prox;
    }

    // Propagar o sinal pelas camadas ocultas até a saída
    camadaAtual = camadaAtual->prox; // Primeira camada oculta
    while (camadaAtual != NULL) {
        TNeuronio *neuronioAtual = camadaAtual->neuronios;
        while (neuronioAtual != NULL) {
            float soma = 0;

            // Calcular a soma ponderada das saídas da camada anterior
            TCamada *camadaAnterior = L->redeNeural->camadas; // Primeira camada (entrada)

            // Encontrar a camada anterior à atual
            while (camadaAnterior->prox != camadaAtual) {
                camadaAnterior = camadaAnterior->prox;
            }

            TNeuronio *neuronioAnterior = camadaAnterior->neuronios;
            while (neuronioAnterior != NULL) {
                // Procurar a sinapse entre o neurônio anterior e o atual
                TSinapse *sinapseAtual = L->redeNeural->pesos;
                while (sinapseAtual != NULL) {
                    if (sinapseAtual->neuronio_origem == neuronioAnterior->neuronio &&
                        sinapseAtual->neuronio_destino == neuronioAtual->neuronio) {
                        soma += neuronioAnterior->soma * sinapseAtual->peso;
                        break; // Encontrou a sinapse correspondente
                    }
                    sinapseAtual = sinapseAtual->prox;
                }
                neuronioAnterior = neuronioAnterior->prox;
            }

            // Aplicar a função de ativação (sigmoid) à soma
            neuronioAtual->soma = sigmoid(soma);
            neuronioAtual = neuronioAtual->prox;
        }

        camadaAtual = camadaAtual->prox;
    }
}


// Função auxiliar para inserir um indivíduo no final da lista
void inserirNoFimAux(TLista *L, TIndividuo *individuo) {
    if (L->populacao == NULL) { // Se a lista estiver vazia
        L->populacao = individuo;
        L->fimLista = individuo;
        L->totalIndividuos++;
        return;
    }
    
    TIndividuo *aux = L->fimLista; // Apontar diretamente para o último da lista
    aux->prox = individuo;         // Conectar o novo indivíduo no final
    L->fimLista = individuo;       // Atualizar o ponteiro do último indivíduo
    L->totalIndividuos++;
}


// Função para conectar camadas da rede neural
void conectaCamadas(TLista *L, int neuronioDe, int neuronioAte) {
    TSinapse *novaSinapse = (TSinapse *)malloc(sizeof(TSinapse));
    novaSinapse->neuronio_origem = neuronioDe;
    novaSinapse->neuronio_destino = neuronioAte;
    novaSinapse->peso = randomFloat(-1.0, 1.0); // Peso inicial aleatório
    novaSinapse->prox = NULL;

    // Adicionar a nova sinapse na lista de sinapses da rede neural
    if (L->redeNeural->pesos == NULL) {
        L->redeNeural->pesos = novaSinapse;
    } else {
        TSinapse *atual = L->redeNeural->pesos;
        while (atual->prox != NULL) {
            atual = atual->prox;
        }
        atual->prox = novaSinapse;
    }
}


// Função para gerar os indivíduos
void geraIndividuos(TLista *L) {
    TIndividuo *novo;
    int i, x;

    srand((unsigned)time(NULL));

    for (i = 0; i < L->Qtd_Populacao; i++) {
        novo = (TIndividuo *)malloc(sizeof(TIndividuo));
        novo->prox = NULL;
        novo->numero = i + 1;
        novo->erros = -1;

        for (x = 0; x < MAX_Pesos; x++) {
            novo->genes[x] = rand() % 101;
            novo->genes[x] = novo->genes[x] / 100;
        }

        if (L->populacao == NULL) {
            L->populacao = novo;
            L->fimLista = novo;  // Inicializa o fim da lista corretamente
        } else {
            L->fimLista->prox = novo; // Usa fimLista para evitar laços desnecessários
            L->fimLista = novo;       // Atualiza o fim da lista
        }

        L->totalIndividuos++;
    }
}


// Função para gerar as lições de entrada
// Função para gerar as lições de entrada
void geraLicoes(TLista *L) {
    insereLicao(L, 0, 0, 0);
    insereLicao(L, 0, 1, 0);
    insereLicao(L, 1, 0, 0);
    insereLicao(L, 1, 1, 1);
}


// Função para inserir uma lição
void insereLicao(TLista *L, int p, int q, int resultado) {
    TLicao *novo = (TLicao *)malloc(sizeof(TLicao));

    novo->prox = NULL;
    novo->p = p;
    novo->q = q;
    novo->resultadoEsperado = resultado;

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

// Avaliação dos indivíduos após a propagação do feedforward
void avaliacaoIndividuos(TLista *L) {
    TIndividuo *atual = L->populacao;

    while (atual != NULL) {
        if (atual->erros == -1) {
            atual->erros = 0;
            TLicao *licaoAtual = L->licoes;
            while (licaoAtual != NULL) {
                int entradas[MAX_Entradas] = {licaoAtual->p, licaoAtual->q};

                // Propaga o sinal pelas camadas
                feedforward(L, entradas);

                // Calcular o erro entre a saída esperada e a saída real
                TCamada *camadaSaida = L->redeNeural->camadas;  // Última camada
                while (camadaSaida->prox != NULL) {
                    camadaSaida = camadaSaida->prox;  // Avança até a última camada
                }
                TNeuronio *neuronioSaida = camadaSaida->neuronios;

                while (neuronioSaida != NULL) {
                    if (licaoAtual->resultadoEsperado != (int)round(neuronioSaida->soma)) {
                        atual->erros++;
                    }
                    neuronioSaida = neuronioSaida->prox;
                }

                licaoAtual = licaoAtual->prox;
            }
        }

        atual = atual->prox;
    }
}

// Função para promover mutações nos indivíduos
void promoveMutacoes(TLista *L, float learningRate) {
    if (L->populacao == NULL) {
        printf("Lista de individuos vazia.\n");
        return;
    }

    // Escolher um indivíduo aleatório
    int index = rand() % L->totalIndividuos;
    TIndividuo *individuo = L->populacao;

    for (int i = 0; i < index; i++) {
        individuo = individuo->prox;
    }

    // Escolher um gene aleatório para mutação
    int geneIndex = rand() % MAX_Pesos;
    individuo->genes[geneIndex] += (rand() % 2 == 0 ? -learningRate : learningRate);
}

// Função de treinamento (loop principal)
void treinamento(TLista *L) {
    printf("\n\n\t\t=====| INICIADO TREINAMENTO |=====\n\n");
    for (int i = 0; i < L->Total_geracoes; i++) {
        cruzamento(L);

        if ((i % L->Qtd_Geracoes_para_Mutacoes) == 0) {
            promoveMutacoes(L, L->learningRate);
        }

        avaliacaoIndividuos(L);
        ordenamentoIndividuos(L);
        poda(L);
        exibeIndividuos(L);
    }

    fclose(L->fp);
}

// Função para cruzamento dos indivíduos
void cruzamento(TLista *L) {
    TIndividuo *pai1 = L->populacao;
    TIndividuo *pai2 = pai1->prox;
    int cont = L->totalIndividuos + 1;
    int individuosCruzados = 0;

    while (individuosCruzados < L->totalIndividuos / 2) {
        TIndividuo *filho1 = (TIndividuo *)malloc(sizeof(TIndividuo));
        TIndividuo *filho2 = (TIndividuo *)malloc(sizeof(TIndividuo));
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
        inserirNoFimAux(L, filho1);
        inserirNoFimAux(L, filho2);

        individuosCruzados += 1;

        pai1 = pai1->prox;
        pai2 = pai2->prox;
    }
}

// Função de poda: Remove os indivíduos menos aptos, mantendo o tamanho da população
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
            // Se anterior for NULL, significa que todos os indivíduos foram removidos
            L->populacao = NULL;
            L->fimLista = NULL;
            L->totalIndividuos = 0;
        }
    }
}


// Função para ordenar os indivíduos com base nos erros (menor número de erros primeiro)
void ordenamentoIndividuos(TLista *L) {
    if (L->populacao == NULL || L->populacao->prox == NULL) {
        // Se a lista estiver vazia ou tiver apenas um indivíduo, não há o que ordenar
        return;
    }

    int trocou;
    TIndividuo *atual;
    TIndividuo *anterior;
    TIndividuo *temp;

    do {
        trocou = 0;
        atual = L->populacao;
        anterior = NULL;

        while (atual->prox != NULL) {
            if (atual->erros > atual->prox->erros) {
                // Troca de posições entre os indivíduos
                temp = atual->prox;
                atual->prox = atual->prox->prox;
                temp->prox = atual;

                if (anterior == NULL) {
                    L->populacao = temp;
                } else {
                    anterior->prox = temp;
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

// Função para exibir a população de indivíduos
void exibeIndividuos(TLista *L) {
    TIndividuo *atual = L->populacao;
    int i = 1;

    printf("\n\t TABELA DE INDIVIDUOS:\n");
    fprintf(L->fp, "\n\t TABELA DE INDIVIDUOS:\n");
    
    while (atual != NULL) {
        printf("| \t(%d) \t| numero = %d \t| erros = %d \t| genes = ", i, atual->numero, atual->erros);
        for (int j = 0; j < MAX_Pesos; j++) {
            printf("%.2f,", atual->genes[j]);
        }
        printf("\n");

        fprintf(L->fp, "| \t(%d) \t| numero = %d \t| erros = %d \t| genes = ", i, atual->numero, atual->erros);
        for (int j = 0; j < MAX_Pesos; j++) {
            fprintf(L->fp, "%.2f,", atual->genes[j]);
        }
        fprintf(L->fp, "\n");

        i++;
        atual = atual->prox;
    }

    printf("\n\nTotal de indivíduos: %d\n", L->totalIndividuos);
    fprintf(L->fp, "\n\nTotal de indivíduos: %d\n", L->totalIndividuos);
}

int main() {
    // Inicializa a lista e configurações da rede neural
    inicializa(&lista);

    // Inicializa a rede neural com 2 camadas ocultas, cada uma com 5 neurônios, e uma camada de saída com 1 neurônio
    inicializaRedeNeural(&lista, 2, 5, 1);

    // Define os parâmetros da simulação
    lista.Qtd_Populacao = 100;           // População de 100 indivíduos
    lista.Total_geracoes = 1000;         // 1000 gerações
    lista.Qtd_Mutacoes_por_vez = 5;      // 5 mutações a cada cilo
    lista.Qtd_Geracoes_para_Mutacoes = 20; // A cada 20 gerações ocorrem mutações
    lista.learningRate = 0.1;            // Taxa de aprendizado

    // Gera indivíduos e lições de exemplo
    geraIndividuos(&lista);
    geraLicoes(&lista);

    // Treina a rede neural
    treinamento(&lista);

    return 0;
}



