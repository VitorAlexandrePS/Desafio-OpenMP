// Coeficiente de variacao (dados agrupados em classes) com OpenMP
// Compilar: gcc -O2 -fopenmp cv_openmp.c -o cv -lm
// Executar: ./cv dados.txt [fator] [tabela]
//   fator  = quantas vezes replicar os dados no teste de desempenho (padrao 10000)
//   tabela = (opcional) qualquer texto: imprime a tabela de frequencias
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define MAXN 1000
#define MAXK 512

typedef struct { double mu, sigma, cv, ini; int k; long f[MAXK]; } Res;

/* Medias e desvio a partir das classes: formulas (1), (2) e (3) do enunciado */
static void finaliza_seq(Res *r, long n, double larg) {
    double s = 0.0, v = 0.0;
    for (int i = 0; i < r->k; i++) s += (r->ini + larg * i + larg / 2.0) * r->f[i];
    double mu = s / n;
    for (int i = 0; i < r->k; i++) {
        double xi = r->ini + larg * i + larg / 2.0;
        v += r->f[i] * (xi - mu) * (xi - mu);
    }
    r->mu = mu; r->sigma = sqrt(v / n); r->cv = r->sigma / mu * 100.0;
}

/* ---------- SEQUENCIAL otimizado: 2 passadas nos dados, sem divisoes ---------- */
static void calc_seq(const double *x, long n, double larg, Res *r) {
    const double inv = 1.0 / larg;
    double mn = x[0], mx = x[0];
    for (long i = 1; i < n; i++) { if (x[i] < mn) mn = x[i]; if (x[i] > mx) mx = x[i]; }
    r->ini = floor(mn * inv) * larg;
    r->k = (int)((mx - r->ini) * inv) + 1;
    for (int j = 0; j < MAXK; j++) r->f[j] = 0;
    for (long i = 0; i < n; i++) r->f[(int)((x[i] - r->ini) * inv)]++;
    finaliza_seq(r, n, larg);
}

/* ---------- PARALELO: uma unica regiao parallel ---------- */
static void calc_par(const double *x, long n, double larg, Res *r) {
    const double inv = 1.0 / larg;
    double mn = x[0], mx = x[0], ini = 0, s = 0.0, v = 0.0, mu = 0;
    int k = 0;
    for (int j = 0; j < MAXK; j++) r->f[j] = 0;

    #pragma omp parallel
    {
        /* 1) minimo e maximo */
        #pragma omp for reduction(min:mn) reduction(max:mx)
        for (long i = 0; i < n; i++) {
            if (x[i] < mn) mn = x[i];
            if (x[i] > mx) mx = x[i];
        }
        #pragma omp single
        { ini = floor(mn * inv) * larg; k = (int)((mx - ini) * inv) + 1; }

        /* 2) frequencias: histograma local por thread, depois junta */
        long loc[MAXK] = {0};
        #pragma omp for
        for (long i = 0; i < n; i++) loc[(int)((x[i] - ini) * inv)]++;
        #pragma omp critical
        for (int j = 0; j < k; j++) r->f[j] += loc[j];
        #pragma omp barrier

        /* 3) soma de xi*fi (media) */
        #pragma omp for reduction(+:s)
        for (int i = 0; i < k; i++) s += (ini + larg * i + larg / 2.0) * r->f[i];
        #pragma omp single
        mu = s / n;

        /* 4) soma de fi*(xi-mu)^2 (desvio padrao) */
        #pragma omp for reduction(+:v)
        for (int i = 0; i < k; i++) {
            double xi = ini + larg * i + larg / 2.0;
            v += r->f[i] * (xi - mu) * (xi - mu);
        }
    }
    r->ini = ini; r->k = k;
    r->mu = s / n; r->sigma = sqrt(v / n); r->cv = r->sigma / r->mu * 100.0;
}

typedef void (*Fn)(const double*, long, double, Res*);

/* tempo medio por execucao (varias repeticoes, divide pelo total) */
static double tempo(Fn fn, const double *x, long n, double larg, int reps) {
    static Res r;
    fn(x, n, larg, &r);                       /* aquecimento */
    double t = omp_get_wtime();
    for (int i = 0; i < reps; i++) fn(x, n, larg, &r);
    return (omp_get_wtime() - t) / reps;
}

static void mostra(const char *nome, const char *un, double larg,
                   const double *x, long n, int tabela) {
    Res a, b; calc_seq(x, n, larg, &a); calc_par(x, n, larg, &b);
    printf("%s (classe %.0f %s): k=%d  media=%.3f  desvio=%.3f  CV=%.2f%%  [paralelo: CV=%.2f%%]\n",
           nome, larg, un, a.k, a.mu, a.sigma, a.cv, b.cv);
    if (tabela)
        for (int j = 0; j < a.k; j++)
            printf("   [%.0f, %.0f) : %ld\n", a.ini + larg * j, a.ini + larg * (j + 1), a.f[j]);
}

int main(int argc, char **argv) {
    const char *arq = argc > 1 ? argv[1] : "dados.txt";
    long fator = argc > 2 ? atol(argv[2]) : 10000;
    int tabela = argc > 3;
    static double h[MAXN], p[MAXN];
    int n = 0;
    FILE *fp = fopen(arq, "r");
    if (!fp) { perror("abrir arquivo"); return 1; }
    while (n < MAXN && fscanf(fp, "%lf %lf", &h[n], &p[n]) == 2) n++;
    fclose(fp);

    printf("Pessoas lidas: %d | threads: %d\n\n", n, omp_get_max_threads());
    mostra("Estatura", "cm", 8.0, h, n, tabela);
    mostra("Peso    ", "kg", 4.0, p, n, tabela);

    printf("\n--- Desempenho ---\n");
    double ts1 = tempo(calc_seq, h, n, 8.0, 100000) + tempo(calc_seq, p, n, 4.0, 100000);
    double tp1 = tempo(calc_par, h, n, 8.0, 100000) + tempo(calc_par, p, n, 4.0, 100000);
    printf("N = %d (original): seq %.2f us | par %.2f us | speedup %.2fx\n",
           n, ts1 * 1e6, tp1 * 1e6, ts1 / tp1);

    long N = (long)n * fator;
    double *H = malloc(N * sizeof(double)), *P = malloc(N * sizeof(double));
    for (long i = 0; i < N; i++) { H[i] = h[i % n]; P[i] = p[i % n]; }
    double ts = tempo(calc_seq, H, N, 8.0, 10) + tempo(calc_seq, P, N, 4.0, 10);
    double tp = tempo(calc_par, H, N, 8.0, 10) + tempo(calc_par, P, N, 4.0, 10);
    printf("N = %ld (replicado %ldx): seq %.4f s | par %.4f s | speedup %.2fx\n",
           N, fator, ts, tp, ts / tp);
    free(H); free(P);
    return 0;
}