/* EL2208 Praktikum Pemecahan Masalah dengan C 2023/2024
* Modul            : Tubes - Travelling Salesmen Problem 
* Nama (NIM)       : M.Faddel (18322003)
* Asisten (NIM)    : Emmanuella Pramudita Rumanti (13220031)
* Nama File        : main.c
* Deskripsi        : penggunaan algoritma PSO
*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <string.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAKS_KOTA 100
#define MAKS_PARTIKEL 100
#define MAKS_ITERASI 1000
#define W 0.5
#define C1 1.0
#define C2 2.0

typedef struct {
    char nama[50];
    double lintang;
    double bujur;
} Kota;

typedef struct {
    int posisi[MAKS_KOTA];
    double biaya;
    int pbest_posisi[MAKS_KOTA];
    double pbest_biaya;
    double kecepatan[MAKS_KOTA];
} Partikel;

Kota kota[MAKS_KOTA];
int jumlah_kota;
Partikel partikel[MAKS_PARTIKEL];
Partikel gbest;
int jumlah_partikel;
int indeks_kota_awal;
double jarak[MAKS_KOTA][MAKS_KOTA];

double haversine(double lintang1, double bujur1, double lintang2, double bujur2) {
    double dLat = (lintang2 - lintang1) * M_PI / 180.0;
    double dLon = (bujur2 - bujur1) * M_PI / 180.0;
    lintang1 = lintang1 * M_PI / 180.0;
    lintang2 = lintang2 * M_PI / 180.0;

    double a = sin(dLat / 2) * sin(dLat / 2) + sin(dLon / 2) * sin(dLon / 2) * cos(lintang1) * cos(lintang2);
    double c = 2 * atan2(sqrt(a), sqrt(1 - a));
    return 6371 * c; // Radius bumi dalam kilometer
}

void baca_csv(const char* nama_file) {
    FILE* file = fopen(nama_file, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    char line[100];
    jumlah_kota = 0;
    while (fgets(line, sizeof(line), file)) jumlah_kota++;
    rewind(file);

    for (int i = 0; i < jumlah_kota; i++) {
        fscanf(file, "%[^,],%lf,%lf\n", kota[i].nama, &kota[i].lintang, &kota[i].bujur);
    }
    fclose(file);
}

void hitung_jarak() {
    for (int i = 0; i < jumlah_kota; i++) {
        for (int j = 0; j < jumlah_kota; j++) {
            if (i != j) {
                jarak[i][j] = haversine(kota[i].lintang, kota[i].bujur, kota[j].lintang, kota[j].bujur);
            } else {
                jarak[i][j] = 0.0;
            }
        }
    }
}

double hitung_biaya(int* rute) {
    double total_jarak = 0;
    for (int i = 0; i < jumlah_kota - 1; i++) {
        total_jarak += jarak[rute[i]][rute[i + 1]];
    }
    total_jarak += jarak[rute[jumlah_kota - 1]][rute[0]];
    return total_jarak;
}

void inisialisasi_partikel() {
    #pragma omp parallel for
    for (int p = 0; p < jumlah_partikel; p++) {
        // Inisialisasi posisi dengan kota awal di posisi pertama
        partikel[p].posisi[0] = indeks_kota_awal;
        for (int i = 1, j = 0; i < jumlah_kota; i++, j++) {
            if (j == indeks_kota_awal) j++; // Lewati indeks kota awal
            partikel[p].posisi[i] = j;
        }
        // Acak urutan kecuali kota awal
        for (int i = 1; i < jumlah_kota; i++) {
            int r = 1 + rand() % (jumlah_kota - 1);
            int temp = partikel[p].posisi[i];
            partikel[p].posisi[i] = partikel[p].posisi[r];
            partikel[p].posisi[r] = temp;
        }
        partikel[p].biaya = hitung_biaya(partikel[p].posisi);
        partikel[p].pbest_biaya = partikel[p].biaya;
        memcpy(partikel[p].pbest_posisi, partikel[p].posisi, sizeof(partikel[p].posisi));
    }
    gbest = partikel[0];
    for (int p = 1; p < jumlah_partikel; p++) {
        if (partikel[p].biaya < gbest.biaya) {
            gbest = partikel[p];
        }
    }
}

void perbarui_partikel() {
    #pragma omp parallel for
    for (int p = 0; p < jumlah_partikel; p++) {
        for (int i = 1; i < jumlah_kota; i++) { // Lewati kota pertama (kota awal)
            double r1 = ((double)rand()) / RAND_MAX;
            double r2 = ((double)rand()) / RAND_MAX;
            partikel[p].kecepatan[i] = W * partikel[p].kecepatan[i] + C1 * r1 * (partikel[p].pbest_posisi[i] - partikel[p].posisi[i]) + C2 * r2 * (gbest.posisi[i] - partikel[p].posisi[i]);
        }

        // Perbarui posisi berdasarkan kecepatan 
        for (int i = 1; i < jumlah_kota; i++) { // Lewati kota pertama (kota awal)
            if (rand() / (double)RAND_MAX < fabs(partikel[p].kecepatan[i])) {
                int swap_idx = 1 + rand() % (jumlah_kota - 1);
                int temp = partikel[p].posisi[i];
                partikel[p].posisi[i] = partikel[p].posisi[swap_idx];
                partikel[p].posisi[swap_idx] = temp;
            }
        }

        partikel[p].biaya = hitung_biaya(partikel[p].posisi);
        if (partikel[p].biaya < partikel[p].pbest_biaya) {
            partikel[p].pbest_biaya = partikel[p].biaya;
            memcpy(partikel[p].pbest_posisi, partikel[p].posisi, sizeof(partikel[p].posisi));
        }

        #pragma omp critical
        {
            if (partikel[p].biaya < gbest.biaya) {
                gbest = partikel[p];
            }
        }
    }
}

void selesaikan_tsp_pso(const char* kota_awal) {
    srand(time(NULL));
    jumlah_partikel = 30;

    // Temukan indeks kota awal
    for (int i = 0; i < jumlah_kota; i++) {
        if (strcmp(kota[i].nama, kota_awal) == 0) {
            indeks_kota_awal = i;
            break;
        }
    }

    hitung_jarak();
    inisialisasi_partikel();
    for (int iterasi = 0; iterasi < MAKS_ITERASI; iterasi++) {
        perbarui_partikel();
    }

    printf("Rute terbaik yang ditemukan:\n");
    for (int i = 0; i < jumlah_kota; i++) {
        printf("%s -> ", kota[gbest.posisi[i]].nama);
    }
    printf("%s\n", kota[gbest.posisi[0]].nama);
    printf("Best route distance: %.5f km\n", gbest.biaya);
}

int main() {
    char nama_file[100];
    char kota_awal[50];

    printf("Enter list of cities file name: ");
    scanf("%s", nama_file);
    printf("Enter starting point: ");
    scanf("%s", kota_awal);

    baca_csv(nama_file);

    clock_t waktu_mulai = clock();
    selesaikan_tsp_pso(kota_awal);
    clock_t waktu_selesai = clock();

    double waktu_berjalan = (double)(waktu_selesai - waktu_mulai) / CLOCKS_PER_SEC;
    printf("Time elapsed: %.5f s\n", waktu_berjalan);

    return 0;
}
    
