// QR-разложение матрицы методом Хаусхолдера с распараллеливанием на OpenMP.
//
// Программа раскладывает матрицы разного размера в 1, 2, 4, 8 и 16 потоков,
// замеряет время, считает ускорение и сравнивает его с теоретическим
// пределом по закону Амдала. Результаты пишутся в CSV, а графики строит
// приложенный скрипт на Python (он генерируется этой же программой).
//
// Сборка: Visual Studio, конфигурация Release x64 (OpenMP включён в свойствах
// проекта). Для построения графиков нужен Python с пакетами matplotlib,
// pandas и numpy: pip install matplotlib pandas numpy

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <string>
#include <random>
#include <algorithm>
#include <omp.h>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace std;
using Matrix = vector<vector<double>>;
using Clock = chrono::high_resolution_clock;

// Одна строка таблицы результатов: размер матрицы, число потоков,
// время и посчитанные по нему метрики.
struct BenchResult {
    int    n;
    int    threads;
    double time_ms;
    double speedup;
    double efficiency;
    double amdahl_speedup;
};

// Случайная матрица n*n. Seed фиксированный, чтобы при каждом запуске
// получались одни и те же числа и замеры можно было сравнивать.
Matrix make_random(int n, unsigned seed = 42) {
    mt19937_64 rng(seed);
    uniform_real_distribution<double> dist(-10.0, 10.0);
    Matrix A(n, vector<double>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i][j] = dist(rng);
    return A;
}

// Последовательное QR-разложение в один поток. Это эталон, относительно
// которого считаем ускорение параллельной версии.
void qr_sequential(Matrix A, Matrix& Q, Matrix& R) {
    int n = (int)A.size();
    Q.assign(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) Q[i][i] = 1.0;
    R = A;
    // Идём по столбцам и каждым отражением Хаусхолдера обнуляем всё,
    // что ниже диагонали в текущем столбце.
    for (int k = 0; k < n - 1; k++) {
        // Строим вектор отражения v для столбца k.
        vector<double> v(n - k);
        double sigma = 0.0;
        for (int i = k; i < n; i++) { v[i - k] = R[i][k]; sigma += v[i - k] * v[i - k]; }
        double nrm = sqrt(sigma);
        v[0] += (v[0] >= 0.0 ? nrm : -nrm);
        double nv2 = 0.0;
        for (double vi : v) nv2 += vi * vi;
        if (nv2 < 1e-14) continue;            // столбец уже почти нулевой, пропускаем
        double inv = 1.0 / nv2;
        // Применяем отражение к оставшимся столбцам R.
        for (int j = k; j < n; j++) {
            double d = 0.0;
            for (int i = k; i < n; i++) d += v[i - k] * R[i][j];
            d *= 2.0 * inv;
            for (int i = k; i < n; i++) R[i][j] -= d * v[i - k];
        }
        // Накапливаем то же отражение в матрицу Q.
        for (int i = 0; i < n; i++) {
            double d = 0.0;
            for (int j = k; j < n; j++) d += Q[i][j] * v[j - k];
            d *= 2.0 * inv;
            for (int j = k; j < n; j++) Q[i][j] -= d * v[j - k];
        }
    }
}

// То же разложение, но два самых тяжёлых внутренних цикла распараллелены
// через OpenMP. Внешний цикл по столбцам остаётся последовательным -
// каждый его шаг зависит от результата предыдущего, и это та самая
// непараллелизуемая часть из закона Амдала.
void qr_parallel(Matrix A, Matrix& Q, Matrix& R, int num_threads) {
    int n = (int)A.size();
    Q.assign(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) Q[i][i] = 1.0;
    R = A;
    omp_set_num_threads(num_threads);
    for (int k = 0; k < n - 1; k++) {
        vector<double> v(n - k);
        double sigma = 0.0;
        for (int i = k; i < n; i++) { v[i - k] = R[i][k]; sigma += v[i - k] * v[i - k]; }
        double nrm = sqrt(sigma);
        v[0] += (v[0] >= 0.0 ? nrm : -nrm);
        double nv2 = 0.0;
        for (double vi : v) nv2 += vi * vi;
        if (nv2 < 1e-14) continue;
        double inv = 1.0 / nv2;
        // Столбцы R обрабатываются независимо друг от друга, поэтому
        // итерации цикла можно спокойно раздать по потокам.
#pragma omp parallel for schedule(static)
        for (int j = k; j < n; j++) {
            double d = 0.0;
            for (int i = k; i < n; i++) d += v[i - k] * R[i][j];
            d *= 2.0 * inv;
            for (int i = k; i < n; i++) R[i][j] -= d * v[i - k];
        }
        // Строки Q тоже независимы - параллелим так же.
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            double d = 0.0;
            for (int j = k; j < n; j++) d += Q[i][j] * v[j - k];
            d *= 2.0 * inv;
            for (int j = k; j < n; j++) Q[i][j] -= d * v[j - k];
        }
    }
}

// Замеряем время разложения. Усредняем по нескольким прогонам, чтобы
// сгладить случайный разброс (фоновые процессы, турбобуст и т.п.).
double bench(int n, int t, int reps = 3) {
    auto A = make_random(n);
    double total = 0.0;
    Matrix Q, R;
    for (int r = 0; r < reps; r++) {
        auto t0 = Clock::now();
        if (t == 1) qr_sequential(A, Q, R);
        else        qr_parallel(A, Q, R, t);
        auto t1 = Clock::now();
        total += chrono::duration<double, milli>(t1 - t0).count();
    }
    return total / reps;
}

// Простой текстовый индикатор ускорения для консоли, вида [####......].
string speedup_bar(double s, double ideal) {
    int filled = max(0, min(20, (int)(s / ideal * 20)));
    return "[" + string(filled, '#') + string(20 - filled, '.') + "]";
}

// Записывает рядом с программой скрипт plot_results.py, который потом
// читает CSV и рисует четыре графика. Сам скрипт лежит ниже как обычный
// текст и подставляет в него посчитанные p и число потоков.
void write_plot_script(const string& script_path, double p_parallel, int hw_threads) {
    ofstream f(script_path);
    f << R"PY(# -*- coding: utf-8 -*-
# Скрипт строит графики по benchmark_results.csv.
# Запуск вручную:  python plot_results.py
# Нужны пакеты:     pip install matplotlib pandas numpy
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os

CSV  = os.path.join(os.path.dirname(__file__), 'benchmark_results.csv')
OUT  = os.path.dirname(__file__)
P    = )PY" << p_parallel << R"PY(
HW   = )PY" << hw_threads << R"PY(

df      = pd.read_csv(CSV)
sizes   = sorted(df['matrix_size'].unique())
threads = sorted(df['threads'].unique())
c5      = ['#1F3864','#2F5496','#4472C4','#70AD47','#ED7D31','#7030A0']
mk      = ['o','s','^','D','v','P']
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})

# 1. Закон Амдала
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Закон Амдала: теория и реальное ускорение', fontsize=15, fontweight='bold')

p_vals  = [0.50, 0.70, P, 0.90, 0.95]
p_lbls  = ['p=0.50', 'p=0.70', f'p={P} (наш QR)', 'p=0.90', 'p=0.95']
pal     = plt.cm.plasma(np.linspace(0.1, 0.9, len(p_vals)))
n_disc  = np.array([1,2,4,8,16,32,64,128])
n_cont  = np.linspace(1, 128, 600)

# Слева - дискретные точки, поверх них реальные замеры
for pv, col, lbl in zip(p_vals, pal, p_lbls):
    sp = 1/((1-pv) + pv/n_disc)
    lw = 3 if pv==P else 1.8
    ls = '-' if pv==P else '--'
    ax1.plot(n_disc, sp, ls, color=col, lw=lw, marker='o',
             ms=6 if pv==P else 4, zorder=5 if pv==P else 3, label=lbl)
ax1.plot(n_disc, np.minimum(n_disc, 8.0), 'k:', lw=1.5, alpha=0.5, label='Идеал S=N')

# Накладываем реальные данные (две самые крупные матрицы)
real_cols = ['#C00000', '#FF6B35']
for ci, n in enumerate(sorted(sizes)[-2:]):
    sub = df[df['matrix_size']==n]
    sym = 'v' if ci==1 else '^'
    ax1.plot(sub['threads'], sub['speedup'], sym,
             color=real_cols[ci], ms=11, zorder=10,
             label=f'Реальные {n}x{n}',
             markeredgecolor='white', markeredgewidth=1.2)

smax = 1/(1-P)
ax1.axhline(y=smax, color='#C00000', ls=':', lw=2, alpha=0.6)
ax1.text(1.2, smax+0.25, f'S_max = {smax:.2f}x (p={P})',
         color='#C00000', fontsize=9, fontweight='bold')
ax1.set_ylim(0, 8)
ax1.set_xscale('log', base=2)
ax1.set_xticks([1,2,4,8,16])
ax1.set_xticklabels([1,2,4,8,16])
ax1.set_xlabel('Число потоков N', fontsize=12)
ax1.set_ylabel('Ускорение S(N)', fontsize=12)
ax1.set_title('Кривые Амдала и реальные замеры', fontsize=12, fontweight='bold')
ax1.legend(fontsize=8, loc='upper left', framealpha=0.9)
ax1.grid(True, alpha=0.3)

# Справа - непрерывные кривые с подписями асимптот
for pv, col, lbl in zip(p_vals, pal, p_lbls):
    sp = 1/((1-pv) + pv/n_cont)
    lw = 3 if pv==P else 1.5
    ls = '-' if pv==P else '--'
    ax2.plot(n_cont, sp, ls, color=col, lw=lw, label=lbl)
    sm = round(1/(1-pv), 1)
    ax2.axhline(y=sm, color=col, ls=':', lw=0.9, alpha=0.5)
    ax2.text(131, sm+0.15, f'{sm}x', color=col, fontsize=8,
             va='bottom', fontweight='bold')
ax2.plot(n_cont, np.minimum(n_cont, 20), 'k:', lw=1.5, alpha=0.4, label='Идеал S=N')
ax2.set_xlim(0, 148); ax2.set_ylim(0, 22)
ax2.annotate(f'Предел нашего QR\nS_max = {smax:.2f}x',
             xy=(128, smax), xytext=(85, smax+3.5),
             arrowprops=dict(arrowstyle='->', color='#C00000', lw=1.5),
             fontsize=9, color='#C00000', fontweight='bold')
ax2.set_xlabel('Число потоков N', fontsize=12)
ax2.set_ylabel('Ускорение S(N)', fontsize=12)
ax2.set_title('Непрерывные кривые и асимптоты S_max', fontsize=12, fontweight='bold')
ax2.legend(fontsize=8, loc='upper left', framealpha=0.9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(OUT, 'chart_amdahl.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print('  [ok] chart_amdahl.png')

# 2. Реальное ускорение против теории по каждому размеру матрицы
fig, axes = plt.subplots(1, len(sizes), figsize=(4*len(sizes), 4.5))
fig.suptitle('Реальное ускорение и теория Амдала по размерам матриц',
             fontsize=13, fontweight='bold')
for idx, (ax, n) in enumerate(zip(axes, sizes)):
    sub = df[df['matrix_size']==n]
    ax.plot(sub['threads'], sub['speedup'],
            'o-', color='#C00000', lw=2.5, ms=8, label='Реальное S', zorder=5)
    ax.plot(sub['threads'], sub['amdahl_speedup'],
            's--', color='#4472C4', lw=2, ms=7, label='Амдал S', zorder=4)
    ax.plot(sub['threads'], sub['threads'].astype(float),
            ':', color='#70AD47', lw=1.5, label='Идеал S=N')
    ax.fill_between(sub['threads'], sub['speedup'], sub['amdahl_speedup'],
                    alpha=0.12, color='orange', label='Потери (overhead)')
    ax.set_title(f'{n}x{n}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Потоки', fontsize=9)
    if idx == 0: ax.set_ylabel('Ускорение S', fontsize=10)
    ax.set_xticks(threads)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
out = os.path.join(OUT, 'chart_speedup.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print('  [ok] chart_speedup.png')

# 3. Тепловая карта эффективности
th4 = [t for t in threads if t > 1]
fig, ax = plt.subplots(figsize=(2.4*len(th4), 5))
mat = np.array([[df[(df['matrix_size']==n)&(df['threads']==t)]['efficiency_pct'].values[0]
                 for t in th4] for n in sizes])
im = ax.imshow(mat, cmap='RdYlGn', aspect='auto', vmin=0, vmax=110)
ax.set_xticks(range(len(th4)))
ax.set_xticklabels([f'T={t}' for t in th4], fontsize=12)
ax.set_yticks(range(len(sizes)))
ax.set_yticklabels([f'{n}x{n}' for n in sizes], fontsize=12)
ax.set_title('Параллельная эффективность E (%)\n>100% = суперлинейно (эффект кэша)',
             fontsize=12, fontweight='bold')
for i in range(len(sizes)):
    for j in range(len(th4)):
        v = mat[i, j]
        ax.text(j, i, f'{v:.1f}%', ha='center', va='center',
                fontsize=11, fontweight='bold',
                color='black' if v > 45 else 'white')
plt.colorbar(im, label='Эффективность (%)')
plt.tight_layout()
out = os.path.join(OUT, 'chart_efficiency.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print('  [ok] chart_efficiency.png')

# 4. Время выполнения
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('QR-разложение: время выполнения от числа потоков',
             fontsize=13, fontweight='bold')
for i, n in enumerate(sizes):
    sub = df[df['matrix_size']==n]
    kw = dict(marker=mk[i%len(mk)], color=c5[i%len(c5)], lw=2, ms=7, label=f'{n}x{n}')
    ax1.plot(sub['threads'], sub['time_ms'], **kw)
    ax2.plot(sub['threads'], sub['time_ms'], **kw)
for ax, ttl, logy in [(ax1, 'Лог. шкала', True), (ax2, 'Линейная шкала', False)]:
    ax.set_xlabel('Потоки OpenMP', fontsize=11)
    ax.set_ylabel('Время (мс)', fontsize=11)
    ax.set_title(ttl, fontsize=11)
    ax.set_xticks(threads)
    ax.legend(title='Размер матрицы', fontsize=9)
    ax.grid(True, alpha=0.4)
    if logy: ax.set_yscale('log')
plt.tight_layout()
out = os.path.join(OUT, 'chart_time.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print('  [ok] chart_time.png')

print('\nВсе графики сохранены в:', OUT)
)PY";
    f.close();
}

// Пытается запустить Python тем способом, который найдётся в системе.
bool run_python(const string& script) {
    vector<string> cmds = { "py", "python", "python3" };
    for (auto& py : cmds) {
        string cmd = py + " \"" + script + "\" 2>nul";
#ifndef _WIN32
        cmd = py + " \"" + script + "\" 2>/dev/null";
#endif
        if (system(cmd.c_str()) == 0) return true;
    }
    return false;
}

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);          // чтобы русский текст в консоли не превращался в кракозябры
#endif

    const double p = 0.82;                // оценка доли параллельного кода для кривой Амдала
    vector<int> sizes = { 64, 128, 256, 512, 1024, 2048 };
    vector<int> threads = { 1, 2, 4, 8, 16 };
    int hw_threads = omp_get_max_threads();

    cout << "\n";
    cout << "==== QR-разложение (метод Хаусхолдера) на OpenMP ====\n";
    cout << "Закон Амдала: S(n) = 1 / ((1-p) + p/n),  p = " << p << "\n";
    cout << "Аппаратных потоков на этой машине: " << hw_threads << "\n";
    cout << "Каждый замер - среднее по 3 прогонам\n\n";
    cout << "Обозначения:\n";
    cout << "  S     - реальное ускорение, T(1 поток) / T(N потоков)\n";
    cout << "  E     - эффективность, S / N * 100%\n";
    cout << "  S_amd - теоретический предел по Амдалу\n\n";

    vector<BenchResult> all;

    for (int n : sizes) {
        cout << "--- Матрица " << n << "x" << n << " ---\n";
        double t1 = bench(n, 1);
        cout << "  T= 1  -> " << fixed << setprecision(2) << setw(10) << t1
            << " мс  (база, один поток)\n";
        all.push_back({ n, 1, t1, 1.0, 100.0, 1.0 });

        for (int t : threads) {
            if (t == 1) continue;
            double tm = bench(n, t);
            double sp = t1 / tm;
            double eff = sp / t * 100.0;
            double am = 1.0 / ((1.0 - p) + p / (double)t);
            string comment;
            if (sp > am * 1.05)        comment = " << суперлинейно (эффект кэша)";
            else if (sp >= am * 0.85)  comment = " << близко к теории Амдала";
            else if (sp < 1.0)         comment = " << потоков больше, чем работы";
            cout << "  T=" << setw(2) << t << "  -> "
                << setw(10) << tm << " мс"
                << "  S=" << setprecision(3) << setw(5) << sp
                << "  E=" << setprecision(1) << setw(5) << eff << "%"
                << "  S_amd=" << setprecision(3) << setw(5) << am
                << "  " << speedup_bar(sp, (double)t)
                << comment << "\n";
            all.push_back({ n, t, tm, sp, eff, am });
        }
        cout << "\n";
    }

    // Сводка: при каком числе потоков ускорение было максимальным.
    cout << "=== Лучшее число потоков по размерам матриц ===\n";
    for (int n : sizes) {
        double best_sp = 0.0; int best_t = 1;
        for (auto& r : all)
            if (r.n == n && r.speedup > best_sp) { best_sp = r.speedup; best_t = r.threads; }
        cout << "  " << setw(6) << n << "x" << n
            << ":  ускорение " << fixed << setprecision(2) << best_sp
            << "x при T=" << best_t << "\n";
    }
    double s_max = 1.0 / (1.0 - p);
    cout << "\nЗакон Амдала: при p=" << p << " предел S_max = "
        << fixed << setprecision(2) << s_max << "x\n";
    cout << "(" << (1 - p) * 100 << "% кода последовательны - это и есть жёсткий потолок)\n\n";

    // Сохраняем результаты в CSV. Имена столбцов оставлены латиницей,
    // потому что по ним обращается скрипт построения графиков.
    {
        ofstream csv("benchmark_results.csv");
        csv << "matrix_size,threads,time_ms,speedup,efficiency_pct,amdahl_speedup\n";
        csv << fixed << setprecision(4);
        for (auto& r : all)
            csv << r.n << "," << r.threads << "," << r.time_ms << ","
            << r.speedup << "," << r.efficiency << "," << r.amdahl_speedup << "\n";
        cout << "  CSV сохранён: benchmark_results.csv\n";
    }

    write_plot_script("plot_results.py", p, hw_threads);
    cout << "  скрипт построения графиков: plot_results.py\n\n";

    cout << "  Строю графики...\n";
    if (run_python("plot_results.py")) {
        cout << "\n  Графики сохранены:\n";
        cout << "    chart_amdahl.png\n";
        cout << "    chart_speedup.png\n";
        cout << "    chart_efficiency.png\n";
        cout << "    chart_time.png\n";
    }
    else {
        cout << "\n  Python не найден. Установите Python и пакеты, затем\n";
        cout << "  постройте графики вручную:\n";
        cout << "    pip install matplotlib pandas numpy\n";
        cout << "    python plot_results.py\n";
    }

    cout << "\n  Нажмите Enter для выхода...";
    cin.get();
    return 0;
}
