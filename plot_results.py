# -*- coding: utf-8 -*-
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
P    = 0.82
HW   = 12

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
