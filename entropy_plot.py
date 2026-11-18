import itertools
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns


def plot_combined_entropy(
    df: pd.DataFrame,
    tokens: list = None,
    ratios: list[tuple[str, str]] = None,
    gaps: list[tuple[str, str]] = None,
    normalize_gen0: bool = False,
    models: list = None,
    temperatures: list = None,
    hue_col: str = 'temperature',
    style_col: str = 'token',
    aggregate_runs: bool = True,
    clean_token_names: bool = True,
    figsize: tuple = (12, 7),
    save_path: str = None,
    dpi: int = 300,
    show: bool = True,
):
  data = df.copy()

  # 1. Filtri su modelli e temperature
  if models is not None:
    data = data[data['model'].isin(models)]
  if temperatures is not None:
    mask = np.zeros(len(data), dtype=bool)
    for t in temperatures:
      mask |= np.isclose(data['temperature'].astype(float), float(t))
    data = data[mask]

  # Normalizzazione rispetto a Gen 0 (prima di calcolare eventuali ratio/gap sui singoli token)
  if normalize_gen0:
    group_cols = [
        c for c in ['model', 'temperature', 'token', 'run'] if c in data.columns
    ]
    # Versione vettorizzata veloce:
    data = data.merge(
        data[data['gen'] == 0][group_cols + ['average_entropy']].rename(
            columns={'average_entropy': 'gen0_val'}
        ),
        on=group_cols,
        how='left',
    )
    data['average_entropy'] = np.where(
        data['gen0_val'] != 0, data['average_entropy'] / data['gen0_val'], np.nan
    )
    data = data.drop(columns=['gen0_val'])

  frames_to_plot = []

  def format_label(val: str) -> str:
    s = str(val)
    if clean_token_names and s.startswith('__') and s.endswith('__'):
      return s.strip('_')
    elif s.startswith('_'):
      return ' ' + s
    return s

  # 2. Estrazione dei token richiesti
  if tokens:
    df_tokens = data[data['token'].isin(tokens)].copy()
    df_tokens['token'] = df_tokens['token'].apply(format_label)
    frames_to_plot.append(df_tokens)

  # 3. Calcolo di Ratios (A / B) e Gaps (A - B)
  pairs_to_compute = [('Ratio', p) for p in (ratios or [])] + [
      ('Gap', p) for p in (gaps or [])
  ]
  if pairs_to_compute:
    index_cols = [
        c for c in ['model', 'temperature', 'gen', 'run'] if c in data.columns
    ]
    needed_tokens = {t for _, pair in pairs_to_compute for t in pair}
    pivot_df = (
        data[data['token'].isin(needed_tokens)]
        .pivot_table(index=index_cols, columns='token', values='average_entropy')
        .reset_index()
    )

    for mode, (a, b) in pairs_to_compute:
      if a in pivot_df.columns and b in pivot_df.columns:
        sub_df = pivot_df[index_cols].copy()
        op_sym = '/' if mode == 'Ratio' else '-'
        sub_df['token'] = f'{mode}: {format_label(a)} {op_sym} {format_label(b)}'
        if mode == 'Ratio':
          sub_df['average_entropy'] = np.where(
              pivot_df[b] != 0, pivot_df[a] / pivot_df[b], np.nan
          )
        else:
          sub_df['average_entropy'] = pivot_df[a] - pivot_df[b]
        frames_to_plot.append(sub_df)
      else:
        print(f"Attenzione: '{a}' o '{b}' non trovati nel dataset filtrato.")

  if not frames_to_plot:
    print('Nessun dato da plottare. Specifica `tokens`, `ratios` o `gaps`.')
    return

  plot_df = pd.concat(frames_to_plot, ignore_index=True)

  if hue_col == 'temperature' or style_col == 'temperature':
    plot_df = plot_df.sort_values(by=['temperature', 'token', 'gen'])

  plot_df[hue_col] = plot_df[hue_col].apply(format_label)
  plot_df[style_col] = plot_df[style_col].apply(format_label)

  n_styles = plot_df[style_col].nunique()
  marker_cycle = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>']
  dash_cycle = [(1, 0), (4, 1.5), (1, 1), (3, 1, 1, 1), (5, 2, 1, 2), (2, 2)]
  markers_list = [
      m for m, _ in zip(itertools.cycle(marker_cycle), range(n_styles))
  ]
  dashes_list = [
      d for d, _ in zip(itertools.cycle(dash_cycle), range(n_styles))
  ]

  # 4. Creazione del grafico
  fig, ax = plt.subplots(figsize=figsize)

  sns.lineplot(
      data=plot_df,
      x='gen',
      y='average_entropy',
      hue=hue_col,
      style=style_col,
      markers=markers_list,
      dashes=dashes_list,
      units=None if aggregate_runs else 'run',
      estimator='mean' if aggregate_runs else None,
      errorbar='sd' if aggregate_runs else None,
      palette='viridis',
      legend='full',
      ax=ax,
  )

  ax.tick_params(axis='both', which='major', labelsize=14)
  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
  ax.minorticks_on()
  ax.grid(True, which='major', linestyle='--', alpha=0.5)
  ax.grid(True, which='minor', linestyle=':', alpha=0.25)

  # Linee di riferimento e label asse Y automatiche
  if ratios and not tokens and not gaps:
    ax.axhline(1.0, color='red', linestyle=':', alpha=0.6)
    ax.set_ylabel('Entropy Ratio', fontsize=16)
  elif gaps and not tokens and not ratios:
    ax.axhline(0.0, color='red', linestyle=':', alpha=0.6)
    ax.set_ylabel('Entropy Gap (bits)', fontsize=16)
  elif normalize_gen0:
    ax.axhline(1.0, color='red', linestyle=':', alpha=0.6)
    ax.set_ylabel('Normalized Entropy (rel. to Gen 0)', fontsize=16)
  else:
    ax.set_ylabel('Average Entropy', fontsize=16)

  ax.set_xlabel('Generation', fontsize=16)

  handles, labels = ax.get_legend_handles_labels()
  if style_col in labels and plot_df[style_col].nunique() <= 1:
    idx = labels.index(style_col)
    handles = handles[:idx]
    labels = labels[:idx]

  if handles:
    ax.legend(
        handles=handles,
        labels=labels,
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        frameon=True,
        fontsize=15,
    )

  plt.tight_layout()

  if save_path:
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f'Grafico salvato in: {save_path}')

  if show:
    plt.show()
  else:
    plt.close(fig)

  return plot_df





import pandas as pd

MACRO_MICRO = 'MICRO'

if MACRO_MICRO == 'MACRO':
    df = pd.read_parquet("hf://datasets/stefanocarrera/autophagy_code_D_results_entropy_per_next_token_macro_avg/data/train-00000-of-00001.parquet")
else:
    df = pd.read_parquet("hf://datasets/stefanocarrera/autophagy_code_D_results_entropy_per_next_token/data/train-00000-of-00001.parquet")



# plot_combined_entropy(
#     df,
#     gaps=[('__ALL_OPERATORS__', '__NON_OPERATORS__')],
#     models=['Qwen3-8B'],
#     temperatures=[0.2, 0.5, 0.75, 1.0, 1.25, 1.5],
#     normalize_gen0=True,
#     save_path=f'plots/gap_op_nonop_{MACRO_MICRO}.png',
#     figsize=(11,7)
# )

plot_combined_entropy(
      df,
      tokens=['__ALL_OPERATORS__', '__NON_OPERATORS__'],
      normalize_gen0=True,
      models=['Qwen3-8B'],
      temperatures=[0.2, 1.0, 1.5],
      hue_col='temperature',
      style_col='token',
      save_path=f'plots/norm_gen0_{MACRO_MICRO}.png',
      figsize=(11,7)
  )

# plot_combined_entropy(
#     df,
#     ratios=[
#         ('__ALL_OPERATORS__', '__NON_OPERATORS__'),
#     ],
#     normalize_gen0=True,
#     models=['Qwen3-8B'],     # Opzionale: filtra un modello specifico
#     temperatures=[0.2, 0.5, 0.75, 1, 1.25, 1.5],
#     hue_col='temperature',   # Colore = Temperatura
#     style_col='token',       # Tratteggio/Marker = quale rapporto stiamo guardando
#     save_path=f"plots/ratios_op_nonop_{MACRO_MICRO}.png",
#     figsize=(11,7)
# )

# plot_combined_entropy(
#     df,
#     ratios=[
#             ('__OP_CONTROL_DECL__', '__NON_OPERATORS__'),
#             ('__OP_WORD__', '__NON_OPERATORS__'),
#             ('__OP_SYMBOLIC__', '__NON_OPERATORS__'),
#             ('__OP_PUNCTUATION__', '__NON_OPERATORS__'),
#         ],
#     models=['Qwen3-8B'],     # Opzionale: filtra un modello specifico
#     temperatures=[0.2],
#     normalize_gen0=True,
#     hue_col='token',   # Colore = Temperatura
#     style_col='token',       # Tratteggio/Marker = quale rapporto stiamo guardando
#     save_path=f"plots/ratios_optypes_nonop_{MACRO_MICRO}.png",
#     figsize=(14,7)
# )

# plot_combined_entropy(
#     df,
#     ratios=[
#             ('Ġreturn', '__NON_OPERATORS__'),
#         ],
#     models=['Qwen3-8B'],     # Opzionale: filtra un modello specifico
#     temperatures=[0.2,0.5,0.75,1, 1.25, 1.5],
#     hue_col='temperature',   # Colore = Temperatura
#     style_col='token',       # Tratteggio/Marker = quale rapporto stiamo guardando
#     save_path=f"plots/_non_op_{MACRO_MICRO}.png",
#     figsize=(12,7)
# )

# plot_combined_entropy(
#     df,
#     tokens = ['__ALL_TOKENS__'],
#     models=['Qwen3-8B'],     # Opzionale: filtra un modello specifico
#     temperatures=[0.2,0.5,0.75,1],
#     hue_col='temperature',   # Colore = Temperatura
#     style_col='token',       # Tratteggio/Marker = quale rapporto stiamo guardando
#     save_path=f"plots/all_{MACRO_MICRO}.png",
#     figsize=(18,7)
# )

# plot_combined_entropy(
#     df,
#     tokens = ['__NON_OPERATORS__'],
#     models=['Qwen3-8B'],     # Opzionale: filtra un modello specifico
#     temperatures=[0.2,0.5,0.75,1],
#     hue_col='temperature',   # Colore = Temperatura
#     style_col='token',       # Tratteggio/Marker = quale rapporto stiamo guardando
#     save_path=f"plots/non_op_{MACRO_MICRO}.png",
#     figsize=(18,7)
# )
