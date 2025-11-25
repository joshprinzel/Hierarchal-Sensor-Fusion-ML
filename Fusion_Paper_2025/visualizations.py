
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import griddata
from scipy.stats import pearsonr, entropy
import scipy.stats as stats
import os
import logging

logger = logging.getLogger('FlightAnomaly')


def extract_dst_uncertainty(pignistics):
    """
    Extract m(Omega) from pignistic probabilities.
    Handles different data structures (dict vs array).
    """
    if isinstance(pignistics, list):
        return np.array([
            p.get('Ω', p.get('omega', 0.35)) if isinstance(p, dict) else 0.35
            for p in pignistics
        ])
    return np.array([0.35] * len(pignistics))  # Fallback


def create_uncertainty_heatmap(results_dict, test_labels, fold_idx=0, outdir="Fig"):
    import os, numpy as np, matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from scipy.stats import pearsonr, spearmanr

    final_pignistics = results_dict['predictions']['final_pignistics']
    xgb_probs = results_dict['predictions']['xgb_probs']
    final_predictions = results_dict['predictions']['final']

    conf = xgb_probs.max(axis=1).astype(float)
    unc  = extract_dst_uncertainty(final_pignistics).astype(float)
    correct = (final_predictions == test_labels).astype(float)

    # Focus range
    conf_min = max(0.5, np.percentile(conf, 5))
    conf_max = min(1.0, np.percentile(conf, 95))
    unc_min  = max(0.2, np.percentile(unc, 5))
    unc_max  = min(0.6, np.percentile(unc, 95))

    conf_bins = np.linspace(conf_min, conf_max, 20)
    unc_bins  = np.linspace(unc_min,  unc_max,  20)

    H_correct = np.zeros((len(unc_bins)-1, len(conf_bins)-1))
    H_total   = np.zeros_like(H_correct)

    j = np.clip(np.searchsorted(conf_bins, conf, side="right")-1, 0, len(conf_bins)-2)
    i = np.clip(np.searchsorted(unc_bins,  unc,  side="right")-1, 0, len(unc_bins)-2)
    for ii, jj, ok in zip(i, j, correct):
        H_total[ii, jj]   += 1
        H_correct[ii, jj] += ok

    alpha = beta = 1.0
    acc_grid = (H_correct + alpha) / (H_total + alpha + beta)
    acc_grid[H_total == 0] = np.nan

    # --- Figure: main heatmap + marginal histograms + two colorbars on the right
    fig = plt.figure(figsize=(18.5, 8.6))
    import matplotlib.gridspec as mgs
    gs = mgs.GridSpec(
        3, 5, figure=fig,
        width_ratios=[0.8, 0.02, 0.02, 0.02, 0.02],  # big heatmap + 4 skinny cbar/space cols
        height_ratios=[0.22, 1.0, 0.22],             # top hist, heatmap, right hist
        wspace=0.30, hspace=0.30
    )

    # Axes
    axtop  = fig.add_subplot(gs[0, 0])   # top hist (confidence)
    axmain = fig.add_subplot(gs[1, 0])   # main heatmap
    axright= fig.add_subplot(gs[1, 4])   # right hist (uncertainty) rotated
    cax1   = fig.add_subplot(gs[1, 2])   # accuracy colorbar
    cax2   = fig.add_subplot(gs[1, 3])   # count colorbar (for contours)

    # Heatmap
    vmin, vcenter, vmax = 0.4, 0.85, 1.0
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    X_edges, Y_edges = conf_bins, unc_bins
    Xc = 0.5*(X_edges[:-1] + X_edges[1:])
    Yc = 0.5*(Y_edges[:-1] + Y_edges[1:])
    masked_acc = np.ma.masked_invalid(acc_grid)

    im = axmain.pcolormesh(X_edges, Y_edges, masked_acc, cmap='RdYlGn', norm=norm, shading='auto')

    # Isocount contours ON the heatmap (no second panel)
    with np.errstate(invalid='ignore'):
        Zlog = np.ma.masked_where(H_total == 0, np.log10(H_total + 1))
    if not np.ma.getmaskarray(Zlog).all():
        CS = axmain.contour(Xc, Yc, Zlog, levels=4, colors='k', linewidths=1.0, alpha=0.55)
        axmain.clabel(CS, inline=True, fmt=lambda v: f'{int(round(10**v-1))}', fontsize=9)

    # Policy quadrants
    unc_th  = float(np.median(unc))
    conf_th = float(np.percentile(conf, 70))
    axmain.axhline(unc_th, color='navy', linestyle='--', linewidth=2.0, alpha=0.7)
    axmain.axvline(conf_th, color='navy', linestyle='--', linewidth=2.0, alpha=0.7)

    # Quadrant labels
    x_high = 0.5*(conf_th + conf_max); x_low = 0.5*(conf_min + conf_th)
    y_high = 0.5*(unc_th + unc_max);   y_low = 0.5*(unc_min  + unc_th)
    style = dict(boxstyle='round,pad=0.6', edgecolor='black', linewidth=1.6)
    axmain.text(x_high, y_low,  'High confidence\nLow uncertainty\n→ Automate',
                ha='center', va='center', fontsize=11, fontweight='bold',
                bbox=dict(**style, facecolor='white', alpha=0.88))
    axmain.text(x_high, y_high, 'High confidence\nHigh uncertainty\n→ Review',
                ha='center', va='center', fontsize=11, fontweight='bold',
                bbox=dict(**style, facecolor='white', alpha=0.88))
    axmain.text(x_low,  y_high, 'Low confidence\nHigh uncertainty\n→ Expert review',
                ha='center', va='center', fontsize=11, fontweight='bold',
                bbox=dict(**style, facecolor='white', alpha=0.88))
    axmain.text(x_low,  y_low,  'Moderate confidence\n→ Manual check',
                ha='center', va='center', fontsize=11, fontweight='bold',
                bbox=dict(**style, facecolor='white', alpha=0.88))

    axmain.set_xlabel('XGBoost confidence (max probability)', fontsize=13, fontweight='bold')
    axmain.set_ylabel('DST uncertainty $m(\\Omega)$', fontsize=13, fontweight='bold')
    axmain.set_title('Prediction accuracy across confidence and uncertainty',
                     fontsize=15, fontweight='bold', pad=12)
    axmain.grid(True, alpha=0.15, linestyle=':', linewidth=0.5)

    # Top marginal histogram (confidence)
    axtop.hist(conf, bins=40, range=(conf_min, conf_max), alpha=0.7, edgecolor='white')
    axtop.set_xlim(conf_min, conf_max)
    axtop.set_xticklabels([]); axtop.set_yticklabels([])
    axtop.set_ylabel('Count', fontsize=9)
    axtop.grid(True, axis='y', alpha=0.2, linestyle=':')

    # Right marginal histogram (uncertainty) — rotate ticks
    axright.hist(unc, bins=40, range=(unc_min, unc_max), orientation='horizontal',
                 alpha=0.7, edgecolor='white')
    axright.set_ylim(unc_min, unc_max)
    axright.set_xticklabels([]); axright.set_yticklabels([])
    axright.set_xlabel('Count', fontsize=9)
    axright.grid(True, axis='x', alpha=0.2, linestyle=':')

    # Colorbars
    cb1 = fig.colorbar(im, cax=cax1)
    cb1.set_label('Prediction accuracy', fontsize=11)
    cb1.ax.tick_params(labelsize=9)

    # Count colorbar label (for contour units)
    sm = plt.cm.ScalarMappable(cmap='Blues')
    sm.set_array([])
    cb2 = fig.colorbar(sm, cax=cax2)
    cb2.set_label('Flight count (contour labels)', fontsize=10)
    cb2.ax.set_yticks([])

    # Orthogonality annotation
    r_p, p_p = pearsonr(conf, unc); r_s, p_s = spearmanr(conf, unc)
    axmain.text(0.015, 0.985,
                f'Orthogonality\nPearson r = {r_p:.3f}, p = {p_p:.4f}\nSpearman $\\rho$ = {r_s:.3f}, p = {p_s:.4f}',
                transform=axmain.transAxes, fontsize=10, fontweight='bold',
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.92,
                          edgecolor='black', linewidth=1.2))

    # Suptitle / save
    fig.suptitle(f'Uncertainty-aware decision landscape (Fold {fold_idx})',
                 fontsize=17, fontweight='bold', y=0.995)
    plt.tight_layout(rect=(0.02, 0.02, 0.98, 0.95))

    os.makedirs(outdir, exist_ok=True)
    fig.savefig(f'{outdir}/uncertainty_heatmap_fold_{fold_idx}.pdf', bbox_inches='tight')
    fig.savefig(f'{outdir}/uncertainty_heatmap_fold_{fold_idx}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)



def create_class_uncertainty_profiles(results_dict, test_labels, fold_idx=0):
    """
    Box plots + scatter showing uncertainty by class.
    Validates the "Late Flaps: rare but certain" insight.
    """
    final_pignistics = results_dict['predictions']['final_pignistics']
    final_predictions = results_dict['predictions']['final']

    dst_uncertainty = extract_dst_uncertainty(final_pignistics)

    class_names = ['Nominal', 'High Speed', 'High Path', 'Late Flaps']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # === LEFT: Box plots ===
    class_uncertainties = [dst_uncertainty[test_labels == c] for c in range(4)]

    bp = ax1.boxplot(
        class_uncertainties, labels=class_names,
        patch_artist=True, showmeans=True,
        boxprops=dict(facecolor='lightblue', alpha=0.7),
        medianprops=dict(color='red', linewidth=2),
        meanprops=dict(marker='D', markerfacecolor='green', markersize=8)
    )

    # Overlay jittered points
    for i, unc in enumerate(class_uncertainties):
        if len(unc) == 0:
            continue
        y = unc
        x = np.random.normal(i + 1, 0.04, size=len(y))
        is_correct = (final_predictions[test_labels == i] == i)
        colors = ['green' if c else 'red' for c in is_correct]
        ax1.scatter(x, y, alpha=0.3, s=20, c=colors, edgecolors='black', linewidth=0.3)

    # Add stats
    for i in range(4):
        mask = test_labels == i
        if mask.sum() == 0:
            continue
        acc = (final_predictions[mask] == i).mean()
        n = mask.sum()
        mean_unc = dst_uncertainty[mask].mean()
        ax1.text(
            i + 1, 0.55, f'Acc: {acc:.1%}\nn={n}\nμ={mean_unc:.2f}',
            ha='center', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

    ax1.set_ylabel('DST Uncertainty m(Ω)', fontsize=12)
    ax1.set_title('Uncertainty Distribution by True Class\n(Green=Correct, Red=Incorrect)',
                  fontsize=14)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0.2, 0.6])

    # === RIGHT: Uncertainty vs Accuracy ===
    class_accs = [(final_predictions[test_labels == c] == c).mean() for c in range(4) if (test_labels == c).sum() > 0]
    class_unc_means = [dst_uncertainty[test_labels == c].mean() for c in range(4) if (test_labels == c).sum() > 0]
    class_sizes = [np.sum(test_labels == c) for c in range(4) if (test_labels == c).sum() > 0]

    scatter = ax2.scatter(
        class_unc_means, class_accs,
        s=[np.sqrt(n) * 50 for n in class_sizes],
        c=['blue', 'orange', 'green', 'red'][:len(class_accs)], alpha=0.7,
        edgecolors='black', linewidths=2
    )

    # Annotate
    for i, name in enumerate([class_names[c] for c in range(4) if (test_labels == c).sum() > 0]):
        ax2.annotate(
            f'{name}\n({class_sizes[i]} flights)',
            (class_unc_means[i], class_accs[i]),
            xytext=(10, 10), textcoords='offset points',
            fontsize=10, ha='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3')
        )

    # Correlation
    # Trend line only (n=4 is too small for meaningful r/p)
    if len(class_unc_means) >= 2:
        z = np.polyfit(class_unc_means, class_accs, 1)
        pfit = np.poly1d(z)
        x_line = np.linspace(min(class_unc_means), max(class_unc_means), 100)
        ax2.plot(x_line, pfit(x_line), 'r--', linewidth=2, alpha=0.8,
                 label='trend only (n=4)')

    ax2.set_xlabel('Mean DST Uncertainty m(Ω)', fontsize=12)
    ax2.set_ylabel('Classification Accuracy', fontsize=12)
    ax2.set_title('Uncertainty Reflects Pattern Distinctiveness, Not Rarity\n(Late Flaps: Rare but Certain)',
                  fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if not os.path.exists('Fig'):
        os.makedirs('Fig')
    plt.savefig(f'Fig/class_uncertainty_profiles_fold_{fold_idx}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'Fig/class_uncertainty_profiles_fold_{fold_idx}.pdf', bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Generated class uncertainty profiles for fold {fold_idx}")

    return None


def plot_reliability_slices(m_omega, xgb_probs, y_true, y_pred, save_path='Fig/reliability_slices.pdf'):
    """
    Show-stopper: Accuracy vs confidence within uncertainty quartiles.
    Demonstrates complementary (orthogonal) signals without relying on a 2D heatmap.

    - x-axis: confidence deciles (XGBoost max prob)
    - curves: uncertainty quartiles (m(Omega))
    - y-axis: empirical accuracy with Beta-smoothed estimates for stability
    """
    import numpy as np
    import matplotlib.pyplot as plt

    conf = xgb_probs.max(axis=1).astype(float)
    unc  = m_omega.astype(float)
    correct = (y_true == y_pred).astype(float)

    # Define uncertainty quartiles
    uq = np.percentile(unc, [0, 25, 50, 75, 100])
    quart_masks = []
    for i in range(4):
        if i == 3:
            mask = (unc >= uq[i]) & (unc <= uq[i+1])
        else:
            mask = (unc >= uq[i]) & (unc < uq[i+1])
        quart_masks.append(mask)

    # Confidence deciles (within overall range, not per quartile)
    cb = np.percentile(conf, np.arange(0, 101, 10))
    centers = 0.5 * (cb[:-1] + cb[1:])

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']  # color-blind tolerable palette
    labels = [f'Q{i+1}  [{uq[i]:.2f}, {uq[i+1]:.2f}]' for i in range(4)]

    for qi, qmask in enumerate(quart_masks):
        acc, lo, hi = [], [], []
        for j in range(len(cb)-1):
            m = qmask & (conf >= cb[j]) & (conf < cb[j+1])
            n = m.sum()
            if n == 0:
                acc.append(np.nan); lo.append(np.nan); hi.append(np.nan)
                continue
            k = correct[m].sum()
            # Beta(1,1) smoothing for stability in sparse bins
            mean = (k + 1) / (n + 2)
            # Approx 95% Wilson interval (good for proportions)
            p = mean
            z = 1.96
            denom = 1 + z**2/n
            center = (p + z**2/(2*n)) / denom
            halfwidth = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / denom
            acc.append(center)
            lo.append(max(0.0, center - halfwidth))
            hi.append(min(1.0, center + halfwidth))

        ax.plot(centers, acc, marker='o', linewidth=2, markersize=5, color=colors[qi], label=labels[qi])
        ax.fill_between(centers, lo, hi, alpha=0.15, color=colors[qi])

    ax.set_xlabel('XGBoost confidence (max probability)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prediction accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Reliability by Uncertainty Quartile\n(Confidence complements DST uncertainty)', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, linestyle=':')
    ax.set_ylim(0.4, 1.01)
    leg = ax.legend(title='m(Ω) quartiles', framealpha=0.9)
    leg.get_title().set_fontweight('bold')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
